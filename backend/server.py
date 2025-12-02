import os
import re
import sys
import json
import time
import uuid
import queue
import torch
import uvicorn
import asyncio
import aiohttp
import logging
import requests
import threading
import numpy as np
import stream2sentence as s2s
from datetime import datetime
from pydantic import BaseModel
from queue import Queue, Empty
from openai import AsyncOpenAI
from collections import defaultdict
from threading import Thread, Event, Lock
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from supabase import create_client, Client
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from typing import Optional, Dict, List, Union, Any, AsyncIterator, AsyncGenerator
from loguru import logger
from enum import Enum

from RealtimeSTT import AudioToTextRecorder
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
from backend.RealtimeTTS.threadsafe_generators import CharIterator, AccumulatingThreadSafeGenerator

from character_manager import (
    CharacterManager,
    Character as DbCharacter,
    CharacterCreate,
    CharacterUpdate
)
from voice_manager import (
    VoiceManager,
    Voice as DbVoice,
    VoiceCreate,
    VoiceUpdate
)
from conversation_manager import (
    ConversationManager,
    Conversation as DbConversation,
    ConversationCreate,
    ConversationUpdate
)
from message_manager import (
    MessageManager,
    Message as DbMessage,
    MessageCreate
)

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jslevsbvapopncjehhva.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpzbGV2c2J2YXBvcG5jamVoaHZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgwNTQwOTMsImV4cCI6MjA3MzYzMDA5M30.DotbJM3IrvdVzwfScxOtsSpxq0xsj7XxI3DvdiqDSrE")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

logging.basicConfig(filename="filelogger.log", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sys.path.append('/workspace/tts/Code')

character_manager = CharacterManager(supabase)
voice_manager = VoiceManager(supabase)
conversation_manager = ConversationManager(supabase)
message_manager = MessageManager(supabase)

########################################
##--           Data Models          --##
########################################

@dataclass
class Character:
    """
    Character/Persona from Supabase characters table.

    Database schema mapping:
    - characters.id → id
    - characters.name → name
    - characters.voice → voice (references voices.voice)
    - characters.system_prompt → system_prompt
    """
    id: str                     # id from DB
    name: str                   # name from DB
    system_prompt: str          # Character's personality/instructions
    voice: str                  # References Voice.voice (from 'voice' field in DB)
    is_active: bool = True
    image_url: str = ""
    images: List[str] = field(default_factory=list)

    @classmethod
    def from_db(cls, row: Dict[str, Any]) -> 'Character':
        """Create from Supabase row"""
        return cls(
            id=row['id'],
            name=row['name'],
            system_prompt=row['system_prompt'],
            voice=row['voice'],  # DB field 'voice' references voices.voice
            is_active=row.get('is_active', False),
            image_url=row.get('image_url', ''),
            images=row.get('images', [])
        )

@dataclass
class Voice:
    """
    Voice configuration from Supabase voices table.

    We use the DESCRIPTION METHOD ONLY:
    - speaker_desc: "feminine; warm; professional"
    - scene_prompt: "Audio is recorded from a quiet room"

    Voice cloning (audio_path + text_path) is for future use.
    """
    voice: str                  # voice from DB (serves as both PK and name)
    method: str                 # "description" or "clone"
    speaker_desc: str           # For description method
    scene_prompt: str           # For description method
    audio_path: str = ""        # For clone method (future)
    text_path: str = ""         # For clone method (future)

    @classmethod
    def from_db(cls, row: Dict[str, Any]) -> 'Voice':
        """Create from Supabase row"""
        return cls(
            voice=row['voice'],
            method=row.get('method', 'description'),
            speaker_desc=row.get('speaker_desc', ''),
            scene_prompt=row.get('scene_prompt', ''),
            audio_path=row.get('audio_path', ''),
            text_path=row.get('text_path', '')
        )

    def get_higgs_system_prompt(self) -> str:
        """Build system prompt for Higgs TTS using description method"""
        parts = ["Generate audio following instruction.\n\n<|scene_desc_start|>"]

        if self.scene_prompt:
            parts.append(self.scene_prompt)

        if self.speaker_desc:
            parts.append(f"\n{self.speaker_desc}")

        parts.append("\n<|scene_desc_end|>")
        return "".join(parts)

@dataclass
class ConversationMessage:
    """Single message in conversation history"""
    role: str                   # "user", "assistant", "system"
    content: str
    character_id: Optional[str] = None
    name: Optional[str] = None  # Speaker name (User, or character name)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ConversationContext:
    """
    Complete conversation state that flows through the pipeline.
    Gets enriched at each service stage.
    """
    conversation_id: str
    active_characters: List[Character] = field(default_factory=list)
    history: List[ConversationMessage] = field(default_factory=list)

    # Populated by LLM Orchestrator
    response_queue: List[Character] = field(default_factory=list)
    current_speaker: Optional[Character] = None
    user_input: str = ""

    def add_user_message(self, text: str):
        """Add user message to history"""
        self.history.append(ConversationMessage(
            role="user",
            content=text,
            name="User"
        ))

    def add_character_message(self, character: Character, text: str):
        """Add character response to history"""
        self.history.append(ConversationMessage(
            role="assistant",
            content=text,
            character_id=character.id,
            name=character.name
        ))

########################################
##--          Data Classes          --##
########################################

@dataclass
class AudioChunk:
    """Raw PCM audio from browser → STT"""
    data: bytes                 # PCM16, mono
    sample_rate: int            # 16000 Hz (for STT input)
    timestamp: float

@dataclass
class Transcription:
    """Transcribed text from STT → LLM Orchestrator"""
    text: str
    is_final: bool
    context: Optional[ConversationContext]
    timestamp: float

@dataclass
class TextChunk:
    """Text chunk from LLM Orchestrator → Browser UI"""
    text: str
    is_final: bool
    speaker_id: str
    speaker_name: str
    sequence_number: int        # Which character's turn (0, 1, 2...)
    chunk_index: int
    timestamp: float

@dataclass
class TTSRequest:
    """TTS request from LLM → TTS Service"""
    text: str
    is_final: bool
    speaker: Character
    voice: Voice
    sequence_number: int
    timestamp: float

@dataclass
class SequencedAudioChunk:
    """Audio chunk from TTS → Audio Sequencer → Browser"""
    data: bytes                 # PCM16, mono, 24kHz
    sample_rate: int
    sequence_number: int        # Which character's turn
    character_chunk_index: int  # Chunk number within this character's audio
    is_final: bool
    speaker_id: str
    speaker_name: str
    timestamp: float

########################################
##--          Queue Manager         --##
########################################

class QueueManager:
    """Manages all inter-service queues and control signals"""

    def __init__(self):
        # Audio from browser → STT
        self.audio_input = asyncio.Queue()

        # STT → LLM Orchestrator
        self.transcription = asyncio.Queue()

        # LLM → Browser (for text display)
        self.text_output = asyncio.Queue()

        # LLM → TTS (with speaker info and sequence)
        self.tts_requests = asyncio.Queue()

        # TTS → Audio Sequencer
        self.audio_output = asyncio.Queue()

        # Control signals
        self.stop_signal = asyncio.Event()
        self.interrupt_signal = asyncio.Event()

    async def clear_all(self):
        """Clear all queues (on interruption)"""
        for q in [self.audio_input, self.transcription, self.text_output,
                  self.tts_requests, self.audio_output]:
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def interrupt(self):
        """Signal interrupt - stops all pending generation"""
        self.interrupt_signal.set()
        await self.clear_all()
        await asyncio.sleep(0.1)
        self.interrupt_signal.clear()

########################################
##--        Character Service       --##
########################################

class CharacterService:
    """
    Character management service - handles character state, turn-taking,
    voice assignment, and prompt building.

    This is NOT an async service - it's a helper used by other services.
    """

    def __init__(self, char_mgr: CharacterManager = None, voice_mgr: VoiceManager = None):
        """
        Args:
            char_mgr: Supabase CharacterManager instance
            voice_mgr: Supabase VoiceManager instance
        """
        self.character_manager = char_mgr if char_mgr else character_manager
        self.voice_manager = voice_mgr if voice_mgr else voice_manager
        self._character_cache: Dict[str, Character] = {}
        self._voice_cache: Dict[str, Voice] = {}

    async def load_active_characters(self, character_ids: List[str]) -> List[Character]:
        """Load characters from database"""
        characters = []
        for char_id in character_ids:
            if char_id in self._character_cache:
                characters.append(self._character_cache[char_id])
            else:
                # Fetch from DB
                if self.character_manager:
                    db_char = await self.character_manager.get_character(char_id)
                    if db_char:
                        character = Character.from_db(db_char.dict())
                        self._character_cache[char_id] = character
                        characters.append(character)
        return characters

    async def get_voice_for_character(self, character: Character) -> Optional[Voice]:
        """Get voice configuration for character"""
        if character.voice in self._voice_cache:
            return self._voice_cache[character.voice]

        if self.voice_manager:
            db_voice = await self.voice_manager.get_voice(character.voice)
            if db_voice:
                voice = Voice.from_db(db_voice.dict())
                self._voice_cache[character.voice] = voice
                return voice
        return None

    def parse_character_mentions(
        self,
        text: str,
        active_characters: List[Character]
    ) -> List[Character]:
        """
        Parse user input to determine which characters are mentioned.
        Returns characters in order of first mention.

        Example:
            "Hey Sarah and Mike, how are you?" → [Sarah, Mike]
            "Mike and Sarah, what do you think?" → [Mike, Sarah]
            "Hello there" → [Sarah, Mike] (all characters, alphabetically)
        """
        mentioned_characters = []
        processed_ids = set()

        # Track all name mentions with positions
        name_mentions = []

        for character in active_characters:
            # Handle multi-word names like "Sarah Johnson"
            name_parts = character.name.lower().split()

            for name_part in name_parts:
                # Find all occurrences of this name part
                pattern = r'\b' + re.escape(name_part) + r'\b'
                for match in re.finditer(pattern, text.lower()):
                    name_mentions.append({
                        'character': character,
                        'position': match.start(),
                        'name_part': name_part
                    })

        # Sort by position in text
        name_mentions.sort(key=lambda x: x['position'])

        # Add characters in order of first mention, avoiding duplicates
        for mention in name_mentions:
            char = mention['character']
            if char.id not in processed_ids:
                mentioned_characters.append(char)
                processed_ids.add(char.id)

        # If no one mentioned, all active characters respond (alphabetically)
        if not mentioned_characters:
            mentioned_characters = sorted(active_characters, key=lambda c: c.name)

        return mentioned_characters

    def build_character_prompt(
        self,
        character: Character,
        conversation_history: List[ConversationMessage]
    ) -> List[Dict[str, str]]:
        """
        Build LLM prompt for a specific character.

        Returns messages array for OpenRouter/OpenAI API format.
        """
        messages = []

        # Add character's system prompt
        messages.append({
            "role": "system",
            "content": character.system_prompt
        })

        # Add conversation history
        for msg in conversation_history:
            message_dict = {
                "role": msg.role,
                "content": msg.content
            }
            # Add name field if available (for multi-character conversations)
            if msg.name:
                message_dict["name"] = msg.name
            messages.append(message_dict)

        # Add instruction for this turn
        instruction = (
            f"Based on the conversation history above, provide the next reply as {character.name}. "
            f"Your response should include only {character.name}'s reply. "
            f"Do not respond for/as anyone else. "
            f"Wrap your entire response in <{character.name}></{character.name}> tags."
        )

        messages.append({
            "role": "system",
            "content": instruction
        })

        return messages

    def extract_character_response(
        self,
        raw_response: str,
        character: Character
    ) -> str:
        """
        Extract character response from tagged LLM output.

        Example:
            Input: "<Sarah>I'm doing great, thanks for asking!</Sarah>"
            Output: "I'm doing great, thanks for asking!"
        """
        # Try to extract from tags
        pattern = f"<{character.name}>(.*?)</{character.name}>"
        match = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()

        # Fallback: return raw response (LLM didn't follow instructions)
        return raw_response.strip()

########################################
##--           STT Service          --##
########################################

class STTService:
    """
    Speech-to-Text Service using RealtimeSTT.
    Consumes audio chunks → produces transcriptions.
    """

    def __init__(self, queue_manager: QueueManager, model: str = "base.en"):
        self.queues = queue_manager
        self.model = model
        self.recorder = None
        self.current_context: Optional[ConversationContext] = None

    async def initialize(self):
        """Initialize STT recorder (runs in thread pool)"""
        loop = asyncio.get_event_loop()
        self.recorder = await loop.run_in_executor(None, self._create_recorder)

    def _create_recorder(self):
        """Create AudioToTextRecorder (blocking operation)"""
        return AudioToTextRecorder(
            model=self.model,
            language="en",
            enable_realtime_transcription=True,
            realtime_processing_pause=0.1,
            on_realtime_transcription_update=self._on_partial,
            on_transcription_complete=self._on_complete,
        )

    def _on_partial(self, text: str):
        """Callback for partial transcriptions"""
        # Can be used for live captions in UI
        pass

    def _on_complete(self, text: str):
        """Callback for final transcriptions"""
        transcription = Transcription(
            text=text,
            is_final=True,
            context=self.current_context,
            timestamp=time.time()
        )
        try:
            self.queues.transcription.put_nowait(transcription)
        except asyncio.QueueFull:
            print("Warning: Transcription queue full")

    def set_context(self, context: ConversationContext):
        """Set current conversation context"""
        self.current_context = context

    async def run(self):
        """Main service loop: consume audio chunks, feed to recorder"""
        await self.initialize()
        print("✓ STT Service initialized")

        while not self.queues.stop_signal.is_set():
            try:
                # Get audio chunk from input queue
                chunk: AudioChunk = await asyncio.wait_for(
                    self.queues.audio_input.get(),
                    timeout=0.1
                )

                # Feed to recorder (runs in thread pool to avoid blocking)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self.recorder.feed_audio,
                    chunk.data,
                    chunk.sample_rate
                )

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"STT Service error: {e}")
                continue

########################################
##--         LLM Orchestrator       --##
########################################

class LLMOrchestrator:
    """
    LLM Orchestrator - Manages multi-character turn-taking with concurrent processing.

    Key behavior:
    - Characters respond in mention order
    - Character N+1 starts LLM generation as soon as Character N's TEXT completes
    - Does NOT wait for Character N's audio to finish
    - Streams text to UI and TTS concurrently
    """

    def __init__(
        self,
        queue_manager: QueueManager,
        character_service: CharacterService,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet"
    ):
        self.queues = queue_manager
        self.character_service = character_service
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model

    async def run(self):
        """Main loop: consume transcriptions, orchestrate character responses"""
        print("✓ LLM Orchestrator initialized")

        while not self.queues.stop_signal.is_set():
            try:
                # Get transcription
                transcription: Transcription = await asyncio.wait_for(
                    self.queues.transcription.get(),
                    timeout=0.1
                )

                if not transcription.is_final:
                    continue

                context = transcription.context
                if not context:
                    continue

                # Update context
                context.user_input = transcription.text
                context.add_user_message(transcription.text)

                # Save user message to database
                try:
                    await message_manager.create_message(MessageCreate(
                        conversation_id=context.conversation_id,
                        role="user",
                        content=transcription.text,
                        name="User",
                        character_id=None
                    ))

                    # Auto-update conversation title from first message if needed
                    if len(context.history) == 1:  # First message
                        await conversation_manager.auto_update_title_from_first_message(
                            context.conversation_id,
                            transcription.text
                        )
                except Exception as e:
                    print(f"Error saving user message: {e}")

                # Determine response queue (who responds)
                response_queue = self.character_service.parse_character_mentions(
                    transcription.text,
                    context.active_characters
                )
                context.response_queue = response_queue

                print(f"Response queue: {[c.name for c in response_queue]}")

                # Generate responses for each character
                await self._orchestrate_character_responses(context, response_queue)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"LLM Orchestrator error: {e}")
                continue

    async def _orchestrate_character_responses(
        self,
        context: ConversationContext,
        response_queue: List[Character]
    ):
        """
        Orchestrate character responses with proper sequencing.

        Each character:
        1. Waits for previous character's TEXT to complete
        2. Generates LLM response (streaming)
        3. Streams text to UI
        4. Sends complete sentences to TTS
        5. Completes and triggers next character
        """
        for sequence_number, character in enumerate(response_queue):
            # Check for interrupt
            if self.queues.interrupt_signal.is_set():
                print(f"Interrupt detected, stopping at character {character.name}")
                break

            # Generate this character's response
            await self._generate_character_response(
                context,
                character,
                sequence_number
            )

            # Brief pause between characters (optional, for natural flow)
            await asyncio.sleep(0.05)

    async def _generate_character_response(
        self,
        context: ConversationContext,
        character: Character,
        sequence_number: int
    ):
        """Generate streaming response for a single character"""

        print(f"🎭 Generating response for {character.name} (seq {sequence_number})")

        # Build character-specific prompt
        messages = self.character_service.build_character_prompt(
            character,
            context.history
        )

        # Track response
        raw_response = ""
        text_buffer = ""
        chunk_index = 0

        try:
            # Stream LLM response
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )

            async for chunk in stream:
                # Check for interrupt
                if self.queues.interrupt_signal.is_set():
                    break

                content = chunk.choices[0].delta.content
                if content:
                    raw_response += content
                    text_buffer += content

                    # Stream text to UI (every chunk)
                    text_chunk = TextChunk(
                        text=content,
                        is_final=False,
                        speaker_id=character.id,
                        speaker_name=character.name,
                        sequence_number=sequence_number,
                        chunk_index=chunk_index,
                        timestamp=time.time()
                    )
                    await self.queues.text_output.put(text_chunk)
                    chunk_index += 1

                    # Send to TTS when we have a reasonable chunk
                    if self._should_send_to_tts(text_buffer):
                        voice = await self.character_service.get_voice_for_character(character)

                        tts_request = TTSRequest(
                            text=text_buffer,
                            is_final=False,
                            speaker=character,
                            voice=voice,
                            sequence_number=sequence_number,
                            timestamp=time.time()
                        )
                        await self.queues.tts_requests.put(tts_request)
                        text_buffer = ""

            # Send final chunk to UI
            final_text_chunk = TextChunk(
                text="",
                is_final=True,
                speaker_id=character.id,
                speaker_name=character.name,
                sequence_number=sequence_number,
                chunk_index=chunk_index,
                timestamp=time.time()
            )
            await self.queues.text_output.put(final_text_chunk)

            # Send remaining text to TTS
            if text_buffer.strip():
                voice = await self.character_service.get_voice_for_character(character)
                tts_request = TTSRequest(
                    text=text_buffer,
                    is_final=True,
                    speaker=character,
                    voice=voice,
                    sequence_number=sequence_number,
                    timestamp=time.time()
                )
                await self.queues.tts_requests.put(tts_request)

            # Extract clean response for TTS/UI, but save raw response with tags to history
            clean_response = self.character_service.extract_character_response(
                raw_response,
                character
            )

            # Add raw response (with character tags) to history
            context.add_character_message(character, raw_response)

            # Save raw response (with character tags) to database
            try:
                await message_manager.create_message(MessageCreate(
                    conversation_id=context.conversation_id,
                    role="assistant",
                    content=raw_response,
                    name=character.name,
                    character_id=character.id
                ))
            except Exception as e:
                print(f"Error saving character message: {e}")

            print(f"✓ {character.name} text generation complete")

        except Exception as e:
            print(f"Error generating response for {character.name}: {e}")

    def _should_send_to_tts(self, text_buffer: str) -> bool:
        """
        Determine if we should send current buffer to TTS.

        Send when:
        - Buffer ends with sentence terminator (. ! ?)
        - Buffer reaches length threshold (50 chars)
        """
        if not text_buffer:
            return False

        # Check for sentence endings
        if text_buffer.rstrip().endswith(('.', '!', '?', '...')):
            return True

        # Check length threshold
        if len(text_buffer) >= 50:
            return True

        return False

########################################
##--           TTS Service          --##
########################################

class HiggsTTSService:
    """
    TTS service using Higgs Audio with concurrent multi-character processing.

    Key behavior:
    - Multiple TTS generation tasks run in parallel
    - Audio chunks tagged with sequence_number for proper playback order
    - Each character's audio generated independently
    """

    def __init__(
        self,
        queue_manager: QueueManager,
        model_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
        tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer",
        device: str = "cuda",
        chunk_size: int = 64,
        max_concurrent_generations: int = 3
    ):
        self.queues = queue_manager
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.chunk_size = chunk_size
        self.serve_engine = None

        # Track ongoing TTS tasks per character
        self.character_tts_tasks = {}

        # Semaphore to limit concurrent generations (avoid GPU OOM)
        self.generation_semaphore = asyncio.Semaphore(max_concurrent_generations)

    async def initialize(self):
        """Initialize Higgs engine (blocking operation)"""
        loop = asyncio.get_event_loop()
        self.serve_engine = await loop.run_in_executor(None, self._create_engine)

    def _create_engine(self):
        """Create HiggsAudioServeEngine"""
        print("Loading Higgs Audio engine...")
        return HiggsAudioServeEngine(
            self.model_path,
            self.tokenizer_path,
            device=self.device,
            torch_dtype=torch.bfloat16
        )

    async def run(self):
        """Main loop: consume TTS requests, spawn concurrent generation tasks"""
        await self.initialize()
        print("✓ Higgs TTS Service initialized")

        # Track text buffers per (speaker, sequence_number)
        text_buffers = defaultdict(str)

        while not self.queues.stop_signal.is_set():
            try:
                # Get TTS request
                request: TTSRequest = await asyncio.wait_for(
                    self.queues.tts_requests.get(),
                    timeout=0.1
                )

                # Check for interrupt
                if self.queues.interrupt_signal.is_set():
                    # Cancel all ongoing TTS tasks
                    for task in self.character_tts_tasks.values():
                        task.cancel()
                    self.character_tts_tasks.clear()
                    text_buffers.clear()
                    continue

                # Buffer text for this character
                buffer_key = (request.speaker.id, request.sequence_number)
                text_buffers[buffer_key] += request.text

                # Generate audio when buffer is ready or final
                if request.is_final and text_buffers[buffer_key].strip():
                    # Spawn TTS generation task
                    task = asyncio.create_task(
                        self._generate_audio_task(
                            text_buffers[buffer_key],
                            request.speaker,
                            request.voice,
                            request.sequence_number
                        )
                    )

                    task_key = (request.speaker.id, request.sequence_number)
                    self.character_tts_tasks[task_key] = task

                    # Clear buffer
                    text_buffers[buffer_key] = ""

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"TTS Service error: {e}")
                continue

    async def _generate_audio_task(
        self,
        text: str,
        character: Character,
        voice: Voice,
        sequence_number: int
    ):
        """
        Generate audio for a character's text (runs as concurrent task).
        Uses semaphore to limit concurrent GPU usage.
        """
        async with self.generation_semaphore:
            print(f"🎤 Starting TTS for {character.name} (seq {sequence_number})")

            # Build system prompt using voice description
            system_prompt = voice.get_higgs_system_prompt() if voice else (
                "Generate audio following instruction.\n\n"
                "<|scene_desc_start|>\n"
                "Audio is recorded from a quiet room.\n"
                "<|scene_desc_end|>"
            )

            # Prepare messages
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=text)
            ]

            # Generate audio tokens
            audio_token_buffer = []
            character_chunk_index = 0

            try:
                # Create streamer
                streamer = self.serve_engine.generate_delta_stream(
                    chat_ml_sample=ChatMLSample(messages=messages),
                    temperature=0.75,
                    top_p=0.95,
                    top_k=50,
                    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                    force_audio_gen=True
                )

                # Stream audio tokens
                async for delta in streamer:
                    # Check for interrupt
                    if self.queues.interrupt_signal.is_set():
                        print(f"⚠️  Interrupted TTS for {character.name}")
                        break

                    if delta.audio_tokens is not None:
                        audio_token_buffer.append(delta.audio_tokens)

                        # Process chunk when buffer is full
                        if len(audio_token_buffer) >= self.chunk_size:
                            pcm_chunk = await self._process_audio_tokens(
                                audio_token_buffer[:self.chunk_size]
                            )

                            if pcm_chunk is not None:
                                # Create sequenced audio chunk
                                audio_chunk = SequencedAudioChunk(
                                    data=pcm_chunk,
                                    sample_rate=self.serve_engine.audio_tokenizer.sampling_rate,
                                    sequence_number=sequence_number,
                                    character_chunk_index=character_chunk_index,
                                    is_final=False,
                                    speaker_id=character.id,
                                    speaker_name=character.name,
                                    timestamp=time.time()
                                )
                                await self.queues.audio_output.put(audio_chunk)
                                character_chunk_index += 1

                            # Keep overlap for continuity
                            num_codebooks = delta.audio_tokens.shape[0]
                            tokens_to_keep = num_codebooks - 1
                            audio_token_buffer = audio_token_buffer[
                                self.chunk_size - tokens_to_keep:
                            ]

                    if delta.text == "<|eot_id|>":
                        break

                # Process remaining tokens
                if audio_token_buffer and not self.queues.interrupt_signal.is_set():
                    pcm_chunk = await self._process_audio_tokens(audio_token_buffer)
                    if pcm_chunk is not None:
                        audio_chunk = SequencedAudioChunk(
                            data=pcm_chunk,
                            sample_rate=self.serve_engine.audio_tokenizer.sampling_rate,
                            sequence_number=sequence_number,
                            character_chunk_index=character_chunk_index,
                            is_final=True,
                            speaker_id=character.id,
                            speaker_name=character.name,
                            timestamp=time.time()
                        )
                        await self.queues.audio_output.put(audio_chunk)

                print(f"✓ TTS complete for {character.name} (seq {sequence_number})")

            except asyncio.CancelledError:
                print(f"⚠️  TTS cancelled for {character.name}")
            except Exception as e:
                print(f"Error in TTS for {character.name}: {e}")

    async def _process_audio_tokens(self, tokens):
        """Convert audio tokens to PCM (run in executor to avoid blocking)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._decode_tokens, tokens)

    def _decode_tokens(self, tokens):
        """Decode audio tokens to PCM16 (blocking operation)"""
        try:
            audio_chunk = torch.stack(tokens, dim=1)
            vq_code = revert_delay_pattern(audio_chunk).clip(
                0, self.serve_engine.audio_codebook_size - 1
            )
            waveform = self.serve_engine.audio_tokenizer.decode(
                vq_code.unsqueeze(0)
            )[0, 0]
            pcm_data = (waveform * 32767).astype(np.int16)
            return pcm_data.tobytes()
        except Exception as e:
            print(f"Token decode error: {e}")
            return None

########################################
##--          Audio Playback        --##
########################################

class AudioPlaybackSequencer:
    """
    Buffers audio chunks and ensures they're sent to browser in correct sequence.

    Key behavior:
    - Buffers chunks from concurrent TTS generations
    - Sends to browser in sequence_number order
    - Waits for Character 0's audio before Character 1's, etc.
    """

    def __init__(self, queue_manager: QueueManager, websocket: WebSocket):
        self.queues = queue_manager
        self.websocket = websocket

        # Buffer chunks by sequence_number
        self.chunk_buffers = defaultdict(list)

        # Track which sequence we're currently playing
        self.current_sequence = 0

        # Track if each sequence is complete
        self.sequence_complete = {}

    async def run(self):
        """Main loop: consume audio chunks, sequence them, send to browser"""

        # Send metadata first
        metadata = {
            "type": "metadata",
            "sample_rate": 24000,
            "channels": 1,
            "format": "int16"
        }
        await self.websocket.send_json(metadata)

        print("✓ Audio Sequencer initialized")

        while not self.queues.stop_signal.is_set():
            try:
                # Get audio chunk
                chunk: SequencedAudioChunk = await asyncio.wait_for(
                    self.queues.audio_output.get(),
                    timeout=0.1
                )

                # Check for interrupt
                if self.queues.interrupt_signal.is_set():
                    # Clear all buffers
                    self.chunk_buffers.clear()
                    self.sequence_complete.clear()
                    self.current_sequence = 0

                    # Send interrupt signal to browser
                    await self.websocket.send_json({
                        "type": "interrupt",
                        "message": "Playback interrupted"
                    })
                    continue

                # Buffer this chunk
                self.chunk_buffers[chunk.sequence_number].append(chunk)

                # Mark sequence as complete if this is the final chunk
                if chunk.is_final:
                    self.sequence_complete[chunk.sequence_number] = True

                # Send chunks in sequence order
                await self._send_sequenced_chunks()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Audio Sequencer error: {e}")
                continue

    async def _send_sequenced_chunks(self):
        """Send buffered chunks in sequence order"""

        while self.current_sequence in self.chunk_buffers:
            chunks = self.chunk_buffers[self.current_sequence]

            if not chunks:
                break

            # Sort chunks by character_chunk_index
            chunks.sort(key=lambda c: c.character_chunk_index)

            # Send all available chunks for current sequence
            for chunk in chunks:
                # Send speaker change notification (first chunk of this sequence)
                if chunk.character_chunk_index == 0:
                    await self.websocket.send_json({
                        "type": "speaker_change",
                        "speaker_name": chunk.speaker_name,
                        "speaker_id": chunk.speaker_id,
                        "sequence_number": chunk.sequence_number
                    })

                # Send audio data
                await self.websocket.send_bytes(chunk.data)

            # Clear sent chunks
            self.chunk_buffers[self.current_sequence] = []

            # Move to next sequence if current is complete
            if self.sequence_complete.get(self.current_sequence, False):
                # Send end marker for this character
                await self.websocket.send_json({
                    "type": "character_end",
                    "sequence_number": self.current_sequence
                })

                # Move to next sequence
                self.current_sequence += 1
            else:
                # Wait for more chunks from current sequence
                break

########################################
##--        WebSocket Handler       --##
########################################

class VoiceChatOrchestrator:
    """
    Main orchestrator that manages all services and WebSocket connections.
    Creates a new instance per WebSocket connection.
    """

    def __init__(self, config: dict):
        self.config = config
        self.queues = QueueManager()

        # Initialize character service
        # TODO: Wire up real CharacterManager and VoiceManager from Supabase
        self.character_service = CharacterService()

        # Initialize services
        self.stt = STTService(
            self.queues,
            model=config.get("stt_model", "base.en")
        )

        self.llm = LLMOrchestrator(
            self.queues,
            self.character_service,
            api_key=config["openrouter_api_key"],
            model=config.get("llm_model", "anthropic/claude-3.5-sonnet")
        )

        self.tts = HiggsTTSService(
            self.queues,
            model_path=config.get(
                "higgs_model_path",
                "bosonai/higgs-audio-v2-generation-3B-base"
            ),
            tokenizer_path=config.get(
                "higgs_tokenizer_path",
                "bosonai/higgs-audio-v2-tokenizer"
            ),
            device=config.get("device", "cuda"),
            max_concurrent_generations=config.get("max_concurrent_tts", 3)
        )

        self.service_tasks = []
        self.context: Optional[ConversationContext] = None

    async def start_services(self, websocket: WebSocket):
        """Start all services including audio sequencer"""

        # Create audio sequencer with websocket
        audio_sequencer = AudioPlaybackSequencer(self.queues, websocket)

        # Start all service tasks
        self.service_tasks = [
            asyncio.create_task(self.stt.run()),
            asyncio.create_task(self.llm.run()),
            asyncio.create_task(self.tts.run()),
            asyncio.create_task(audio_sequencer.run()),
            asyncio.create_task(self._stream_text_to_browser(websocket))
        ]

        print("✓ All services started")

    async def stop_services(self):
        """Stop all services gracefully"""
        print("Stopping services...")
        self.queues.stop_signal.set()

        # Cancel all tasks
        for task in self.service_tasks:
            task.cancel()

        # Wait for cancellation
        await asyncio.gather(*self.service_tasks, return_exceptions=True)

        # Clear queues
        await self.queues.clear_all()

        print("✓ All services stopped")

    async def handle_websocket(
        self,
        websocket: WebSocket,
        conversation_id: Optional[str] = None,
        character_ids: Optional[List[str]] = None
    ):
        """Handle WebSocket connection for audio I/O with conversation persistence"""
        await websocket.accept()
        print("WebSocket connection accepted")

        # Determine conversation ID (load existing or create new)
        if conversation_id:
            # Load existing conversation
            try:
                conversation = await conversation_manager.get_conversation(conversation_id)
                print(f"Loaded existing conversation: {conversation_id}")

                # Use conversation's active characters if character_ids not provided
                if not character_ids:
                    character_ids = conversation.active_characters

                # Load message history
                messages = await message_manager.get_messages(conversation_id)
                print(f"Loaded {len(messages)} messages from history")

                # Convert DB messages to ConversationMessage format
                history = []
                for msg in messages:
                    history.append(ConversationMessage(
                        role=msg.role,
                        content=msg.content,
                        character_id=msg.character_id,
                        name=msg.name,
                        timestamp=time.time()  # Use current time for loaded messages
                    ))

            except HTTPException as e:
                if e.status_code == 404:
                    print(f"Conversation {conversation_id} not found, creating new one")
                    conversation = None
                    history = []
                else:
                    raise
        else:
            conversation = None
            history = []

        # Get character IDs (use active characters if not specified)
        if not character_ids:
            # Default to active characters
            active_chars = await character_manager.get_active_characters()
            character_ids = [c.id for c in active_chars[:2]]  # Default to first 2 active
            print(f"Using active characters: {character_ids}")

        # Create or update conversation in database
        if not conversation:
            conversation = await conversation_manager.create_conversation(
                ConversationCreate(
                    title=None,  # Will be auto-generated from first message
                    active_characters=character_ids
                )
            )
            conversation_id = conversation.conversation_id
            print(f"Created new conversation: {conversation_id}")
        else:
            # Update active characters if they changed
            if character_ids != conversation.active_characters:
                conversation = await conversation_manager.update_active_characters(
                    conversation_id,
                    character_ids
                )
                print(f"Updated active characters: {character_ids}")

        # Create conversation context
        self.context = ConversationContext(
            conversation_id=conversation_id,
            active_characters=await self.character_service.load_active_characters(
                character_ids
            ),
            history=history
        )

        # Set context for STT
        self.stt.set_context(self.context)

        try:
            # Start services for this session
            await self.start_services(websocket)

            # Create tasks for bidirectional communication
            receive_task = asyncio.create_task(
                self._receive_audio(websocket)
            )
            send_task = asyncio.create_task(
                self._send_text_chunks(websocket)
            )

            # Wait for completion or error
            await asyncio.gather(receive_task, send_task)

        except WebSocketDisconnect:
            print("Client disconnected")
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            await self.stop_services()

    async def _receive_audio(self, websocket: WebSocket):
        """Receive audio from browser and queue for STT"""
        while True:
            try:
                data = await websocket.receive()

                if "bytes" in data:
                    # Binary audio data
                    audio_chunk = AudioChunk(
                        data=data["bytes"],
                        sample_rate=16000,  # Browser sends 16kHz
                        timestamp=time.time()
                    )
                    await self.queues.audio_input.put(audio_chunk)

                elif "text" in data:
                    # Control messages (e.g., stop, interrupt)
                    message = json.loads(data["text"])
                    if message.get("type") == "interrupt":
                        await self.queues.interrupt()

            except Exception as e:
                print(f"Receive error: {e}")
                break

    async def _stream_text_to_browser(self, websocket: WebSocket):
        """Stream text chunks to browser for UI display"""
        while not self.queues.stop_signal.is_set():
            try:
                text_chunk: TextChunk = await asyncio.wait_for(
                    self.queues.text_output.get(),
                    timeout=0.1
                )

                # Send text to browser
                await websocket.send_json({
                    "type": "text_chunk",
                    "text": text_chunk.text,
                    "is_final": text_chunk.is_final,
                    "speaker_name": text_chunk.speaker_name,
                    "speaker_id": text_chunk.speaker_id,
                    "sequence_number": text_chunk.sequence_number,
                    "chunk_index": text_chunk.chunk_index
                })

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Text streaming error: {e}")
                break

    async def _send_text_chunks(self, websocket: WebSocket):
        """Alias for _stream_text_to_browser for task creation"""
        await self._stream_text_to_browser(websocket)

########################################
##--           FastAPI App          --##
########################################

app = FastAPI(title="Low-Latency Voice Chat Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
# TODO: Load from environment variables or config file
CONFIG = {
    "openrouter_api_key": "YOUR_OPENROUTER_API_KEY",  # Set this!
    "stt_model": "base.en",
    "llm_model": "anthropic/claude-3.5-sonnet",
    "higgs_model_path": "bosonai/higgs-audio-v2-generation-3B-base",
    "higgs_tokenizer_path": "bosonai/higgs-audio-v2-tokenizer",
    "device": "cuda",
    "max_concurrent_tts": 3
}

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    conversation_id: Optional[str] = None,
    character_ids: Optional[str] = None
):
    """
    WebSocket endpoint for voice chat

    Query Parameters:
    - conversation_id (optional): UUID of existing conversation to resume
    - character_ids (optional): Comma-separated list of character IDs (e.g., "char-001,char-002")
    """
    orchestrator = VoiceChatOrchestrator(CONFIG)

    # Parse character IDs if provided
    parsed_character_ids = None
    if character_ids:
        parsed_character_ids = [cid.strip() for cid in character_ids.split(",")]

    await orchestrator.handle_websocket(
        websocket,
        conversation_id=conversation_id,
        character_ids=parsed_character_ids
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Low-Latency Voice Chat Server",
        "status": "running",
        "websocket_endpoint": "/ws",
        "api_docs": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

########################################
##--       Character Endpoints      --##
########################################

@app.get("/api/characters", response_model=List[DbCharacter])
async def get_all_characters():
    """Get all characters"""
    return await character_manager.get_all_characters()

@app.get("/api/characters/active", response_model=List[DbCharacter])
async def get_active_characters():
    """Get all active characters"""
    return await character_manager.get_active_characters()

@app.get("/api/characters/search", response_model=List[DbCharacter])
async def search_characters(query: str = Query(..., description="Search query")):
    """Search characters by name"""
    return await character_manager.search_characters(query)

@app.get("/api/characters/{character_id}", response_model=DbCharacter)
async def get_character(character_id: str):
    """Get a specific character by ID"""
    return await character_manager.get_character(character_id)

@app.post("/api/characters", response_model=DbCharacter)
async def create_character(character_data: CharacterCreate):
    """Create a new character"""
    return await character_manager.create_character(character_data)

@app.put("/api/characters/{character_id}", response_model=DbCharacter)
async def update_character(character_id: str, character_data: CharacterUpdate):
    """Update an existing character"""
    return await character_manager.update_character(character_id, character_data)

@app.put("/api/characters/{character_id}/active")
async def set_character_active(character_id: str, is_active: bool = Query(...)):
    """Set character active status"""
    return await character_manager.set_character_active(character_id, is_active)

@app.delete("/api/characters/{character_id}")
async def delete_character(character_id: str):
    """Delete a character"""
    success = await character_manager.delete_character(character_id)
    return {"success": success, "message": f"Character {character_id} deleted"}

########################################
##--         Voice Endpoints        --##
########################################

@app.get("/api/voices", response_model=List[DbVoice])
async def get_all_voices():
    """Get all voices"""
    return await voice_manager.get_all_voices()

@app.get("/api/voices/{voice}", response_model=DbVoice)
async def get_voice(voice: str):
    """Get a specific voice by name"""
    return await voice_manager.get_voice(voice)

@app.post("/api/voices", response_model=DbVoice)
async def create_voice(voice_data: VoiceCreate):
    """Create a new voice"""
    return await voice_manager.create_voice(voice_data)

@app.put("/api/voices/{voice}", response_model=DbVoice)
async def update_voice(voice: str, voice_data: VoiceUpdate):
    """Update an existing voice"""
    return await voice_manager.update_voice(voice, voice_data)

@app.delete("/api/voices/{voice}")
async def delete_voice(voice: str):
    """Delete a voice"""
    success = await voice_manager.delete_voice(voice)
    return {"success": success, "message": f"Voice {voice} deleted"}

########################################
##--      Conversation Endpoints    --##
########################################

@app.get("/api/conversations")
async def get_conversations(limit: Optional[int] = None, offset: int = 0):
    """Get all conversations ordered by most recent first"""
    return await conversation_manager.get_all_conversations(limit=limit, offset=offset)

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation by ID with its metadata"""
    return await conversation_manager.get_conversation(conversation_id)

@app.post("/api/conversations")
async def create_conversation(conversation_data: ConversationCreate):
    """Create a new conversation"""
    return await conversation_manager.create_conversation(conversation_data)

@app.put("/api/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, conversation_data: ConversationUpdate):
    """Update conversation metadata (title, active_characters)"""
    return await conversation_manager.update_conversation(conversation_id, conversation_data)

@app.put("/api/conversations/{conversation_id}/characters")
async def update_conversation_characters(conversation_id: str, character_ids: List[str]):
    """Update the active characters in a conversation"""
    return await conversation_manager.update_active_characters(conversation_id, character_ids)

@app.post("/api/conversations/{conversation_id}/characters/{character_id}")
async def add_character_to_conversation(conversation_id: str, character_id: str):
    """Add a character to a conversation"""
    return await conversation_manager.add_character(conversation_id, character_id)

@app.delete("/api/conversations/{conversation_id}/characters/{character_id}")
async def remove_character_from_conversation(conversation_id: str, character_id: str):
    """Remove a character from a conversation"""
    return await conversation_manager.remove_character(conversation_id, character_id)

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages"""
    await conversation_manager.delete_conversation(conversation_id)
    return {"message": "Conversation deleted successfully"}

########################################
##--        Message Endpoints       --##
########################################

@app.get("/api/messages")
async def get_messages(conversation_id: str, limit: Optional[int] = None, offset: int = 0):
    """Get all messages for a conversation"""
    return await message_manager.get_messages(conversation_id, limit=limit, offset=offset)

@app.get("/api/messages/recent")
async def get_recent_messages(conversation_id: str, n: int = 10):
    """Get the last N messages from a conversation"""
    return await message_manager.get_recent_messages(conversation_id, n=n)

@app.get("/api/messages/last")
async def get_last_message(conversation_id: str, n: int = 1):
    """Get the last message from a conversation"""
    return await message_manager.get_last_message(conversation_id, n=n)

@app.post("/api/messages")
async def create_message(message_data: MessageCreate):
    """Create a single message in a conversation"""
    return await message_manager.create_message(message_data)

@app.post("/api/messages/batch")
async def create_messages_batch(messages: List[MessageCreate]):
    """Create multiple messages in a batch"""
    return await message_manager.create_messages_batch(messages)

########################################
##--           Run Server           --##
########################################

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
