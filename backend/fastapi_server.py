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
import multiprocessing
import stream2sentence as s2s
from datetime import datetime
from pydantic import BaseModel
from queue import Queue, Empty
from openai import AsyncOpenAI
from collections import defaultdict
from collections.abc import Awaitable
from threading import Thread, Event, Lock
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from supabase import create_client, Client
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from typing import Callable, Optional, Dict, List, Union, Any, AsyncIterator, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from enum import Enum, auto

from backend.RealtimeSTT import AudioToTextRecorder
from backend.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from backend.boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from backend.boson_multimodal.data_types import ChatMLSample, Message, AudioContent
from backend.RealtimeTTS.threadsafe_generators import CharIterator, AccumulatingThreadSafeGenerator

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jslevsbvapopncjehhva.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpzbGV2c2J2YXBvcG5jamVoaHZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgwNTQwOTMsImV4cCI6MjA3MzYzMDA5M30.DotbJM3IrvdVzwfScxOtsSpxq0xsj7XxI3DvdiqDSrE")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

logging.basicConfig(filename="filelogger.log", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sys.path.append('/workspace/tts/Code')


class Character(BaseModel):
    id: str
    name: str
    voice: str = ""
    system_prompt: str = ""
    image_url: str = ""
    images: List[str] = []
    is_active: bool

@dataclass
class Voice:
    voice: str                  
    method: str
    speaker_desc: str
    scene_prompt: str
    audio_path: str = ""
    text_path: str = ""

@dataclass
class ConversationMessage:
    role: str
    name: str
    content: str

@dataclass
class ModelSettings:
    model: str
    temperature: float 
    top_p: float
    min_p: float
    top_k: int
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float

@dataclass
class SentenceTTS:
    text: str
    speaker: Character
    voice: Optional[Voice]
    sentence_index: int
    is_final: bool
    timestamp: float = field(default_factory=time.time)

@dataclass 
class TextChunk:
    text: str
    message_id: str
    character_name: str
    chunk_index: int
    is_final: bool
    timestamp: float

@dataclass
class AudioChunk:
    """Represents a single audio chunk for streaming playback"""
    chunk_id: str              # Unique chunk identifier (e.g., "msg-001-chunk-0")
    message_id: str            # Parent message ID
    character_id: str          # Which character is speaking
    character_name: str        # Character name for display
    audio_data: bytes          # PCM16 @ 24kHz audio data
    chunk_index: int           # Position in message (0, 1, 2...)
    is_final: bool             # Last chunk in this message?
    timestamp: float = field(default_factory=time.time)

@dataclass
class STTCallbacks:
    """Callback functions for STT events"""
    # Transcription callbacks
    on_realtime_update: Optional[Callable[[str], Any]] = None
    on_realtime_stabilized: Optional[Callable[[str], Any]] = None
    on_final_transcription: Optional[Callable[[str], Any]] = None
    
    # VAD callbacks - useful for interrupt detection
    on_vad_start: Optional[Callable[[], Any]] = None
    on_vad_stop: Optional[Callable[[], Any]] = None
    on_vad_detect_start: Optional[Callable[[], Any]] = None
    on_vad_detect_stop: Optional[Callable[[], Any]] = None
    
    # Recording lifecycle callbacks
    on_recording_start: Optional[Callable[[], Any]] = None
    on_recording_stop: Optional[Callable[[], Any]] = None
    
    # Turn detection (silence during speech)
    on_turn_start: Optional[Callable[[], Any]] = None
    on_turn_end: Optional[Callable[[], Any]] = None

########################################
##--             Queues             --##
########################################

class Queues:
    """Queue Management for various pipeline stages"""

    def __init__(self):

        self.transcribed_text = asyncio.Queue()

        self.text_queue = asyncio.Queue()

        self.sentence_queue = asyncio.Queue()

        self.tts_sentence_queue = asyncio.Queue()

        self.audio_queue = asyncio.Queue()

########################################
##--           STT Service          --##
########################################

class STTService:
    """Speech-to-Text using RealtimeSTT"""

    def __init__(self):
        self.recorder: Optional[AudioToTextRecorder] = None
        self.recording_thread: Optional[threading.Thread] = None
        self.callbacks: Dict[str, Any] = {}
        self.is_listening = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def initialize(self):
        """Initialize the STT recorder"""
        logger.info("Initializing STT service...")

        try:
            self.recorder = AudioToTextRecorder(
                model="small.en",
                language="en",
                enable_realtime_transcription=True,
                realtime_processing_pause=0.1,
                realtime_model_type="small.en",
                on_realtime_transcription_update=self._on_realtime_update,
                on_realtime_transcription_stabilized=self._on_realtime_stabilized,
                on_recording_start=self._on_recording_start,
                on_recording_stop=self._on_recording_stop,
                on_vad_detect_start=self._on_vad_detect_start,
                on_vad_detect_stop=self._on_vad_detect_stop,
                silero_sensitivity=0.4,
                webrtc_sensitivity=3,
                post_speech_silence_duration=0.7,
                min_length_of_recording=0.5,
                spinner=False,
                level=logging.WARNING,
                use_microphone=False
            )

            self.recording_thread = threading.Thread(target=self.transcription_loop, daemon=True)
            self.recording_thread.start()

            logger.info("STT service initialized")

        except Exception as e:
            logger.error(f"Failed to initialize STT service: {e}")
            raise

    def transcription_loop(self):
        """Main recording/transcription loop running in separate thread"""
        logger.info("STT recording loop started")

        while True:
            if self.is_listening:
                try:
                    # Get transcription from recorder
                    text = self.recorder.text()

                    if text and text.strip():
                        logger.info(f"Final transcription: {text}")
                        self._on_final_transcription(text)
                    else:
                        print(e)
                    
                    self.is_listening.clear()
                    
                except Exception as e:
                    self.is_listening.clear()

            else:
                time.sleep(0.05)
    
    def _on_realtime_update(self, text: str):
        """Called when realtime transcription updates"""
        if self.callbacks.on_realtime_update:
            self._schedule_callback(self.callbacks.on_realtime_update, text)
    
    def _on_realtime_stabilized(self, text: str):
        """Called when realtime transcription stabilizes"""
        if self.callbacks.on_realtime_stabilized:
            self._schedule_callback(self.callbacks.on_realtime_stabilized, text)

    def _on_final_transcription(self, user_message: str):
        if self.callbacks.on_final_transcription:
            self._schedule_callback(self.callbacks.on_final_transcription, user_message)
    
    def _on_vad_detect_start(self):
        """Called when system starts listening for voice activity."""
        if self.callbacks.on_vad_detect_start:
            self._schedule_callback(self.callbacks.on_vad_detect_start)
    
    def _on_vad_detect_stop(self):
        """Called when system stops listening for voice activity"""
        if self.callbacks.on_vad_detect_stop:
            self._schedule_callback(self.callbacks.on_vad_detect_stop)
    
    def _on_vad_start(self):
        """Called when actual voice activity is detected during recording."""
        if self.callbacks.on_vad_start:
            self._schedule_callback(self.callbacks.on_vad_start)
    
    def _on_vad_stop(self):
        """Called when voice activity stops during recording"""
        if self.callbacks.on_vad_stop:
            self._schedule_callback(self.callbacks.on_vad_stop)
    
    def _on_recording_start(self):
        """Called when recording starts"""
        if self.callbacks.on_recording_start:
            self._schedule_callback(self.callbacks.on_recording_start)
    
    def _on_recording_stop(self):
        """Called when recording stops"""
        if self.callbacks.on_recording_stop:
            self._schedule_callback(self.callbacks.on_recording_stop)

########################################
##--       Callback Utilities       --##
########################################
    
    def _schedule_callback(self, callback: Callable, *args):
        """Schedule a callback to run, handling both sync and async callbacks"""
        if self._loop is None:
            return
            
        try:
            if asyncio.iscoroutinefunction(callback):
                # Schedule async callback
                self._loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(callback(*args))
                )

            else:
                # Schedule sync callback
                self._loop.call_soon_threadsafe(callback, *args)
        except Exception as e:
            logger.error(f"Error scheduling callback: {e}")
    
    async def _run_callback_async(self, callback: Callable, *args):
        """Run a callback that may be sync or async"""
        if asyncio.iscoroutinefunction(callback):
            await callback(*args)
        else:
            callback(*args)

########################################
##--   Text to Sentence Extractor   --##
########################################

class TextToSentence:
    """
    Bridges async LLM text stream to synchronous stream2sentence library.
    
    Architecture:
    - Async side: Receives text chunks, feeds to CharIterator
    - Sync side: stream2sentence extracts sentences in background thread
    - Output: Sentences are pushed to an async queue as they're detected
    
    This enables true streaming: first sentence starts TTS while LLM is still
    generating subsequent text.
    """
    
    def __init__(
        self,
        sentence_callback: Callable[[str, int], None] = None,
        minimum_sentence_length: int = 25,
        minimum_first_fragment_length: int = 10,
        quick_yield_single_sentence_fragment: bool = True,
        cleanup_text_links: bool = True,
        cleanup_text_emojis: bool = False,
        tokenize_sentences: bool = False,
    ):

        self.sentence_callback = sentence_callback
        
        self.s2s_config = {
            "minimum_sentence_length": minimum_sentence_length,
            "minimum_first_fragment_length": minimum_first_fragment_length,
            "quick_yield_single_sentence_fragment": quick_yield_single_sentence_fragment,
            "cleanup_text_links": cleanup_text_links,
            "cleanup_text_emojis": cleanup_text_emojis,
            "tokenize_sentences": tokenize_sentences,
        }
        
        # Thread-safe components
        self.char_iter: Optional[CharIterator] = None
        self.thread_safe_iter: Optional[AccumulatingThreadSafeGenerator] = None
        
        # Sentence output queue (thread-safe)
        
        # Control
        self.sentences_thread: Optional[threading.Thread] = None
        self._is_running = False
        self._is_complete = False
        self._sentence_count = 0
        self._accumulated_text = ""
        
        # Thread pool for non-blocking operations
        self._executor = ThreadPoolExecutor(max_workers=1)
    
    def start(self):
        """Initialize and start the sentence extraction pipeline"""
        self.char_iter = CharIterator()
        self.thread_safe_iter = AccumulatingThreadSafeGenerator(self.char_iter)
        
        self._is_running = True
        self._is_complete = False
        self._sentence_count = 0
        self._accumulated_text = ""
        
        # Start sentence extraction thread
        self.sentences_thread = threading.Thread(target=self.run_sentence_generator, daemon=True)
        self.sentences_thread.start()
    
    def feed_text(self, text: str):
        """
        Feed text chunk from LLM stream.
        Thread-safe, can be called from async context.
        """
        if not self._is_running:
            return
            
        self._accumulated_text += text
        self.char_iter.add(text)

    def get_accumulated_text(self) -> str:
        """Get all text that was fed to the sentence extractor"""
        return self._accumulated_text
    
    def finish(self):
        """
        Signal that LLM stream is complete.
        Flushes any remaining text as final sentence.
        """
        if not self._is_running:
            return
            
        self._is_complete = True
        
        # Signal end to CharIterator
        if self.char_iter:
            self.char_iter.add("")  # Empty string can signal completion
            self.char_iter.complete()  # If this method exists
    
    def run_sentence_generator(self):
        """
        Background thread: Extract sentences using stream2sentence.
        Runs until stream is complete and all sentences are extracted.
        """
        try:
            sentence_generator = s2s.generate_sentences(
                self.thread_safe_iter,
                **self.s2s_config
            )
            
            for sentence in sentence_generator:
                if not self._is_running:
                    break
                    
                sentence = sentence.strip()
                if sentence:
                    # Push to queue
                    self.sentence_queue.put((sentence, self._sentence_count))
                    
                    # Callback if provided
                    if self.sentence_callback:
                        self.sentence_callback(sentence, self._sentence_count)
                    
                    self._sentence_count += 1
                    
        except Exception as e:
            print(f"Sentence extraction error: {e}")
        finally:
            # Signal completion
            self.sentence_queue.put(None)  # Sentinel value
            self._is_running = False
    
    async def get_sentences(self) -> AsyncIterator[tuple[str, int]]:
        """
        Async generator that yields sentences as they become available.
        Yields: (sentence_text, sentence_index)
        """
        loop = asyncio.get_event_loop()
        
        while True:
            try:
                # Non-blocking check with short timeout
                sentence = await loop.run_in_executor(self._executor, lambda: self.sentence_queue.get(timeout=0.02))
                
                if sentence is None:  # Sentinel
                    break
                    
                yield sentence
                
            except Empty:
                # No sentence ready yet, yield control
                await asyncio.sleep(0.005)
                
                # Check if we should exit
                if not self._is_running and self.sentence_queue.empty():
                    break
    
    def shutdown(self):
        """Clean shutdown"""
        self._is_running = False
        if self.sentences_thread and self.sentences_thread.is_alive():
            self.sentences_thread.join(timeout=1.0)
        self._executor.shutdown(wait=False)

########################################
##--           LLM Service          --##
########################################

class LLMService:
    """LLM Service"""

    def __init__(self, character: Character, queues: Queues, api_key: str, model: str):
        
        self.is_initialized = False
        self.queues = queues
        self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model = model

        self.conversation_history: List[Message] = []
        self.character: Optional[Character] = None
        self.speaker_number = 0

        self.sentence_extractor = TextToSentence()
        
        # Tracking
        self.chunk_index = 0
        self.sentence_index = 0
        self.response_text = ""
        self.is_complete = False

    async def initialize(self):
        self.is_initialized = True
        logger.info("LLMService initialized")

    def strip_character_tags(self, text: str) -> str:
        """Strip character tags from text for display/TTS purposes"""
        return re.sub(r'<[^>]+>', '', text).strip()

    def add_user_message(self, name: str = "Jay", content = "user_message"):
        """Add user message to history"""
        self.conversation_history.append(content(
            role="user",
            name=name,
            content="user_message",
        ))

    def add_character_message(self, character: Character, text: str):
        """Add character response to history"""
        self.conversation_history.append(ConversationMessage(
            role="assistant",
            name=character.name,
            content=text
        ))

    def add_message_to_conversation_history(self, role: str, name: str, content: str):
        """Add (user or character) message to conversation history"""
        self.conversation_history.append(ConversationMessage(
            role=role,
            name=name,
            content=content
        ))

    def create_character_instruction_message(self, character: Character) -> Dict[str, str]:
        """Create character instruction message for group chat with character tags."""
        return {
            'role': 'system',
            'content': f'Based on the conversation history above provide the next reply as {character.name}. Your response should include only {character.name}\'s reply. Do not respond for/as anyone else. Wrap your entire response in <{character.name}></{character.name}> tags.'
        }

    def parse_character_mentions(self, message: str, active_characters: List[Character]) -> List[Character]:
        """Parse a message for character mentions in order of appearance"""
        mentioned_characters = []
        processed_characters = set()

        name_mentions = []

        for character in active_characters:
            name_parts = character.name.lower().split()

            for name_part in name_parts:
                pattern = r'\b' + re.escape(name_part) + r'\b'
                for match in re.finditer(pattern, message, re.IGNORECASE):
                    name_mentions.append({
                        'character': character,
                        'position': match.start(),
                        'name_part': name_part
                    })

        name_mentions.sort(key=lambda x: x['position'])

        for mention in name_mentions:
            if mention['character'].id not in processed_characters:
                mentioned_characters.append(mention['character'])
                processed_characters.add(mention['character'].id)

        if not mentioned_characters:
            mentioned_characters = sorted(active_characters, key=lambda c: c.name)

        return mentioned_characters
    
    def get_model_settings(self):
        """Get model with parameters from client message"""

        # function that has model and model parameters for the message we are about to send.

    def build_character_message_request(self, user_message: str, character:Character):
        """Build LLM prompt for a specific character."""

        # build entire message body to send to openrouter - user_message, model with settings

        messages = []

        # Add character's system prompt
        


    async def main_llm_loop(self, model_settings: ModelSettings):
        """Run LLM loop"""

        while True:
            try:

                user_message = await self.queues.transcribed_text.get()
                
                if not user_message or not user_message.strip():
                    continue

                # Add user message to conversation history
                self.conversation_history.append({"role": "user", "name": "Jay", "content": user_message})

                # Parse which characters should respond
                mentioned_characters = self.parse_character_mentions(user_message, self.active_characters)

                for character in mentioned_characters:
                    if self.queues.interrupt_signal.is_set():
                        break

                    messages = []

                    messages.append({"role": "system", "name": character.name, "content": character.system_prompt})

                    messages.extend(self.conversation_history)

                    # Add character instruction message at the end
                    messages.append(self.create_character_instruction_message(character))

                    text_stream = await self.client.chat.completions.create({
                        "model": model_settings.model,
                        "messages": messages,
                        "temperature": model_settings.temperature,
                        "top_p": model_settings.top_p,
                        "min_p": model_settings.min_p,
                        "top_k": model_settings.top_k,
                        "frequency_penalty": model_settings.frequency_penalty,
                        "presence_penalty": model_settings.presence_penalty,
                        "repetition_penalty": model_settings.repetition_penalty,
                        "stream":True
                    })

                    # Generate responses for each character in sequence
                    response_text = await self.character_response_stream(self, character=character, text_stream=text_stream)

                    # add response to conversation history
                    if response_text: 
                        self.add_message_to_conversation_history("assistant", character.name, response_text)
                
            except Exception as e:
                logger.error(f"Error in LLM loop: {e}")


    async def character_response_stream(self, character: Character, text_stream: AsyncIterator) -> str:
        """Generate and stream a single character's response"""

        self.sentence_extractor.start()
        
        # Create task for sentence-to-TTS processing
        sentence_task = asyncio.create_task(self.process_sentences_for_tts())
        
        try:
            # Stream from LLM
            async for chunk in text_stream:
                # Check for interrupt
                if self.interrupt_signal.is_set():
                    break
                
                # Extract content from chunk
                content = chunk.choices[0].delta.content
                if content:
                    self.response_text += content
                    
                    # Feed to sentence extractor (non-blocking)
                    self.sentence_extractor.feed_text(content)
                    
                    # Stream to UI immediately
                    display_text = TextChunk(
                        text=content,
                        message_id=self.message.id,
                        character_name=self.character.name,
                        chunk_index=self.chunk_index,
                        is_final=False,
                        timestamp=time.time()
                    )
                    await self.text_queue.put(display_text)
                    self.chunk_index += 1
            
            # Signal LLM stream complete
            self.sentence_extractor.finish()
            
            # Send final text chunk to UI
            final_display_text = TextChunk(
                text="",
                message_id=self.message.id,
                character_name=self.character.name,
                chunk_index=self.chunk_index,
                is_final=False,
                timestamp=time.time()
            )
            await self.text_queue.put(final_display_text)
            
            # Wait for sentence processing to complete
            await sentence_task
            
        except Exception as e:
            print(f"Error in stream_response for {self.character.name}: {e}")
            raise
        finally:
            self.sentence_extractor.shutdown()
            self.is_complete = True
        
        return self.response_text
    
    async def process_sentences_for_tts(self):
        """Process sentences as they're extracted and queue for TTS. Runs concurrently with LLM text stream."""

        sentences_queued = []
        
        try:
            async for sentence, sentence_index in self.sentence_extractor.get_sentences():
                if self.interrupt_signal.is_set():
                    break

                sentence_index = 0
                sentences_queued.append(sentence)
                
                # Determine if this might be the last sentence
                # (We don't know for sure until sentence streamer finishes)
                is_final = False  # Will be corrected below
                
                # Queue for TTS immediately
                sentence = SentenceTTS(
                    text=sentence,
                    speaker=self.character,
                    voice=self.voice,
                    sentence_index=sentence_index,
                    is_final=is_final,
                    timestamp=time.time()
                )
                await self.tts_sentence_queue.put(sentence)
                
                print(f"📝 Sentence {sentence_index} queued for TTS ({self.character.name}): "
                      f"{sentence[:50]}{'...' if len(sentence) > 50 else ''}")
            
            # Mark the last sentence as final
            if sentences_queued:
                # Send a "final" marker for this character's TTS
                final_marker = SentenceTTS(
                    text="",  # Empty text signals completion
                    speaker=self.character,
                    voice=self.voice,
                    sentence_index=self.sentence_index + 1,
                    is_final=True,
                    timestamp=time.time()
                )
                await self.tts_sentence_queue.put(final_marker)
                
        except Exception as e:
            print(f"Error processing sentences for {self.character.name}: {e}")



########################################
##--           TTS Service          --##
########################################

class TTSService:
    """TTS Service"""


########################################
##--        WebSocket Manager       --##
########################################

class WebSocketManager:
    """Manages WebSocket"""
    
    def __init__(self):
        self.stt_service: Optional[STTService] = None
        self.llm_service: Optional[LLMService] = None
        self.tts_service: Optional[TTSService] = None
        self.websocket: Optional[WebSocket] = None
        self.queues: Optional[Queues] = None

    async def initialize(self):
        """Initialize all services with proper callbacks"""

        self.queues = Queues()

        # Setup STT callbacks
        stt_callbacks = STTCallbacks(
            on_realtime_update=self.on_realtime_update,
            on_realtime_stabilized=self.on_realtime_stabilized,
            on_final_transcription=self.on_final_transcription,
        )
        
        self.stt_service = STTService()
        self.llm_service = LLMService(character=Character, queues=Queues, api_key=str, model=str)
        self.tts_service = TTSService()

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.websocket = websocket
        logger.info("WebSocket connected")

        await self.start_service_tasks()

    async def start_service_tasks(self):
        """Start all services"""

        self.service_tasks = [
            asyncio.create_task(self.llm_service.main_llm_loop()),
            asyncio.create_task(self.tts_service.main_tts_loop())
        ]
    
    async def handle_audio_message(self, audio_data: bytes):
        """Feed audio for transcription"""
        if self.stt_service:
            self.stt_service.feed_audio(audio_data)

    async def handle_text_message(self, message: str):
        """Handle incoming text messages"""
        try:
            data = json.loads(message)
            message_type = data.get("type", "")
            payload = data.get("data", {})
            
            if message_type == "user_message":
                user_message = payload.get("text", "")
                await self.handle_user_message(user_message)
            
            elif message_type == "start_listening":
                self.stt_service.start_listening()
            
            elif message_type == "stop_listening":
                self.stt_service.stop_listening()

            elif message_type == "model_settings":
                model_settings = data.get("model", "")
                # need to add all model parameters (temperature, top_p, min_p, top_k, frequency_penalty, presense_penalty, repetition_penalty)
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def handle_user_message(self, user_message: str):
        """Process manually sent user message"""
        await self.queues.transcribed_text.put(user_message)

    async def send_text_to_client(self, data: dict):
        """Send JSON message to client"""
        if self.websocket:
            await self.websocket.send_text(json.dumps(data))
    
    async def stream_audio_to_client(self, audio_data: bytes):
        """Send binary audio to client (TTS)"""
        if self.websocket:
            await self.websocket.send_bytes(audio_data)

    async def on_realtime_update(self, text: str):
        await self.send_text_to_client({"type": "stt_update", "text": text})
    
    async def on_realtime_stabilized(self, text: str):
        await self.send_text_to_client({"type": "stt_stabilized", "text": text})
    
    async def on_final_transcription(self, user_message: str):
        # put transcribed text into queue for llm to get
        await self.queues.transcribed_text.put(user_message)
        # send final text to client for user's prompt UI display
        await self.send_text_to_client({"type": "stt_final", "text": user_message})

    async def disconnect(self):
        """Handle WebSocket disconnection"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        logger.info("WebSocket disconnected")

########################################
##--           FastAPI App          --##
########################################

ws_manager = WebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up services...")
    await ws_manager.initialize()
    print("All services initialized!")
    yield
    print("Shutting down services...")
    await ws_manager.shutdown()
    print("All services shut down!")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

########################################
##--       WebSocket Endpoint       --##
########################################

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    
    try:
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                await ws_manager.handle_text_message(message["text"])
            
            elif "bytes" in message:
                await ws_manager.handle_audio_message(message["bytes"])
    
    except WebSocketDisconnect:
        await ws_manager.disconnect()
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws_manager.disconnect()

########################################
##--           Run Server           --##
########################################

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
