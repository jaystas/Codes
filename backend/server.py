"""
FastAPI WebSocket Server Framework
==================================
A single WebSocket endpoint handling both binary audio and text messages.

Structure:
1. Imports
2. Configuration
3. Service Classes (stubs)
4. WebSocket Connection Manager
5. Message Handlers
6. WebSocket Endpoint
7. Lifecycle Events
8. Main Entry Point
"""

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

class Message(BaseModel):
    """Single message in conversation history"""
    role: str
    name: str
    content: str

class STTState(Enum):
    """STT Service states"""
    IDLE = auto()
    LISTENING = auto()      # Waiting for voice activity
    RECORDING = auto()      # Voice detected, recording in progress
    PROCESSING = auto()     # Processing final transcription
    ERROR = auto()


@dataclass
class STTConfig:
    """Configuration for STT Service"""
    # Model settings
    model: str = "small.en"
    realtime_model: str = "tiny.en"
    language: str = "en"
    device: str = "cuda"
    compute_type: str = "float16"
    
    # Audio settings
    sample_rate: int = 16000
    buffer_size: int = 512
    
    # VAD settings
    silero_sensitivity: float = 0.4
    webrtc_sensitivity: int = 3
    post_speech_silence_duration: float = 0.6  # Lower for faster response
    min_length_of_recording: float = 0.5       # Minimum utterance length
    pre_recording_buffer_duration: float = 0.5
    
    # Realtime transcription settings
    enable_realtime_transcription: bool = True
    realtime_processing_pause: float = 0.1     # How often to run realtime STT
    init_realtime_after_seconds: float = 0.1   # Start realtime after this delay
    
    # Performance settings
    beam_size: int = 5
    beam_size_realtime: int = 3
    batch_size: int = 16
    realtime_batch_size: int = 8
    
    # Behavior settings
    ensure_sentence_starting_uppercase: bool = True
    ensure_sentence_ends_with_period: bool = False  # Don't force period for streaming
    spinner: bool = False  # Disable spinner for server use
    no_log_file: bool = True


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
    """Manages all queues"""

    def __init__(self):

        # STT → LLM Orchestrator
        self.transcribed_text = asyncio.Queue()

        # LLM → Browser (for text display)
        self.display_queue = asyncio.Queue()

        # LLM → TTS (with speaker info and sequence)
        self.sentence_queue = asyncio.Queue()

        # TTS → Audio Sequencer
        self.audio_queue = asyncio.Queue()

        # Control signals
        self.stop_signal = asyncio.Event()
        self.interrupt_signal = asyncio.Event()

########################################
##--           STT Service          --##
########################################

class STTService:
    """
    Speech-to-Text Service using RealtimeSTT
    
    Designed for remote GPU processing with external audio feed.
    Provides async-compatible callbacks for FastAPI integration.
    
    Usage:
        stt = STTService(config, callbacks)
        await stt.initialize()
        
        # Start listening for voice
        stt.start_listening()
        
        # Feed audio from WebSocket
        stt.feed_audio(audio_chunk)
        
        # Get final transcription
        text = await stt.transcribe_audio_message()
    """
    
    def __init__(
        self,
        config: Optional[STTConfig] = None,
        callbacks: Optional[STTCallbacks] = None
    ):
        self.config = config or STTConfig()
        self.callbacks = callbacks or STTCallbacks()
        
        self._recorder = None
        self._state = STTState.IDLE
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Thread-safe state tracking
        self._lock = threading.Lock()
        self._is_initialized = False
        self._is_shutting_down = False
        
        # For async transcription results
        self._transcription_future: Optional[asyncio.Future] = None
        self._current_transcription: str = ""
        
        # VAD state for interrupt detection
        self._voice_active = False
        self._tts_interrupt_requested = False
        
    @property
    def state(self) -> STTState:
        return self._state
    
    @property
    def is_voice_active(self) -> bool:
        """Check if voice activity is currently detected - useful for interrupt logic"""
        return self._voice_active
    
    @property
    def is_recording(self) -> bool:
        """Check if actively recording"""
        return self._recorder is not None and self._recorder.is_recording
    
    @property
    def interrupt_requested(self) -> bool:
        """Check if TTS interrupt was requested due to voice detection"""
        return self._tts_interrupt_requested
    
    def clear_interrupt(self):
        """Clear the interrupt flag after handling"""
        self._tts_interrupt_requested = False
        
    async def initialize(self) -> bool:
        """
        Initialize the STT recorder.
        Must be called before using the service.
        """
        if self._is_initialized:
            logger.warning("STT Service already initialized")
            return True
            
        try:
            self._loop = asyncio.get_running_loop()
             
            self._recorder = AudioToTextRecorder(
                # Model configuration
                model=self.config.model,
                realtime_model_type=self.config.realtime_model,
                language=self.config.language,
                device=self.config.device,
                compute_type=self.config.compute_type,
                
                # Critical: disable microphone for remote operation
                use_microphone=False,
                
                # Audio settings
                sample_rate=self.config.sample_rate,
                buffer_size=self.config.buffer_size,
                
                # VAD settings
                silero_sensitivity=self.config.silero_sensitivity,
                webrtc_sensitivity=self.config.webrtc_sensitivity,
                post_speech_silence_duration=self.config.post_speech_silence_duration,
                min_length_of_recording=self.config.min_length_of_recording,
                pre_recording_buffer_duration=self.config.pre_recording_buffer_duration,
                
                # Realtime transcription
                enable_realtime_transcription=self.config.enable_realtime_transcription,
                realtime_processing_pause=self.config.realtime_processing_pause,
                init_realtime_after_seconds=self.config.init_realtime_after_seconds,
                on_realtime_transcription_update=self._on_realtime_update,
                on_realtime_transcription_stabilized=self._on_realtime_stabilized,
                
                # VAD callbacks
                on_vad_detect_start=self._on_vad_detect_start,
                on_vad_detect_stop=self._on_vad_detect_stop,
                on_vad_start=self._on_vad_start,
                on_vad_stop=self._on_vad_stop,
                
                # Recording callbacks
                on_recording_start=self._on_recording_start,
                on_recording_stop=self._on_recording_stop,
                
                # Turn detection
                on_turn_detection_start=self._on_turn_start,
                on_turn_detection_stop=self._on_turn_end,
                
                # Performance settings
                beam_size=self.config.beam_size,
                beam_size_realtime=self.config.beam_size_realtime,
                batch_size=self.config.batch_size,
                realtime_batch_size=self.config.realtime_batch_size,
                
                # Behavior
                ensure_sentence_starting_uppercase=self.config.ensure_sentence_starting_uppercase,
                ensure_sentence_ends_with_period=self.config.ensure_sentence_ends_with_period,
                spinner=self.config.spinner,
                no_log_file=self.config.no_log_file,
                
                # Run callbacks in new thread to avoid blocking
                start_callback_in_new_thread=True,
            )
            
            self._is_initialized = True
            self._state = STTState.IDLE
            logger.info("STT Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize STT Service: {e}", exc_info=True)
            self._state = STTState.ERROR
            return False
    
    def start_listening(self):
        """
        Start listening for voice activity.
        The recorder will automatically start recording when voice is detected.
        """
        if not self._is_initialized or self._recorder is None:
            logger.warning("Cannot start listening: STT Service not initialized")
            return
            
        with self._lock:
            self._state = STTState.LISTENING
            self._current_transcription = ""
            self._tts_interrupt_requested = False
            
        # Put recorder in listening state
        self._recorder.listen()
        logger.debug("STT Service: Started listening for voice activity")
    
    def stop_listening(self):
        """Stop listening and abort any ongoing recording"""
        if self._recorder is None:
            return
            
        try:
            self._recorder.abort()
            with self._lock:
                self._state = STTState.IDLE
                self._voice_active = False
        except Exception as e:
            logger.error(f"Error stopping STT: {e}")
    
    def feed_audio(self, audio_data: bytes):
        """Feed raw PCM audio bytes (16kHz, 16-bit, mono)"""
        if self._recorder:
            try:
                self._recorder.feed_audio(audio_data, original_sample_rate=16000)
            except Exception as e:
                logger.error(f"Failed to feed audio to recorder: {e}")

    async def transcribe_audio_message(self) -> str:
        """transcribes user audio message"""
        if not self._is_initialized or self._recorder is None:
            return ""
        
        try:
            loop = asyncio.get_running_loop()
            
            with self._lock:
                self._state = STTState.LISTENING
                
            user_message = await asyncio.wait_for(
                loop.run_in_executor(None, self._recorder.text),
            )
            
            with self._lock:
                self._state = STTState.IDLE
                self._current_transcription = user_message or ""
                
            # Fire final transcription callback
            if user_message and self.callbacks.on_final_transcription:
                await self._run_callback_async(self.callbacks.on_final_transcription, user_message)
                
            return user_message or ""
            
        except asyncio.TimeoutError:
            self.stop_listening()
            return ""
        except Exception as e:
            logger.error(f"Error getting transcription: {e}")
            return ""
    
    def start_recording(self):
        """Manually start recording (bypasses VAD wait)"""
        if self._recorder is None:
            return
        self._recorder.start()
        with self._lock:
            self._state = STTState.RECORDING
    
    def stop_recording(self):
        """Manually stop recording"""
        if self._recorder is None:
            return
        self._recorder.stop()
        with self._lock:
            self._state = STTState.PROCESSING
    
    async def shutdown(self):
        """Shutdown the STT service and release resources"""
        if self._is_shutting_down:
            return
            
        self._is_shutting_down = True
        logger.info("Shutting down STT Service...")
        
        try:
            if self._recorder is not None:
                # Run shutdown in executor to not block
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._recorder.shutdown)
                self._recorder = None
                
            self._is_initialized = False
            self._state = STTState.IDLE
            logger.info("STT Service shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during STT shutdown: {e}")
        finally:
            self._is_shutting_down = False
    
########################################
##--      STT Callback Handlers     --##
########################################
    
    def _on_realtime_update(self, text: str):
        """Called when realtime transcription updates"""
        if self.callbacks.on_realtime_update:
            self._schedule_callback(self.callbacks.on_realtime_update, text)
    
    def _on_realtime_stabilized(self, text: str):
        """Called when realtime transcription stabilizes"""
        if self.callbacks.on_realtime_stabilized:
            self._schedule_callback(self.callbacks.on_realtime_stabilized, text)
    
    def _on_vad_detect_start(self):
        """
        Called when system starts listening for voice activity.
        This is triggered when entering the listening state.
        """
        with self._lock:
            self._voice_active = True
            
        if self.callbacks.on_vad_detect_start:
            self._schedule_callback(self.callbacks.on_vad_detect_start)
    
    def _on_vad_detect_stop(self):
        """Called when system stops listening for voice activity"""
        with self._lock:
            self._voice_active = False
            
        if self.callbacks.on_vad_detect_stop:
            self._schedule_callback(self.callbacks.on_vad_detect_stop)
    
    def _on_vad_start(self):
        """
        Called when actual voice activity is detected during recording.
        Key callback for interrupt detection!
        """
        with self._lock:
            self._voice_active = True
            self._tts_interrupt_requested = True  # Signal for TTS interrupt
            
        logger.debug("VAD Start: Voice activity detected")
        
        if self.callbacks.on_vad_start:
            self._schedule_callback(self.callbacks.on_vad_start)
    
    def _on_vad_stop(self):
        """Called when voice activity stops during recording"""
        with self._lock:
            self._voice_active = False
            
        logger.debug("VAD Stop: Voice activity ended")
        
        if self.callbacks.on_vad_stop:
            self._schedule_callback(self.callbacks.on_vad_stop)
    
    def _on_recording_start(self):
        """Called when recording starts"""
        with self._lock:
            self._state = STTState.RECORDING
            
        logger.debug("Recording started")
        
        if self.callbacks.on_recording_start:
            self._schedule_callback(self.callbacks.on_recording_start)
    
    def _on_recording_stop(self):
        """Called when recording stops"""
        with self._lock:
            self._state = STTState.PROCESSING
            
        logger.debug("Recording stopped")
        
        if self.callbacks.on_recording_stop:
            self._schedule_callback(self.callbacks.on_recording_stop)
    
    def _on_turn_start(self):
        """Called when silence is detected during speech (possible turn end)"""
        if self.callbacks.on_turn_start:
            self._schedule_callback(self.callbacks.on_turn_start)
    
    def _on_turn_end(self):
        """Called when speech resumes after silence"""
        if self.callbacks.on_turn_end:
            self._schedule_callback(self.callbacks.on_turn_end)
    
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
##--           LLM Service          --##
########################################

class LLMService:
    """LLM Service"""

    def __init__(self, api_key: str, queues=Queues):
        self.is_initialized = False
        self.queues = queues
        self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model: str 
    
    async def initialize(self):
        self.is_initialized = True
        print("LLMService initialized")
    
    def set_model(self, model: str):
        """Set the current LLM model"""
        self.model = model
        logger.info(f"LLM model set to: {model}")

    def strip_character_tags(self, text: str) -> str:
        """Strip character tags from text for display/TTS purposes"""
        return re.sub(r'<[^>]+>', '', text).strip()
    
    def parse_character_mentions(self, message: str, active_characters: List[Character]) -> List[Character]:
        """Parse a message for character mentions in order of appearance"""
        mentioned_characters = []
        processed_characters = set()

        # Create an array of all possible name mentions with their positions
        name_mentions = []

        for character in active_characters:
            name_parts = character.name.lower().split()

            for name_part in name_parts:
                # Find all occurrences of this name part in the message
                pattern = r'\b' + re.escape(name_part) + r'\b'
                for match in re.finditer(pattern, message, re.IGNORECASE):
                    name_mentions.append({
                        'character': character,
                        'position': match.start(),
                        'name_part': name_part
                    })

        # Sort by position in the message
        name_mentions.sort(key=lambda x: x['position'])

        # Add characters in order of first mention, avoiding duplicates
        for mention in name_mentions:
            if mention['character'].id not in processed_characters:
                mentioned_characters.append(mention['character'])
                processed_characters.add(mention['character'].id)

        # If no one was mentioned, all active characters respond (in order)
        if not mentioned_characters:
            mentioned_characters = sorted(active_characters, key=lambda c: c.name)

        return mentioned_characters 

    def build_character_prompt(
        self,
        character: Character,
        conversation_history: List[Message]
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
    

    async def run_llm_loop(self):
        """Continuously running LLM processing loop"""
        
        while True:
            try:

                user_message = await self.queues.transcribed_text.get()





            except Exception as e:
                logger.error(f"LLM loop error: {e}")


    async def shutdown(self):
        """Clean up LLM resources."""
        self.is_initialized = False
        print("LLMService shut down")

########################################
##--           TTS Service          --##
########################################

class TTSService:
    """Text-to-Speech Service - converts text to audio."""
    
    def __init__(self):
        self.is_initialized = False
        self.sample_rate: int = 24000
    
    async def initialize(self):
        """Initialize the TTS model/service."""
        # TODO: Load your TTS model here (e.g., Higgs Audio)
        self.is_initialized = True
        print("TTSService initialized")
    
    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes."""
        # TODO: Implement text-to-speech
        pass
    
    async def synthesize_stream(self, text: str):
        """Stream audio chunks as they're generated (async generator)."""
        # TODO: Implement streaming TTS
        # yield audio chunks as they're generated
        yield b""
    
    async def shutdown(self):
        """Clean up TTS resources."""
        self.is_initialized = False
        print("TTSService shut down")


class Pipeline:
    """Run TextToAudio pipeline"""

    def __init__(self):
        self.is_initialized = False

    async def initialize(self):
        """Initialize the text to audio pipeline."""
        self.is_initialized = True

    async def start_services(self):
        """Start all services including audio sequencer"""

        self.pipeline_tasks = [
            asyncio.create_task(self.stt.run_stt_loop()),
            asyncio.create_task(self.llm.run_llm_loop()),
            asyncio.create_task(self.tts.run_tts_loop())
        ]

    async def run_pipeline_loop(self, audio_data: bytes, stt:STTService, llm: LLMService, tts: TTSService):
        """continuously runs text to audio stream pipeline"""

        # I'm trying to figure out if should run here or in one of stt / llm loop.

        stt = STTService(config, callbacks)
        await stt.initialize()
        
        # Start listening for voice
        stt.start_listening()
        
        # Feed audio from WebSocket
        stt.feed_audio(audio_data)
        
        # Get final transcription
        text = await stt.transcribe_audio_message()

########################################
##--        WebSocket Manager       --##
########################################

class WebSocketManager:
    """Manages WebSocket connections and their associated state."""
    
    def __init__(self):
        self.stt_service: Optional[STTService] = None
        self.llm_service: Optional[LLMService] = None
        self.tts_service: Optional[TTSService] = None
        self.websocket: Optional[WebSocket] = None
        self.queues: Optional[Queues] = None

    async def initialize(self):
        """Initialize all services with proper callbacks"""
        # Setup STT callbacks
        stt_callbacks = STTCallbacks(
            on_realtime_update=self.on_realtime_update,
            on_realtime_stabilized=self.on_realtime_stabilized,
            on_final_transcription=self.on_final_transcription,
        )
        
        self.stt_service = STTService(callbacks=stt_callbacks)
        self.llm_service = LLMService()
        self.tts_service = TTSService()
        self.queues = Queues()

        await self.queues.initialize()
        await self.stt_service.initialize()
        await self.llm_service.initialize()
        await self.tts_service.initialize()

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.websocket = websocket
        logger.info("WebSocket connected")
    
    async def disconnect(self):
        """Handle WebSocket disconnection"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        logger.info("WebSocket disconnected")

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
            
            elif message_type == "set_model":
                model = data.get("model", "")
                self.llm_service.set_model(model)
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def handle_user_message(self, user_message: str):
        """Process manually sent user message"""
        await self.queues.transcribed_text.put(user_message)

    async def send_text_message(self, data: dict):
        """Send JSON message to client"""
        if self.websocket:
            await self.websocket.send_text(json.dumps(data))
    
    async def stream_audio_to_client(self, audio_data: bytes):
        """Send binary audio to client"""
        if self.websocket:
            await self.websocket.send_bytes(audio_data)

    async def on_realtime_update(self, text: str):
        await self.send_text_message({"type": "stt_update", "text": text})
    
    async def on_realtime_stabilized(self, text: str):
        await self.send_text_message({"type": "stt_stabilized", "text": text})
    
    async def on_final_transcription(self, user_message: str):
        
        # put transcribed text into queue for llm to get
        await self.queues.transcribed_text.put(user_message)

        # send final text to client for user's prompt UI display
        await self.send_text_message({"type": "stt_final", "text": user_message})

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
