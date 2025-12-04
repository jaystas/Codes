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
from backend.boson_multimodal.data_types import ChatMLSample, AudioContent
from backend.boson_multimodal.data_types import Message as HiggsMessage  # Alias to avoid conflict
from backend.RealtimeTTS.threadsafe_generators import CharIterator, AccumulatingThreadSafeGenerator

# Streaming pipeline imports
from concurrent.futures import ThreadPoolExecutor
import stream2sentence as s2s

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
##--      Streaming Data Classes    --##
########################################

@dataclass
class SentenceChunk:
    """A complete sentence ready for TTS"""
    text: str
    speaker: Character
    voice: Optional[str]
    sequence_number: int
    sentence_index: int      # Which sentence in this response (0, 1, 2...)
    is_final: bool           # Last sentence of this character's response
    timestamp: float = field(default_factory=time.time)


@dataclass
class TextChunk:
    """Text chunk for UI streaming"""
    text: str
    is_final: bool
    speaker_name: str
    sequence_number: int
    chunk_index: int
    timestamp: float = field(default_factory=time.time)


########################################
##--             Queues             --##
########################################

class Queues:
    """Manages all queues"""

    def __init__(self):

        # STT → LLM Orchestrator
        self.transcribed_text = asyncio.Queue()

        # LLM → Browser (for text display) - renamed for consistency
        self.text_output = asyncio.Queue()

        # LLM → TTS (sentence chunks, not full responses) - renamed for clarity
        self.tts_requests = asyncio.Queue()

        # TTS → Audio Sequencer - renamed for consistency
        self.audio_output = asyncio.Queue()

        # Control signals
        self.stop_signal = asyncio.Event()
        self.interrupt_signal = asyncio.Event()


########################################
##-- Streaming Sentence Extractor   --##
########################################

class StreamingSentenceExtractor:
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
        # stream2sentence parameters - tuned for TTS latency
        minimum_sentence_length: int = 10,
        minimum_first_fragment_length: int = 10,
        quick_yield_single_sentence_fragment: bool = True,
        cleanup_text_links: bool = True,
        cleanup_text_emojis: bool = False,
        tokenize_sentences: bool = False,
    ):
        """
        Args:
            sentence_callback: Called with (sentence_text, sentence_index) for each sentence
            minimum_sentence_length: Min chars before yielding a sentence
            minimum_first_fragment_length: Min chars for first fragment
            quick_yield_single_sentence_fragment: Yield single sentences quickly
            cleanup_text_links: Remove URLs from text
            cleanup_text_emojis: Remove emojis from text
            tokenize_sentences: Use NLTK tokenization (slower, more accurate)
        """
        self.sentence_callback = sentence_callback

        # stream2sentence config
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
        self.sentence_queue: Queue = Queue()

        # Control
        self._extraction_thread: Optional[threading.Thread] = None
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

        # Start extraction thread
        self._extraction_thread = threading.Thread(
            target=self._extraction_loop,
            daemon=True
        )
        self._extraction_thread.start()

    def feed_text(self, text: str):
        """
        Feed text chunk from LLM stream.
        Thread-safe, can be called from async context.
        """
        if not self._is_running:
            return

        self._accumulated_text += text
        self.char_iter.add(text)

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
            try:
                self.char_iter.complete()  # If this method exists
            except AttributeError:
                pass

    def _extraction_loop(self):
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
            logger.error(f"Sentence extraction error: {e}", exc_info=True)
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
                result = await loop.run_in_executor(
                    self._executor,
                    lambda: self.sentence_queue.get(timeout=0.02)
                )

                if result is None:  # Sentinel - extraction complete
                    break

                yield result

            except Empty:
                # No sentence ready yet, yield control
                await asyncio.sleep(0.005)

                # Check if we should exit
                if not self._is_running and self.sentence_queue.empty():
                    break

    def get_accumulated_text(self) -> str:
        """Get all text that was fed to the extractor"""
        return self._accumulated_text

    def shutdown(self):
        """Clean shutdown"""
        self._is_running = False
        if self._extraction_thread and self._extraction_thread.is_alive():
            self._extraction_thread.join(timeout=1.0)
        self._executor.shutdown(wait=False)


########################################
##--    LLM Response Streamer       --##
########################################

class LLMResponseStreamer:
    """
    Manages the complete streaming pipeline for a single character response.

    Flow:
    1. Starts LLM generation
    2. Streams text chunks to UI queue AND sentence extractor
    3. Sentences are queued for TTS as soon as detected
    4. Tracks completion and handles cleanup

    This is instantiated once per character response within LLM Service.
    """

    def __init__(
        self,
        character: Character,
        voice: Optional[str],
        sequence_number: int,
        text_output_queue: asyncio.Queue,
        tts_request_queue: asyncio.Queue,
        interrupt_signal: asyncio.Event,
    ):
        self.character = character
        self.voice = voice
        self.sequence_number = sequence_number
        self.text_output_queue = text_output_queue
        self.tts_request_queue = tts_request_queue
        self.interrupt_signal = interrupt_signal

        # Sentence extractor
        self.extractor = StreamingSentenceExtractor()

        # Tracking
        self.chunk_index = 0
        self.sentence_index = 0
        self.total_text = ""
        self.is_complete = False

    async def stream_response(
        self,
        llm_stream: AsyncIterator,  # OpenAI streaming response
    ) -> str:
        """
        Process LLM stream, extract sentences, queue for TTS.

        Args:
            llm_stream: Async iterator from OpenAI client

        Returns:
            Complete response text
        """
        # Start sentence extraction pipeline
        self.extractor.start()

        # Create task for sentence-to-TTS processing
        sentence_task = asyncio.create_task(self._process_sentences())

        try:
            # Stream from LLM
            async for chunk in llm_stream:
                # Check for interrupt
                if self.interrupt_signal.is_set():
                    logger.warning(f"⚠️ Interrupt during LLM stream for {self.character.name}")
                    break

                # Extract content from OpenAI chunk
                content = chunk.choices[0].delta.content
                if content:
                    self.total_text += content

                    # Feed to sentence extractor (non-blocking)
                    self.extractor.feed_text(content)

                    # Stream to UI immediately
                    text_chunk = TextChunk(
                        text=content,
                        is_final=False,
                        speaker_name=self.character.name,
                        sequence_number=self.sequence_number,
                        chunk_index=self.chunk_index,
                        timestamp=time.time()
                    )
                    await self.text_output_queue.put(text_chunk)
                    self.chunk_index += 1

            # Signal LLM stream complete
            self.extractor.finish()

            # Send final text chunk to UI
            final_text_chunk = TextChunk(
                text="",
                is_final=True,
                speaker_name=self.character.name,
                sequence_number=self.sequence_number,
                chunk_index=self.chunk_index,
                timestamp=time.time()
            )
            await self.text_output_queue.put(final_text_chunk)

            # Wait for sentence processing to complete
            await sentence_task

        except Exception as e:
            logger.error(f"Error in stream_response for {self.character.name}: {e}", exc_info=True)
            raise
        finally:
            self.extractor.shutdown()
            self.is_complete = True

        return self.total_text

    async def _process_sentences(self):
        """
        Process sentences as they're extracted and queue for TTS.
        Runs concurrently with LLM streaming.
        """
        sentences_queued = []

        try:
            async for sentence, index in self.extractor.get_sentences():
                # Check for interrupt
                if self.interrupt_signal.is_set():
                    break

                sentences_queued.append(sentence)
                self.sentence_index = index

                # Determine if this might be the last sentence
                # (We don't know for sure until extractor finishes)
                is_final = False  # Will be corrected below

                # Queue for TTS immediately
                sentence_chunk = SentenceChunk(
                    text=sentence,
                    speaker=self.character,
                    voice=self.voice,
                    sequence_number=self.sequence_number,
                    sentence_index=index,
                    is_final=is_final,
                    timestamp=time.time()
                )
                await self.tts_request_queue.put(sentence_chunk)

                logger.info(f"📝 Sentence {index} queued for TTS ({self.character.name}): "
                      f"{sentence[:50]}{'...' if len(sentence) > 50 else ''}")

            # Mark the last sentence as final
            if sentences_queued:
                # Send a "final" marker for this character's TTS
                final_marker = SentenceChunk(
                    text="",  # Empty text signals completion
                    speaker=self.character,
                    voice=self.voice,
                    sequence_number=self.sequence_number,
                    sentence_index=self.sentence_index + 1,
                    is_final=True,
                    timestamp=time.time()
                )
                await self.tts_request_queue.put(final_marker)

        except Exception as e:
            logger.error(f"Error processing sentences for {self.character.name}: {e}", exc_info=True)


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

    async def run_transcription_loop(self):
        """
        Active loop that waits for and processes final transcriptions.

        This is necessary because RealtimeSTT doesn't provide an on_final_transcription
        callback. Instead, recorder.text() is a blocking call that returns when
        speech ends and transcription is complete.
        """
        logger.info("🎤 STT transcription loop started")

        while not self._is_shutting_down:
            try:
                if not self._is_initialized or self._recorder is None:
                    await asyncio.sleep(0.1)
                    continue

                # Run blocking text() call in executor
                loop = asyncio.get_running_loop()
                user_message = await loop.run_in_executor(
                    None,
                    self._recorder.text
                )

                # Got a transcription!
                if user_message:
                    logger.debug(f"STT loop got transcription: {user_message}")

                    with self._lock:
                        self._current_transcription = user_message

                    # Fire the on_final_transcription callback
                    if self.callbacks.on_final_transcription:
                        await self._run_callback_async(
                            self.callbacks.on_final_transcription,
                            user_message
                        )

            except Exception as e:
                if not self._is_shutting_down:
                    logger.error(f"Error in STT transcription loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

        logger.info("🎤 STT transcription loop ended")

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
    """LLM Service with streaming sentence extraction"""

    def __init__(self, api_key: str, queues: Queues):
        self.is_initialized = False
        self.queues = queues
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model: str = "anthropic/claude-3.5-sonnet"

        # Conversation state (simplified for now)
        self.conversation_history: List[Message] = []
        self.current_character: Optional[Character] = None
        self.sequence_number = 0

    async def initialize(self):
        self.is_initialized = True
        logger.info("LLMService initialized")

    def set_model(self, model: str):
        """Set the current LLM model"""
        self.model = model
        logger.info(f"LLM model set to: {model}")

    def set_character(self, character: Character):
        """Set current active character"""
        self.current_character = character
        logger.info(f"Active character set to: {character.name}")

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

        return messages

    async def run_llm_loop(self):
        """Main loop: consume transcriptions, generate streaming responses"""
        logger.info("🚀 LLM Service loop started")

        while not self.queues.stop_signal.is_set():
            try:
                # Wait for user transcription
                user_message = await asyncio.wait_for(
                    self.queues.transcribed_text.get(),
                    timeout=0.1
                )

                logger.info(f"💬 LLM received transcription: {user_message}")

                # Add to conversation history
                self.conversation_history.append(Message(
                    role="user",
                    name="User",
                    content=user_message
                ))

                # Generate streaming response
                await self._generate_streaming_response(user_message)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"LLM loop error: {e}", exc_info=True)
                continue

    async def _generate_streaming_response(self, user_message: str):
        """Generate streaming response with real-time sentence extraction"""

        if not self.current_character:
            logger.warning("No character set for LLM response")
            return

        self.sequence_number += 1
        character = self.current_character

        logger.info(f"🎭 Generating streaming response for {character.name} (seq {self.sequence_number})")

        # Build prompt
        messages = self.build_character_prompt(character, self.conversation_history)

        try:
            # Create LLM stream
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.7,
            )

            # Create response streamer
            streamer = LLMResponseStreamer(
                character=character,
                voice=character.voice,  # Pass voice string
                sequence_number=self.sequence_number,
                text_output_queue=self.queues.text_output,
                tts_request_queue=self.queues.tts_requests,
                interrupt_signal=self.queues.interrupt_signal,
            )

            # Process stream - sentences queued for TTS as they're detected
            response_text = await streamer.stream_response(stream)

            # Strip character tags if present
            clean_text = self.strip_character_tags(response_text)

            # Add to conversation history
            self.conversation_history.append(Message(
                role="assistant",
                name=character.name,
                content=clean_text
            ))

            logger.info(f"✅ {character.name} streaming complete ({streamer.sentence_index + 1} sentences)")

        except Exception as e:
            logger.error(f"Error generating response for {character.name}: {e}", exc_info=True)

    async def shutdown(self):
        """Clean up LLM resources"""
        self.is_initialized = False
        logger.info("LLMService shut down")

########################################
##--           TTS Service          --##
########################################

class TTSService:
    """Text-to-Speech Service with streaming sentence processing"""

    def __init__(self, queues: Queues):
        self.is_initialized = False
        self.queues = queues
        self.sample_rate: int = 24000
        self.serve_engine = None

        # Higgs model paths
        self.model_path = "bosonai/higgs-audio-v2-generation-3B-base"
        self.tokenizer_path = "bosonai/higgs-audio-v2-tokenizer"
        self.device = "cuda"
        self.chunk_size = 64  # Audio tokens per chunk

        # Semaphore to limit concurrent GPU operations
        self.generation_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent

        # Track active generation tasks
        self.active_tasks: Dict[tuple, asyncio.Task] = {}

    async def initialize(self):
        """Initialize the TTS model/service (Higgs)"""
        try:
            logger.info("Initializing Higgs Audio TTS engine...")
            loop = asyncio.get_event_loop()
            self.serve_engine = await loop.run_in_executor(None, self._create_engine)

            self.is_initialized = True
            logger.info("✅ TTSService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}", exc_info=True)
            raise

    def _create_engine(self):
        """Create Higgs engine (runs in thread pool)"""
        logger.info("Loading Higgs Audio engine...")
        engine = HiggsAudioServeEngine(
            self.model_path,
            self.tokenizer_path,
            device=self.device,
            torch_dtype=torch.bfloat16
        )
        logger.info("Higgs Audio engine loaded")
        return engine

    async def run_tts_loop(self):
        """Main loop: process sentence chunks as they arrive"""
        logger.info("🚀 TTS Service loop started")

        while not self.queues.stop_signal.is_set():
            try:
                # Get sentence chunk
                sentence: SentenceChunk = await asyncio.wait_for(
                    self.queues.tts_requests.get(),
                    timeout=0.05  # Short timeout for responsiveness
                )

                # Check for interrupt
                if self.queues.interrupt_signal.is_set():
                    await self._cancel_all_tasks()
                    self.queues.interrupt_signal.clear()
                    continue

                # Skip empty final markers
                if not sentence.text.strip():
                    if sentence.is_final:
                        logger.info(f"✅ TTS complete signal for {sentence.speaker.name}")
                    continue

                logger.info(f"🎤 TTS received sentence {sentence.sentence_index}: {sentence.text[:50]}...")

                # Start TTS generation immediately for this sentence (concurrent)
                task_key = (
                    sentence.speaker.name,
                    sentence.sequence_number,
                    sentence.sentence_index
                )

                task = asyncio.create_task(
                    self._generate_sentence_audio(sentence)
                )
                self.active_tasks[task_key] = task

                # Cleanup completed tasks
                self._cleanup_completed_tasks()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"TTS loop error: {e}", exc_info=True)
                continue

    async def _generate_sentence_audio(self, sentence: SentenceChunk):
        """Generate audio for a single sentence with semaphore control"""
        async with self.generation_semaphore:
            logger.info(f"🎙️ Generating TTS: {sentence.speaker.name} sentence {sentence.sentence_index}")

            # Build system prompt (simplified - you can enhance with voice configs)
            system_prompt = (
                "Generate audio following instruction.\n\n"
                "<|scene_desc_start|>\n"
                "Audio is recorded from a quiet room.\n"
                f"Speaker: {sentence.speaker.name}\n"
                "<|scene_desc_end|>"
            )

            # Build messages for Higgs
            messages = [
                HiggsMessage(
                    role="system",
                    content=system_prompt
                ),
                HiggsMessage(
                    role="user",
                    content=sentence.text
                )
            ]

            audio_token_buffer = []
            chunk_index = 0

            try:
                # Generate audio stream
                streamer = self.serve_engine.generate_delta_stream(
                    chat_ml_sample=ChatMLSample(messages=messages),
                    temperature=0.75,
                    top_p=0.95,
                    top_k=50,
                    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                    force_audio_gen=True
                )

                async for delta in streamer:
                    # Check for interrupt
                    if self.queues.interrupt_signal.is_set():
                        logger.warning(f"⚠️ TTS interrupted for {sentence.speaker.name}")
                        break

                    # Buffer audio tokens
                    if delta.audio_tokens is not None:
                        audio_token_buffer.append(delta.audio_tokens)

                        # Emit chunk when buffer full
                        if len(audio_token_buffer) >= self.chunk_size:
                            pcm_chunk = await self._process_audio_tokens(
                                audio_token_buffer[:self.chunk_size]
                            )

                            if pcm_chunk:
                                await self._emit_audio_chunk(
                                    pcm_chunk,
                                    sentence,
                                    chunk_index,
                                    is_final=False
                                )
                                chunk_index += 1

                            # Keep overlap for smooth transitions
                            num_codebooks = delta.audio_tokens.shape[0]
                            tokens_to_keep = num_codebooks - 1
                            audio_token_buffer = audio_token_buffer[
                                self.chunk_size - tokens_to_keep:
                            ]

                    # Check for end token
                    if delta.text == "<|eot_id|>":
                        break

                # Process remaining tokens
                if audio_token_buffer and not self.queues.interrupt_signal.is_set():
                    pcm_chunk = await self._process_audio_tokens(audio_token_buffer)
                    if pcm_chunk:
                        await self._emit_audio_chunk(
                            pcm_chunk,
                            sentence,
                            chunk_index,
                            is_final=True
                        )

                logger.info(f"✅ TTS complete: {sentence.speaker.name} sentence {sentence.sentence_index}")

            except asyncio.CancelledError:
                logger.warning(f"⚠️ TTS cancelled for {sentence.speaker.name}")
            except Exception as e:
                logger.error(f"TTS generation error: {e}", exc_info=True)

    async def _emit_audio_chunk(
        self,
        pcm_data: bytes,
        sentence: SentenceChunk,
        chunk_index: int,
        is_final: bool
    ):
        """Emit audio chunk to output queue"""
        audio_chunk = {
            "data": pcm_data,
            "sample_rate": self.serve_engine.audio_tokenizer.sampling_rate,
            "sequence_number": sentence.sequence_number,
            "sentence_index": sentence.sentence_index,
            "chunk_index": chunk_index,
            "is_final": is_final,
            "speaker_name": sentence.speaker.name,
            "timestamp": time.time()
        }

        await self.queues.audio_output.put(audio_chunk)
        logger.debug(f"Audio chunk emitted: seq={sentence.sequence_number} sent={sentence.sentence_index} chunk={chunk_index}")

    async def _process_audio_tokens(self, tokens):
        """Convert audio tokens to PCM (run in executor)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._decode_tokens, tokens)

    def _decode_tokens(self, tokens):
        """Decode audio tokens to PCM16"""
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
            logger.error(f"Token decode error: {e}")
            return None

    async def _cancel_all_tasks(self):
        """Cancel all active TTS tasks"""
        logger.info("Cancelling all TTS tasks")
        for task in self.active_tasks.values():
            task.cancel()
        self.active_tasks.clear()

    def _cleanup_completed_tasks(self):
        """Remove completed tasks from tracking dict"""
        completed = [k for k, v in self.active_tasks.items() if v.done()]
        for key in completed:
            del self.active_tasks[key]

    async def shutdown(self):
        """Clean up TTS resources"""
        await self._cancel_all_tasks()
        self.is_initialized = False
        logger.info("TTSService shut down")

########################################
##--        WebSocket Manager       --##
########################################

class WebSocketManager:
    """Manages WebSocket connections and their associated state"""

    def __init__(self):
        self.stt_service: Optional[STTService] = None
        self.llm_service: Optional[LLMService] = None
        self.tts_service: Optional[TTSService] = None
        self.websocket: Optional[WebSocket] = None
        self.queues: Optional[Queues] = None

        # Track service tasks
        self.service_tasks: List[asyncio.Task] = []

        # Current character (simplified - can enhance later)
        self.current_character: Optional[Character] = None

    async def initialize(self):
        """Initialize all services with proper callbacks"""

        # Create queues first
        self.queues = Queues()

        # Setup STT callbacks
        stt_callbacks = STTCallbacks(
            on_realtime_update=self.on_realtime_update,
            on_realtime_stabilized=self.on_realtime_stabilized,
            on_final_transcription=self.on_final_transcription,
            on_vad_start=self.on_vad_start,  # For interrupt detection
        )

        # Initialize services
        self.stt_service = STTService(callbacks=stt_callbacks)

        # Get OpenRouter API key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.error("OPENROUTER_API_KEY environment variable not set!")
            raise ValueError("OPENROUTER_API_KEY environment variable required")

        self.llm_service = LLMService(
            api_key=api_key,
            queues=self.queues
        )
        self.tts_service = TTSService(queues=self.queues)

        # Initialize each service
        await self.stt_service.initialize()
        await self.llm_service.initialize()
        await self.tts_service.initialize()

        logger.info("✅ All services initialized")

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection and start service tasks"""
        await websocket.accept()
        self.websocket = websocket

        # Start service background tasks
        await self.start_service_tasks()

        logger.info("✅ WebSocket connected and services started")

    async def start_service_tasks(self):
        """Start background service loops"""

        # STT: Start listening AND transcription loop
        if self.stt_service:
            self.stt_service.start_listening()
            logger.info("🎤 STT listening started")

            # Start transcription loop that calls recorder.text()
            stt_task = asyncio.create_task(self.stt_service.run_transcription_loop())
            self.service_tasks.append(stt_task)
            logger.info("🎤 STT transcription loop task started")

        # LLM: Active loop consuming transcription queue
        if self.llm_service:
            llm_task = asyncio.create_task(self.llm_service.run_llm_loop())
            self.service_tasks.append(llm_task)
            logger.info("🧠 LLM loop task started")

        # TTS: Active loop consuming sentence queue
        if self.tts_service:
            tts_task = asyncio.create_task(self.tts_service.run_tts_loop())
            self.service_tasks.append(tts_task)
            logger.info("🔊 TTS loop task started")

        # Audio output: Send to WebSocket
        audio_task = asyncio.create_task(self.audio_output_loop())
        self.service_tasks.append(audio_task)
        logger.info("📡 Audio output loop task started")

        # Text output: Send to WebSocket
        text_task = asyncio.create_task(self.text_output_loop())
        self.service_tasks.append(text_task)
        logger.info("💬 Text output loop task started")

        logger.info(f"✅ All {len(self.service_tasks)} service tasks running")

    async def disconnect(self):
        """Handle WebSocket disconnection and cleanup"""
        logger.info("🔌 Disconnecting WebSocket...")

        # Signal stop
        if self.queues:
            self.queues.stop_signal.set()

        # Cancel all service tasks
        for task in self.service_tasks:
            task.cancel()

        # Wait for cancellation (with timeout)
        if self.service_tasks:
            await asyncio.wait(self.service_tasks, timeout=2.0)

        # Shutdown services
        if self.stt_service:
            await self.stt_service.shutdown()
        if self.llm_service:
            await self.llm_service.shutdown()
        if self.tts_service:
            await self.tts_service.shutdown()

        # Close WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None

        logger.info("✅ WebSocket disconnected and services shut down")

    # =========================================================================
    # Message Handlers
    # =========================================================================

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

            if message_type == "set_character":
                # Set current character for responses
                char_data = payload.get("character", {})
                self.current_character = Character(
                    id=char_data.get("id", "default"),
                    name=char_data.get("name", "Assistant"),
                    voice=char_data.get("voice", ""),
                    system_prompt=char_data.get("system_prompt", "You are a helpful assistant."),
                    image_url=char_data.get("image_url", ""),
                    images=char_data.get("images", []),
                    is_active=True
                )
                self.llm_service.set_character(self.current_character)
                logger.info(f"Character set: {self.current_character.name}")

            elif message_type == "set_model":
                model = payload.get("model", "")
                self.llm_service.set_model(model)

            elif message_type == "start_listening":
                if self.stt_service:
                    self.stt_service.start_listening()

            elif message_type == "stop_listening":
                if self.stt_service:
                    self.stt_service.stop_listening()

            elif message_type == "interrupt":
                # User interrupted - stop TTS
                if self.queues:
                    self.queues.interrupt_signal.set()
                logger.info("Interrupt signal set")

        except Exception as e:
            logger.error(f"Error handling text message: {e}", exc_info=True)

    # =========================================================================
    # Output Loops (send to WebSocket)
    # =========================================================================

    async def audio_output_loop(self):
        """Send audio chunks to WebSocket client"""
        logger.info("📡 Audio output loop started")

        while not self.queues.stop_signal.is_set():
            try:
                audio_chunk = await asyncio.wait_for(
                    self.queues.audio_output.get(),
                    timeout=0.1
                )

                if self.websocket:
                    # Send binary audio data
                    await self.websocket.send_bytes(audio_chunk["data"])
                    logger.debug(f"Sent audio chunk: {len(audio_chunk['data'])} bytes")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Audio output error: {e}", exc_info=True)
                break

        logger.info("📡 Audio output loop ended")

    async def text_output_loop(self):
        """Send text chunks to WebSocket client for UI display"""
        logger.info("💬 Text output loop started")

        while not self.queues.stop_signal.is_set():
            try:
                text_chunk: TextChunk = await asyncio.wait_for(
                    self.queues.text_output.get(),
                    timeout=0.1
                )

                if self.websocket:
                    # Send text chunk as JSON
                    await self.send_text_message({
                        "type": "llm_chunk",
                        "text": text_chunk.text,
                        "is_final": text_chunk.is_final,
                        "speaker": text_chunk.speaker_name,
                        "sequence": text_chunk.sequence_number,
                        "chunk_index": text_chunk.chunk_index
                    })

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Text output error: {e}", exc_info=True)
                break

        logger.info("💬 Text output loop ended")

    async def send_text_message(self, data: dict):
        """Send JSON message to client"""
        if self.websocket:
            try:
                await self.websocket.send_text(json.dumps(data))
            except Exception as e:
                logger.error(f"Error sending text message: {e}")

    # =========================================================================
    # STT Callbacks
    # =========================================================================

    async def on_realtime_update(self, text: str):
        """Realtime STT update (may change)"""
        await self.send_text_message({"type": "stt_update", "text": text})

    async def on_realtime_stabilized(self, text: str):
        """Realtime STT stabilized (less likely to change)"""
        await self.send_text_message({"type": "stt_stabilized", "text": text})

    async def on_final_transcription(self, user_message: str):
        """Final transcription ready - feed to LLM"""
        logger.info(f"✅ Final transcription: {user_message}")

        # Put into queue for LLM to consume
        await self.queues.transcribed_text.put(user_message)

        # Send to client for UI display
        await self.send_text_message({"type": "stt_final", "text": user_message})

    async def on_vad_start(self):
        """Voice activity detected - potentially interrupt TTS"""
        logger.debug("VAD: Voice detected")
        # Could trigger interrupt here if desired
        # self.queues.interrupt_signal.set()

########################################
##--           FastAPI App          --##
########################################

ws_manager = WebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 FastAPI application starting up...")
    await ws_manager.initialize()
    logger.info("✅ All services initialized!")
    yield
    logger.info("🔌 FastAPI application shutting down...")
    # Services are shut down per-connection in disconnect()
    logger.info("✅ Shutdown complete!")

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
