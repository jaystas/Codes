# FastAPI Server Refactoring Plan

## Executive Summary

This document outlines a systematic approach to refactoring `backend/fastapi_server.py` into a clean, maintainable, low-latency architecture based on async generators, asyncio queues, and callbacks for concurrent generation across STT, LLM, and TTS services.

---

## Part 1: Current State Analysis

### 1.1 File Structure Overview

The current `fastapi_server.py` (~1134 lines) contains:

| Component | Lines | Purpose |
|-----------|-------|---------|
| Imports | 1-42 | Mixed stdlib, third-party, local |
| Data Models | 55-140 | Pydantic models & dataclasses |
| Queues | 146-159 | Queue container class |
| STTService | 165-324 | Speech-to-text (threaded) |
| TextToSentence | 329-488 | Sentence extraction bridge |
| LLMService | 494-830 | Multi-character LLM loop |
| TTSService | 835-836 | **Placeholder only** |
| WebSocketManager | 842-1073 | Main orchestrator |
| FastAPI App | 1079-1133 | Server setup & endpoints |

### 1.2 Current Data Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  WebSocket  │───▶│ STTService  │───▶│ LLMService  │───▶│ TTSService  │
│   Client    │    │  (Thread)   │    │   (Async)   │    │ (Placeholder)│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                 │                  │                   │
       │                 ▼                  ▼                   ▼
       │          transcribed_text   response_queue       audio_queue
       │              Queue              Queue               Queue
       │                                   │
       │                            TextToSentence
       │                           (Thread + Queue)
       │                                   │
       │                                   ▼
       │                          tts_sentence_queue
       │                               Queue
       └───────────────────────────────────────────────────────────────▶
                              WebSocket send (text/binary)
```

### 1.3 Current Queue Usage

| Queue | Producer | Consumer | Purpose |
|-------|----------|----------|---------|
| `transcribed_text` | STTService callback | LLMService.main_llm_loop | User speech → text |
| `response_queue` | LLMService | WebSocketManager.stream_text_to_client | Text chunks to UI |
| `sentence_queue` | TextToSentence | (unused in current code) | Extracted sentences |
| `tts_sentence_queue` | LLMService.process_sentences_for_tts | TTSService (missing) | Sentences for TTS |
| `audio_queue` | TTSService (missing) | WebSocketManager | Audio chunks to client |

---

## Part 2: Critical Issues Identified

### 2.1 Bugs & Errors

| Location | Issue | Severity |
|----------|-------|----------|
| Line 804 | `voice` variable undefined in `process_sentences_for_tts` | **Critical** |
| Line 42 | Imports `tts_service.py` but file was deleted | **Critical** |
| Line 270, 276 | `is_listening` changed from `Event` to `bool` (type mismatch) | **High** |
| Line 882-883 | Calls `TTSService(queues=...)` but class takes no args | **Critical** |
| Line 904 | Calls `tts_service.main_tts_loop()` which doesn't exist | **Critical** |

### 2.2 Architectural Problems

| Issue | Description |
|-------|-------------|
| **Single global WebSocketManager** | Line 1079: One instance shared across all connections - no client isolation |
| **Mixed concurrency models** | Threading (STT), asyncio (LLM), thread pools (TextToSentence) |
| **No error boundaries** | Service failures cascade without graceful degradation |
| **Hardcoded credentials** | Lines 44-45 (Supabase), Line 501 (OpenRouter API key) |
| **TTSService is a stub** | Lines 835-836: Only a placeholder class, no implementation |
| **Interrupt event never cleared** | Once set, remains set forever |

### 2.3 Code Quality Issues

```python
# Unused imports (lines 1-42):
import queue          # Uses asyncio.Queue instead
import multiprocessing # Not used
import aiohttp        # Not used
import requests       # Not used
from loguru import logger  # Overwritten by logging.getLogger on line 50

# Duplicate/conflicting logging setup:
from loguru import logger          # Line 34
logger = logging.getLogger(__name__)  # Line 50 (overwrites loguru)
```

### 2.4 Missing Functionality

- **TTSService**: Complete implementation needed
- **Connection isolation**: Each WebSocket needs independent service instances
- **Graceful shutdown**: Proper cleanup of threads and tasks
- **Health checks**: No monitoring or status endpoints
- **Configuration management**: No centralized config

---

## Part 3: Target Architecture

### 3.1 Proposed File Structure

```
backend/
├── server/
│   ├── __init__.py
│   ├── app.py                 # FastAPI app & lifespan
│   ├── config.py              # Configuration management
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── websocket.py       # WebSocket endpoint
│   │   └── health.py          # Health check endpoints
│   └── middleware/
│       └── __init__.py
├── services/
│   ├── __init__.py
│   ├── base.py                # Base service class
│   ├── stt_service.py         # Speech-to-Text
│   ├── llm_service.py         # Language Model
│   ├── tts_service.py         # Text-to-Speech
│   └── sentence_extractor.py  # TextToSentence
├── managers/
│   ├── __init__.py
│   ├── websocket_manager.py   # WebSocket orchestration
│   ├── queue_manager.py       # Queue coordination
│   └── connection_manager.py  # Multi-client handling
├── models/
│   ├── __init__.py
│   ├── characters.py          # Character, Voice models
│   ├── messages.py            # Message dataclasses
│   ├── settings.py            # ModelSettings
│   └── audio.py               # AudioChunk, etc.
└── utils/
    ├── __init__.py
    ├── callbacks.py           # Callback utilities
    └── logging.py             # Logging configuration
```

### 3.2 Target Data Flow

```
                              ┌──────────────────────────────────────────┐
                              │         ConnectionManager                │
                              │   (Manages per-client sessions)          │
                              └──────────────────────────────────────────┘
                                              │
                     ┌──────────────────────────────────────────┐
                     │         Per-Connection Session           │
                     └──────────────────────────────────────────┘
                                              │
    ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
    │                                         │                                         │
    ▼                                         ▼                                         ▼
┌────────────┐                        ┌────────────────┐                        ┌────────────┐
│ STTService │                        │   LLMService   │                        │ TTSService │
│            │                        │                │                        │            │
│ async feed_audio()                  │ async stream_response()                 │ async main_loop()
│   │                                 │   │                                     │   │
│   ▼                                 │   ▼                                     │   ▼
│ ┌──────────────┐                    │ ┌──────────────┐                        │ ┌──────────────┐
│ │ Transcription│                    │ │ OpenAI Stream│                        │ │ TTS Engine   │
│ │    Thread    │                    │ │   (async)    │                        │ │   Thread     │
│ └──────────────┘                    │ └──────────────┘                        │ └──────────────┘
│   │                                 │   │                                     │   │
│   ▼ callback                        │   ▼ async generator                     │   ▼ callback
└───┼────────────┘                    └───┼────────────────┘                    └───┼────────────┘
    │                                     │                                         │
    ▼                                     ▼                                         ▼
┌────────────┐                     ┌─────────────────┐                       ┌────────────┐
│transcribed │                     │SentenceExtractor│                       │audio_queue │
│   Queue    │                     │                 │                       │  (asyncio) │
│  (asyncio) │                     │ async sentences()                       └────────────┘
└────────────┘                     │      │                                        │
    │                              │      ▼                                        │
    │                              │┌─────────────┐                                │
    │                              ││tts_sentence │                                │
    │                              ││   Queue     │                                │
    │                              │└─────────────┘                                │
    │                              └───────┼───────┘                               │
    │                                      │                                       │
    ▼                                      ▼                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              WebSocketManager                                        │
│                                                                                      │
│  • Orchestrates service lifecycle                                                    │
│  • Routes messages between services                                                  │
│  • Handles client communication                                                      │
│  • Manages interrupts & cancellation                                                 │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                                        ┌──────────┐
                                        │WebSocket │
                                        │  Client  │
                                        └──────────┘
```

### 3.3 Service Interface Pattern

Each service follows a consistent async interface:

```python
class BaseService(ABC):
    """Base class for all services"""

    def __init__(self, queues: QueueManager, config: ServiceConfig):
        self.queues = queues
        self.config = config
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

    @abstractmethod
    async def initialize(self) -> None:
        """One-time initialization"""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the service main loop"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Graceful shutdown"""
        pass

    async def interrupt(self) -> None:
        """Handle generation interrupt"""
        pass
```

---

## Part 4: Implementation Plan

### Phase 1: Foundation (Prepare for Refactor)

#### Step 1.1: Create Configuration Module

Create `backend/server/config.py`:

```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # API Keys
    openrouter_api_key: str = ""
    supabase_url: str = ""
    supabase_anon_key: str = ""

    # STT Settings
    stt_model: str = "small.en"
    stt_language: str = "en"

    # LLM Defaults
    default_llm_model: str = "meta-llama/llama-3.1-8b-instruct"
    default_temperature: float = 0.7

    # TTS Settings
    tts_engine: str = "kokoro"  # or "orpheus", "openai", etc.
    tts_sample_rate: int = 24000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

#### Step 1.2: Create Models Module

Move dataclasses to `backend/models/`:

```python
# backend/models/characters.py
from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import List, Optional
import time

class Character(BaseModel):
    id: str
    name: str
    voice: str = ""
    system_prompt: str = ""
    image_url: str = ""
    images: List[str] = []
    is_active: bool = True

@dataclass
class Voice:
    voice: str
    method: str
    speaker_desc: str
    scene_prompt: str
    audio_path: str = ""
    text_path: str = ""
```

```python
# backend/models/messages.py
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class TextChunk:
    text: str
    message_id: str
    character_name: str
    chunk_index: int
    is_final: bool
    timestamp: float = field(default_factory=time.time)

@dataclass
class SentenceTTS:
    text: str
    speaker: 'Character'
    voice: Optional['Voice']
    sentence_index: int
    is_final: bool
    timestamp: float = field(default_factory=time.time)

@dataclass
class AudioChunk:
    chunk_id: str
    message_id: str
    character_id: str
    character_name: str
    audio_data: bytes
    chunk_index: int
    is_final: bool
    timestamp: float = field(default_factory=time.time)
```

### Phase 2: Service Refactoring

#### Step 2.1: Implement Queue Manager

Create `backend/managers/queue_manager.py`:

```python
import asyncio
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class QueueManager:
    """Centralized queue management for pipeline stages"""

    # STT → LLM
    transcribed_text: asyncio.Queue = field(default_factory=asyncio.Queue)

    # LLM → WebSocket (text streaming)
    response_chunks: asyncio.Queue = field(default_factory=asyncio.Queue)

    # LLM → TTS (sentences)
    tts_sentences: asyncio.Queue = field(default_factory=asyncio.Queue)

    # TTS → WebSocket (audio)
    audio_chunks: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Control queues
    interrupt_signal: asyncio.Event = field(default_factory=asyncio.Event)

    def clear_all(self):
        """Clear all queues (for interrupt handling)"""
        for queue in [self.transcribed_text, self.response_chunks,
                      self.tts_sentences, self.audio_chunks]:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    def reset_interrupt(self):
        """Reset interrupt signal for new generation"""
        self.interrupt_signal.clear()
```

#### Step 2.2: Refactor STTService

Create `backend/services/stt_service.py`:

```python
import asyncio
import threading
import logging
from typing import Optional, Callable, Any
from dataclasses import dataclass

from backend.RealtimeSTT import AudioToTextRecorder
from backend.managers.queue_manager import QueueManager
from backend.server.config import settings

logger = logging.getLogger(__name__)

@dataclass
class STTCallbacks:
    on_realtime_update: Optional[Callable[[str], Any]] = None
    on_realtime_stabilized: Optional[Callable[[str], Any]] = None
    on_final_transcription: Optional[Callable[[str], Any]] = None
    on_vad_detect_start: Optional[Callable[[], Any]] = None
    on_vad_detect_stop: Optional[Callable[[], Any]] = None
    on_recording_start: Optional[Callable[[], Any]] = None
    on_recording_stop: Optional[Callable[[], Any]] = None

class STTService:
    """Speech-to-Text service using RealtimeSTT"""

    def __init__(self, queues: QueueManager):
        self.queues = queues
        self.recorder: Optional[AudioToTextRecorder] = None
        self._recording_thread: Optional[threading.Thread] = None
        self._is_listening = threading.Event()
        self._should_stop = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.callbacks = STTCallbacks()

    async def initialize(self) -> None:
        """Initialize STT service"""
        self._loop = asyncio.get_event_loop()

        self.recorder = AudioToTextRecorder(
            model=settings.stt_model,
            language=settings.stt_language,
            enable_realtime_transcription=True,
            realtime_processing_pause=0.1,
            realtime_model_type=settings.stt_model,
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

        self._recording_thread = threading.Thread(
            target=self._transcription_loop,
            daemon=True
        )
        self._recording_thread.start()
        logger.info("STT service initialized")

    def _transcription_loop(self) -> None:
        """Background thread for transcription"""
        while not self._should_stop.is_set():
            if self._is_listening.is_set():
                try:
                    text = self.recorder.text()
                    if text and text.strip():
                        logger.info(f"Final transcription: {text}")
                        self._on_final_transcription(text)
                    self._is_listening.clear()
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    self._is_listening.clear()
            else:
                self._should_stop.wait(timeout=0.05)

    def start_listening(self) -> None:
        """Start listening for voice"""
        self._is_listening.set()
        if self.recorder:
            self.recorder.listen()

    def stop_listening(self) -> None:
        """Stop listening"""
        self._is_listening.clear()

    def feed_audio(self, audio_bytes: bytes) -> None:
        """Feed raw PCM audio (16kHz, 16-bit, mono)"""
        if self.recorder:
            try:
                self.recorder.feed_audio(audio_bytes, original_sample_rate=16000)
            except Exception as e:
                logger.error(f"Failed to feed audio: {e}")

    async def shutdown(self) -> None:
        """Graceful shutdown"""
        self._should_stop.set()
        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join(timeout=2.0)
        logger.info("STT service shut down")

    # Callback handlers with thread-safe async scheduling
    def _schedule_callback(self, callback: Callable, *args):
        if self._loop is None or callback is None:
            return
        try:
            if asyncio.iscoroutinefunction(callback):
                self._loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(callback(*args))
                )
            else:
                self._loop.call_soon_threadsafe(callback, *args)
        except Exception as e:
            logger.error(f"Callback scheduling error: {e}")

    def _on_realtime_update(self, text: str):
        if self.callbacks.on_realtime_update:
            self._schedule_callback(self.callbacks.on_realtime_update, text)

    def _on_realtime_stabilized(self, text: str):
        if self.callbacks.on_realtime_stabilized:
            self._schedule_callback(self.callbacks.on_realtime_stabilized, text)

    def _on_final_transcription(self, text: str):
        # Put transcription in queue for LLM
        if self._loop:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    self.queues.transcribed_text.put(text)
                )
            )
        if self.callbacks.on_final_transcription:
            self._schedule_callback(self.callbacks.on_final_transcription, text)

    def _on_recording_start(self):
        if self.callbacks.on_recording_start:
            self._schedule_callback(self.callbacks.on_recording_start)

    def _on_recording_stop(self):
        if self.callbacks.on_recording_stop:
            self._schedule_callback(self.callbacks.on_recording_stop)

    def _on_vad_detect_start(self):
        if self.callbacks.on_vad_detect_start:
            self._schedule_callback(self.callbacks.on_vad_detect_start)

    def _on_vad_detect_stop(self):
        if self.callbacks.on_vad_detect_stop:
            self._schedule_callback(self.callbacks.on_vad_detect_stop)
```

#### Step 2.3: Implement TTSService

Create `backend/services/tts_service.py`:

```python
import asyncio
import logging
from typing import Optional, Callable, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from backend.RealtimeTTS import TextToAudioStream, HeadlessPlayer
from backend.RealtimeTTS.engines import KokoroEngine  # or other engine
from backend.managers.queue_manager import QueueManager
from backend.models.messages import SentenceTTS, AudioChunk
from backend.server.config import settings

logger = logging.getLogger(__name__)

class TTSService:
    """Text-to-Speech service using RealtimeTTS"""

    def __init__(self, queues: QueueManager):
        self.queues = queues
        self.engine = None
        self.stream: Optional[TextToAudioStream] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._is_running = False
        self._task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize TTS engine"""
        loop = asyncio.get_event_loop()

        # Initialize engine in thread pool (can be slow)
        self.engine = await loop.run_in_executor(
            self._executor,
            self._create_engine
        )

        # Create stream with headless player for WebSocket output
        self.stream = TextToAudioStream(
            engine=self.engine,
            player_backend="headless",
            on_audio_stream_start=self._on_audio_start,
            on_audio_stream_stop=self._on_audio_stop,
        )

        logger.info(f"TTS service initialized with {settings.tts_engine} engine")

    def _create_engine(self):
        """Create TTS engine (runs in thread pool)"""
        # Example with Kokoro - adjust based on your preferred engine
        from backend.RealtimeTTS.engines.kokoro_engine import KokoroEngine
        return KokoroEngine()

    async def start(self) -> None:
        """Start the main TTS processing loop"""
        self._is_running = True
        self._task = asyncio.create_task(self._main_loop())

    async def _main_loop(self) -> None:
        """Main loop: consume sentences, produce audio"""
        logger.info("TTS main loop started")

        while self._is_running:
            try:
                # Wait for sentence with timeout
                try:
                    sentence: SentenceTTS = await asyncio.wait_for(
                        self.queues.tts_sentences.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                # Check for interrupt
                if self.queues.interrupt_signal.is_set():
                    continue

                # Skip final markers with empty text
                if sentence.is_final and not sentence.text:
                    continue

                # Synthesize sentence to audio
                await self._synthesize_sentence(sentence)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"TTS loop error: {e}")

    async def _synthesize_sentence(self, sentence: SentenceTTS) -> None:
        """Synthesize a single sentence and stream audio chunks"""
        loop = asyncio.get_event_loop()
        message_id = f"audio-{sentence.sentence_index}"
        chunk_index = 0

        # Set voice if specified
        if sentence.voice:
            await loop.run_in_executor(
                self._executor,
                lambda: self.engine.set_voice(sentence.voice.voice)
            )

        # Create audio chunk callback
        async def on_audio_chunk(chunk_data: bytes):
            nonlocal chunk_index

            audio_chunk = AudioChunk(
                chunk_id=f"{message_id}-{chunk_index}",
                message_id=message_id,
                character_id=sentence.speaker.id,
                character_name=sentence.speaker.name,
                audio_data=chunk_data,
                chunk_index=chunk_index,
                is_final=False
            )
            await self.queues.audio_chunks.put(audio_chunk)
            chunk_index += 1

        # Synthesize with callback
        def synthesize_sync():
            self.stream.feed(sentence.text)
            self.stream.play(
                on_audio_chunk=lambda chunk: asyncio.run_coroutine_threadsafe(
                    on_audio_chunk(chunk), loop
                ),
                muted=True  # We handle audio via callback
            )

        await loop.run_in_executor(self._executor, synthesize_sync)

        # Send final chunk marker
        final_chunk = AudioChunk(
            chunk_id=f"{message_id}-final",
            message_id=message_id,
            character_id=sentence.speaker.id,
            character_name=sentence.speaker.name,
            audio_data=b"",
            chunk_index=chunk_index,
            is_final=True
        )
        await self.queues.audio_chunks.put(final_chunk)

    async def stop(self) -> None:
        """Stop TTS processing"""
        self._is_running = False
        if self.stream:
            self.stream.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def shutdown(self) -> None:
        """Full shutdown"""
        await self.stop()
        if self.engine:
            self.engine.shutdown()
        self._executor.shutdown(wait=False)
        logger.info("TTS service shut down")

    def _on_audio_start(self):
        logger.debug("TTS audio stream started")

    def _on_audio_stop(self):
        logger.debug("TTS audio stream stopped")
```

#### Step 2.4: Refactor LLMService

Key changes needed in `backend/services/llm_service.py`:

```python
import asyncio
import re
import time
import logging
from typing import List, Dict, Optional, AsyncIterator
from openai import AsyncOpenAI

from backend.managers.queue_manager import QueueManager
from backend.models.characters import Character, Voice
from backend.models.messages import TextChunk, SentenceTTS
from backend.models.settings import ModelSettings
from backend.services.sentence_extractor import SentenceExtractor
from backend.server.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """LLM Service for multi-character conversations"""

    def __init__(self, queues: QueueManager):
        self.queues = queues
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key
        )

        self.active_characters: List[Character] = []
        self.conversation_history: List[Dict[str, str]] = []
        self.model_settings: Optional[ModelSettings] = None

        # Per-response state
        self._current_extractor: Optional[SentenceExtractor] = None
        self._is_running = False
        self._task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize LLM service"""
        logger.info("LLM service initialized")

    async def start(self) -> None:
        """Start the main LLM loop"""
        self._is_running = True
        self._task = asyncio.create_task(self._main_loop())

    async def _main_loop(self, user_name: str = "User") -> None:
        """Main conversation loop"""
        logger.info("LLM main loop started")

        while self._is_running:
            try:
                # Wait for transcribed text
                try:
                    user_message = await asyncio.wait_for(
                        self.queues.transcribed_text.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                if not user_message or not user_message.strip():
                    continue

                # Reset interrupt for new generation
                self.queues.reset_interrupt()

                # Add to history
                self.conversation_history.append({
                    "role": "user",
                    "name": user_name,
                    "content": user_message
                })

                # Get characters to respond
                mentioned = self._parse_character_mentions(user_message)

                for character in mentioned:
                    if self.queues.interrupt_signal.is_set():
                        break

                    response = await self._generate_character_response(character)

                    if response:
                        self.conversation_history.append({
                            "role": "assistant",
                            "name": character.name,
                            "content": response
                        })

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"LLM loop error: {e}")

    async def _generate_character_response(self, character: Character) -> Optional[str]:
        """Generate streaming response for a character"""
        message_id = f"msg-{character.id}-{int(time.time() * 1000)}"

        # Build messages
        messages = [
            {"role": "system", "name": character.name, "content": character.system_prompt}
        ]
        messages.extend(self.conversation_history)
        messages.append(self._create_character_instruction(character))

        # Get model settings
        model_settings = self._get_model_settings()

        # Create sentence extractor
        self._current_extractor = SentenceExtractor()
        self._current_extractor.start()

        # Start sentence processing task
        sentence_task = asyncio.create_task(
            self._process_sentences(character, message_id)
        )

        full_response = ""
        chunk_index = 0

        try:
            # Stream from LLM
            stream = await self.client.chat.completions.create(
                model=model_settings.model,
                messages=messages,
                temperature=model_settings.temperature,
                top_p=model_settings.top_p,
                frequency_penalty=model_settings.frequency_penalty,
                presence_penalty=model_settings.presence_penalty,
                stream=True
            )

            async for chunk in stream:
                if self.queues.interrupt_signal.is_set():
                    break

                content = chunk.choices[0].delta.content
                if content:
                    full_response += content

                    # Feed to sentence extractor
                    self._current_extractor.feed_text(content)

                    # Stream to UI
                    text_chunk = TextChunk(
                        text=content,
                        message_id=message_id,
                        character_name=character.name,
                        chunk_index=chunk_index,
                        is_final=False
                    )
                    await self.queues.response_chunks.put(text_chunk)
                    chunk_index += 1

            # Signal stream complete
            self._current_extractor.finish()

            # Send final marker
            final_chunk = TextChunk(
                text="",
                message_id=message_id,
                character_name=character.name,
                chunk_index=chunk_index,
                is_final=True
            )
            await self.queues.response_chunks.put(final_chunk)

            # Wait for sentence processing
            await sentence_task

        except Exception as e:
            logger.error(f"Error generating response for {character.name}: {e}")
            raise
        finally:
            if self._current_extractor:
                self._current_extractor.shutdown()

        return full_response

    async def _process_sentences(self, character: Character, message_id: str) -> None:
        """Process sentences as they're extracted and queue for TTS"""
        sentence_index = 0

        try:
            async for sentence, idx in self._current_extractor.get_sentences():
                if self.queues.interrupt_signal.is_set():
                    break

                # Get voice for character (could be from character.voice or default)
                voice = self._get_voice_for_character(character)

                sentence_tts = SentenceTTS(
                    text=sentence,
                    speaker=character,
                    voice=voice,
                    sentence_index=sentence_index,
                    is_final=False
                )
                await self.queues.tts_sentences.put(sentence_tts)

                logger.debug(f"Sentence {sentence_index} queued: {sentence[:50]}...")
                sentence_index += 1

            # Send completion marker
            final_marker = SentenceTTS(
                text="",
                speaker=character,
                voice=None,
                sentence_index=sentence_index,
                is_final=True
            )
            await self.queues.tts_sentences.put(final_marker)

        except Exception as e:
            logger.error(f"Sentence processing error: {e}")

    def _get_voice_for_character(self, character: Character) -> Optional[Voice]:
        """Get voice configuration for a character"""
        if character.voice:
            return Voice(
                voice=character.voice,
                method="default",
                speaker_desc="",
                scene_prompt=""
            )
        return None

    def _parse_character_mentions(self, message: str) -> List[Character]:
        """Parse message for character mentions"""
        mentioned = []
        processed_ids = set()

        for character in self.active_characters:
            name_parts = character.name.lower().split()
            for part in name_parts:
                if re.search(rf'\b{re.escape(part)}\b', message, re.IGNORECASE):
                    if character.id not in processed_ids:
                        mentioned.append(character)
                        processed_ids.add(character.id)
                    break

        # Default to all active if none mentioned
        if not mentioned:
            return sorted(self.active_characters, key=lambda c: c.name)

        return mentioned

    def _create_character_instruction(self, character: Character) -> Dict[str, str]:
        """Create instruction message for character response"""
        return {
            'role': 'system',
            'content': f'Provide the next reply as {character.name}. '
                      f'Only respond as {character.name}. '
                      f'Wrap your response in <{character.name}></{character.name}> tags.'
        }

    def _get_model_settings(self) -> ModelSettings:
        """Get current model settings"""
        if self.model_settings:
            return self.model_settings
        return ModelSettings(
            model=settings.default_llm_model,
            temperature=settings.default_temperature,
            top_p=0.9,
            min_p=0.0,
            top_k=40,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0
        )

    # Public API methods
    def set_active_characters(self, characters: List[Character]) -> None:
        self.active_characters = characters

    def set_model_settings(self, settings: ModelSettings) -> None:
        self.model_settings = settings

    def clear_history(self) -> None:
        self.conversation_history = []

    async def stop(self) -> None:
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def shutdown(self) -> None:
        await self.stop()
        logger.info("LLM service shut down")
```

### Phase 3: WebSocket Manager Refactor

#### Step 3.1: Connection Manager

Create `backend/managers/connection_manager.py`:

```python
import asyncio
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field
from fastapi import WebSocket

from backend.managers.queue_manager import QueueManager
from backend.services.stt_service import STTService
from backend.services.llm_service import LLMService
from backend.services.tts_service import TTSService

logger = logging.getLogger(__name__)

@dataclass
class ClientSession:
    """Per-client session with isolated services"""
    websocket: WebSocket
    queues: QueueManager
    stt_service: STTService
    llm_service: LLMService
    tts_service: TTSService
    tasks: list = field(default_factory=list)

    async def shutdown(self):
        """Shutdown all services for this session"""
        for task in self.tasks:
            if not task.done():
                task.cancel()

        await self.stt_service.shutdown()
        await self.llm_service.shutdown()
        await self.tts_service.shutdown()

class ConnectionManager:
    """Manages multiple WebSocket connections with isolated sessions"""

    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}

    async def create_session(self, websocket: WebSocket) -> ClientSession:
        """Create a new session for a WebSocket connection"""
        session_id = str(id(websocket))

        # Create isolated queues
        queues = QueueManager()

        # Create services with shared queues
        stt = STTService(queues)
        llm = LLMService(queues)
        tts = TTSService(queues)

        # Initialize all services
        await stt.initialize()
        await llm.initialize()
        await tts.initialize()

        session = ClientSession(
            websocket=websocket,
            queues=queues,
            stt_service=stt,
            llm_service=llm,
            tts_service=tts
        )

        self.sessions[session_id] = session
        logger.info(f"Created session {session_id}")

        return session

    async def remove_session(self, websocket: WebSocket) -> None:
        """Remove and cleanup a session"""
        session_id = str(id(websocket))

        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            await session.shutdown()
            logger.info(f"Removed session {session_id}")

    def get_session(self, websocket: WebSocket) -> Optional[ClientSession]:
        """Get session for a WebSocket"""
        return self.sessions.get(str(id(websocket)))
```

#### Step 3.2: Refactored WebSocket Manager

Create `backend/managers/websocket_manager.py`:

```python
import json
import asyncio
import logging
from typing import Optional
from fastapi import WebSocket

from backend.managers.connection_manager import ConnectionManager, ClientSession
from backend.models.characters import Character
from backend.models.settings import ModelSettings
from backend.models.messages import TextChunk, AudioChunk

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Orchestrates WebSocket communication and service coordination"""

    def __init__(self):
        self.connection_manager = ConnectionManager()

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Handle a WebSocket connection lifecycle"""
        await websocket.accept()

        session = await self.connection_manager.create_session(websocket)

        try:
            # Start service loops
            await self._start_session_tasks(session)

            # Handle messages
            await self._message_loop(session)

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await self.connection_manager.remove_session(websocket)

    async def _start_session_tasks(self, session: ClientSession) -> None:
        """Start all background tasks for a session"""
        # Start service main loops
        await session.llm_service.start()
        await session.tts_service.start()

        # Start output streaming tasks
        session.tasks.extend([
            asyncio.create_task(self._stream_text(session)),
            asyncio.create_task(self._stream_audio(session)),
        ])

    async def _message_loop(self, session: ClientSession) -> None:
        """Main message handling loop"""
        while True:
            message = await session.websocket.receive()

            if "text" in message:
                await self._handle_text_message(session, message["text"])
            elif "bytes" in message:
                await self._handle_audio_message(session, message["bytes"])

    async def _handle_text_message(self, session: ClientSession, raw: str) -> None:
        """Handle incoming text message"""
        try:
            data = json.loads(raw)
            msg_type = data.get("type", "")
            payload = data.get("data", {})

            handlers = {
                "user_message": self._handle_user_message,
                "start_listening": self._handle_start_listening,
                "stop_listening": self._handle_stop_listening,
                "model_settings": self._handle_model_settings,
                "set_characters": self._handle_set_characters,
                "clear_history": self._handle_clear_history,
                "interrupt": self._handle_interrupt,
                "ping": self._handle_ping,
            }

            handler = handlers.get(msg_type)
            if handler:
                await handler(session, payload)
            else:
                logger.warning(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {raw}")
        except Exception as e:
            logger.error(f"Message handling error: {e}")

    async def _handle_user_message(self, session: ClientSession, payload: dict) -> None:
        text = payload.get("text", "")
        if text:
            await session.queues.transcribed_text.put(text)

    async def _handle_start_listening(self, session: ClientSession, _: dict) -> None:
        session.stt_service.start_listening()

    async def _handle_stop_listening(self, session: ClientSession, _: dict) -> None:
        session.stt_service.stop_listening()

    async def _handle_model_settings(self, session: ClientSession, payload: dict) -> None:
        settings = ModelSettings(
            model=payload.get("model", "meta-llama/llama-3.1-8b-instruct"),
            temperature=float(payload.get("temperature", 0.7)),
            top_p=float(payload.get("top_p", 0.9)),
            min_p=float(payload.get("min_p", 0.0)),
            top_k=int(payload.get("top_k", 40)),
            frequency_penalty=float(payload.get("frequency_penalty", 0.0)),
            presence_penalty=float(payload.get("presence_penalty", 0.0)),
            repetition_penalty=float(payload.get("repetition_penalty", 1.0))
        )
        session.llm_service.set_model_settings(settings)

    async def _handle_set_characters(self, session: ClientSession, payload: dict) -> None:
        characters = [
            Character(**char) for char in payload.get("characters", [])
        ]
        session.llm_service.set_active_characters(characters)

    async def _handle_clear_history(self, session: ClientSession, _: dict) -> None:
        session.llm_service.clear_history()

    async def _handle_interrupt(self, session: ClientSession, _: dict) -> None:
        session.queues.interrupt_signal.set()
        session.queues.clear_all()

    async def _handle_ping(self, session: ClientSession, _: dict) -> None:
        await self._send_json(session, {"type": "pong"})

    async def _handle_audio_message(self, session: ClientSession, audio: bytes) -> None:
        """Handle incoming audio data"""
        session.stt_service.feed_audio(audio)

    # Output streaming
    async def _stream_text(self, session: ClientSession) -> None:
        """Stream text chunks to client"""
        while True:
            try:
                chunk: TextChunk = await session.queues.response_chunks.get()
                await self._send_json(session, {
                    "type": "response_chunk",
                    "data": {
                        "text": chunk.text,
                        "character_name": chunk.character_name,
                        "message_id": chunk.message_id,
                        "is_final": chunk.is_final
                    }
                })
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Text streaming error: {e}")

    async def _stream_audio(self, session: ClientSession) -> None:
        """Stream audio chunks to client"""
        while True:
            try:
                chunk: AudioChunk = await session.queues.audio_chunks.get()
                if chunk.audio_data:
                    await session.websocket.send_bytes(chunk.audio_data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audio streaming error: {e}")

    async def _send_json(self, session: ClientSession, data: dict) -> None:
        """Send JSON message to client"""
        await session.websocket.send_text(json.dumps(data))
```

### Phase 4: FastAPI App Refactor

Create `backend/server/app.py`:

```python
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.managers.websocket_manager import WebSocketManager
from backend.server.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global manager
ws_manager = WebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Starting FastAPI server...")
    yield
    logger.info("Shutting down FastAPI server...")

app = FastAPI(
    title="Multi-Character Voice Chat API",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint"""
    try:
        await ws_manager.handle_connection(websocket)
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Static files (mount last)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

def main():
    import uvicorn
    uvicorn.run(
        "backend.server.app:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )

if __name__ == "__main__":
    main()
```

---

## Part 5: Migration Checklist

### Pre-Migration

- [ ] Create `.env` file with all required credentials
- [ ] Backup current `fastapi_server.py`
- [ ] Set up test environment

### Phase 1: Foundation

- [ ] Create `backend/server/config.py`
- [ ] Create `backend/models/` package with all models
- [ ] Create `backend/managers/queue_manager.py`
- [ ] Test configuration loading

### Phase 2: Services

- [ ] Create `backend/services/stt_service.py`
- [ ] Create `backend/services/sentence_extractor.py`
- [ ] Create `backend/services/llm_service.py`
- [ ] Create `backend/services/tts_service.py`
- [ ] Unit test each service

### Phase 3: Managers

- [ ] Create `backend/managers/connection_manager.py`
- [ ] Create `backend/managers/websocket_manager.py`
- [ ] Integration test multi-client handling

### Phase 4: App

- [ ] Create `backend/server/app.py`
- [ ] Add health check endpoint
- [ ] End-to-end test

### Post-Migration

- [ ] Remove old `fastapi_server.py`
- [ ] Update imports across codebase
- [ ] Performance testing
- [ ] Documentation update

---

## Part 6: Key Design Decisions

### 6.1 Why Per-Connection Services?

The current design shares one set of services across all connections:
- **Problem**: Client A's interrupt affects Client B
- **Solution**: Each WebSocket gets isolated `QueueManager` + services
- **Tradeoff**: More memory per connection, but proper isolation

### 6.2 Queue vs Callback Architecture

We use **both**:
- **Queues**: For producer/consumer decoupling (STT→LLM→TTS pipeline)
- **Callbacks**: For event notification (VAD events, audio stream events)

### 6.3 Threading vs Asyncio

| Component | Model | Reason |
|-----------|-------|--------|
| STT | Thread | RealtimeSTT is synchronous/blocking |
| LLM | Asyncio | OpenAI client is async |
| TTS | Thread + Asyncio | Engine is sync, orchestration is async |
| Sentence Extractor | Thread | stream2sentence is synchronous |

The pattern: Wrap synchronous libraries in threads, expose async interfaces.

### 6.4 Interrupt Handling

```python
# On interrupt:
1. Set interrupt_signal event
2. Clear all queues
3. Cancel in-flight generations
4. Reset interrupt_signal before next generation
```

---

## Part 7: Testing Strategy

### Unit Tests

```python
# test_stt_service.py
async def test_stt_initialization():
    queues = QueueManager()
    stt = STTService(queues)
    await stt.initialize()
    assert stt.recorder is not None

# test_llm_service.py
async def test_character_mention_parsing():
    llm = LLMService(QueueManager())
    llm.active_characters = [
        Character(id="1", name="Alice", is_active=True),
        Character(id="2", name="Bob", is_active=True),
    ]
    mentioned = llm._parse_character_mentions("Hey Alice, how are you?")
    assert len(mentioned) == 1
    assert mentioned[0].name == "Alice"
```

### Integration Tests

```python
# test_pipeline.py
async def test_full_pipeline():
    queues = QueueManager()

    # Simulate user message
    await queues.transcribed_text.put("Hello, tell me a joke")

    # Verify text flows through
    # ...
```

---

## Appendix A: Import Cleanup

Remove these unused imports from current code:

```python
# Remove
import queue              # Using asyncio.Queue
import multiprocessing    # Not used
import aiohttp           # Not used
import requests          # Not used
from loguru import logger # Overwritten

# Keep
import asyncio
import threading
import logging
import json
import time
import uuid
import re
import os
```

---

## Appendix B: Environment Variables

Create `.env`:

```env
# Server
HOST=0.0.0.0
PORT=8000

# API Keys
OPENROUTER_API_KEY=sk-or-v1-...
SUPABASE_URL=https://...supabase.co
SUPABASE_ANON_KEY=eyJ...

# STT
STT_MODEL=small.en
STT_LANGUAGE=en

# LLM
DEFAULT_LLM_MODEL=meta-llama/llama-3.1-8b-instruct
DEFAULT_TEMPERATURE=0.7

# TTS
TTS_ENGINE=kokoro
TTS_SAMPLE_RATE=24000
```

---

## Next Steps

Once this plan is reviewed and approved:

1. **Create directory structure** and empty files
2. **Implement Phase 1** (configuration + models)
3. **Iteratively implement services** with tests
4. **Wire up WebSocket manager**
5. **End-to-end testing**
6. **Deploy and monitor**

Ready to proceed when you are!
