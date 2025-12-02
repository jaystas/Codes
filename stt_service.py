"""
STT Service Module for Low-Latency Voice Chat Application

This module provides a Speech-to-Text service component using RealtimeSTT
for integration with a FastAPI WebSocket-based voice chat application.

Features:
- Real-time transcription with streaming callbacks (update, stabilized, final)
- VAD-based voice activity detection for user interrupt capabilities
- External audio feed support (use_microphone=False) for remote GPU processing
- Async-compatible event system for FastAPI integration

Audio Input: PCM16 @ 16kHz from browser WebSocket
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Any
from collections.abc import Awaitable

import numpy as np

logger = logging.getLogger(__name__)


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
    on_voice_detected: Optional[Callable[[], Any]] = None  # Maps to on_vad_detect_start
    on_voice_ended: Optional[Callable[[], Any]] = None     # Maps to on_vad_detect_stop
    
    # Recording lifecycle callbacks
    on_recording_start: Optional[Callable[[], Any]] = None
    on_recording_stop: Optional[Callable[[], Any]] = None
    
    # Turn detection (silence during speech)
    on_turn_start: Optional[Callable[[], Any]] = None
    on_turn_end: Optional[Callable[[], Any]] = None


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
        text = await stt.get_transcription()
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
            
            # Import here to avoid loading heavy dependencies on import
            from RealtimeSTT import AudioToTextRecorder
            
            logger.info("Initializing STT Service with RealtimeSTT...")
            
            # Create recorder with external audio feed (use_microphone=False)
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
    
    def feed_audio(self, audio_data: bytes, sample_rate: int = 16000):
        """
        Feed audio data from WebSocket into the STT pipeline.
        
        Args:
            audio_data: Raw PCM16 audio bytes
            sample_rate: Sample rate of input audio (will resample if not 16kHz)
        """
        if not self._is_initialized or self._recorder is None:
            logger.warning("Cannot feed audio: STT Service not initialized")
            return
            
        if self._is_shutting_down:
            return
            
        try:
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            else:
                audio_array = audio_data
                
            # Feed to recorder (handles resampling internally if needed)
            self._recorder.feed_audio(audio_array, original_sample_rate=sample_rate)
            
        except Exception as e:
            logger.error(f"Error feeding audio: {e}")
    
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
    
    async def get_transcription(self, timeout: float = 30.0) -> str:
        """
        Wait for and return the final transcription.
        
        This method blocks until:
        - Voice activity is detected and recording completes
        - The transcription is processed
        
        Args:
            timeout: Maximum time to wait for transcription
            
        Returns:
            Final transcription text
        """
        if not self._is_initialized or self._recorder is None:
            logger.warning("Cannot get transcription: STT Service not initialized")
            return ""
            
        try:
            # Run the blocking text() call in a thread pool
            loop = asyncio.get_running_loop()
            
            with self._lock:
                self._state = STTState.LISTENING
                
            text = await asyncio.wait_for(
                loop.run_in_executor(None, self._recorder.text),
                timeout=timeout
            )
            
            with self._lock:
                self._state = STTState.IDLE
                self._current_transcription = text or ""
                
            # Fire final transcription callback
            if text and self.callbacks.on_final_transcription:
                await self._run_callback_async(
                    self.callbacks.on_final_transcription, 
                    text
                )
                
            return text or ""
            
        except asyncio.TimeoutError:
            logger.warning("Transcription timed out")
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
    
    # =========================================================================
    # Internal callback handlers
    # These bridge RealtimeSTT callbacks to async-compatible user callbacks
    # =========================================================================
    
    def _on_realtime_update(self, text: str):
        """Called when realtime transcription updates (may change)"""
        if self.callbacks.on_realtime_update:
            self._schedule_callback(self.callbacks.on_realtime_update, text)
    
    def _on_realtime_stabilized(self, text: str):
        """Called when realtime transcription stabilizes (less likely to change)"""
        if self.callbacks.on_realtime_stabilized:
            self._schedule_callback(self.callbacks.on_realtime_stabilized, text)
    
    def _on_vad_detect_start(self):
        """
        Called when system starts listening for voice activity.
        This is triggered when entering the listening state.
        """
        with self._lock:
            self._voice_active = True
            
        if self.callbacks.on_voice_detected:
            self._schedule_callback(self.callbacks.on_voice_detected)
    
    def _on_vad_detect_stop(self):
        """Called when system stops listening for voice activity"""
        with self._lock:
            self._voice_active = False
            
        if self.callbacks.on_voice_ended:
            self._schedule_callback(self.callbacks.on_voice_ended)
    
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
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
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

# =============================================================================
# Example usage and integration patterns
# =============================================================================

async def example_usage():
    """Example showing how to use STTService with FastAPI WebSocket"""
    
    # Define callbacks
    async def on_realtime_update(text: str):
        print(f"[Realtime Update] {text}")
        # Send to WebSocket client
        # await websocket.send_json({"type": "stt_update", "text": text})
    
    async def on_realtime_stabilized(text: str):
        print(f"[Stabilized] {text}")
        # await websocket.send_json({"type": "stt_stabilized", "text": text})
    
    async def on_final_transcription(text: str):
        print(f"[Final] {text}")
        # await websocket.send_json({"type": "stt_final", "text": text})
    
    async def on_vad_start():
        print("[VAD] Voice activity detected - Consider interrupting TTS")
        # This is where you'd trigger TTS interrupt
    
    # Create config
    config = STTConfig(
        model="small.en",
        realtime_model="tiny.en",
        post_speech_silence_duration=0.5,  # Fast response
        enable_realtime_transcription=True,
    )
    
    # Create callbacks
    callbacks = STTCallbacks(
        on_realtime_update=on_realtime_update,
        on_realtime_stabilized=on_realtime_stabilized,
        on_final_transcription=on_final_transcription,
        on_vad_start=on_vad_start,
    )
    
    # Create and initialize service
    stt = STTService(config, callbacks)
    await stt.initialize()
    
    try:
        # Start listening
        stt.start_listening()
        
        # In real usage, you'd receive audio from WebSocket:
        # async for message in websocket:
        #     if isinstance(message, bytes):
        #         stt.feed_audio(message)
        
        # Get final transcription (blocks until speech ends)
        text = await stt.get_transcription()
        print(f"Got transcription: {text}")
        
    finally:
        await stt.shutdown()

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
