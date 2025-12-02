"""
FastAPI Integration Example for STT Service

Shows how to integrate the STT service with a FastAPI WebSocket endpoint,
including the TTS interrupt pattern for responsive voice chat.

Architecture:
- Single WebSocket connection handles both binary audio and text messages
- Audio flows: Browser (PCM16 16kHz) -> WebSocket -> STT Service -> Transcription
- Interrupt flow: VAD detects voice -> Signal TTS to stop -> Start new recording
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from stt_service import STTService, STTConfig, STTCallbacks, STTState

logger = logging.getLogger(__name__)


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Global application state"""
    stt_service: Optional[STTService] = None
    tts_is_playing: bool = False
    current_websocket: Optional[WebSocket] = None


app_state = AppState()


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup services"""
    logger.info("Starting up Voice Chat application...")
    
    # Initialize STT with optimized settings for low latency
    config = STTConfig(
        model="small.en",
        realtime_model="tiny.en",
        device="cuda",
        compute_type="float16",
        
        # Optimized for low latency
        post_speech_silence_duration=0.5,
        min_length_of_recording=0.3,
        pre_recording_buffer_duration=0.3,
        
        # Realtime settings
        enable_realtime_transcription=True,
        realtime_processing_pause=0.1,
        init_realtime_after_seconds=0.1,
        
        # Performance
        beam_size=5,
        beam_size_realtime=3,
    )
    
    # Create callbacks that will send to WebSocket
    callbacks = STTCallbacks(
        on_realtime_update=on_stt_realtime_update,
        on_realtime_stabilized=on_stt_stabilized,
        on_vad_start=on_voice_activity_start,
        on_vad_stop=on_voice_activity_stop,
        on_recording_start=on_recording_start,
        on_recording_stop=on_recording_stop,
    )
    
    app_state.stt_service = STTService(config, callbacks)
    await app_state.stt_service.initialize()
    
    logger.info("Voice Chat application started")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Voice Chat application...")
    if app_state.stt_service:
        await app_state.stt_service.shutdown()


# =============================================================================
# STT Callback Handlers
# =============================================================================

async def on_stt_realtime_update(text: str):
    """Send realtime transcription updates to client"""
    if app_state.current_websocket:
        try:
            await app_state.current_websocket.send_json({
                "type": "stt_update",
                "text": text,
                "is_final": False
            })
        except Exception as e:
            logger.error(f"Failed to send realtime update: {e}")


async def on_stt_stabilized(text: str):
    """Send stabilized transcription to client"""
    if app_state.current_websocket:
        try:
            await app_state.current_websocket.send_json({
                "type": "stt_stabilized", 
                "text": text,
                "is_final": False
            })
        except Exception as e:
            logger.error(f"Failed to send stabilized text: {e}")


async def on_voice_activity_start():
    """
    Called when voice activity is detected.
    This is the key point for implementing TTS interrupt!
    """
    logger.info("Voice activity detected")
    
    # If TTS is playing, trigger interrupt
    if app_state.tts_is_playing:
        logger.info("Interrupting TTS playback due to user voice")
        await trigger_tts_interrupt()
    
    # Notify client of voice detection
    if app_state.current_websocket:
        try:
            await app_state.current_websocket.send_json({
                "type": "vad_start",
                "tts_interrupted": app_state.tts_is_playing
            })
        except Exception:
            pass


async def on_voice_activity_stop():
    """Called when voice activity stops"""
    logger.info("Voice activity stopped")
    
    if app_state.current_websocket:
        try:
            await app_state.current_websocket.send_json({
                "type": "vad_stop"
            })
        except Exception:
            pass


async def on_recording_start():
    """Called when recording begins"""
    logger.info("Recording started")
    
    if app_state.current_websocket:
        try:
            await app_state.current_websocket.send_json({
                "type": "recording_start"
            })
        except Exception:
            pass


async def on_recording_stop():
    """Called when recording ends"""
    logger.info("Recording stopped")
    
    if app_state.current_websocket:
        try:
            await app_state.current_websocket.send_json({
                "type": "recording_stop"  
            })
        except Exception:
            pass


async def trigger_tts_interrupt():
    """
    Trigger TTS playback to stop.
    Implement this based on your TTS service architecture.
    """
    app_state.tts_is_playing = False
    
    if app_state.current_websocket:
        try:
            await app_state.current_websocket.send_json({
                "type": "tts_interrupt",
                "action": "stop"
            })
        except Exception:
            pass
    
    # If you have a TTS service reference, call its stop method:
    # await app_state.tts_service.stop_playback()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Voice Chat API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """
    Main WebSocket endpoint for voice chat.
    
    Protocol:
    - Binary messages: Raw PCM16 audio data at 16kHz
    - Text messages: JSON control messages
    
    Control Messages (client -> server):
    - {"type": "start_listening"}: Begin listening for voice
    - {"type": "stop_listening"}: Stop listening
    - {"type": "tts_started"}: Client started playing TTS
    - {"type": "tts_ended"}: Client finished playing TTS
    
    Server Messages (server -> client):
    - {"type": "stt_update", "text": "..."}: Realtime transcription update
    - {"type": "stt_stabilized", "text": "..."}: Stabilized transcription
    - {"type": "stt_final", "text": "..."}: Final transcription
    - {"type": "vad_start"}: Voice activity detected
    - {"type": "vad_stop"}: Voice activity ended
    - {"type": "tts_interrupt"}: Client should stop TTS playback
    - {"type": "recording_start"}: Recording started
    - {"type": "recording_stop"}: Recording stopped
    """
    await websocket.accept()
    app_state.current_websocket = websocket
    
    logger.info("WebSocket client connected")
    
    try:
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                break
                
            # Handle binary audio data
            if "bytes" in message:
                audio_data = message["bytes"]
                
                # Feed audio to STT service
                if app_state.stt_service:
                    app_state.stt_service.feed_audio(audio_data, sample_rate=16000)
                    
            # Handle text control messages
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    await handle_control_message(data, websocket)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message: {message['text']}")
                    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        app_state.current_websocket = None
        
        # Stop STT if active
        if app_state.stt_service:
            app_state.stt_service.stop_listening()


async def handle_control_message(data: dict, websocket: WebSocket):
    """Handle JSON control messages from client"""
    msg_type = data.get("type")
    
    if msg_type == "start_listening":
        # Start listening for voice activity
        if app_state.stt_service:
            app_state.stt_service.start_listening()
            await websocket.send_json({"type": "listening_started"})
            
    elif msg_type == "stop_listening":
        # Stop listening
        if app_state.stt_service:
            app_state.stt_service.stop_listening()
            await websocket.send_json({"type": "listening_stopped"})
            
    elif msg_type == "tts_started":
        # Client started playing TTS audio
        app_state.tts_is_playing = True
        logger.debug("TTS playback started")
        
    elif msg_type == "tts_ended":
        # Client finished playing TTS audio
        app_state.tts_is_playing = False
        logger.debug("TTS playback ended")
        
        # Automatically start listening after TTS finishes
        if app_state.stt_service:
            app_state.stt_service.start_listening()
            await websocket.send_json({"type": "listening_started"})
            
    elif msg_type == "get_transcription":
        # Request final transcription
        if app_state.stt_service:
            text = await app_state.stt_service.get_transcription(timeout=30.0)
            await websocket.send_json({
                "type": "stt_final",
                "text": text,
                "is_final": True
            })
            
    elif msg_type == "ping":
        await websocket.send_json({"type": "pong"})
        
    else:
        logger.warning(f"Unknown message type: {msg_type}")


# =============================================================================
# Alternative: Continuous Conversation Loop
# =============================================================================

async def conversation_loop(websocket: WebSocket):
    """
    Alternative pattern: continuous conversation loop.
    
    This shows how to run a continuous voice conversation where:
    1. Listen for user speech
    2. Get transcription
    3. Generate LLM response
    4. Generate and play TTS
    5. Repeat
    
    With interrupt handling built in.
    """
    stt = app_state.stt_service
    
    while True:
        try:
            # Start listening
            stt.start_listening()
            await websocket.send_json({"type": "listening_started"})
            
            # Wait for user to speak and get transcription
            user_text = await stt.get_transcription(timeout=60.0)
            
            if not user_text:
                continue
                
            # Send final transcription
            await websocket.send_json({
                "type": "stt_final",
                "text": user_text,
                "is_final": True
            })
            
            # Here you would:
            # 1. Send user_text to LLM
            # 2. Stream LLM response
            # 3. Generate TTS audio
            # 4. Send audio to client
            
            # Signal TTS started (client should also send this)
            app_state.tts_is_playing = True
            
            # ... TTS generation and streaming ...
            
            # When TTS is done
            app_state.tts_is_playing = False
            
            # Check if user interrupted during TTS
            if stt.interrupt_requested:
                stt.clear_interrupt()
                # User interrupted, their speech is already being recorded
                # Loop will continue and get their transcription
                
        except Exception as e:
            logger.error(f"Conversation loop error: {e}")
            break


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stt_status = "ready" if app_state.stt_service and app_state.stt_service._is_initialized else "not_ready"
    
    return {
        "status": "healthy",
        "stt_service": stt_status,
        "stt_state": app_state.stt_service.state.name if app_state.stt_service else "N/A"
    }


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=30,
        ws_ping_timeout=30,
    )
