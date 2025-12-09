"""
TTSService: Streaming Text-to-Speech using HiggsAudio with Voice Consistency

This service provides streaming TTS audio generation that maintains voice consistency
across multiple generations using a rolling context buffer of:
- Speaker description text (voice characteristics)
- Scene prompts (environment/style descriptions)
- Reference audio tokens (for voice cloning)
- Previously generated audio tokens (rolling buffer)
"""

import os
import asyncio
import torch
import numpy as np
import base64
import librosa
from io import BytesIO
from dataclasses import dataclass, field
from typing import Optional, Dict, List, AsyncGenerator, Union
from loguru import logger

from backend.boson_multimodal.serve.serve_engine import (
    HiggsAudioServeEngine,
    HiggsAudioStreamerDelta,
)
from backend.boson_multimodal.data_types import (
    ChatMLSample,
    Message,
    AudioContent,
    TextContent,
)
from backend.boson_multimodal.model.higgs_audio.utils import revert_delay_pattern


# Default configuration
DEFAULT_CHUNK_SIZE = 64  # Audio tokens per chunk (~1.28 seconds at 50 Hz)
DEFAULT_BUFFER_SIZE = 3  # Number of generation chunks to keep in rolling buffer
DEFAULT_CROSS_FADE_DURATION = 0.04  # 40ms cross-fade between chunks
DEFAULT_SAMPLE_RATE = 24000
NUM_CODEBOOKS = 12


@dataclass
class VoiceContext:
    """
    Maintains voice consistency state for a character/speaker.

    The rolling buffer mechanism accumulates:
    1. Reference audio tokens (from voice cloning, if provided)
    2. Generated audio tokens from previous generations
    3. Message history for context continuity

    This ensures the model generates audio with consistent voice characteristics
    across multiple TTS calls.
    """
    character_id: str
    speaker_desc: str  # Voice description (e.g., "Male, American accent, friendly tone")
    scene_prompt: str  # Scene description (e.g., "Audio is recorded from a quiet room")

    # Reference audio for voice cloning (optional)
    ref_audio_ids: List[torch.Tensor] = field(default_factory=list)
    ref_audio_messages: List[Message] = field(default_factory=list)  # User/assistant pairs

    # Rolling buffer of generated audio for voice consistency
    generated_audio_ids: List[torch.Tensor] = field(default_factory=list)
    generation_messages: List[Message] = field(default_factory=list)

    # Configuration
    buffer_size: int = DEFAULT_BUFFER_SIZE  # Max chunks to keep in rolling buffer

    def get_system_message(self) -> Message:
        """Build system message with scene description and speaker info."""
        content_parts = ["Generate audio following instruction."]

        scene_desc_parts = []
        if self.scene_prompt:
            scene_desc_parts.append(self.scene_prompt)
        if self.speaker_desc:
            scene_desc_parts.append(f"SPEAKER0: {self.speaker_desc}")

        if scene_desc_parts:
            content_parts.append(
                f"<|scene_desc_start|>\n{chr(10).join(scene_desc_parts)}\n<|scene_desc_end|>"
            )

        return Message(
            role="system",
            content="\n\n".join(content_parts)
        )

    def get_context_audio_ids(self) -> List[torch.Tensor]:
        """Get combined reference + generated audio tokens for context."""
        return self.ref_audio_ids + self.generated_audio_ids

    def get_context_messages(self) -> List[Message]:
        """Get messages including system, reference audio, and generation history."""
        messages = [self.get_system_message()]
        messages.extend(self.ref_audio_messages)
        messages.extend(self.generation_messages)
        return messages

    def update_after_generation(
        self,
        generated_audio_tokens: torch.Tensor,
        text: str
    ):
        """
        Update rolling buffer after a generation.

        Args:
            generated_audio_tokens: Audio tokens from the generation (shape: [num_codebooks, seq_len])
            text: The text that was synthesized
        """
        # Add user message (the text) and assistant message (the audio)
        self.generation_messages.append(
            Message(role="user", content=text)
        )
        self.generation_messages.append(
            Message(role="assistant", content=AudioContent(audio_url=""))
        )

        # Add generated audio tokens to buffer
        self.generated_audio_ids.append(generated_audio_tokens)

        # Apply sliding window to limit memory usage
        if len(self.generated_audio_ids) > self.buffer_size:
            self.generated_audio_ids = self.generated_audio_ids[-self.buffer_size:]
            # Messages come in pairs (user + assistant per generation)
            self.generation_messages = self.generation_messages[(-2 * self.buffer_size):]

    def clear_history(self):
        """Clear generation history but keep reference audio and voice settings."""
        self.generated_audio_ids = []
        self.generation_messages = []


class TTSService:
    """
    Streaming Text-to-Speech service with voice consistency.

    Features:
    - Async streaming audio generation
    - Voice consistency via rolling context buffer
    - Support for voice cloning via reference audio
    - Cross-fade between audio chunks for smooth playback

    Usage:
        service = TTSService(model_path, audio_tokenizer_path)
        await service.initialize()

        # Create voice context for a character
        voice_ctx = await service.create_voice_context(
            character_id="char_001",
            speaker_desc="Male, American accent, friendly tone",
            scene_prompt="Audio is recorded from a quiet room"
        )

        # Stream audio for text
        async for audio_chunk in service.generate_audio_stream("Hello world!", voice_ctx):
            # audio_chunk is PCM16 bytes at 24kHz
            websocket.send_bytes(audio_chunk)
    """

    def __init__(
        self,
        model_path: str = None,
        audio_tokenizer_path: str = None,
        device: str = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        kv_cache_lengths: List[int] = [1024, 4096, 8192],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        cross_fade_duration: float = DEFAULT_CROSS_FADE_DURATION,
    ):
        """
        Initialize TTS Service.

        Args:
            model_path: Path to HiggsAudio model (default: bosonai/higgs-audio-v2-generation-3B-base)
            audio_tokenizer_path: Path to audio tokenizer (default: bosonai/higgs-audio-v2-tokenizer)
            device: Device to use ("cuda" or "cpu", auto-detected if None)
            torch_dtype: Model dtype
            kv_cache_lengths: KV cache sizes for CUDA graph optimization
            chunk_size: Number of audio tokens per streaming chunk
            cross_fade_duration: Cross-fade duration in seconds between chunks
        """
        self.model_path = model_path or os.getenv(
            "TTS_MODEL_PATH",
            "bosonai/higgs-audio-v2-generation-3B-base"
        )
        self.audio_tokenizer_path = audio_tokenizer_path or os.getenv(
            "TTS_AUDIO_TOKENIZER_PATH",
            "bosonai/higgs-audio-v2-tokenizer"
        )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.torch_dtype = torch_dtype
        self.kv_cache_lengths = kv_cache_lengths
        self.chunk_size = chunk_size
        self.cross_fade_duration = cross_fade_duration

        # Will be initialized in initialize()
        self.serve_engine: Optional[HiggsAudioServeEngine] = None
        self.voice_contexts: Dict[str, VoiceContext] = {}

        self.is_initialized = False

    async def initialize(self):
        """Initialize the HiggsAudio serve engine."""
        if self.is_initialized:
            logger.warning("TTSService already initialized")
            return

        logger.info(f"Initializing TTSService on device: {self.device}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Audio tokenizer path: {self.audio_tokenizer_path}")

        # Initialize in executor to not block event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_serve_engine)

        self.is_initialized = True
        logger.info("TTSService initialized successfully")

    def _init_serve_engine(self):
        """Initialize serve engine (runs in thread pool)."""
        self.serve_engine = HiggsAudioServeEngine(
            model_name_or_path=self.model_path,
            audio_tokenizer_name_or_path=self.audio_tokenizer_path,
            device=self.device,
            torch_dtype=self.torch_dtype,
            kv_cache_lengths=self.kv_cache_lengths,
        )

        # Pre-compute cross-fade windows
        cross_fade_samples = int(self.cross_fade_duration * self.serve_engine.audio_tokenizer.sampling_rate)
        self.cross_fade_samples = cross_fade_samples
        self.fade_out = np.linspace(1, 0, cross_fade_samples).astype(np.float32)
        self.fade_in = np.linspace(0, 1, cross_fade_samples).astype(np.float32)

    async def create_voice_context(
        self,
        character_id: str,
        speaker_desc: str = "",
        scene_prompt: str = "Audio is recorded from a quiet room.",
        ref_audio_path: Optional[str] = None,
        ref_audio_base64: Optional[str] = None,
        ref_audio_text: Optional[str] = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> VoiceContext:
        """
        Create a voice context for a character.

        Args:
            character_id: Unique identifier for this voice context
            speaker_desc: Voice description (e.g., "Male, American accent, friendly tone")
            scene_prompt: Scene/environment description
            ref_audio_path: Path to reference audio file for voice cloning
            ref_audio_base64: Base64-encoded reference audio for voice cloning
            ref_audio_text: Transcript of reference audio (required if ref audio provided)
            buffer_size: Number of generation chunks to keep in rolling buffer

        Returns:
            VoiceContext configured for consistent voice generation
        """
        if not self.is_initialized:
            raise RuntimeError("TTSService not initialized. Call initialize() first.")

        voice_ctx = VoiceContext(
            character_id=character_id,
            speaker_desc=speaker_desc,
            scene_prompt=scene_prompt,
            buffer_size=buffer_size,
        )

        # Add reference audio for voice cloning if provided
        if ref_audio_path or ref_audio_base64:
            if not ref_audio_text:
                raise ValueError("ref_audio_text required when providing reference audio")

            loop = asyncio.get_event_loop()
            audio_ids = await loop.run_in_executor(
                None,
                self._encode_reference_audio,
                ref_audio_path,
                ref_audio_base64
            )

            if audio_ids is not None:
                voice_ctx.ref_audio_ids.append(audio_ids)

                # Add as user/assistant message pair for few-shot learning
                voice_ctx.ref_audio_messages.append(
                    Message(role="user", content=ref_audio_text)
                )
                voice_ctx.ref_audio_messages.append(
                    Message(
                        role="assistant",
                        content=AudioContent(
                            audio_url=ref_audio_path or "",
                            raw_audio=ref_audio_base64
                        )
                    )
                )

        # Store in registry
        self.voice_contexts[character_id] = voice_ctx

        logger.info(f"Created voice context for {character_id}: "
                   f"speaker_desc='{speaker_desc[:50]}...', "
                   f"has_ref_audio={len(voice_ctx.ref_audio_ids) > 0}")

        return voice_ctx

    def _encode_reference_audio(
        self,
        ref_audio_path: Optional[str],
        ref_audio_base64: Optional[str]
    ) -> Optional[torch.Tensor]:
        """Encode reference audio to tokens (runs in thread pool)."""
        try:
            if ref_audio_path and os.path.exists(ref_audio_path):
                raw_audio, _ = librosa.load(
                    ref_audio_path,
                    sr=self.serve_engine.audio_tokenizer.sampling_rate
                )
            elif ref_audio_base64:
                raw_audio, _ = librosa.load(
                    BytesIO(base64.b64decode(ref_audio_base64)),
                    sr=self.serve_engine.audio_tokenizer.sampling_rate
                )
            else:
                return None

            audio_ids = self.serve_engine.audio_tokenizer.encode(
                raw_audio,
                self.serve_engine.audio_tokenizer.sampling_rate
            )
            return audio_ids.squeeze(0).cpu()

        except Exception as e:
            logger.error(f"Failed to encode reference audio: {e}")
            return None

    def get_voice_context(self, character_id: str) -> Optional[VoiceContext]:
        """Get an existing voice context by character ID."""
        return self.voice_contexts.get(character_id)

    async def generate_audio_stream(
        self,
        text: str,
        voice_context: VoiceContext,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        force_audio_gen: bool = True,
        ras_win_len: int = 7,
        ras_win_max_num_repeat: int = 2,
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate streaming audio for text with voice consistency.

        Args:
            text: Text to synthesize
            voice_context: VoiceContext with voice settings and history
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            force_audio_gen: Force audio output (vs. text)
            ras_win_len: Repetition amplification sampling window
            ras_win_max_num_repeat: Max repetitions in RAS window

        Yields:
            PCM16 audio bytes at 24kHz sample rate
        """
        if not self.is_initialized:
            raise RuntimeError("TTSService not initialized")

        # Build messages with context
        messages = voice_context.get_context_messages()
        messages.append(Message(role="user", content=text))

        # Prepare audio context
        context_audio_ids = voice_context.get_context_audio_ids()

        # Create ChatMLSample
        chat_ml_sample = self._build_chatml_sample(messages, context_audio_ids)

        # Streaming generation
        audio_token_buffer = []
        all_generated_tokens = []

        streamer = self.serve_engine.generate_delta_stream(
            chat_ml_sample=chat_ml_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            force_audio_gen=force_audio_gen,
            ras_win_len=ras_win_len,
            ras_win_max_num_repeat=ras_win_max_num_repeat,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )

        previous_chunk = None

        async for delta in streamer:
            if delta.audio_tokens is not None:
                audio_token_buffer.append(delta.audio_tokens)
                all_generated_tokens.append(delta.audio_tokens)

                # Check if we have enough tokens for a chunk
                if len(audio_token_buffer) >= self.chunk_size:
                    # Decode chunk
                    audio_chunk = await self._decode_audio_chunk(
                        audio_token_buffer[:self.chunk_size]
                    )

                    # Apply cross-fade with previous chunk
                    if previous_chunk is not None:
                        audio_chunk = self._apply_cross_fade(previous_chunk, audio_chunk)
                    else:
                        # First chunk: trim start for clean beginning
                        audio_chunk = audio_chunk[self.cross_fade_samples:]

                    # Convert to PCM16 bytes
                    pcm_bytes = self._to_pcm16_bytes(audio_chunk)
                    yield pcm_bytes

                    # Keep overlap for continuity
                    tokens_to_keep = NUM_CODEBOOKS - 1  # 11 tokens
                    audio_token_buffer = audio_token_buffer[self.chunk_size - tokens_to_keep:]

                    # Store end of current chunk for next cross-fade
                    previous_chunk = audio_chunk

        # Process remaining tokens
        if len(audio_token_buffer) > NUM_CODEBOOKS:
            audio_chunk = await self._decode_audio_chunk(audio_token_buffer)

            if previous_chunk is not None:
                audio_chunk = self._apply_cross_fade(previous_chunk, audio_chunk)

            # Final chunk: include ending
            pcm_bytes = self._to_pcm16_bytes(audio_chunk)
            yield pcm_bytes

        # Update voice context with generated audio for consistency
        if all_generated_tokens:
            generated_tensor = torch.stack(all_generated_tokens, dim=1)
            generated_tensor = revert_delay_pattern(generated_tensor)
            generated_tensor = generated_tensor.clip(
                0, self.serve_engine.audio_codebook_size - 1
            )[:, 1:-1]  # Trim BOS/EOS

            voice_context.update_after_generation(
                generated_tensor.cpu(),
                text
            )

    def _build_chatml_sample(
        self,
        messages: List[Message],
        context_audio_ids: List[torch.Tensor]
    ) -> ChatMLSample:
        """Build ChatMLSample with audio context."""
        # Create sample with messages
        sample = ChatMLSample(messages=messages)

        # Audio IDs will be handled by serve_engine._prepare_inputs
        # through AudioContent objects in messages
        return sample

    async def _decode_audio_chunk(
        self,
        audio_tokens: List[torch.Tensor]
    ) -> np.ndarray:
        """Decode audio tokens to waveform."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._decode_audio_chunk_sync,
            audio_tokens
        )

    def _decode_audio_chunk_sync(
        self,
        audio_tokens: List[torch.Tensor]
    ) -> np.ndarray:
        """Synchronous audio decoding."""
        # Stack tokens: shape (num_codebooks, seq_len)
        audio_chunk = torch.stack(audio_tokens, dim=1)

        # Revert delay pattern
        vq_code = revert_delay_pattern(audio_chunk)
        vq_code = vq_code.clip(0, self.serve_engine.audio_codebook_size - 1)

        # Decode to waveform
        wv_numpy = self.serve_engine.audio_tokenizer.decode(
            vq_code.unsqueeze(0).to(self.device)
        )[0, 0]

        return wv_numpy

    def _apply_cross_fade(
        self,
        previous_chunk: np.ndarray,
        current_chunk: np.ndarray
    ) -> np.ndarray:
        """Apply cross-fade between chunks for smooth transition."""
        # Cross-fade overlap region
        overlap_end_prev = previous_chunk[-self.cross_fade_samples:]
        overlap_start_curr = current_chunk[:self.cross_fade_samples]

        cross_faded = overlap_end_prev * self.fade_out + overlap_start_curr * self.fade_in

        # Return current chunk with cross-faded start
        result = current_chunk.copy()
        result[:self.cross_fade_samples] = cross_faded

        return result[self.cross_fade_samples:]  # Skip the overlap we already sent

    def _to_pcm16_bytes(self, audio: np.ndarray) -> bytes:
        """Convert float audio to PCM16 bytes."""
        # Clip and scale to int16 range
        audio_clipped = np.clip(audio, -1.0, 1.0)
        pcm_data = (audio_clipped * 32767).astype(np.int16)
        return pcm_data.tobytes()

    @property
    def sample_rate(self) -> int:
        """Audio sample rate."""
        if self.serve_engine:
            return self.serve_engine.audio_tokenizer.sampling_rate
        return DEFAULT_SAMPLE_RATE

    async def shutdown(self):
        """Cleanup resources."""
        logger.info("Shutting down TTSService")
        self.voice_contexts.clear()
        # Model cleanup handled by Python GC
        self.is_initialized = False


# Convenience function for simple use cases
async def create_tts_service(
    model_path: str = None,
    audio_tokenizer_path: str = None,
    device: str = None,
) -> TTSService:
    """Create and initialize a TTSService instance."""
    service = TTSService(
        model_path=model_path,
        audio_tokenizer_path=audio_tokenizer_path,
        device=device,
    )
    await service.initialize()
    return service
