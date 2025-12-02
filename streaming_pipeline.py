"""
Streaming Sentence Pipeline for Low-Latency Voice Chat

This module provides a properly streaming text-to-sentence-to-TTS pipeline that:
1. Receives streaming text chunks from LLM (OpenAI async client)
2. Feeds them into stream2sentence for real-time sentence detection
3. Queues complete sentences for TTS immediately as they're detected
4. Enables concurrent audio generation while text is still streaming

Key components:
- StreamingSentenceExtractor: Bridges async LLM stream to sync stream2sentence
- LLMResponseStreamer: Manages the full streaming pipeline for a character response
- Refactored LLMOrchestrator: Uses the new streaming pipeline
"""

import asyncio
import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import Optional, Dict, List, AsyncIterator, Callable, Any
from concurrent.futures import ThreadPoolExecutor

import stream2sentence as s2s
from backend.RealtimeTTS.threadsafe_generators import CharIterator, AccumulatingThreadSafeGenerator


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Character:
    """Character representation (mirror from main server)"""
    id: str
    name: str
    system_prompt: str
    voice: str
    is_active: bool = True
    image_url: str = ""
    images: List[str] = field(default_factory=list)


@dataclass
class Voice:
    """Voice configuration (mirror from main server)"""
    voice: str
    method: str
    speaker_desc: str
    scene_prompt: str
    audio_path: str = ""
    text_path: str = ""

    def get_higgs_system_prompt(self) -> str:
        """Build system prompt for Higgs TTS"""
        parts = ["Generate audio following instruction.\n\n<|scene_desc_start|>"]
        if self.scene_prompt:
            parts.append(self.scene_prompt)
        if self.speaker_desc:
            parts.append(f"\n{self.speaker_desc}")
        parts.append("\n<|scene_desc_end|>")
        return "".join(parts)


@dataclass
class SentenceChunk:
    """A complete sentence ready for TTS"""
    text: str
    speaker: Character
    voice: Optional[Voice]
    sequence_number: int
    sentence_index: int      # Which sentence in this response (0, 1, 2...)
    is_final: bool           # Last sentence of this character's response
    timestamp: float = field(default_factory=time.time)


@dataclass 
class TextChunk:
    """Text chunk for UI streaming"""
    text: str
    is_final: bool
    speaker_id: str
    speaker_name: str
    sequence_number: int
    chunk_index: int
    timestamp: float


# =============================================================================
# Streaming Sentence Extractor
# =============================================================================

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
            self.char_iter.complete()  # If this method exists
    
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


# =============================================================================
# LLM Response Streamer
# =============================================================================

class LLMResponseStreamer:
    """
    Manages the complete streaming pipeline for a single character response.
    
    Flow:
    1. Starts LLM generation
    2. Streams text chunks to UI queue AND sentence extractor
    3. Sentences are queued for TTS as soon as detected
    4. Tracks completion and handles cleanup
    
    This is instantiated once per character response within LLMOrchestrator.
    """
    
    def __init__(
        self,
        character: Character,
        voice: Optional[Voice],
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
                    print(f"⚠️ Interrupt during LLM stream for {self.character.name}")
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
                        speaker_id=self.character.id,
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
                speaker_id=self.character.id,
                speaker_name=self.character.name,
                sequence_number=self.sequence_number,
                chunk_index=self.chunk_index,
                timestamp=time.time()
            )
            await self.text_output_queue.put(final_text_chunk)
            
            # Wait for sentence processing to complete
            await sentence_task
            
        except Exception as e:
            print(f"Error in stream_response for {self.character.name}: {e}")
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
                
                print(f"📝 Sentence {index} queued for TTS ({self.character.name}): "
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
            print(f"Error processing sentences for {self.character.name}: {e}")


# =============================================================================
# Refactored LLM Orchestrator
# =============================================================================

class StreamingLLMOrchestrator:
    """
    LLM Orchestrator with proper streaming sentence extraction.
    
    Key improvements over original:
    - Uses stream2sentence for robust sentence detection
    - Sentences are queued for TTS as soon as detected (not at end)
    - Concurrent sentence extraction and TTS generation
    - Pre-cached voices to eliminate async lookups in hot path
    """
    
    def __init__(
        self,
        queue_manager,  # QueueManager instance
        character_service,  # CharacterService instance
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet"
    ):
        from openai import AsyncOpenAI
        
        self.queues = queue_manager
        self.character_service = character_service
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model
        
        # Pre-cached voices (set by VoiceChatOrchestrator at session start)
        self.voice_cache: Dict[str, Voice] = {}
    
    def set_voice_cache(self, cache: Dict[str, Voice]):
        """Set pre-loaded voice cache to avoid async lookups"""
        self.voice_cache = cache
    
    def get_voice(self, character: Character) -> Optional[Voice]:
        """Get voice from cache (sync, no DB call)"""
        return self.voice_cache.get(character.id)
    
    async def run(self):
        """Main loop: consume transcriptions, orchestrate character responses"""
        print("✓ Streaming LLM Orchestrator initialized")
        
        while not self.queues.stop_signal.is_set():
            try:
                transcription = await asyncio.wait_for(
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
                
                # Queue database write (non-blocking) - implement BackgroundPersistence
                # self.persistence.queue_message(MessageCreate(...))
                
                # Determine response queue
                response_queue = self.character_service.parse_character_mentions(
                    transcription.text,
                    context.active_characters
                )
                context.response_queue = response_queue
                
                print(f"Response queue: {[c.name for c in response_queue]}")
                
                # Generate responses with streaming sentences
                await self._orchestrate_streaming_responses(context, response_queue)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"LLM Orchestrator error: {e}")
                continue
    
    async def _orchestrate_streaming_responses(
        self,
        context,  # ConversationContext
        response_queue: List[Character]
    ):
        """
        Orchestrate character responses with streaming sentence extraction.
        
        Each character:
        1. Starts LLM streaming
        2. Sentences extracted and queued for TTS in real-time
        3. Next character starts when current character's TEXT is complete
           (TTS may still be generating - that's fine, it runs concurrently)
        """
        for sequence_number, character in enumerate(response_queue):
            if self.queues.interrupt_signal.is_set():
                print(f"Interrupt detected, stopping at {character.name}")
                break
            
            await self._generate_streaming_response(
                context,
                character,
                sequence_number
            )
    
    async def _generate_streaming_response(
        self,
        context,
        character: Character,
        sequence_number: int
    ):
        """Generate streaming response with real-time sentence extraction"""
        
        print(f"🎭 Generating streaming response for {character.name} (seq {sequence_number})")
        
        # Get cached voice (no async DB call!)
        voice = self.get_voice(character)
        
        # Build prompt
        messages = self.character_service.build_character_prompt(
            character,
            context.history
        )
        
        try:
            # Create LLM stream
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            # Create response streamer
            streamer = LLMResponseStreamer(
                character=character,
                voice=voice,
                sequence_number=sequence_number,
                text_output_queue=self.queues.text_output,
                tts_request_queue=self.queues.tts_requests,
                interrupt_signal=self.queues.interrupt_signal,
            )
            
            # Process stream - sentences are queued for TTS as detected
            response_text = await streamer.stream_response(stream)
            
            # Add to conversation history
            context.add_character_message(character, response_text)
            
            # Queue database write (non-blocking)
            # self.persistence.queue_message(MessageCreate(...))
            
            print(f"✓ {character.name} streaming complete ({streamer.sentence_index + 1} sentences)")
            
        except Exception as e:
            print(f"Error generating response for {character.name}: {e}")


# =============================================================================
# Refactored TTS Service
# =============================================================================

class StreamingHiggsTTSService:
    """
    TTS Service that processes sentence chunks as they arrive.
    
    Key improvements:
    - Processes SentenceChunk immediately (no buffering until final)
    - Concurrent generation with semaphore limiting
    - Proper sequencing maintained via sequence_number and sentence_index
    """
    
    def __init__(
        self,
        queue_manager,
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
        
        # Semaphore to limit concurrent GPU operations
        self.generation_semaphore = asyncio.Semaphore(max_concurrent_generations)
        
        # Track active generation tasks
        self.active_tasks: Dict[tuple, asyncio.Task] = {}
    
    async def initialize(self):
        """Initialize Higgs engine"""
        import torch
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        
        loop = asyncio.get_event_loop()
        self.serve_engine = await loop.run_in_executor(None, self._create_engine)
    
    def _create_engine(self):
        import torch
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        
        print("Loading Higgs Audio engine...")
        return HiggsAudioServeEngine(
            self.model_path,
            self.tokenizer_path,
            device=self.device,
            torch_dtype=torch.bfloat16
        )
    
    async def run(self):
        """Main loop: process sentence chunks as they arrive"""
        await self.initialize()
        print("✓ Streaming TTS Service initialized")
        
        while not self.queues.stop_signal.is_set():
            try:
                # Get sentence chunk (now using SentenceChunk, not TTSRequest)
                sentence: SentenceChunk = await asyncio.wait_for(
                    self.queues.tts_requests.get(),
                    timeout=0.05  # Reduced timeout for lower latency
                )
                
                # Check for interrupt
                if self.queues.interrupt_signal.is_set():
                    await self._cancel_all_tasks()
                    continue
                
                # Skip empty final markers (just track completion)
                if not sentence.text.strip():
                    if sentence.is_final:
                        print(f"✓ TTS complete signal for {sentence.speaker.name}")
                    continue
                
                # Start TTS generation immediately for this sentence
                task_key = (
                    sentence.speaker.id,
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
                print(f"TTS Service error: {e}")
                continue
    
    async def _generate_sentence_audio(self, sentence: SentenceChunk):
        """
        Generate audio for a single sentence.
        Uses semaphore to limit concurrent GPU usage.
        """
        async with self.generation_semaphore:
            print(f"🎤 TTS generating: {sentence.speaker.name} sentence {sentence.sentence_index}")
            
            # Build system prompt
            if sentence.voice:
                system_prompt = sentence.voice.get_higgs_system_prompt()
            else:
                system_prompt = (
                    "Generate audio following instruction.\n\n"
                    "<|scene_desc_start|>\n"
                    "Audio is recorded from a quiet room.\n"
                    "<|scene_desc_end|>"
                )
            
            from boson_multimodal.data_types import ChatMLSample, Message
            
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=sentence.text)
            ]
            
            audio_token_buffer = []
            chunk_index = 0
            
            try:
                streamer = self.serve_engine.generate_delta_stream(
                    chat_ml_sample=ChatMLSample(messages=messages),
                    temperature=0.75,
                    top_p=0.95,
                    top_k=50,
                    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                    force_audio_gen=True
                )
                
                async for delta in streamer:
                    if self.queues.interrupt_signal.is_set():
                        print(f"⚠️ TTS interrupted for {sentence.speaker.name}")
                        break
                    
                    if delta.audio_tokens is not None:
                        audio_token_buffer.append(delta.audio_tokens)
                        
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
                            
                            # Keep overlap
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
                    if pcm_chunk:
                        await self._emit_audio_chunk(
                            pcm_chunk,
                            sentence,
                            chunk_index,
                            is_final=True
                        )
                
                print(f"✓ TTS done: {sentence.speaker.name} sentence {sentence.sentence_index}")
                
            except asyncio.CancelledError:
                print(f"⚠️ TTS cancelled for {sentence.speaker.name}")
            except Exception as e:
                print(f"TTS error for {sentence.speaker.name}: {e}")
    
    async def _emit_audio_chunk(
        self,
        pcm_data: bytes,
        sentence: SentenceChunk,
        chunk_index: int,
        is_final: bool
    ):
        """Emit audio chunk to output queue"""
        from dataclasses import dataclass
        
        # Create SequencedAudioChunk (compatible with AudioPlaybackSequencer)
        @dataclass
        class SequencedAudioChunk:
            data: bytes
            sample_rate: int
            sequence_number: int
            character_chunk_index: int
            is_final: bool
            speaker_id: str
            speaker_name: str
            timestamp: float
            # New: track sentence within sequence
            sentence_index: int = 0
        
        audio_chunk = SequencedAudioChunk(
            data=pcm_data,
            sample_rate=self.serve_engine.audio_tokenizer.sampling_rate,
            sequence_number=sentence.sequence_number,
            character_chunk_index=chunk_index,
            is_final=is_final,
            speaker_id=sentence.speaker.id,
            speaker_name=sentence.speaker.name,
            timestamp=time.time(),
            sentence_index=sentence.sentence_index
        )
        
        await self.queues.audio_output.put(audio_chunk)
    
    async def _process_audio_tokens(self, tokens):
        """Convert audio tokens to PCM"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._decode_tokens, tokens)
    
    def _decode_tokens(self, tokens):
        """Decode audio tokens to PCM16"""
        import torch
        import numpy as np
        from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
        
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
    
    async def _cancel_all_tasks(self):
        """Cancel all active TTS tasks"""
        for task in self.active_tasks.values():
            task.cancel()
        self.active_tasks.clear()
    
    def _cleanup_completed_tasks(self):
        """Remove completed tasks from tracking dict"""
        completed = [k for k, v in self.active_tasks.items() if v.done()]
        for key in completed:
            del self.active_tasks[key]


# =============================================================================
# Integration Example
# =============================================================================

def create_streaming_orchestrator(config: dict, queue_manager, character_service):
    """
    Factory function to create properly configured streaming orchestrator.
    
    Usage in VoiceChatOrchestrator:
    
        self.llm = create_streaming_orchestrator(
            config, 
            self.queues, 
            self.character_service
        )
    """
    return StreamingLLMOrchestrator(
        queue_manager=queue_manager,
        character_service=character_service,
        api_key=config["openrouter_api_key"],
        model=config.get("llm_model", "anthropic/claude-3.5-sonnet")
    )


def create_streaming_tts(config: dict, queue_manager):
    """
    Factory function to create streaming TTS service.
    
    Usage in VoiceChatOrchestrator:
    
        self.tts = create_streaming_tts(config, self.queues)
    """
    return StreamingHiggsTTSService(
        queue_manager=queue_manager,
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
