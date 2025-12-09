/**
 * Audio Manager - Functional Pattern
 * Handles microphone capture (STT) and audio playback (TTS)
 * Focus: Audio processing and streaming
 */

import { websocket, MESSAGE_TYPES } from './websocket.js';
import { chat } from './chat.js';

// ============================================================================
// Constants
// ============================================================================

const AUDIO_CONFIG = {
  // STT (Microphone Input)
  sttSampleRate: 16000,        // Backend expects 16kHz
  sttChunkSize: 320,           // 320 samples @ 16kHz = 20ms

  // TTS (Audio Playback)
  ttsSampleRate: 24000,        // Backend sends 24kHz

  // Audio Context
  audioConstraints: {
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  },
};

const CAPTURE_STATE = {
  IDLE: 'idle',
  STARTING: 'starting',
  CAPTURING: 'capturing',
  STOPPING: 'stopping',
};

const PLAYBACK_STATE = {
  IDLE: 'idle',
  PLAYING: 'playing',
  PAUSED: 'paused',
};

// ============================================================================
// State Management (Immutable)
// ============================================================================

/**
 * Create initial audio state
 */
const createAudioState = () => ({
  // Audio Context
  audioContext: null,

  // Microphone Capture
  captureState: CAPTURE_STATE.IDLE,
  mediaStream: null,
  microphoneSource: null,
  audioWorkletNode: null,

  // Audio Playback
  playbackState: PLAYBACK_STATE.IDLE,
  playbackQueue: [],
  currentPlayback: null,
  nextPlaybackTime: 0,
});

/**
 * Update audio state immutably
 */
const updateAudioState = (state, updates) => ({
  ...state,
  ...updates,
});

/**
 * Set capture state
 */
const setCaptureState = (state, captureState) =>
  updateAudioState(state, { captureState });

/**
 * Set playback state
 */
const setPlaybackState = (state, playbackState) =>
  updateAudioState(state, { playbackState });

/**
 * Add to playback queue
 */
const addToPlaybackQueue = (state, audioData) => ({
  ...state,
  playbackQueue: [...state.playbackQueue, audioData],
});

/**
 * Remove from playback queue
 */
const removeFromPlaybackQueue = (state) => {
  const [, ...rest] = state.playbackQueue;
  return updateAudioState(state, { playbackQueue: rest });
};

/**
 * Clear playback queue
 */
const clearPlaybackQueue = (state) =>
  updateAudioState(state, { playbackQueue: [] });

// ============================================================================
// Audio Format Conversion
// ============================================================================

/**
 * Convert Float32 audio to PCM16 (Int16)
 * @param {Float32Array} float32Array - Audio samples in range [-1.0, 1.0]
 * @returns {Int16Array} PCM16 audio samples
 */
const float32ToPCM16 = (float32Array) => {
  const pcm16 = new Int16Array(float32Array.length);

  for (let i = 0; i < float32Array.length; i++) {
    // Clamp to [-1.0, 1.0]
    const clamped = Math.max(-1.0, Math.min(1.0, float32Array[i]));

    // Convert to 16-bit integer
    pcm16[i] = clamped < 0
      ? clamped * 0x8000
      : clamped * 0x7FFF;
  }

  return pcm16;
};

/**
 * Convert PCM16 (Int16) to Float32 audio
 * @param {Int16Array} pcm16Array - PCM16 audio samples
 * @returns {Float32Array} Audio samples in range [-1.0, 1.0]
 */
const pcm16ToFloat32 = (pcm16Array) => {
  const float32 = new Float32Array(pcm16Array.length);

  for (let i = 0; i < pcm16Array.length; i++) {
    float32[i] = pcm16Array[i] / (pcm16Array[i] < 0 ? 0x8000 : 0x7FFF);
  }

  return float32;
};

/**
 * Convert Int16Array to ArrayBuffer for WebSocket transmission
 */
const int16ToArrayBuffer = (int16Array) => {
  return int16Array.buffer.slice(
    int16Array.byteOffset,
    int16Array.byteOffset + int16Array.byteLength
  );
};

// ============================================================================
// Audio Context Management
// ============================================================================

/**
 * Create and initialize audio context
 */
const createAudioContext = async () => {
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContext();

    // Resume context if suspended (browser autoplay policy)
    if (context.state === 'suspended') {
      await context.resume();
    }

    return context;
  } catch (error) {
    console.error('Failed to create audio context:', error);
    return null;
  }
};

/**
 * Close audio context
 */
const closeAudioContext = async (audioContext) => {
  if (audioContext && audioContext.state !== 'closed') {
    await audioContext.close();
  }
};

// ============================================================================
// Microphone Capture
// ============================================================================

/**
 * Request microphone access
 */
const requestMicrophoneAccess = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia(AUDIO_CONFIG.audioConstraints);
    return stream;
  } catch (error) {
    console.error('Failed to access microphone:', error);
    return null;
  }
};

/**
 * Create AudioWorklet for microphone processing
 */
const createMicrophoneWorklet = async (audioManager) => {
  const { audioContext } = audioManager.state;

  if (!audioContext) return null;

  try {
    // Load audio processor worklet
    await audioContext.audioWorklet.addModule('audio-processor.js');

    // Create worklet node
    const workletNode = new AudioWorkletNode(audioContext, 'microphone-processor');

    // Configure worklet
    workletNode.port.postMessage({
      type: 'setSampleRate',
      sampleRate: AUDIO_CONFIG.sttSampleRate,
    });

    workletNode.port.postMessage({
      type: 'setChunkSize',
      chunkSize: AUDIO_CONFIG.sttChunkSize,
    });

    // Handle processed audio data
    workletNode.port.onmessage = (event) => {
      if (event.data.type === 'audioData') {
        handleAudioChunk(audioManager, event.data.samples);
      }
    };

    return workletNode;
  } catch (error) {
    console.error('Failed to create audio worklet:', error);
    return null;
  }
};

/**
 * Handle audio chunk from worklet
 */
const handleAudioChunk = (audioManager, float32Samples) => {
  // Convert to PCM16
  const pcm16Samples = float32ToPCM16(float32Samples);

  // Convert to ArrayBuffer for WebSocket
  const arrayBuffer = int16ToArrayBuffer(pcm16Samples);

  // Send to backend via WebSocket
  websocket.sendBinary(arrayBuffer);
};

/**
 * Start microphone capture
 */
const startCapture = async (audioManager) => {
  if (audioManager.state.captureState !== CAPTURE_STATE.IDLE) {
    console.warn('Capture already in progress');
    return false;
  }

  audioManager.state = setCaptureState(audioManager.state, CAPTURE_STATE.STARTING);

  // Create audio context if needed
  if (!audioManager.state.audioContext) {
    const audioContext = await createAudioContext();
    if (!audioContext) {
      audioManager.state = setCaptureState(audioManager.state, CAPTURE_STATE.IDLE);
      return false;
    }
    audioManager.state = updateAudioState(audioManager.state, { audioContext });
  }

  // Request microphone access
  const mediaStream = await requestMicrophoneAccess();
  if (!mediaStream) {
    audioManager.state = setCaptureState(audioManager.state, CAPTURE_STATE.IDLE);
    return false;
  }

  // Create audio source from microphone
  const microphoneSource = audioManager.state.audioContext.createMediaStreamSource(mediaStream);

  // Create audio worklet for processing
  const audioWorkletNode = await createMicrophoneWorklet(audioManager);
  if (!audioWorkletNode) {
    mediaStream.getTracks().forEach(track => track.stop());
    audioManager.state = setCaptureState(audioManager.state, CAPTURE_STATE.IDLE);
    return false;
  }

  // Connect audio pipeline: microphone → worklet
  microphoneSource.connect(audioWorkletNode);

  // Update state
  audioManager.state = updateAudioState(audioManager.state, {
    captureState: CAPTURE_STATE.CAPTURING,
    mediaStream,
    microphoneSource,
    audioWorkletNode,
  });

  console.log('Microphone capture started');
  return true;
};

/**
 * Stop microphone capture
 */
const stopCapture = (audioManager) => {
  if (audioManager.state.captureState !== CAPTURE_STATE.CAPTURING) {
    return false;
  }

  audioManager.state = setCaptureState(audioManager.state, CAPTURE_STATE.STOPPING);

  // Disconnect audio pipeline
  if (audioManager.state.microphoneSource) {
    audioManager.state.microphoneSource.disconnect();
  }

  if (audioManager.state.audioWorkletNode) {
    audioManager.state.audioWorkletNode.disconnect();
    audioManager.state.audioWorkletNode.port.close();
  }

  // Stop media stream
  if (audioManager.state.mediaStream) {
    audioManager.state.mediaStream.getTracks().forEach(track => track.stop());
  }

  // Clear state
  audioManager.state = updateAudioState(audioManager.state, {
    captureState: CAPTURE_STATE.IDLE,
    mediaStream: null,
    microphoneSource: null,
    audioWorkletNode: null,
  });

  console.log('Microphone capture stopped');
  return true;
};

// ============================================================================
// Audio Playback (TTS)
// ============================================================================

/**
 * Convert Blob to ArrayBuffer
 */
const blobToArrayBuffer = async (blob) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsArrayBuffer(blob);
  });
};

/**
 * Create AudioBuffer from PCM16 data
 */
const createAudioBuffer = (audioContext, pcm16Data, sampleRate) => {
  // Convert PCM16 to Float32
  const float32Data = pcm16ToFloat32(new Int16Array(pcm16Data));

  // Create audio buffer
  const audioBuffer = audioContext.createBuffer(
    1, // mono
    float32Data.length,
    sampleRate
  );

  // Copy data to buffer
  audioBuffer.getChannelData(0).set(float32Data);

  return audioBuffer;
};

/**
 * Play audio buffer
 */
const playAudioBuffer = (audioManager, audioBuffer) => {
  const { audioContext } = audioManager.state;

  if (!audioContext) return;

  // Create buffer source
  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);

  // Schedule playback
  const startTime = Math.max(audioContext.currentTime, audioManager.state.nextPlaybackTime);
  source.start(startTime);

  // Update next playback time for gapless playback
  const duration = audioBuffer.duration;
  audioManager.state = updateAudioState(audioManager.state, {
    nextPlaybackTime: startTime + duration,
    currentPlayback: source,
  });

  // Handle playback end
  source.onended = () => {
    processPlaybackQueue(audioManager);
  };
};

/**
 * Process playback queue
 */
const processPlaybackQueue = async (audioManager) => {
  if (audioManager.state.playbackQueue.length === 0) {
    audioManager.state = setPlaybackState(audioManager.state, PLAYBACK_STATE.IDLE);
    return;
  }

  // Get next item from queue
  const audioData = audioManager.state.playbackQueue[0];
  audioManager.state = removeFromPlaybackQueue(audioManager.state);

  try {
    // Convert Blob to ArrayBuffer
    const arrayBuffer = await blobToArrayBuffer(audioData);

    // Create audio buffer
    const audioBuffer = createAudioBuffer(
      audioManager.state.audioContext,
      arrayBuffer,
      AUDIO_CONFIG.ttsSampleRate
    );

    // Play audio
    playAudioBuffer(audioManager, audioBuffer);

    audioManager.state = setPlaybackState(audioManager.state, PLAYBACK_STATE.PLAYING);
  } catch (error) {
    console.error('Failed to play audio:', error);
    // Continue with next item in queue
    processPlaybackQueue(audioManager);
  }
};

/**
 * Queue audio for playback
 */
const queueAudio = async (audioManager, audioBlob) => {
  // Add to queue
  audioManager.state = addToPlaybackQueue(audioManager.state, audioBlob);

  // Start processing if idle
  if (audioManager.state.playbackState === PLAYBACK_STATE.IDLE) {
    await processPlaybackQueue(audioManager);
  }
};

/**
 * Stop audio playback
 */
const stopPlayback = (audioManager) => {
  // Stop current playback
  if (audioManager.state.currentPlayback) {
    try {
      audioManager.state.currentPlayback.stop();
    } catch (error) {
      // Already stopped
    }
  }

  // Clear queue and state
  audioManager.state = updateAudioState(audioManager.state, {
    playbackState: PLAYBACK_STATE.IDLE,
    playbackQueue: [],
    currentPlayback: null,
    nextPlaybackTime: 0,
  });

  console.log('Audio playback stopped');
};

// ============================================================================
// Integration with Chat & WebSocket
// ============================================================================

/**
 * Setup WebSocket audio handler
 */
const setupWebSocketHandler = (audioManager) => {
  websocket.on(MESSAGE_TYPES.AUDIO, async (audioBlob) => {
    await queueAudio(audioManager, audioBlob);
  });
};

/**
 * Sync with chat mic state
 */
const syncMicStateWithChat = (audioManager, micState) => {
  // Update chat mic button visual state
  if (chat.setMicState) {
    chat.setMicState(micState);
  }
};

// ============================================================================
// Initialization & Cleanup
// ============================================================================

/**
 * Initialize audio manager
 */
const initialize = async (audioManager) => {
  // Create audio context
  const audioContext = await createAudioContext();
  if (!audioContext) {
    console.error('Failed to initialize audio manager');
    return false;
  }

  audioManager.state = updateAudioState(audioManager.state, { audioContext });

  // Setup WebSocket handler
  setupWebSocketHandler(audioManager);

  console.log('Audio manager initialized');
  return true;
};

/**
 * Cleanup audio resources
 */
const cleanup = async (audioManager) => {
  // Stop capture if active
  if (audioManager.state.captureState === CAPTURE_STATE.CAPTURING) {
    stopCapture(audioManager);
  }

  // Stop playback
  stopPlayback(audioManager);

  // Close audio context
  if (audioManager.state.audioContext) {
    await closeAudioContext(audioManager.state.audioContext);
  }

  audioManager.state = createAudioState();
  console.log('Audio manager cleaned up');
};

// ============================================================================
// Public API
// ============================================================================

/**
 * Create an audio manager instance
 */
export const createAudioManager = () => {
  const audioManager = {
    state: createAudioState(),
  };

  return {
    // Initialization
    initialize: async () => await initialize(audioManager),
    cleanup: async () => await cleanup(audioManager),

    // Microphone capture
    startCapture: async () => await startCapture(audioManager),
    stopCapture: () => stopCapture(audioManager),
    isCapturing: () => audioManager.state.captureState === CAPTURE_STATE.CAPTURING,

    // Audio playback
    queueAudio: async (audioBlob) => await queueAudio(audioManager, audioBlob),
    stopPlayback: () => stopPlayback(audioManager),
    isPlaying: () => audioManager.state.playbackState === PLAYBACK_STATE.PLAYING,

    // State access
    getState: () => audioManager.state,
    getCaptureState: () => audioManager.state.captureState,
    getPlaybackState: () => audioManager.state.playbackState,

    // Internal reference
    _audioManager: audioManager,
  };
};

// ============================================================================
// Default Export (Singleton)
// ============================================================================

export const audio = createAudioManager();
