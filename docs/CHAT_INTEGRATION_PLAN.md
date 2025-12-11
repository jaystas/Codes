# Frontend Chat Integration Plan

## Executive Summary

This document outlines the implementation plan for integrating the frontend chat interface with the FastAPI WebSocket server. The goal is to create a seamless voice-first conversational experience with automatic speech-to-text, real-time LLM responses, and text-to-speech audio playback.

---

## 1. Backend Analysis

### 1.1 WebSocket Server Overview

**Location:** `backend/fastapi_server.py`
**Endpoint:** `/ws`

The server implements a full-duplex WebSocket connection supporting:
- **Text messages (JSON):** Commands and text data
- **Binary messages:** Audio data (PCM16)

### 1.2 Server-to-Client Message Types

| Message Type | Format | Description |
|-------------|--------|-------------|
| `stt_update` | `{type: "stt_update", text: string}` | Real-time transcription update |
| `stt_stabilized` | `{type: "stt_stabilized", text: string}` | Stabilized transcription text |
| `stt_final` | `{type: "stt_final", text: string}` | Final transcription result |
| `response_chunk` | `{type: "response_chunk", data: {text, character_name, message_id, is_final}}` | LLM response text chunk |
| **Binary** | `ArrayBuffer` | TTS audio chunk (PCM16 @ 24kHz) |

### 1.3 Client-to-Server Message Types

| Message Type | Payload | Description |
|-------------|---------|-------------|
| `user_message` | `{text: string}` | Manual text message submission |
| `start_listening` | `{}` | Begin STT listening mode |
| `stop_listening` | `{}` | Stop STT listening |
| `model_settings` | `{model, temperature, top_p, ...}` | Update LLM parameters |
| `set_characters` | `{characters: Character[]}` | Set active characters |
| `clear_history` | `{}` | Clear conversation history |
| `interrupt` | `{}` | Stop current generation |
| `refresh_active_characters` | `{}` | Reload characters from database |
| **Binary** | `ArrayBuffer` | Audio data (PCM16 @ 16kHz, mono) |

### 1.4 Audio Specifications

| Direction | Sample Rate | Format | Channels |
|-----------|-------------|--------|----------|
| Client → Server (STT) | 16,000 Hz | PCM16 (Int16) | Mono |
| Server → Client (TTS) | 24,000 Hz | PCM16 (Int16) | Mono |

---

## 2. The WebSocket "Zombie Connection" Issue

### 2.1 Problem Description

You mentioned experiencing a WebSocket that appears connected but stops responding after certain operations. This is a known issue typically caused by:

1. **Stale Connection State:** The WebSocket object reports `readyState === OPEN` but the underlying TCP connection has died
2. **Message Handler Exhaustion:** Single-use message handlers that don't re-register
3. **Async Operation Blocking:** Pending promises preventing new message processing
4. **Missing Heartbeat/Ping-Pong:** No mechanism to detect dead connections

### 2.2 Solution Strategy

The `websocket.js` implementation will include:

```javascript
// Key strategies to prevent zombie connections:

1. Heartbeat/Ping-Pong System
   - Client sends ping every 30 seconds
   - Server responds with pong (or vice versa)
   - If no pong received within 5 seconds, reconnect

2. Connection Health Monitoring
   - Track last message received timestamp
   - Automatic reconnection if silent for too long

3. Proper Message Loop Architecture
   - Single persistent onmessage handler
   - Never replace/remove the handler

4. Clean Reconnection Logic
   - Properly close stale connections before reconnecting
   - Exponential backoff for reconnection attempts
   - State restoration after reconnection
```

---

## 3. File Structure & Responsibilities

### 3.1 New Files to Create

```
frontend/
├── websocket.js      # WebSocket connection management
├── chat.js           # Chat UI/display and conversation logic
├── stt-audio.js      # Audio input capture (MediaRecorder/AudioWorklet)
├── stt-processor.js  # AudioWorkletProcessor for STT
└── tts-audio.js      # TTS audio playback management
```

### 3.2 Files to Modify

```
frontend/
├── main.js           # Import and initialize chat system
├── editor.js         # Connect mic/send buttons to chat system
└── styles.css        # Add chat message styles
```

---

## 4. Implementation Specifications

### 4.1 websocket.js - Connection Management

**Purpose:** Robust WebSocket connection with automatic reconnection and health monitoring

```javascript
// ============================================
// EXPORTS
// ============================================
export {
  connect,           // (url?: string) => Promise<void>
  disconnect,        // () => void
  sendText,          // (message: object) => void
  sendAudio,         // (audioData: ArrayBuffer) => void
  isConnected,       // () => boolean
  onMessage,         // (handler: Function) => unsubscribe
  onConnectionChange,// (handler: Function) => unsubscribe
  getState,          // () => ConnectionState
}

// ============================================
// STATE MANAGEMENT
// ============================================
const state = {
  socket: null,
  status: 'disconnected', // 'connecting' | 'connected' | 'disconnected' | 'error'
  reconnectAttempts: 0,
  lastMessageTime: 0,
  pingInterval: null,
  reconnectTimeout: null,
}

// ============================================
// CONFIGURATION
// ============================================
const config = {
  url: `ws://${window.location.host}/ws`,
  reconnect: true,
  maxReconnectAttempts: 10,
  reconnectBaseDelay: 1000,     // Start with 1 second
  reconnectMaxDelay: 30000,     // Max 30 seconds
  pingInterval: 30000,          // Ping every 30 seconds
  pongTimeout: 5000,            // Wait 5s for pong response
  messageTimeout: 60000,        // Reconnect if no messages for 60s
}

// ============================================
// KEY IMPLEMENTATION DETAILS
// ============================================

// 1. Single persistent message handler
function setupMessageHandler(socket) {
  socket.onmessage = (event) => {
    state.lastMessageTime = Date.now()

    if (event.data instanceof ArrayBuffer) {
      // Binary = TTS audio
      notifyListeners('audio', event.data)
    } else {
      // Text = JSON message
      const message = JSON.parse(event.data)
      notifyListeners('message', message)
    }
  }
}

// 2. Health monitoring with ping/pong
function startHeartbeat() {
  state.pingInterval = setInterval(() => {
    if (state.socket?.readyState === WebSocket.OPEN) {
      // Send ping
      sendText({ type: 'ping' })

      // Set timeout for pong
      const pongTimeout = setTimeout(() => {
        console.warn('Pong timeout - reconnecting')
        reconnect()
      }, config.pongTimeout)

      // Clear timeout when pong received (handled in message listener)
    }
  }, config.pingInterval)
}

// 3. Exponential backoff reconnection
function reconnect() {
  if (state.reconnectAttempts >= config.maxReconnectAttempts) {
    state.status = 'error'
    return
  }

  const delay = Math.min(
    config.reconnectBaseDelay * Math.pow(2, state.reconnectAttempts),
    config.reconnectMaxDelay
  )

  state.reconnectTimeout = setTimeout(() => {
    state.reconnectAttempts++
    connect()
  }, delay)
}

// 4. Clean connection closure
function disconnect() {
  clearInterval(state.pingInterval)
  clearTimeout(state.reconnectTimeout)

  if (state.socket) {
    state.socket.onclose = null // Prevent reconnection attempt
    state.socket.close(1000, 'Client disconnect')
    state.socket = null
  }

  state.status = 'disconnected'
}
```

**Critical Anti-Patterns to Avoid:**
```javascript
// DON'T: Replace onmessage handler
socket.onmessage = handler1
socket.onmessage = handler2  // handler1 is now gone!

// DO: Use event emitter pattern
const listeners = new Set()
socket.onmessage = (e) => listeners.forEach(fn => fn(e))

// DON'T: Block the message handler with await
socket.onmessage = async (e) => {
  await heavyOperation()  // Blocks subsequent messages!
}

// DO: Queue async operations
socket.onmessage = (e) => {
  queue.push(() => heavyOperation(e.data))
  processQueue()  // Non-blocking
}
```

---

### 4.2 chat.js - Chat UI & Conversation Logic

**Purpose:** Manage chat UI, message display, and conversation state

```javascript
// ============================================
// EXPORTS
// ============================================
export {
  initChat,              // () => void
  addUserMessage,        // (text: string) => void
  addAssistantMessage,   // (text: string, characterName: string) => Element
  updateAssistantMessage,// (element: Element, text: string) => void
  setTypingIndicator,    // (characterName: string, show: boolean) => void
  clearChat,             // () => void
  sendMessage,           // (text: string) => void
  getCurrentConversation,// () => Message[]
}

// ============================================
// STATE
// ============================================
const state = {
  conversation: [],              // Local conversation history
  currentStreamElement: null,    // Element being streamed to
  currentStreamText: '',         // Accumulated stream text
  isStreaming: false,
}

// ============================================
// MESSAGE HANDLING
// ============================================

function handleServerMessage(message) {
  switch (message.type) {
    case 'stt_update':
      updateSTTPreview(message.text, false)
      break

    case 'stt_stabilized':
      updateSTTPreview(message.text, true)
      break

    case 'stt_final':
      finalizeUserMessage(message.text)
      break

    case 'response_chunk':
      handleResponseChunk(message.data)
      break
  }
}

function handleResponseChunk(data) {
  const { text, character_name, message_id, is_final } = data

  if (!state.isStreaming || state.currentMessageId !== message_id) {
    // Start new message stream
    state.currentStreamElement = createAssistantMessageElement(character_name)
    state.currentStreamText = ''
    state.currentMessageId = message_id
    state.isStreaming = true
  }

  // Append text chunk
  state.currentStreamText += text
  renderMarkdown(state.currentStreamElement, state.currentStreamText)

  if (is_final) {
    // Finalize message
    state.conversation.push({
      role: 'assistant',
      name: character_name,
      content: state.currentStreamText,
    })
    state.isStreaming = false
    scrollToBottom()
  }
}

// ============================================
// UI RENDERING
// ============================================

function createAssistantMessageElement(characterName) {
  const messagesArea = document.querySelector('.messages-area')

  const messageEl = document.createElement('div')
  messageEl.className = 'chat-message assistant'
  messageEl.innerHTML = `
    <div class="message-header">
      <span class="character-name">${escapeHtml(characterName)}</span>
      <span class="message-time">${formatTime(new Date())}</span>
    </div>
    <div class="message-content"></div>
  `

  messagesArea.appendChild(messageEl)
  return messageEl.querySelector('.message-content')
}

// ============================================
// STT PREVIEW (Live Transcription Display)
// ============================================

function updateSTTPreview(text, isStabilized) {
  let preview = document.getElementById('stt-preview')

  if (!preview) {
    preview = createSTTPreviewElement()
  }

  preview.textContent = text
  preview.classList.toggle('stabilized', isStabilized)
}

function finalizeUserMessage(text) {
  removeSTTPreview()
  addUserMessage(text)

  // Send via websocket if not already sent (for voice input)
  // Text input uses sendMessage() directly
}
```

**UI State Flow:**
```
[Idle] → (user speaks) → [Showing STT Preview]
      → (speech ends) → [User Message Added]
      → (LLM streaming) → [Assistant Message Streaming]
      → (stream complete) → [Idle]
```

---

### 4.3 stt-audio.js - Audio Input Capture

**Purpose:** Capture microphone audio and stream to server

```javascript
// ============================================
// EXPORTS
// ============================================
export {
  initAudioCapture,     // () => Promise<boolean>
  startRecording,       // () => void
  stopRecording,        // () => void
  isRecording,          // () => boolean
  setVADEnabled,        // (enabled: boolean) => void
  onStateChange,        // (handler: Function) => unsubscribe
}

// ============================================
// STATE
// ============================================
const state = {
  audioContext: null,
  mediaStream: null,
  workletNode: null,
  status: 'idle',        // 'idle' | 'listening' | 'recording' | 'paused'
  vadEnabled: true,
  isTTSPlaying: false,   // Pause recording during TTS playback
}

// ============================================
// CONFIGURATION
// ============================================
const config = {
  sampleRate: 16000,       // Required by backend STT
  channelCount: 1,         // Mono
  bufferSize: 4096,        // Samples per buffer (256ms at 16kHz)
  vadThreshold: 0.01,      // Voice activity detection threshold
  vadSilenceTimeout: 700,  // ms of silence before stopping
}

// ============================================
// INITIALIZATION
// ============================================

async function initAudioCapture() {
  try {
    // Request microphone access
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: config.sampleRate,
        channelCount: config.channelCount,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      }
    })

    // Create audio context at correct sample rate
    state.audioContext = new AudioContext({
      sampleRate: config.sampleRate,
    })

    // Load and connect AudioWorklet processor
    await state.audioContext.audioWorklet.addModule('stt-processor.js')

    state.workletNode = new AudioWorkletNode(
      state.audioContext,
      'stt-processor',
      {
        processorOptions: {
          bufferSize: config.bufferSize,
        }
      }
    )

    // Connect audio pipeline
    const source = state.audioContext.createMediaStreamSource(state.mediaStream)
    source.connect(state.workletNode)

    // Handle audio data from worklet
    state.workletNode.port.onmessage = handleWorkletMessage

    return true

  } catch (error) {
    console.error('Failed to initialize audio capture:', error)
    return false
  }
}

// ============================================
// AUDIO FLOW CONTROL
// ============================================

function startRecording() {
  if (state.status === 'recording') return
  if (state.isTTSPlaying) {
    state.status = 'paused'
    return
  }

  state.status = 'listening'
  state.workletNode.port.postMessage({ command: 'start' })

  // Tell server to start listening
  websocket.sendText({ type: 'start_listening' })
}

function stopRecording() {
  state.status = 'idle'
  state.workletNode.port.postMessage({ command: 'stop' })
  websocket.sendText({ type: 'stop_listening' })
}

// Called by tts-audio.js
function setTTSPlaying(isPlaying) {
  state.isTTSPlaying = isPlaying

  if (isPlaying && state.status === 'recording') {
    // Pause recording during TTS
    state.workletNode.port.postMessage({ command: 'pause' })
    state.status = 'paused'
  } else if (!isPlaying && state.status === 'paused') {
    // Resume recording after TTS
    state.workletNode.port.postMessage({ command: 'resume' })
    state.status = 'listening'
  }
}

// ============================================
// WORKLET MESSAGE HANDLING
// ============================================

function handleWorkletMessage(event) {
  const { type, data } = event.data

  switch (type) {
    case 'audio':
      // data is Int16Array
      if (state.status === 'recording') {
        websocket.sendAudio(data.buffer)
      }
      break

    case 'vad_start':
      // Voice activity detected
      if (state.status === 'listening') {
        state.status = 'recording'
        notifyStateChange('recording')
      }
      break

    case 'vad_stop':
      // Silence detected after speech
      state.status = 'listening'
      notifyStateChange('listening')
      break
  }
}
```

**Audio State Machine:**
```
                    ┌──────────────┐
                    │     IDLE     │
                    └──────┬───────┘
                           │ startRecording()
                           ▼
                    ┌──────────────┐
            ┌───────│  LISTENING   │◄──────┐
            │       └──────┬───────┘       │
            │              │ VAD start     │ VAD stop
            │              ▼               │
            │       ┌──────────────┐       │
            │       │  RECORDING   │───────┘
            │       └──────┬───────┘
            │              │ TTS starts
            │              ▼
            │       ┌──────────────┐
            │       │    PAUSED    │
            │       └──────┬───────┘
            │              │ TTS ends
            │              ▼
            │       ┌──────────────┐
            └──────►│  LISTENING   │
                    └──────────────┘
```

---

### 4.4 stt-processor.js - AudioWorkletProcessor

**Purpose:** Process audio samples in real-time on a separate thread

```javascript
// stt-processor.js - AudioWorkletProcessor
// NOTE: This file runs in a separate AudioWorklet thread

class STTProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()

    this.bufferSize = options.processorOptions?.bufferSize || 4096
    this.buffer = new Float32Array(this.bufferSize)
    this.bufferIndex = 0

    this.isActive = false
    this.isPaused = false

    // VAD state
    this.vadThreshold = 0.01
    this.silenceCounter = 0
    this.silenceThreshold = 700 / (1000 / (16000 / 128)) // ~44 frames at 16kHz
    this.isSpeaking = false

    // Handle commands from main thread
    this.port.onmessage = this.handleCommand.bind(this)
  }

  handleCommand(event) {
    switch (event.data.command) {
      case 'start':
        this.isActive = true
        this.isPaused = false
        break
      case 'stop':
        this.isActive = false
        this.isSpeaking = false
        break
      case 'pause':
        this.isPaused = true
        break
      case 'resume':
        this.isPaused = false
        break
    }
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0]
    if (!input || !input[0]) return true

    const samples = input[0]

    if (!this.isActive || this.isPaused) {
      return true
    }

    // Calculate RMS for VAD
    let sumSquares = 0
    for (let i = 0; i < samples.length; i++) {
      sumSquares += samples[i] * samples[i]
    }
    const rms = Math.sqrt(sumSquares / samples.length)

    // Voice Activity Detection
    const wasSpeeking = this.isSpeaking

    if (rms > this.vadThreshold) {
      this.isSpeaking = true
      this.silenceCounter = 0

      if (!wasSpeeking) {
        this.port.postMessage({ type: 'vad_start' })
      }
    } else if (this.isSpeaking) {
      this.silenceCounter++

      if (this.silenceCounter > this.silenceThreshold) {
        this.isSpeaking = false
        this.port.postMessage({ type: 'vad_stop' })
      }
    }

    // Buffer samples for transmission
    for (let i = 0; i < samples.length; i++) {
      this.buffer[this.bufferIndex++] = samples[i]

      if (this.bufferIndex >= this.bufferSize) {
        // Convert Float32 to Int16
        const int16Buffer = this.float32ToInt16(this.buffer)
        this.port.postMessage({ type: 'audio', data: int16Buffer }, [int16Buffer.buffer])
        this.bufferIndex = 0
      }
    }

    return true // Keep processor alive
  }

  float32ToInt16(float32Array) {
    const int16Array = new Int16Array(float32Array.length)
    for (let i = 0; i < float32Array.length; i++) {
      // Clamp and convert
      const s = Math.max(-1, Math.min(1, float32Array[i]))
      int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
    }
    return int16Array
  }
}

registerProcessor('stt-processor', STTProcessor)
```

---

### 4.5 tts-audio.js - TTS Audio Playback

**Purpose:** Manage TTS audio queue and playback

```javascript
// ============================================
// EXPORTS
// ============================================
export {
  initTTSPlayback,      // () => Promise<void>
  queueAudioChunk,      // (audioData: ArrayBuffer) => void
  play,                 // () => void
  pause,                // () => void
  stop,                 // () => void
  setVolume,            // (volume: number) => void
  isPlaying,            // () => boolean
  onPlaybackChange,     // (handler: Function) => unsubscribe
}

// ============================================
// STATE
// ============================================
const state = {
  audioContext: null,
  gainNode: null,
  audioQueue: [],           // Queue of AudioBuffer objects
  currentSource: null,      // Currently playing AudioBufferSourceNode
  isPlaying: false,
  isPaused: false,
  volume: 1.0,
}

// ============================================
// CONFIGURATION
// ============================================
const config = {
  sampleRate: 24000,        // TTS output sample rate
  channelCount: 1,          // Mono
  minBufferMs: 100,         // Start playback after this much audio buffered
}

// ============================================
// INITIALIZATION
// ============================================

async function initTTSPlayback() {
  state.audioContext = new AudioContext({
    sampleRate: config.sampleRate,
  })

  // Create gain node for volume control
  state.gainNode = state.audioContext.createGain()
  state.gainNode.connect(state.audioContext.destination)
  state.gainNode.gain.value = state.volume

  // Resume context on user interaction (browser policy)
  if (state.audioContext.state === 'suspended') {
    document.addEventListener('click', () => {
      state.audioContext.resume()
    }, { once: true })
  }
}

// ============================================
// AUDIO QUEUE MANAGEMENT
// ============================================

function queueAudioChunk(pcm16Data) {
  // Convert PCM16 ArrayBuffer to AudioBuffer
  const int16Array = new Int16Array(pcm16Data)
  const float32Array = int16ToFloat32(int16Array)

  const audioBuffer = state.audioContext.createBuffer(
    config.channelCount,
    float32Array.length,
    config.sampleRate
  )
  audioBuffer.getChannelData(0).set(float32Array)

  state.audioQueue.push(audioBuffer)

  // Auto-start playback if not playing and have enough buffer
  if (!state.isPlaying && !state.isPaused) {
    const bufferedMs = getBufferedDuration() * 1000
    if (bufferedMs >= config.minBufferMs) {
      play()
    }
  }
}

function getBufferedDuration() {
  return state.audioQueue.reduce((sum, buf) => sum + buf.duration, 0)
}

// ============================================
// PLAYBACK CONTROL
// ============================================

function play() {
  if (state.audioQueue.length === 0) {
    state.isPlaying = false
    notifyPlaybackChange(false)
    return
  }

  // Resume audio context if suspended
  if (state.audioContext.state === 'suspended') {
    state.audioContext.resume()
  }

  state.isPlaying = true
  state.isPaused = false
  notifyPlaybackChange(true)

  // Notify STT to pause recording
  sttAudio.setTTSPlaying(true)

  playNextChunk()
}

function playNextChunk() {
  if (state.audioQueue.length === 0 || state.isPaused) {
    if (state.audioQueue.length === 0) {
      state.isPlaying = false
      notifyPlaybackChange(false)

      // Notify STT to resume recording
      sttAudio.setTTSPlaying(false)
    }
    return
  }

  const audioBuffer = state.audioQueue.shift()

  state.currentSource = state.audioContext.createBufferSource()
  state.currentSource.buffer = audioBuffer
  state.currentSource.connect(state.gainNode)

  state.currentSource.onended = () => {
    playNextChunk()
  }

  state.currentSource.start()
}

function stop() {
  state.audioQueue = []

  if (state.currentSource) {
    state.currentSource.onended = null
    state.currentSource.stop()
    state.currentSource = null
  }

  state.isPlaying = false
  state.isPaused = false
  notifyPlaybackChange(false)

  // Resume STT recording
  sttAudio.setTTSPlaying(false)
}

function pause() {
  state.isPaused = true

  if (state.currentSource) {
    state.currentSource.onended = null
    state.currentSource.stop()
    state.currentSource = null
  }

  // Keep isPlaying true so we know to resume
}

// ============================================
// UTILITY
// ============================================

function int16ToFloat32(int16Array) {
  const float32Array = new Float32Array(int16Array.length)
  for (let i = 0; i < int16Array.length; i++) {
    float32Array[i] = int16Array[i] / 32768.0
  }
  return float32Array
}

function setVolume(volume) {
  state.volume = Math.max(0, Math.min(1, volume))
  if (state.gainNode) {
    state.gainNode.gain.value = state.volume
  }
}
```

**TTS Playback Flow:**
```
[Receive PCM16] → [Convert to Float32] → [Create AudioBuffer]
      ↓
[Add to Queue] → [Check Buffer Threshold]
      ↓
[Play Queue] → [Notify STT to Pause]
      ↓
[Chunk Ends] → [Play Next] → [Queue Empty?]
      ↓                            ↓
[Continue]                   [Notify STT to Resume]
```

---

## 5. Integration Points

### 5.1 Initialization Sequence

```javascript
// main.js - Add to DOMContentLoaded

import * as websocket from './websocket.js'
import * as chat from './chat.js'
import * as sttAudio from './stt-audio.js'
import * as ttsAudio from './tts-audio.js'

async function initChatSystem() {
  // 1. Initialize TTS playback (can start before websocket)
  await ttsAudio.initTTSPlayback()

  // 2. Connect to WebSocket server
  await websocket.connect()

  // 3. Set up message handlers
  websocket.onMessage((message) => {
    chat.handleServerMessage(message)
  })

  websocket.onMessage((message, isBinary) => {
    if (isBinary) {
      ttsAudio.queueAudioChunk(message)
    }
  })

  // 4. Initialize chat UI
  chat.initChat()

  // 5. Initialize audio capture (requires user interaction)
  // Defer until mic button clicked
}
```

### 5.2 Editor Integration (editor.js modifications)

```javascript
// editor.js - Modify handleMic and handleSend

import * as websocket from './websocket.js'
import * as sttAudio from './stt-audio.js'
import * as chat from './chat.js'

let isMicActive = false
let isAudioInitialized = false

export async function handleMic() {
  // Initialize audio on first click (browser requirement)
  if (!isAudioInitialized) {
    const success = await sttAudio.initAudioCapture()
    if (!success) {
      showMicError('Microphone access denied')
      return
    }
    isAudioInitialized = true
  }

  if (isMicActive) {
    // Stop listening
    sttAudio.stopRecording()
    isMicActive = false
    updateMicButtonState(false)
  } else {
    // Start listening
    sttAudio.startRecording()
    isMicActive = true
    updateMicButtonState(true)
  }
}

export function handleSend() {
  const content = getEditorContent()
  if (!content || !content.trim()) return

  // Add to chat UI
  chat.addUserMessage(content)

  // Send to server
  websocket.sendText({
    type: 'user_message',
    data: { text: content }
  })

  // Clear editor
  clearEditorContent()
}
```

### 5.3 Model Settings Sync

```javascript
// main.js - Connect settings drawer to websocket

function syncModelSettings() {
  const settings = {
    model: localStorage.getItem('selectedModel') || 'meta-llama/llama-3.1-8b-instruct',
    temperature: parseFloat(localStorage.getItem('temperature') || '1.0'),
    top_p: parseFloat(localStorage.getItem('top-p') || '1.0'),
    min_p: parseFloat(localStorage.getItem('min-p') || '0.0'),
    top_k: parseInt(localStorage.getItem('top-k') || '0'),
    frequency_penalty: parseFloat(localStorage.getItem('frequency-penalty') || '0.0'),
    presence_penalty: parseFloat(localStorage.getItem('presence-penalty') || '0.0'),
    repetition_penalty: parseFloat(localStorage.getItem('repetition-penalty') || '1.0'),
  }

  websocket.sendText({
    type: 'model_settings',
    data: settings
  })
}

// Call on connection established and settings change
websocket.onConnectionChange((connected) => {
  if (connected) {
    syncModelSettings()
  }
})
```

---

## 6. CSS Additions (styles.css)

```css
/* ============================================
   CHAT MESSAGE STYLES
   ============================================ */

.messages-area {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding: 1rem;
}

.chat-message {
  display: flex;
  flex-direction: column;
  max-width: 85%;
  padding: 0.75rem 1rem;
  border-radius: 12px;
  animation: messageSlideIn 0.2s ease-out;
}

@keyframes messageSlideIn {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.chat-message.user {
  align-self: flex-end;
  background: #2563eb;
  color: white;
  border-bottom-right-radius: 4px;
}

.chat-message.assistant {
  align-self: flex-start;
  background: #2a2d35;
  color: var(--text);
  border-bottom-left-radius: 4px;
}

.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  font-size: 0.75rem;
}

.character-name {
  font-weight: 600;
  color: var(--primary);
}

.message-time {
  color: var(--muted);
}

.message-content {
  line-height: 1.5;
  word-wrap: break-word;
}

.message-content p {
  margin: 0;
}

.message-content p + p {
  margin-top: 0.5rem;
}

/* ============================================
   STT PREVIEW (Live Transcription)
   ============================================ */

#stt-preview {
  padding: 0.5rem 1rem;
  background: rgba(37, 99, 235, 0.1);
  border-left: 3px solid #2563eb;
  border-radius: 0 8px 8px 0;
  color: var(--muted);
  font-style: italic;
  animation: pulse 1.5s ease-in-out infinite;
}

#stt-preview.stabilized {
  color: var(--text);
  font-style: normal;
  animation: none;
}

@keyframes pulse {
  0%, 100% { opacity: 0.7; }
  50% { opacity: 1; }
}

/* ============================================
   MIC BUTTON STATES
   ============================================ */

.mic-button.active {
  background-color: #dc2626;
  animation: micPulse 1s ease-in-out infinite;
}

.mic-button.active svg {
  color: white;
}

@keyframes micPulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
  50% { box-shadow: 0 0 0 8px rgba(220, 38, 38, 0); }
}

.mic-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* ============================================
   TYPING INDICATOR
   ============================================ */

.typing-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  color: var(--muted);
  font-size: 0.875rem;
}

.typing-dots {
  display: flex;
  gap: 4px;
}

.typing-dots span {
  width: 6px;
  height: 6px;
  background: var(--muted);
  border-radius: 50%;
  animation: typingBounce 1.4s ease-in-out infinite;
}

.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes typingBounce {
  0%, 60%, 100% { transform: translateY(0); }
  30% { transform: translateY(-4px); }
}

/* ============================================
   CONNECTION STATUS
   ============================================ */

.connection-status {
  position: fixed;
  top: 1rem;
  right: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: var(--bg-darker);
  border: 1px solid var(--border);
  border-radius: 20px;
  font-size: 0.75rem;
  z-index: 1000;
  transition: all 0.3s ease;
}

.connection-status.connected { border-color: #22c55e; }
.connection-status.disconnected { border-color: #ef4444; }
.connection-status.connecting { border-color: #eab308; }

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-dot.connected { background: #22c55e; }
.status-dot.disconnected { background: #ef4444; }
.status-dot.connecting {
  background: #eab308;
  animation: pulse 1s ease-in-out infinite;
}
```

---

## 7. Questions for Clarification

Before proceeding with implementation, I'd like to clarify a few points:

### 7.1 Character Handling
1. **Multi-character responses:** The backend supports multi-character conversations. Should the chat UI display character avatars alongside messages?
2. **Character selection:** Should users be able to select which characters are active from the chat interface, or only from the Characters page?

### 7.2 Audio Behavior
1. **Interrupt behavior:** When the user starts speaking during TTS playback, should the TTS:
   - Stop immediately and process new input?
   - Finish current sentence then stop?
   - Continue playing while recording (not recommended)?

2. **Push-to-talk vs Always-on:** The current design assumes "always listening after mic click." Would you prefer:
   - Toggle mode (click to start, click to stop)
   - Push-to-hold mode (hold mic button to record)
   - Automatic (always listening when idle, pause during TTS)

### 7.3 Error Handling
1. **Reconnection notification:** Should users see a toast/banner when connection is lost and being restored?
2. **Audio permission denied:** How prominently should we display the microphone permission error?

### 7.4 Backend Ping/Pong
1. Does the backend currently support a `ping`/`pong` message type for connection health checks? If not, should we implement it, or use a different heartbeat mechanism?

### 7.5 Message Persistence
1. Should conversation history persist across page reloads (localStorage)?
2. Should we sync with the server's conversation history on reconnection?

---

## 8. Implementation Priority

### Phase 1: Core WebSocket (Day 1)
- [ ] `websocket.js` - Connection management with reconnection
- [ ] Basic connection status indicator

### Phase 2: Chat UI (Day 1-2)
- [ ] `chat.js` - Message display and streaming
- [ ] CSS for chat messages
- [ ] Text input integration (handleSend)

### Phase 3: Audio Input (Day 2)
- [ ] `stt-audio.js` - Microphone capture
- [ ] `stt-processor.js` - AudioWorklet
- [ ] STT preview display
- [ ] Mic button integration

### Phase 4: Audio Output (Day 2-3)
- [ ] `tts-audio.js` - TTS playback
- [ ] STT/TTS coordination (pause during playback)

### Phase 5: Polish (Day 3)
- [ ] Model settings sync
- [ ] Error handling
- [ ] Connection status UI
- [ ] Testing and debugging

---

## 9. Testing Checklist

- [ ] WebSocket connects successfully
- [ ] WebSocket reconnects after disconnect
- [ ] Text messages send and receive correctly
- [ ] STT audio streams to server
- [ ] Real-time transcription displays
- [ ] Final transcription becomes user message
- [ ] LLM response streams to chat
- [ ] TTS audio plays correctly
- [ ] Recording pauses during TTS
- [ ] Recording resumes after TTS
- [ ] Interrupt (speaking during TTS) works
- [ ] Model settings sync on change
- [ ] Characters load and display
- [ ] Error states handled gracefully

---

*Document Version: 1.0*
*Created: 2025-12-11*
*Author: Claude Code Analysis*
