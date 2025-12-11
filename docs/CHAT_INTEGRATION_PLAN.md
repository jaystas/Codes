# Frontend Chat Interface Integration Plan

## Executive Summary

This document outlines the integration plan for connecting the frontend chat interface with the FastAPI WebSocket server. The goal is to create a real-time, bidirectional communication system supporting text chat, speech-to-text (STT), and text-to-speech (TTS) functionality.

---

## Part 1: Backend Architecture Analysis

### 1.1 FastAPI Server Overview (`backend/fastapi_server.py`)

The backend is a comprehensive real-time conversational AI system with three main services:

| Service | Class | Purpose |
|---------|-------|---------|
| **STT Service** | `STTService` | Speech-to-text using RealtimeSTT library |
| **LLM Service** | `LLMService` | Multi-character conversation management via OpenRouter |
| **TTS Service** | `TTSServiceManager` | Text-to-speech generation and streaming |

### 1.2 WebSocket Endpoint

**Endpoint:** `ws://localhost:8000/ws`

The WebSocket accepts two message types:
1. **Text Messages (JSON)** - Commands and text input
2. **Binary Messages (bytes)** - Raw PCM audio data for STT

### 1.3 Server-to-Client Message Types

```javascript
// Messages FROM the server
const SERVER_MESSAGES = {
  // STT Events
  "stt_update": { text: string },        // Real-time transcription update
  "stt_stabilized": { text: string },    // Stabilized transcription
  "stt_final": { text: string },         // Final transcription

  // LLM/Chat Events
  "text_chunk": {                         // Streaming text from character
    text: string,
    character_name: string,
    message_id: string,
    is_final: boolean
  },

  // Legacy (may be deprecated)
  "response_chunk": { text: string },
  "response_text": { text: string }
};
```

### 1.4 Client-to-Server Message Types

```javascript
// Messages TO the server
const CLIENT_MESSAGES = {
  // User Input
  "user_message": { text: string },       // Send text message

  // STT Control
  "start_listening": {},                  // Begin voice capture
  "stop_listening": {},                   // Stop voice capture

  // Model Configuration
  "model_settings": {                     // Update LLM settings
    model: string,
    temperature: number,
    top_p: number,
    min_p: number,
    top_k: number,
    frequency_penalty: number,
    presence_penalty: number,
    repetition_penalty: number
  },

  // Character Management
  "set_characters": {                     // Set active characters
    characters: Character[]
  },
  "refresh_active_characters": {},        // Reload from database

  // Conversation Control
  "clear_history": {},                    // Clear conversation
  "interrupt": {}                         // Stop current generation
};
```

### 1.5 Audio Specifications

| Direction | Format | Sample Rate | Bit Depth | Channels |
|-----------|--------|-------------|-----------|----------|
| **STT Input** | PCM | 16 kHz | 16-bit | Mono |
| **TTS Output** | PCM | 24 kHz | 16-bit | Mono |

### 1.6 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  stt-audio   │    │    chat      │    │  tts-audio   │                   │
│  │  (capture)   │    │   (display)  │    │  (playback)  │                   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                   │                            │
│         └───────────────────┼───────────────────┘                            │
│                             │                                                │
│                    ┌────────▼────────┐                                       │
│                    │   websocket.js   │                                       │
│                    │  (connection)    │                                       │
│                    └────────┬────────┘                                       │
│                             │                                                │
└─────────────────────────────┼────────────────────────────────────────────────┘
                              │ WebSocket (ws://localhost:8000/ws)
┌─────────────────────────────┼────────────────────────────────────────────────┐
│                             │           BACKEND                               │
├─────────────────────────────▼────────────────────────────────────────────────┤
│                    ┌──────────────────┐                                       │
│                    │ WebSocketManager │                                       │
│                    └────────┬─────────┘                                       │
│           ┌─────────────────┼─────────────────┐                               │
│           │                 │                 │                               │
│    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐                        │
│    │ STTService  │   │ LLMService  │   │ TTSService  │                        │
│    │             │   │             │   │   Manager   │                        │
│    └─────────────┘   └─────────────┘   └─────────────┘                        │
│                                                                               │
│    Queue System: transcribed_text → response_queue → tts_sentence_queue      │
│                                                  → audio_queue               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Frontend Architecture Analysis

### 2.1 Current File Structure

```
frontend/
├── index.html              # Main HTML (single-page app shell)
├── main.js                 # Navigation, drawers, model settings
├── editor.js               # Tiptap rich text editor
├── supabase.js             # Supabase client configuration
├── characters.js           # Character CRUD operations
├── characterCache.js       # Client-side cache with optimistic updates
├── realtimeSync.js         # Supabase real-time subscriptions
├── styles.css              # Main styles
├── editor.css              # Editor-specific styles
├── characters.css          # Character page styles
└── components/
    ├── components.html     # Reusable component templates
    ├── components.js       # Component logic
    └── components.css      # Component styles
```

### 2.2 Existing Patterns to Follow

1. **ES Modules** - All JS files use ES module syntax (`import`/`export`)
2. **CDN Dependencies** - Libraries loaded via `esm.sh` CDN
3. **Singleton Pattern** - Services exported as singleton instances (see `characterCache`, `realtimeSync`)
4. **Event Emitter Pattern** - Custom event system for decoupled components
5. **Async/Await** - Consistent use of modern async patterns
6. **CSS Variables** - Theming via CSS custom properties

### 2.3 Key UI Elements (from `main.js` home page template)

```javascript
// Chat interface elements
".messages-area"      // Container for chat messages
".editor-area"        // Text input area
"#editor"             // Tiptap editor instance
".mic-button"         // Microphone button (voice input)
".send-button"        // Send button

// Settings drawer
"#settings-drawer"    // Model settings panel
"#temperature-slider" // And other model setting controls
```

---

## Part 3: Implementation Plan

### 3.1 File: `websocket.js` - WebSocket Connection Manager

**Purpose:** Centralized WebSocket connection handling with auto-reconnect, message routing, and state management.

```javascript
// websocket.js - Proposed API

class WebSocketManager {
  constructor() {
    this.ws = null;
    this.url = 'ws://localhost:8000/ws';
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
    this.eventHandlers = new Map();
    this.messageQueue = [];  // Queue messages while disconnected
  }

  // Connection lifecycle
  connect(): Promise<void>
  disconnect(): void
  reconnect(): void

  // Message sending
  sendText(type: string, data: object): void
  sendAudio(audioData: ArrayBuffer): void

  // Event system
  on(event: string, handler: Function): void
  off(event: string, handler: Function): void
  emit(event: string, data: any): void

  // Convenience methods (wrapping sendText)
  sendUserMessage(text: string): void
  startListening(): void
  stopListening(): void
  updateModelSettings(settings: object): void
  setCharacters(characters: array): void
  refreshActiveCharacters(): void
  clearHistory(): void
  interrupt(): void

  // State
  isConnected(): boolean
  getConnectionState(): string  // 'connecting' | 'connected' | 'disconnected'
}

export const websocket = new WebSocketManager();
```

**Key Implementation Details:**

1. **Auto-reconnection** with exponential backoff
2. **Message queuing** when disconnected
3. **Binary/text message routing**
4. **Connection state events** for UI updates
5. **Heartbeat/ping** for connection health monitoring

**Events Emitted:**
- `connection:open` - WebSocket connected
- `connection:close` - WebSocket disconnected
- `connection:error` - Connection error
- `stt:update` - Real-time transcription
- `stt:stabilized` - Stabilized transcription
- `stt:final` - Final transcription
- `text:chunk` - Streaming text chunk
- `text:complete` - Message complete
- `audio:chunk` - TTS audio received

---

### 3.2 File: `chat.js` - Chat UI & Display Logic

**Purpose:** Manages chat message display, conversation state, and user input handling.

```javascript
// chat.js - Proposed API

class ChatManager {
  constructor() {
    this.messages = [];           // Array of Message objects
    this.currentMessageId = null; // For streaming messages
    this.isGenerating = false;    // Is LLM currently generating?
  }

  // Initialization
  init(): void                    // Called when home page loads
  destroy(): void                 // Cleanup when leaving page

  // Message management
  addUserMessage(content: string): Message
  addAssistantMessage(characterName: string, messageId: string): Message
  appendToMessage(messageId: string, text: string): void
  finalizeMessage(messageId: string): void

  // UI rendering
  renderMessages(): void
  renderMessage(message: Message): HTMLElement
  scrollToBottom(): void

  // User input
  handleSend(): void              // Send button clicked
  handleKeyboardShortcut(e: KeyboardEvent): void
  getEditorContent(): string
  clearEditor(): void

  // State
  setGenerating(isGenerating: boolean): void
  getConversation(): Message[]
  clearConversation(): void
}

// Message data structure
interface Message {
  id: string;
  role: 'user' | 'assistant';
  characterName?: string;
  content: string;
  timestamp: Date;
  isComplete: boolean;
}

export const chatManager = new ChatManager();
```

**UI Components to Create:**

1. **Message Bubble** - User vs Assistant styling
2. **Character Avatar** - Show character image/icon
3. **Typing Indicator** - While waiting for response
4. **Streaming Text** - Real-time text display with cursor effect
5. **Error State** - Display connection/generation errors

**Integration Points:**
- Imports `websocket.js` for communication
- Imports `editor.js` for getting user input
- Listens to websocket events for incoming messages
- Triggers UI updates on message events

---

### 3.3 File: `stt-audio.js` - Audio Capture Manager

**Purpose:** Handles microphone access, audio capture, and streaming to the WebSocket.

```javascript
// stt-audio.js - Proposed API

class STTAudioManager {
  constructor() {
    this.audioContext = null;
    this.mediaStream = null;
    this.workletNode = null;
    this.isRecording = false;
    this.hasPermission = false;
  }

  // Initialization
  async init(): Promise<boolean>   // Request mic permission, setup AudioContext
  destroy(): void                  // Cleanup resources

  // Recording control
  async startRecording(): Promise<void>
  stopRecording(): void
  toggleRecording(): void

  // Audio processing configuration
  setSampleRate(rate: number): void      // Default: 16000
  setBufferSize(size: number): void      // Default: 4096

  // State
  isRecording(): boolean
  hasPermission(): boolean

  // Events (via callback or EventEmitter)
  onAudioData(callback: (data: ArrayBuffer) => void): void
  onVoiceActivityStart(callback: () => void): void
  onVoiceActivityEnd(callback: () => void): void
  onError(callback: (error: Error) => void): void
}

export const sttAudio = new STTAudioManager();
```

**Technical Implementation:**

1. **MediaDevices API** - `navigator.mediaDevices.getUserMedia()`
2. **AudioWorklet** - Modern audio processing (uses `stt-processor.js`)
3. **Resampling** - Browser's native rate → 16kHz for STT
4. **Buffer Management** - Efficient audio chunk handling
5. **Voice Activity Detection (VAD)** - Optional client-side VAD

**Audio Pipeline:**
```
Microphone → MediaStream → AudioContext → AudioWorkletNode → Float32 →
→ Resample to 16kHz → Convert to PCM16 → websocket.sendAudio()
```

---

### 3.4 File: `stt-processor.js` - Audio Worklet Processor

**Purpose:** AudioWorklet processor running in a separate thread for low-latency audio processing.

```javascript
// stt-processor.js - AudioWorklet Processor

class STTProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.bufferSize = options.processorOptions?.bufferSize || 4096;
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
    this.targetSampleRate = 16000;  // STT expects 16kHz
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0];  // Mono channel

    // Accumulate samples
    for (let i = 0; i < channelData.length; i++) {
      this.buffer[this.bufferIndex++] = channelData[i];

      if (this.bufferIndex >= this.bufferSize) {
        // Send buffer to main thread
        this.port.postMessage({
          type: 'audio-data',
          data: this.buffer.slice()  // Copy the buffer
        });
        this.bufferIndex = 0;
      }
    }

    return true;  // Keep processor alive
  }
}

registerProcessor('stt-processor', STTProcessor);
```

**Registration in `stt-audio.js`:**
```javascript
await audioContext.audioWorklet.addModule('stt-processor.js');
const workletNode = new AudioWorkletNode(audioContext, 'stt-processor', {
  processorOptions: { bufferSize: 4096 }
});
workletNode.port.onmessage = (event) => {
  if (event.data.type === 'audio-data') {
    // Resample and send via WebSocket
  }
};
```

---

### 3.5 File: `tts-audio.js` - TTS Audio Playback Manager

**Purpose:** Handles TTS audio playback with queuing, streaming, and interruption support.

```javascript
// tts-audio.js - Proposed API

class TTSAudioManager {
  constructor() {
    this.audioContext = null;
    this.audioQueue = [];         // Queue of audio chunks
    this.isPlaying = false;
    this.currentSource = null;    // Currently playing AudioBufferSourceNode
    this.gainNode = null;         // Volume control
    this.nextPlayTime = 0;        // For gapless playback scheduling
  }

  // Initialization
  init(): void                    // Create AudioContext (must be after user gesture)
  destroy(): void                 // Cleanup resources

  // Audio queue management
  queueAudio(audioData: ArrayBuffer): void  // Add PCM16 24kHz data to queue
  processQueue(): void                       // Internal: process queued audio
  clearQueue(): void                         // Clear pending audio

  // Playback control
  play(): void                    // Start/resume playback
  pause(): void                   // Pause playback
  stop(): void                    // Stop and clear queue
  interrupt(): void               // Immediately stop for user input

  // Volume control
  setVolume(level: number): void  // 0.0 to 1.0
  getVolume(): number

  // State
  isPlaying(): boolean
  getQueueLength(): number

  // Events
  onPlaybackStart(callback: () => void): void
  onPlaybackEnd(callback: () => void): void
  onChunkPlayed(callback: (chunkId: string) => void): void
}

export const ttsAudio = new TTSAudioManager();
```

**Audio Pipeline:**
```
WebSocket Binary → PCM16 ArrayBuffer → Convert to Float32 →
→ AudioBuffer (24kHz) → AudioBufferSourceNode → GainNode → Destination
```

**Key Features:**

1. **Streaming Playback** - Play audio as it arrives, don't wait for complete message
2. **Gapless Scheduling** - Use `source.start(nextPlayTime)` for seamless transitions
3. **Queue Management** - Handle out-of-order chunks
4. **Interruption** - Immediately stop playback when user speaks
5. **Volume Control** - Via GainNode

**PCM16 to Float32 Conversion:**
```javascript
function pcm16ToFloat32(pcm16Buffer) {
  const int16Array = new Int16Array(pcm16Buffer);
  const float32Array = new Float32Array(int16Array.length);
  for (let i = 0; i < int16Array.length; i++) {
    float32Array[i] = int16Array[i] / 32768.0;
  }
  return float32Array;
}
```

---

## Part 4: Integration Points

### 4.1 Module Dependencies

```
websocket.js (standalone)
    ↑
    ├── chat.js (imports websocket)
    ├── stt-audio.js (imports websocket)
    │       ↑
    │       └── stt-processor.js (loaded via AudioWorklet)
    └── tts-audio.js (imports websocket)
```

### 4.2 Initialization Sequence

```javascript
// In main.js, when home page loads:

import { websocket } from './websocket.js';
import { chatManager } from './chat.js';
import { sttAudio } from './stt-audio.js';
import { ttsAudio } from './tts-audio.js';

async function initChatInterface() {
  // 1. Connect WebSocket
  await websocket.connect();

  // 2. Initialize chat UI
  chatManager.init();

  // 3. Request mic permission (deferred until user clicks mic button)
  // sttAudio.init() called on first mic click

  // 4. Initialize TTS (requires user gesture, defer to first audio)
  // ttsAudio.init() called on first audio received
}
```

### 4.3 Event Wiring

```javascript
// In chat.js
websocket.on('text:chunk', (data) => {
  chatManager.appendToMessage(data.message_id, data.text);
});

websocket.on('text:complete', (data) => {
  chatManager.finalizeMessage(data.message_id);
});

// In stt-audio.js
sttAudio.onAudioData((audioBuffer) => {
  websocket.sendAudio(audioBuffer);
});

websocket.on('stt:update', (data) => {
  // Show real-time transcription in UI
});

websocket.on('stt:final', (data) => {
  // User finished speaking, transcription complete
});

// In tts-audio.js
websocket.on('audio:chunk', (audioData) => {
  ttsAudio.queueAudio(audioData);
});
```

### 4.4 UI Button Handlers

```javascript
// Update handleMic() in editor.js or main.js
export async function handleMic() {
  if (!sttAudio.hasPermission) {
    await sttAudio.init();
  }

  if (sttAudio.isRecording) {
    sttAudio.stopRecording();
    websocket.stopListening();
    // Update mic button UI to inactive
  } else {
    sttAudio.startRecording();
    websocket.startListening();
    // Update mic button UI to active/recording
  }
}

// Update handleSend() in editor.js
export function handleSend() {
  const content = getEditorContent();
  if (!content.trim()) return;

  // Add to chat UI
  chatManager.addUserMessage(content);

  // Send via WebSocket
  websocket.sendUserMessage(content);

  // Clear editor
  clearEditorContent();

  // Interrupt any TTS playback
  ttsAudio.interrupt();
}
```

---

## Part 5: Additional Considerations

### 5.1 Error Handling

Each module should handle errors gracefully:

- **websocket.js**: Connection failures, reconnection logic
- **chat.js**: Display error states, retry options
- **stt-audio.js**: Mic permission denied, no audio input
- **tts-audio.js**: Audio context errors, playback failures

### 5.2 State Synchronization

Consider adding a central state manager or using the existing event system:

```javascript
// Example state events
const STATES = {
  IDLE: 'idle',
  LISTENING: 'listening',      // Mic active, STT processing
  PROCESSING: 'processing',    // LLM generating response
  SPEAKING: 'speaking',        // TTS playing audio
};
```

### 5.3 Model Settings Integration

The current `main.js` has model settings sliders. These need to be wired to send `model_settings` messages:

```javascript
// In main.js initSliders(), add:
slider.addEventListener('change', () => {
  const settings = gatherAllSettings();
  websocket.updateModelSettings(settings);
});
```

### 5.4 Character Selection Integration

The `characters.js` already has `handleChatWithCharacter()` which tries to use WebSocket. Update to use the new websocket module:

```javascript
// In characters.js
import { websocket } from './websocket.js';

// Replace websocket.isConnected() check with:
if (websocket.isConnected()) {
  websocket.refreshActiveCharacters();
}
```

### 5.5 Browser Compatibility

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| WebSocket | Yes | Yes | Yes | Yes |
| AudioWorklet | Yes | Yes | Yes (14.1+) | Yes |
| getUserMedia | Yes | Yes | Yes | Yes |
| AudioContext | Yes | Yes | Yes | Yes |

**Fallback:** For Safari < 14.1, use ScriptProcessorNode instead of AudioWorklet (deprecated but widely supported).

---

## Part 6: Recommended Implementation Order

1. **Phase 1: WebSocket Foundation**
   - Implement `websocket.js`
   - Test connection, reconnection, message sending

2. **Phase 2: Chat UI**
   - Implement `chat.js`
   - Wire to websocket events
   - Test text message flow

3. **Phase 3: TTS Playback**
   - Implement `tts-audio.js`
   - Test audio chunk playback
   - Verify gapless streaming

4. **Phase 4: STT Input**
   - Implement `stt-processor.js` (AudioWorklet)
   - Implement `stt-audio.js`
   - Test full voice input → transcription flow

5. **Phase 5: Integration & Polish**
   - Wire model settings
   - Add error handling
   - UI state synchronization
   - Testing all flows together

---

## Questions for Clarification

1. **Authentication**: Will there be user authentication? The backend currently has Supabase but doesn't seem to require auth for WebSocket connections.

2. **Multiple Characters**: The backend supports multi-character conversations. Should the UI show which character is speaking with avatars/names?

3. **Conversation Persistence**: Should conversations be saved to Supabase? The backend has tables defined but doesn't seem to use them yet.

4. **Voice Activity Detection (VAD)**: The backend has VAD via RealtimeSTT. Should the frontend also implement client-side VAD for the mic button UI (showing when voice is detected)?

5. **Rich Text**: The editor supports rich text (Tiptap). Should formatted content be sent to the LLM, or plain text only?

6. **Interruption Behavior**: When user starts speaking while TTS is playing, should it:
   - Immediately stop TTS and start listening?
   - Wait for a pause in TTS?
   - Require explicit mic button press?

7. **Offline Handling**: Should the app work offline in any capacity (queuing messages)?

8. **Mobile Support**: Are there mobile-specific considerations (touch events, mobile audio constraints)?

---

## Appendix A: Backend Message Reference

### Complete Server→Client Message Types

| Type | Data Structure | When Sent |
|------|----------------|-----------|
| `stt_update` | `{ text: string }` | Real-time transcription update |
| `stt_stabilized` | `{ text: string }` | Transcription stabilized |
| `stt_final` | `{ text: string }` | Final transcription |
| `text_chunk` | `{ text, character_name, message_id, is_final }` | LLM streaming text |
| `response_chunk` | `{ text: string }` | Legacy chunk format |
| `response_text` | `{ text: string }` | Legacy text format |
| Binary | `ArrayBuffer` (PCM16 @ 24kHz) | TTS audio chunk |

### Complete Client→Server Message Types

| Type | Data Structure | Purpose |
|------|----------------|---------|
| `user_message` | `{ text: string }` | Send text message |
| `start_listening` | `{}` | Begin STT |
| `stop_listening` | `{}` | Stop STT |
| `model_settings` | See Section 1.4 | Update LLM params |
| `set_characters` | `{ characters: Character[] }` | Set active characters |
| `refresh_active_characters` | `{}` | Reload from DB |
| `clear_history` | `{}` | Clear conversation |
| `interrupt` | `{}` | Stop generation |
| Binary | `ArrayBuffer` (PCM16 @ 16kHz) | Audio for STT |

---

## Appendix B: Data Type Definitions

```typescript
// Character (matches Supabase schema)
interface Character {
  id: string;
  name: string;
  voice: string;
  system_prompt: string;
  image_url: string;
  images: string[];
  is_active: boolean;
}

// Model Settings
interface ModelSettings {
  model: string;
  temperature: number;     // 0-2
  top_p: number;           // 0-1
  min_p: number;           // 0-1
  top_k: number;           // 0-100
  frequency_penalty: number; // -2 to 2
  presence_penalty: number;  // -2 to 2
  repetition_penalty: number; // 0-2
}

// Chat Message (frontend)
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  characterName?: string;
  characterId?: string;
  content: string;
  timestamp: Date;
  isStreaming: boolean;
  isComplete: boolean;
}

// Audio Chunk (from backend)
interface AudioChunk {
  chunk_id: string;
  message_id: string;
  character_id: string;
  character_name: string;
  audio_data: ArrayBuffer;  // PCM16 @ 24kHz
  chunk_index: number;
  is_final: boolean;
}
```

---

*Document created: 2025-12-11*
*For: aiChat Frontend-Backend Integration*
