# Architecture & Integration Plan
## aiChat - Low-Latency Character System with Real-Time Chat Pipeline

**Created:** 2025-12-09
**Status:** Planning Phase
**Goal:** Integrate direct Supabase access with WebSocket-based chat pipeline for optimal performance

---

## Table of Contents
1. [Current Architecture](#current-architecture)
2. [Data Flow Overview](#data-flow-overview)
3. [Frontend Enhancements](#frontend-enhancements)
4. [Backend Integration](#backend-integration)
5. [Implementation Phases](#implementation-phases)
6. [Testing Strategy](#testing-strategy)

---

## Current Architecture

### What We Have Now

**Frontend:**
- ✅ Direct Supabase connection for character/voice CRUD operations (fast!)
- ✅ WebSocket manager ready for real-time chat
- ✅ Rich text editor for user input
- ✅ Character management UI (create, edit, delete, search)

**Backend:**
- ✅ FastAPI server with WebSocket endpoint (`/ws`)
- ✅ STT Service (RealtimeSTT)
- ✅ LLM Service (OpenRouter)
- ✅ TTS Service (stub - needs implementation)
- ✅ Supabase client initialized
- ❌ No character management REST endpoints (by design)

**Database (Supabase):**
- ✅ `characters` table (id, name, voice, system_prompt, image_url, images, is_active)
- ✅ `voices` table (voice, method, speaker_desc, scene_prompt, audio_path, text_path)

---

## Data Flow Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         FRONTEND                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌─────────────────┐         │
│  │  Character UI    │         │   Chat UI       │         │
│  │  (CRUD)          │         │   (Conversation)│         │
│  └────────┬─────────┘         └────────┬────────┘         │
│           │                            │                   │
│           │ Direct DB Access           │ Real-time         │
│           ▼                            ▼                   │
│  ┌──────────────────┐         ┌─────────────────┐         │
│  │  supabase.js     │         │  websocket.js   │         │
│  └────────┬─────────┘         └────────┬────────┘         │
│           │                            │                   │
└───────────┼────────────────────────────┼───────────────────┘
            │                            │
            │ HTTPS                      │ WSS
            ▼                            ▼
┌───────────────────────┐    ┌───────────────────────────────┐
│   SUPABASE            │    │   FASTAPI BACKEND             │
│   (Database)          │    │   (Chat Pipeline)             │
│                       │    │                               │
│  • characters         │◄───┤  Read character/voice data    │
│  • voices             │    │  for chat context             │
│  • chats (future)     │    │                               │
│  • messages (future)  │    │  ┌─────────────────────────┐ │
│                       │    │  │  Chat Pipeline:         │ │
│  Real-time subs ──────┼───►│  │  1. STT (audio→text)    │ │
│  (character updates)  │    │  │  2. LLM (text→response) │ │
└───────────────────────┘    │  │  3. TTS (text→audio)    │ │
                             │  └─────────────────────────┘ │
                             └───────────────────────────────┘
```

### Data Flow by Operation

#### 1. Character Management (Frontend → Supabase)
```
User creates/edits character
    ↓
Frontend UI (characters.js)
    ↓
Supabase client (supabase.js)
    ↓
Supabase Database
    ↓
Real-time subscription (optional)
    ↓
Frontend updates automatically
```

#### 2. Chat Session Start (Frontend → Backend)
```
User selects character(s) to chat with
    ↓
Frontend sends via WebSocket:
  {
    type: "start_chat",
    data: {
      character_ids: ["uuid-1", "uuid-2"],
      model_settings: { ... }
    }
  }
    ↓
Backend receives message
    ↓
Backend loads character data from Supabase:
  - Character name, system_prompt
  - Voice settings (for TTS)
    ↓
Backend initializes LLM context
Backend caches character data in memory
```

#### 3. User Message Flow (Full Pipeline)
```
USER INPUT (Voice or Text)
    ↓
┌─────────────────────────────────────┐
│ FRONTEND                            │
├─────────────────────────────────────┤
│ • Capture audio (mic) OR            │
│ • Get text (editor)                 │
│     ↓                               │
│ WebSocket.sendAudio(audioData)      │
│   OR                                │
│ WebSocket.sendUserMessage(text)     │
└──────────────┬──────────────────────┘
               │ WebSocket
               ▼
┌─────────────────────────────────────┐
│ BACKEND (FastAPI)                   │
├─────────────────────────────────────┤
│ 1. STT Service                      │
│    • Audio → Text transcription     │
│    • Send real-time updates to UI   │
│      {type: "stt_update", text}     │
│                                     │
│ 2. LLM Service                      │
│    • Get active character(s)        │
│    • Build prompt with:             │
│      - Character system_prompt      │
│      - Conversation history         │
│      - User message                 │
│    • Stream LLM response            │
│    • Send text chunks to UI         │
│      {type: "text_chunk", data}     │
│                                     │
│ 3. TTS Service                      │
│    • Extract sentences from stream  │
│    • Get character's voice settings │
│    • Generate audio                 │
│    • Stream audio chunks to UI      │
│      {type: "audio", blob}          │
└──────────────┬──────────────────────┘
               │ WebSocket
               ▼
┌─────────────────────────────────────┐
│ FRONTEND (Chat UI)                  │
├─────────────────────────────────────┤
│ • Display STT updates (typing...)   │
│ • Display LLM text (character msg)  │
│ • Play audio (character voice)      │
└─────────────────────────────────────┘
```

---

## Frontend Enhancements

### 1. Character Caching System

**Goal:** Load characters once, keep in memory, sync with database

**Implementation:**
```javascript
// frontend/characterCache.js

class CharacterCache {
  constructor() {
    this.characters = new Map();  // id → character object
    this.voices = new Map();      // voice → voice object
    this.lastSync = null;
    this.isInitialized = false;
  }

  // Initial load from Supabase
  async initialize() {
    const [chars, voices] = await Promise.all([
      supabase.from('characters').select('*'),
      supabase.from('voices').select('*')
    ]);

    chars.data.forEach(c => this.characters.set(c.id, c));
    voices.data.forEach(v => this.voices.set(v.voice, v));

    this.isInitialized = true;
    this.lastSync = Date.now();
  }

  // Get from cache (instant)
  getCharacter(id) {
    return this.characters.get(id);
  }

  getAllCharacters() {
    return Array.from(this.characters.values());
  }

  // Update cache + database
  async upsertCharacter(character) {
    // Optimistic update (UI feels instant)
    this.characters.set(character.id, character);

    // Sync to database in background
    try {
      const { data } = await supabase
        .from('characters')
        .upsert(character)
        .select()
        .single();

      // Update cache with server response (has timestamps, etc.)
      this.characters.set(data.id, data);
      return data;
    } catch (error) {
      // Rollback on error
      this.characters.delete(character.id);
      throw error;
    }
  }
}

export const characterCache = new CharacterCache();
```

**Benefits:**
- ⚡ Instant access (no DB roundtrip)
- 🔄 Automatic sync in background
- 🎯 Optimistic UI updates (feels instant)

### 2. Real-Time Subscriptions

**Goal:** Characters update instantly across all tabs/sessions

**Implementation:**
```javascript
// frontend/realtimeSync.js

import { supabase } from './supabase.js';
import { characterCache } from './characterCache.js';

class RealtimeSync {
  constructor() {
    this.subscriptions = [];
    this.eventHandlers = new Map();
  }

  // Subscribe to character changes
  subscribeToCharacters() {
    const subscription = supabase
      .channel('characters-channel')
      .on(
        'postgres_changes',
        { event: '*', schema: 'public', table: 'characters' },
        (payload) => this.handleCharacterChange(payload)
      )
      .subscribe();

    this.subscriptions.push(subscription);
  }

  handleCharacterChange(payload) {
    const { eventType, new: newRecord, old: oldRecord } = payload;

    switch (eventType) {
      case 'INSERT':
        characterCache.characters.set(newRecord.id, newRecord);
        this.emit('character:created', newRecord);
        break;

      case 'UPDATE':
        characterCache.characters.set(newRecord.id, newRecord);
        this.emit('character:updated', newRecord);
        break;

      case 'DELETE':
        characterCache.characters.delete(oldRecord.id);
        this.emit('character:deleted', oldRecord);
        break;
    }
  }

  // Event system for UI updates
  on(eventName, handler) {
    if (!this.eventHandlers.has(eventName)) {
      this.eventHandlers.set(eventName, []);
    }
    this.eventHandlers.get(eventName).push(handler);
  }

  emit(eventName, data) {
    const handlers = this.eventHandlers.get(eventName) || [];
    handlers.forEach(handler => handler(data));
  }

  cleanup() {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }
}

export const realtimeSync = new RealtimeSync();
```

**Usage in characters.js:**
```javascript
// Subscribe to updates
realtimeSync.on('character:updated', (character) => {
  // Re-render character in list
  updateCharacterInUI(character);
});

realtimeSync.on('character:created', (character) => {
  // Add to UI
  addCharacterToUI(character);
});
```

**Benefits:**
- ⚡ Instant updates across tabs
- 🔄 No manual refresh needed
- 🎯 Only re-render what changed

### 3. Optimistic UI Updates

**Goal:** UI updates immediately, syncs in background

**Implementation:**
```javascript
// Example: Saving a character

async function saveCharacter(characterData) {
  const tempId = characterData.id || `temp-${Date.now()}`;
  const optimisticCharacter = { ...characterData, id: tempId };

  // 1. Update UI immediately (optimistic)
  addCharacterToUI(optimisticCharacter);
  showNotification('Saving...', '', 'info');

  try {
    // 2. Save to database in background
    const savedCharacter = await characterCache.upsertCharacter(characterData);

    // 3. Update UI with real data (has server-generated fields)
    replaceCharacterInUI(tempId, savedCharacter);
    showNotification('Saved!', '', 'success');

  } catch (error) {
    // 4. Rollback on error
    removeCharacterFromUI(tempId);
    showNotification('Save failed', error.message, 'error');
  }
}
```

**Benefits:**
- ⚡ Zero perceived latency
- 🔄 Automatic rollback on errors
- 🎯 Better user experience

---

## Backend Integration

### 1. Character Context Loading

**Backend needs to:**
- Receive character IDs from frontend via WebSocket
- Load character + voice data from Supabase
- Cache in memory for the chat session
- Use in LLM prompts and TTS generation

**Implementation:**

#### A. WebSocket Message Handling (backend/fastapi_server.py)

```python
# New message types to handle

class ChatSessionStart(BaseModel):
    character_ids: List[str]
    model_settings: ModelSettings

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type")

                # New handler: Start chat session
                if msg_type == "start_chat":
                    await ws_manager.start_chat_session(data["data"])

                # Existing handlers
                elif msg_type == "user_message":
                    await ws_manager.handle_user_message(data["data"]["text"])

                # ... other handlers
```

#### B. Session Management (WebSocketManager)

```python
class WebSocketManager:
    def __init__(self):
        # ... existing init
        self.active_characters: List[Character] = []
        self.character_voices: Dict[str, Voice] = {}

    async def start_chat_session(self, session_data: dict):
        """Load characters from database and initialize chat session"""
        character_ids = session_data["character_ids"]
        model_settings = session_data["model_settings"]

        # Load characters from Supabase
        self.active_characters = await self.load_characters(character_ids)

        # Load voice settings for each character
        for char in self.active_characters:
            if char.voice:
                voice = await self.load_voice(char.voice)
                self.character_voices[char.id] = voice

        # Initialize LLM with character context
        await self.llm_service.initialize_with_characters(
            self.active_characters,
            model_settings
        )

        # Send confirmation to client
        await self.send_text_to_client({
            "type": "chat_ready",
            "characters": [c.name for c in self.active_characters]
        })

    async def load_characters(self, character_ids: List[str]) -> List[Character]:
        """Load characters from Supabase"""
        try:
            response = supabase.table("characters").select("*").in_("id", character_ids).execute()

            characters = []
            for row in response.data:
                char = Character(
                    id=row["id"],
                    name=row["name"],
                    voice=row["voice"],
                    system_prompt=row["system_prompt"],
                    image_url=row["image_url"],
                    images=row.get("images", []),
                    is_active=row["is_active"]
                )
                characters.append(char)

            return characters
        except Exception as e:
            logger.error(f"Error loading characters: {e}")
            return []

    async def load_voice(self, voice_name: str) -> Optional[Voice]:
        """Load voice settings from Supabase"""
        try:
            response = supabase.table("voices").select("*").eq("voice", voice_name).single().execute()

            return Voice(
                voice=response.data["voice"],
                method=response.data["method"],
                speaker_desc=response.data["speaker_desc"],
                scene_prompt=response.data["scene_prompt"],
                audio_path=response.data.get("audio_path", ""),
                text_path=response.data.get("text_path", "")
            )
        except Exception as e:
            logger.error(f"Error loading voice: {e}")
            return None
```

#### C. LLM Integration (LLMService)

```python
class LLMService:
    async def initialize_with_characters(
        self,
        characters: List[Character],
        model_settings: ModelSettings
    ):
        """Initialize LLM with character context"""
        self.active_characters = characters
        self.model_settings = model_settings

        # Clear conversation history for new session
        self.conversation_history = []

        logger.info(f"LLM initialized with {len(characters)} character(s)")

    def build_prompt(self, user_message: str, character: Character) -> List[dict]:
        """Build LLM prompt with character context"""
        messages = []

        # Add character's system prompt
        messages.append({
            "role": "system",
            "name": character.name,
            "content": character.system_prompt
        })

        # Add conversation history
        messages.extend(self.conversation_history)

        # Add user message
        messages.append({
            "role": "user",
            "name": "Jay",
            "content": user_message
        })

        # Add instruction for character response
        messages.append(self.create_character_instruction_message(character))

        return messages
```

#### D. TTS Integration (TTSService)

```python
class TTSService:
    async def generate_speech(
        self,
        text: str,
        character: Character,
        voice: Voice
    ) -> bytes:
        """Generate TTS audio using character's voice settings"""

        if voice.method == "clone":
            # Use voice cloning with audio file
            audio_data = await self.higgs_engine.generate_with_clone(
                text=text,
                audio_path=voice.audio_path,
                text_path=voice.text_path
            )
        else:  # description method
            # Use speaker description
            audio_data = await self.higgs_engine.generate_with_description(
                text=text,
                speaker_desc=voice.speaker_desc,
                scene_prompt=voice.scene_prompt
            )

        return audio_data
```

### 2. Frontend → Backend Chat Flow

**Step-by-step integration:**

#### Frontend (Chat UI)

```javascript
// frontend/chat.js (new file)

import { websocket, MESSAGE_TYPES } from './websocket.js';
import { characterCache } from './characterCache.js';

class ChatManager {
  constructor() {
    this.activeCharacters = [];
    this.isSessionActive = false;
  }

  async startChat(characterIds) {
    // Load characters from cache
    this.activeCharacters = characterIds.map(id =>
      characterCache.getCharacter(id)
    );

    // Get model settings from UI
    const modelSettings = this.getModelSettings();

    // Connect WebSocket if not connected
    if (!websocket.isConnected) {
      websocket.connect();
      await this.waitForConnection();
    }

    // Send start_chat message
    websocket.sendMessage('start_chat', {
      character_ids: characterIds,
      model_settings: modelSettings
    });

    // Set up message handlers
    this.setupMessageHandlers();

    this.isSessionActive = true;
  }

  setupMessageHandlers() {
    // STT updates (real-time transcription)
    websocket.on(MESSAGE_TYPES.STT_UPDATE, (data) => {
      this.displayTranscription(data.text, 'updating');
    });

    // Final transcription
    websocket.on(MESSAGE_TYPES.STT_FINAL, (data) => {
      this.displayTranscription(data.text, 'final');
    });

    // LLM text chunks
    websocket.on(MESSAGE_TYPES.TEXT_CHUNK, (data) => {
      this.appendCharacterMessage(data.character_name, data.text);
    });

    // TTS audio
    websocket.on(MESSAGE_TYPES.AUDIO, (audioBlob) => {
      this.playAudio(audioBlob);
    });
  }

  sendUserMessage(text) {
    websocket.sendUserMessage(text);
    this.displayUserMessage(text);
  }

  // ... UI update methods
}

export const chatManager = new ChatManager();
```

---

## Implementation Phases

### Phase 1: Frontend Enhancements (Week 1)
**Goal:** Add caching, real-time sync, optimistic updates

**Tasks:**
1. ✅ Create `frontend/characterCache.js`
   - Implement CharacterCache class
   - Initial load + caching
   - Optimistic updates

2. ✅ Create `frontend/realtimeSync.js`
   - Set up Supabase subscriptions
   - Event handlers for character changes
   - Integration with cache

3. ✅ Update `frontend/characters.js`
   - Use characterCache instead of direct Supabase calls
   - Subscribe to real-time events
   - Implement optimistic UI updates

4. ✅ Test caching & sync
   - Verify instant access
   - Test real-time updates across tabs
   - Test optimistic updates + rollback

**Success Criteria:**
- Characters load instantly after first fetch
- Changes sync across tabs in < 1 second
- UI updates feel instant (no perceived delay)

### Phase 2: Chat UI Integration (Week 2)
**Goal:** Create chat interface connected to WebSocket

**Tasks:**
1. ✅ Create `frontend/chat.js`
   - ChatManager class
   - Character selection flow
   - WebSocket message handling

2. ✅ Update home page UI (`main.js`)
   - Add character selector
   - Add chat message area
   - Add audio playback UI

3. ✅ Test WebSocket connection
   - Verify connection/reconnection
   - Test message sending/receiving
   - Test error handling

**Success Criteria:**
- Can select characters for chat
- WebSocket connects and maintains connection
- UI updates with WebSocket messages

### Phase 3: Backend Character Loading (Week 2-3)
**Goal:** Backend loads and uses character data

**Tasks:**
1. ✅ Update `backend/fastapi_server.py`
   - Add start_chat message handler
   - Implement load_characters() method
   - Implement load_voice() method

2. ✅ Update LLMService
   - Add initialize_with_characters()
   - Update build_prompt() to use character context
   - Test with character system prompts

3. ✅ Test character integration
   - Verify characters load from DB
   - Verify LLM uses character prompts
   - Test with multiple characters

**Success Criteria:**
- Backend successfully loads characters from Supabase
- LLM responses use character system prompts
- Can chat with multiple characters

### Phase 4: Full Pipeline Integration (Week 3-4)
**Goal:** Complete STT → LLM → TTS pipeline

**Tasks:**
1. ✅ Implement TTS with character voices
   - Connect TTSService to voice settings
   - Generate audio with correct voice
   - Stream audio to frontend

2. ✅ Test full pipeline
   - User speaks → STT → display text
   - LLM generates response → display in UI
   - TTS generates audio → play in browser

3. ✅ Polish & optimize
   - Add loading states
   - Error handling
   - Performance optimization

**Success Criteria:**
- Complete conversation flow works end-to-end
- Character voices are used correctly
- Audio plays smoothly in browser

### Phase 5: Persistence & History (Week 4-5)
**Goal:** Save conversations to database

**Tasks:**
1. ✅ Create database tables
   - `chats` table (session info)
   - `messages` table (conversation history)

2. ✅ Backend: Save messages
   - Save user messages
   - Save character responses
   - Link to chat session

3. ✅ Frontend: Display history
   - Load past conversations
   - Display in UI
   - Search/filter

**Success Criteria:**
- Conversations persist across sessions
- Can view past conversations
- Can continue previous conversations

---

## Testing Strategy

### Unit Tests
- **Frontend:** Character cache, real-time sync, optimistic updates
- **Backend:** Character loading, prompt building, voice selection

### Integration Tests
- **WebSocket:** Connection, reconnection, message flow
- **Database:** CRUD operations, real-time subscriptions
- **Pipeline:** STT → LLM → TTS full flow

### Performance Tests
- **Latency:** Measure DB query times (target < 100ms)
- **Real-time:** Measure sync delay (target < 1s)
- **Memory:** Monitor cache size with many characters

### User Acceptance Tests
- Create character → Save → Verify in DB
- Start chat → Send message → Receive response
- Character update → Verify UI updates automatically
- Close/reopen app → Verify characters still loaded

---

## Key Design Decisions

### Why Direct Supabase for Characters?
✅ **Pros:**
- Sub-100ms latency (no backend hop)
- Simpler architecture (fewer moving parts)
- Built-in real-time subscriptions
- Automatic caching via PostgREST

❌ **Cons:**
- Frontend has database credentials (mitigated: anon key + RLS)
- Less control over queries (mitigated: not needed for simple CRUD)

**Decision:** Use direct Supabase for CRUD, backend reads for chat context

### Why WebSocket for Chat?
✅ **Pros:**
- True bi-directional streaming (STT updates, LLM chunks, TTS audio)
- Single persistent connection (efficient)
- Built-in connection management

❌ **Cons:**
- More complex than HTTP (mitigated: websocket.js abstracts complexity)
- Need reconnection logic (mitigated: auto-reconnect implemented)

**Decision:** WebSocket is essential for real-time chat pipeline

### Why Cache Characters in Frontend?
✅ **Pros:**
- Zero latency for character list/display
- Works offline (after first load)
- Reduces database load

❌ **Cons:**
- Memory usage (mitigated: characters are small objects)
- Stale data risk (mitigated: real-time sync)

**Decision:** Cache with real-time sync gives best UX

---

## Next Steps

1. **Review this plan** - Ensure architecture makes sense
2. **Prioritize phases** - Decide what to build first
3. **Set up testing** - Prepare test environment
4. **Start Phase 1** - Begin with frontend enhancements

**Questions to Address:**
- Do we need conversation history now or later? (Suggested: Phase 5)
- Should we support multi-character conversations immediately? (Suggested: Yes)
- What error states need UI? (Connection lost, character not found, etc.)
- Do we need offline support? (Suggested: Basic caching only)

---

## Appendix: Message Types Reference

### Frontend → Backend (Outgoing)
| Type | Payload | Purpose |
|------|---------|---------|
| `start_chat` | `{character_ids, model_settings}` | Initialize chat session |
| `user_message` | `{text}` | Send text message |
| `audio` | Binary audio data | Send audio for STT |
| `model_settings` | `{model, temperature, ...}` | Update LLM settings |
| `start_listening` | - | Start STT |
| `stop_listening` | - | Stop STT |

### Backend → Frontend (Incoming)
| Type | Payload | Purpose |
|------|---------|---------|
| `chat_ready` | `{characters}` | Chat session initialized |
| `stt_update` | `{text}` | Real-time transcription update |
| `stt_stabilized` | `{text}` | Stabilized transcription |
| `stt_final` | `{text}` | Final transcription |
| `text_chunk` | `{character_name, text, chunk_index}` | LLM response chunk |
| `audio` | Binary audio data | TTS audio chunk |

---

**End of Architecture Plan**
