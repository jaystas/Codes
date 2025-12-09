/**
 * Chat Manager - Functional Pattern
 * Handles message display, real-time updates, and chat UI
 * Focus: Message rendering and chat interaction
 */

import { websocket, MESSAGE_TYPES } from './websocket.js';
import { getEditor, clearEditorContent } from './editor.js';

// ============================================================================
// Constants
// ============================================================================

const MESSAGE_STATUS = {
  TRANSCRIBING: 'transcribing',    // Real-time STT updates
  STREAMING: 'streaming',          // Assistant streaming response
  COMPLETE: 'complete',            // Final message
};

const MESSAGE_ROLE = {
  USER: 'user',
  ASSISTANT: 'assistant',
  SYSTEM: 'system',
};

const MIC_STATE = {
  IDLE: 'idle',           // Ready to start
  LISTENING: 'listening', // Green pulse - waiting for voice
  RECORDING: 'recording', // Red pulse - actively recording
};

// ============================================================================
// State Management (Immutable)
// ============================================================================

/**
 * Create initial chat state
 */
const createChatState = () => ({
  messages: [],
  currentTranscription: null,  // Temporary STT message being updated
  currentAssistantMessage: null, // Temporary assistant message being streamed
  micState: MIC_STATE.IDLE,
  isTyping: false,
});

/**
 * Update chat state immutably
 */
const updateChatState = (state, updates) => ({
  ...state,
  ...updates,
});

/**
 * Add message to state
 */
const addMessage = (state, message) => ({
  ...state,
  messages: [...state.messages, message],
});

/**
 * Update last message in state
 */
const updateLastMessage = (state, updates) => {
  if (state.messages.length === 0) return state;

  const messages = [...state.messages];
  messages[messages.length - 1] = { ...messages[messages.length - 1], ...updates };

  return { ...state, messages };
};

/**
 * Clear all messages
 */
const clearMessages = (state) => ({
  ...state,
  messages: [],
  currentTranscription: null,
  currentAssistantMessage: null,
});

/**
 * Set mic state
 */
const setMicState = (state, micState) => ({
  ...state,
  micState,
});

// ============================================================================
// Message Construction
// ============================================================================

/**
 * Create a message object
 */
const createMessage = ({
  id = generateMessageId(),
  role,
  content = '',
  characterName = null,
  characterAvatar = null,
  status = MESSAGE_STATUS.COMPLETE,
  timestamp = Date.now(),
}) => ({
  id,
  role,
  content,
  characterName,
  characterAvatar,
  status,
  timestamp,
});

/**
 * Generate unique message ID
 */
const generateMessageId = () => `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

// ============================================================================
// DOM Elements
// ============================================================================

/**
 * Get chat DOM elements
 */
const getChatElements = () => ({
  messagesArea: document.querySelector('.messages-area'),
  micButton: document.querySelector('.mic-button'),
  sendButton: document.querySelector('.send-button'),
  editorFooter: document.querySelector('.editor-footer'),
});

/**
 * Create new chat button element
 */
const createNewChatButton = () => {
  const button = document.createElement('button');
  button.className = 'new-chat-button';
  button.innerHTML = `
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <line x1="12" y1="5" x2="12" y2="19"></line>
      <line x1="5" y1="12" x2="19" y2="12"></line>
    </svg>
  `;
  button.title = 'New Chat';
  return button;
};

/**
 * Create typing indicator element
 */
const createTypingIndicator = () => {
  const indicator = document.createElement('div');
  indicator.className = 'typing-indicator';
  indicator.innerHTML = `
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
  `;
  return indicator;
};

// ============================================================================
// Message Rendering
// ============================================================================

/**
 * Create message bubble HTML
 */
const createMessageBubble = (message) => {
  const isUser = message.role === MESSAGE_ROLE.USER;
  const isTranscribing = message.status === MESSAGE_STATUS.TRANSCRIBING;
  const isStreaming = message.status === MESSAGE_STATUS.STREAMING;

  const bubbleClasses = [
    'message-bubble',
    isUser ? 'message-user' : 'message-assistant',
    isTranscribing ? 'message-transcribing' : '',
    isStreaming ? 'message-streaming' : '',
  ].filter(Boolean).join(' ');

  let content = '';

  // Assistant messages with character info
  if (!isUser && message.characterName) {
    content += `
      <div class="message-header">
        ${message.characterAvatar
          ? `<img src="${message.characterAvatar}" alt="${message.characterName}" class="character-avatar" />`
          : `<div class="character-avatar-placeholder">${message.characterName[0]}</div>`
        }
        <span class="character-name">${message.characterName}</span>
      </div>
    `;
  }

  content += `<div class="message-content">${escapeHtml(message.content)}</div>`;

  return `<div class="${bubbleClasses}" data-message-id="${message.id}">${content}</div>`;
};

/**
 * Render a message to the DOM
 */
const renderMessage = (messagesArea, message) => {
  const messageHtml = createMessageBubble(message);
  messagesArea.insertAdjacentHTML('beforeend', messageHtml);
  scrollToBottom(messagesArea);
};

/**
 * Update existing message in DOM
 */
const updateMessageInDom = (message) => {
  const messageElement = document.querySelector(`[data-message-id="${message.id}"]`);
  if (!messageElement) return;

  const contentElement = messageElement.querySelector('.message-content');
  if (contentElement) {
    contentElement.textContent = message.content;
  }

  // Update status classes
  messageElement.classList.remove('message-transcribing', 'message-streaming');
  if (message.status === MESSAGE_STATUS.TRANSCRIBING) {
    messageElement.classList.add('message-transcribing');
  } else if (message.status === MESSAGE_STATUS.STREAMING) {
    messageElement.classList.add('message-streaming');
  }
};

/**
 * Remove message from DOM
 */
const removeMessageFromDom = (messageId) => {
  const messageElement = document.querySelector(`[data-message-id="${messageId}"]`);
  if (messageElement) {
    messageElement.remove();
  }
};

/**
 * Clear all messages from DOM
 */
const clearMessagesFromDom = (messagesArea) => {
  messagesArea.innerHTML = '';
};

/**
 * Show typing indicator
 */
const showTypingIndicator = (messagesArea) => {
  // Remove existing indicator if present
  hideTypingIndicator();

  const indicator = createTypingIndicator();
  messagesArea.appendChild(indicator);
  scrollToBottom(messagesArea);
};

/**
 * Hide typing indicator
 */
const hideTypingIndicator = () => {
  const indicator = document.querySelector('.typing-indicator');
  if (indicator) {
    indicator.remove();
  }
};

/**
 * Scroll messages area to bottom
 */
const scrollToBottom = (messagesArea) => {
  requestAnimationFrame(() => {
    messagesArea.scrollTop = messagesArea.scrollHeight;
  });
};

/**
 * Escape HTML for safe rendering
 */
const escapeHtml = (text) => {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
};

// ============================================================================
// Mic Button State Management
// ============================================================================

/**
 * Update mic button visual state
 */
const updateMicButtonState = (micButton, state) => {
  micButton.classList.remove('mic-idle', 'mic-listening', 'mic-recording');

  switch (state) {
    case MIC_STATE.IDLE:
      micButton.classList.add('mic-idle');
      break;
    case MIC_STATE.LISTENING:
      micButton.classList.add('mic-listening');
      break;
    case MIC_STATE.RECORDING:
      micButton.classList.add('mic-recording');
      break;
  }
};

// ============================================================================
// Message Handlers
// ============================================================================

/**
 * Handle STT update (real-time transcription)
 */
const handleSttUpdate = (chatManager, data) => {
  const { messagesArea } = getChatElements();
  const text = data.text || '';

  // If no current transcription message, create one
  if (!chatManager.state.currentTranscription) {
    const message = createMessage({
      role: MESSAGE_ROLE.USER,
      content: text,
      status: MESSAGE_STATUS.TRANSCRIBING,
    });

    chatManager.state = addMessage(chatManager.state, message);
    chatManager.state = updateChatState(chatManager.state, { currentTranscription: message });

    renderMessage(messagesArea, message);
  } else {
    // Update existing transcription message
    const updatedMessage = {
      ...chatManager.state.currentTranscription,
      content: text,
    };

    chatManager.state = updateChatState(chatManager.state, { currentTranscription: updatedMessage });
    updateMessageInDom(updatedMessage);
  }
};

/**
 * Handle STT stabilized (more confident transcription)
 */
const handleSttStabilized = (chatManager, data) => {
  handleSttUpdate(chatManager, data); // Same as update for now
};

/**
 * Handle STT final (complete transcription)
 */
const handleSttFinal = (chatManager, data) => {
  const text = data.text || '';

  if (chatManager.state.currentTranscription) {
    // Convert transcription to final user message
    const finalMessage = {
      ...chatManager.state.currentTranscription,
      content: text,
      status: MESSAGE_STATUS.COMPLETE,
    };

    // Update in state
    const messageIndex = chatManager.state.messages.findIndex(
      m => m.id === chatManager.state.currentTranscription.id
    );

    if (messageIndex !== -1) {
      const messages = [...chatManager.state.messages];
      messages[messageIndex] = finalMessage;
      chatManager.state = updateChatState(chatManager.state, {
        messages,
        currentTranscription: null,
      });
    }

    updateMessageInDom(finalMessage);
  } else {
    // Create new user message if no transcription was tracked
    const message = createMessage({
      role: MESSAGE_ROLE.USER,
      content: text,
      status: MESSAGE_STATUS.COMPLETE,
    });

    chatManager.state = addMessage(chatManager.state, message);
    renderMessage(getChatElements().messagesArea, message);
  }

  // Show typing indicator for assistant response
  showTypingIndicator(getChatElements().messagesArea);
};

/**
 * Handle text chunk (assistant streaming response)
 */
const handleTextChunk = (chatManager, data) => {
  const { messagesArea } = getChatElements();
  const text = data.text || '';
  const characterName = data.character_name || data.characterName || 'Assistant';
  const characterAvatar = data.character_avatar || data.characterAvatar || null;
  const isFinal = data.is_final || false;

  hideTypingIndicator();

  // If no current assistant message, create one
  if (!chatManager.state.currentAssistantMessage) {
    const message = createMessage({
      role: MESSAGE_ROLE.ASSISTANT,
      content: text,
      characterName,
      characterAvatar,
      status: MESSAGE_STATUS.STREAMING,
    });

    chatManager.state = addMessage(chatManager.state, message);
    chatManager.state = updateChatState(chatManager.state, { currentAssistantMessage: message });

    renderMessage(messagesArea, message);
  } else {
    // Append to existing streaming message
    const updatedMessage = {
      ...chatManager.state.currentAssistantMessage,
      content: chatManager.state.currentAssistantMessage.content + text,
      status: isFinal ? MESSAGE_STATUS.COMPLETE : MESSAGE_STATUS.STREAMING,
    };

    chatManager.state = updateChatState(chatManager.state, { currentAssistantMessage: updatedMessage });
    updateMessageInDom(updatedMessage);

    // Clear current assistant message if final
    if (isFinal) {
      chatManager.state = updateChatState(chatManager.state, { currentAssistantMessage: null });
    }
  }

  scrollToBottom(messagesArea);
};

// ============================================================================
// Event Handlers
// ============================================================================

/**
 * Handle send button click
 */
const handleSendMessage = (chatManager) => {
  const editor = getEditor();
  if (!editor) return;

  const content = editor.getText().trim();
  if (!content) return;

  // Create and display user message
  const message = createMessage({
    role: MESSAGE_ROLE.USER,
    content,
    status: MESSAGE_STATUS.COMPLETE,
  });

  chatManager.state = addMessage(chatManager.state, message);
  renderMessage(getChatElements().messagesArea, message);

  // Send to backend
  websocket.sendUserMessage(content);

  // Clear editor
  clearEditorContent();

  // Show typing indicator
  showTypingIndicator(getChatElements().messagesArea);
};

/**
 * Handle new chat button click
 */
const handleNewChat = (chatManager) => {
  const { messagesArea } = getChatElements();

  chatManager.state = clearMessages(chatManager.state);
  clearMessagesFromDom(messagesArea);
  hideTypingIndicator();

  console.log('New chat started');
};

/**
 * Handle mic button click
 */
const handleMicClick = (chatManager) => {
  const { micButton } = getChatElements();

  if (chatManager.state.micState === MIC_STATE.IDLE) {
    // Start listening
    chatManager.state = setMicState(chatManager.state, MIC_STATE.LISTENING);
    updateMicButtonState(micButton, MIC_STATE.LISTENING);
    websocket.startListening();
  } else {
    // Stop listening/recording
    chatManager.state = setMicState(chatManager.state, MIC_STATE.IDLE);
    updateMicButtonState(micButton, MIC_STATE.IDLE);
    websocket.stopListening();
  }
};

// ============================================================================
// Initialization
// ============================================================================

/**
 * Setup WebSocket message handlers
 */
const setupWebSocketHandlers = (chatManager) => {
  websocket.on(MESSAGE_TYPES.STT_UPDATE, (data) => handleSttUpdate(chatManager, data));
  websocket.on(MESSAGE_TYPES.STT_STABILIZED, (data) => handleSttStabilized(chatManager, data));
  websocket.on(MESSAGE_TYPES.STT_FINAL, (data) => handleSttFinal(chatManager, data));
  websocket.on(MESSAGE_TYPES.TEXT_CHUNK, (data) => handleTextChunk(chatManager, data));
};

/**
 * Setup UI event listeners
 */
const setupEventListeners = (chatManager) => {
  const { micButton, sendButton, editorFooter } = getChatElements();

  // Mic button
  if (micButton) {
    micButton.addEventListener('click', () => handleMicClick(chatManager));
  }

  // Send button
  if (sendButton) {
    sendButton.addEventListener('click', () => handleSendMessage(chatManager));
  }

  // New chat button
  if (editorFooter) {
    const newChatButton = createNewChatButton();
    editorFooter.insertBefore(newChatButton, editorFooter.firstChild);

    newChatButton.addEventListener('click', () => handleNewChat(chatManager));
  }

  // Enter key to send (Ctrl+Enter or Cmd+Enter)
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      handleSendMessage(chatManager);
    }
  });
};

/**
 * Initialize mic button state
 */
const initializeMicButton = () => {
  const { micButton } = getChatElements();
  if (micButton) {
    updateMicButtonState(micButton, MIC_STATE.IDLE);
  }
};

// ============================================================================
// Public API
// ============================================================================

/**
 * Create a chat manager instance
 */
export const createChatManager = () => {
  const chatManager = {
    state: createChatState(),
  };

  return {
    // State access
    getState: () => chatManager.state,

    // Message operations
    sendMessage: () => handleSendMessage(chatManager),
    newChat: () => handleNewChat(chatManager),

    // Mic control
    toggleMic: () => handleMicClick(chatManager),
    setMicState: (state) => {
      chatManager.state = setMicState(chatManager.state, state);
      updateMicButtonState(getChatElements().micButton, state);
    },

    // Initialize
    initialize: () => {
      setupWebSocketHandlers(chatManager);
      setupEventListeners(chatManager);
      initializeMicButton();
      console.log('Chat manager initialized');
    },

    // Internal reference for handlers
    _chatManager: chatManager,
  };
};

// ============================================================================
// Default Export (Singleton)
// ============================================================================

export const chat = createChatManager();
