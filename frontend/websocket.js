/**
 * WebSocket Connection Manager - Functional Pattern
 * Handles real-time communication with FastAPI backend
 * Focus: Connection lifecycle and message routing only
 */

// ============================================================================
// Constants
// ============================================================================

const WS_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'ws://localhost:8000'
  : `wss://${window.location.host}`;

const WS_ENDPOINT = '/ws';

const DEFAULT_CONFIG = {
  maxReconnectAttempts: 5,
  initialReconnectDelay: 1000,
  maxReconnectDelay: 30000,
  reconnectBackoffMultiplier: 2,
};

export const MESSAGE_TYPES = {
  // Outgoing (client → server)
  USER_MESSAGE: 'user_message',
  MODEL_SETTINGS: 'model_settings',
  START_LISTENING: 'start_listening',
  STOP_LISTENING: 'stop_listening',

  // Incoming (server → client)
  STT_UPDATE: 'stt_update',
  STT_STABILIZED: 'stt_stabilized',
  STT_FINAL: 'stt_final',
  TEXT_CHUNK: 'text_chunk',
  AUDIO: 'audio',
};

// ============================================================================
// State Management (Immutable)
// ============================================================================

/**
 * Create initial connection state
 */
const createConnectionState = () => ({
  ws: null,
  status: 'disconnected', // disconnected | connecting | connected | error
  reconnectAttempts: 0,
  reconnectDelay: DEFAULT_CONFIG.initialReconnectDelay,
  reconnectTimeoutId: null,
  messageHandlers: {},
  eventCallbacks: {
    onOpen: null,
    onClose: null,
    onError: null,
  },
});

/**
 * Update connection state immutably
 */
const updateState = (state, updates) => ({
  ...state,
  ...updates,
});

/**
 * Reset reconnection state
 */
const resetReconnectionState = (state) =>
  updateState(state, {
    reconnectAttempts: 0,
    reconnectDelay: DEFAULT_CONFIG.initialReconnectDelay,
    reconnectTimeoutId: null,
  });

/**
 * Increment reconnection attempt with exponential backoff
 */
const incrementReconnectionAttempt = (state) => {
  const nextDelay = Math.min(
    state.reconnectDelay * DEFAULT_CONFIG.reconnectBackoffMultiplier,
    DEFAULT_CONFIG.maxReconnectDelay
  );

  return updateState(state, {
    reconnectAttempts: state.reconnectAttempts + 1,
    reconnectDelay: nextDelay,
  });
};

// ============================================================================
// Message Handling
// ============================================================================

/**
 * Create a message router that dispatches to registered handlers
 */
const createMessageRouter = (handlers) => (event) => {
  // Binary message (audio)
  if (event.data instanceof Blob) {
    const audioHandler = handlers[MESSAGE_TYPES.AUDIO];
    if (audioHandler) {
      audioHandler(event.data);
    } else {
      console.warn('No handler registered for audio messages');
    }
    return;
  }

  // Text message (JSON)
  try {
    const message = JSON.parse(event.data);
    const messageType = message.type;

    const handler = handlers[messageType];
    if (handler) {
      handler(message.data || message);
    } else {
      console.warn('No handler registered for message type:', messageType);
    }
  } catch (error) {
    console.error('Error parsing WebSocket message:', error);
  }
};

/**
 * Register a message handler
 */
const registerHandler = (state, messageType, handler) =>
  updateState(state, {
    messageHandlers: {
      ...state.messageHandlers,
      [messageType]: handler,
    },
  });

/**
 * Unregister a message handler
 */
const unregisterHandler = (state, messageType) => {
  const { [messageType]: removed, ...remainingHandlers } = state.messageHandlers;
  return updateState(state, { messageHandlers: remainingHandlers });
};

/**
 * Register an event callback
 */
const registerEventCallback = (state, eventType, callback) =>
  updateState(state, {
    eventCallbacks: {
      ...state.eventCallbacks,
      [eventType]: callback,
    },
  });

// ============================================================================
// Message Construction
// ============================================================================

/**
 * Create a JSON message payload
 */
const createMessage = (type, data = {}) => JSON.stringify({ type, data });

/**
 * Create user message
 */
export const createUserMessage = (text) => createMessage(MESSAGE_TYPES.USER_MESSAGE, { text });

/**
 * Create model settings message
 */
export const createModelSettingsMessage = (settings) =>
  createMessage(MESSAGE_TYPES.MODEL_SETTINGS, settings);

/**
 * Create start listening message
 */
export const createStartListeningMessage = () =>
  createMessage(MESSAGE_TYPES.START_LISTENING);

/**
 * Create stop listening message
 */
export const createStopListeningMessage = () =>
  createMessage(MESSAGE_TYPES.STOP_LISTENING);

// ============================================================================
// WebSocket Operations
// ============================================================================

/**
 * Check if WebSocket is connected and ready
 */
const isConnected = (state) =>
  state.ws !== null && state.ws.readyState === WebSocket.OPEN && state.status === 'connected';

/**
 * Send a message through WebSocket
 */
const send = (state, message) => {
  if (!isConnected(state)) {
    console.error('Cannot send message: WebSocket not connected');
    return false;
  }

  try {
    state.ws.send(message);
    return true;
  } catch (error) {
    console.error('Error sending WebSocket message:', error);
    return false;
  }
};

/**
 * Send a text message
 */
export const sendMessage = (state, type, data = {}) => {
  const message = createMessage(type, data);
  return send(state, message);
};

/**
 * Send binary data (audio)
 */
export const sendBinary = (state, data) => send(state, data);

// ============================================================================
// Connection Lifecycle
// ============================================================================

/**
 * Build WebSocket URL
 */
const buildWebSocketUrl = () => `${WS_BASE_URL}${WS_ENDPOINT}`;

/**
 * Create WebSocket instance
 */
const createWebSocket = (url) => {
  try {
    return new WebSocket(url);
  } catch (error) {
    console.error('Failed to create WebSocket:', error);
    return null;
  }
};

/**
 * Setup WebSocket event handlers
 */
const setupWebSocketHandlers = (ws, state, connectionManager) => {
  const router = createMessageRouter(state.messageHandlers);

  ws.onopen = (event) => {
    console.log('WebSocket connected');
    connectionManager.state = updateState(state, { status: 'connected' });
    connectionManager.state = resetReconnectionState(connectionManager.state);

    if (state.eventCallbacks.onOpen) {
      state.eventCallbacks.onOpen(event);
    }
  };

  ws.onclose = (event) => {
    console.log('WebSocket closed:', event.code, event.reason);
    const wasConnected = state.status === 'connected';
    connectionManager.state = updateState(state, { status: 'disconnected', ws: null });

    if (state.eventCallbacks.onClose) {
      state.eventCallbacks.onClose(event);
    }

    // Attempt reconnection if not a normal closure
    if (wasConnected && event.code !== 1000) {
      scheduleReconnect(connectionManager);
    }
  };

  ws.onerror = (event) => {
    console.error('WebSocket error:', event);
    connectionManager.state = updateState(state, { status: 'error' });

    if (state.eventCallbacks.onError) {
      state.eventCallbacks.onError(event);
    }
  };

  ws.onmessage = router;
};

/**
 * Connect to WebSocket server
 */
const connect = (connectionManager) => {
  const { state } = connectionManager;

  if (state.status === 'connecting' || state.status === 'connected') {
    console.warn('WebSocket already connecting or connected');
    return;
  }

  const url = buildWebSocketUrl();
  console.log('Connecting to WebSocket:', url);

  connectionManager.state = updateState(state, { status: 'connecting' });

  const ws = createWebSocket(url);
  if (!ws) {
    connectionManager.state = updateState(state, { status: 'error' });
    return;
  }

  connectionManager.state = updateState(state, { ws });
  setupWebSocketHandlers(ws, connectionManager.state, connectionManager);
};

/**
 * Schedule reconnection with exponential backoff
 */
const scheduleReconnect = (connectionManager) => {
  const { state } = connectionManager;

  if (state.reconnectAttempts >= DEFAULT_CONFIG.maxReconnectAttempts) {
    console.error('Max reconnection attempts reached');
    connectionManager.state = updateState(state, { status: 'error' });
    return;
  }

  connectionManager.state = incrementReconnectionAttempt(state);
  const { reconnectDelay, reconnectAttempts } = connectionManager.state;

  console.log(
    `Reconnecting in ${reconnectDelay}ms... (attempt ${reconnectAttempts}/${DEFAULT_CONFIG.maxReconnectAttempts})`
  );

  const timeoutId = setTimeout(() => {
    connect(connectionManager);
  }, reconnectDelay);

  connectionManager.state = updateState(connectionManager.state, { reconnectTimeoutId: timeoutId });
};

/**
 * Disconnect from WebSocket server
 */
const disconnect = (connectionManager) => {
  const { state } = connectionManager;

  // Clear reconnection timeout if scheduled
  if (state.reconnectTimeoutId) {
    clearTimeout(state.reconnectTimeoutId);
  }

  if (state.ws) {
    state.ws.close(1000, 'Client disconnecting');
  }

  connectionManager.state = updateState(state, {
    ws: null,
    status: 'disconnected',
    reconnectTimeoutId: null,
  });

  console.log('WebSocket disconnected');
};

// ============================================================================
// Public API
// ============================================================================

/**
 * Create a WebSocket connection manager
 */
export const createWebSocketConnection = (config = {}) => {
  const connectionManager = {
    state: createConnectionState(),
    config: { ...DEFAULT_CONFIG, ...config },
  };

  return {
    // Connection control
    connect: () => connect(connectionManager),
    disconnect: () => disconnect(connectionManager),

    // State queries
    isConnected: () => isConnected(connectionManager.state),
    getStatus: () => connectionManager.state.status,

    // Message handlers
    on: (messageType, handler) => {
      connectionManager.state = registerHandler(connectionManager.state, messageType, handler);
    },
    off: (messageType) => {
      connectionManager.state = unregisterHandler(connectionManager.state, messageType);
    },

    // Event callbacks
    onOpen: (callback) => {
      connectionManager.state = registerEventCallback(connectionManager.state, 'onOpen', callback);
    },
    onClose: (callback) => {
      connectionManager.state = registerEventCallback(connectionManager.state, 'onClose', callback);
    },
    onError: (callback) => {
      connectionManager.state = registerEventCallback(connectionManager.state, 'onError', callback);
    },

    // Message sending
    sendMessage: (type, data) => sendMessage(connectionManager.state, type, data),
    sendBinary: (data) => sendBinary(connectionManager.state, data),

    // Convenience methods
    sendUserMessage: (text) => sendMessage(connectionManager.state, MESSAGE_TYPES.USER_MESSAGE, { text }),
    sendModelSettings: (settings) => sendMessage(connectionManager.state, MESSAGE_TYPES.MODEL_SETTINGS, settings),
    startListening: () => sendMessage(connectionManager.state, MESSAGE_TYPES.START_LISTENING),
    stopListening: () => sendMessage(connectionManager.state, MESSAGE_TYPES.STOP_LISTENING),

    // Internal state access (for debugging)
    _getState: () => connectionManager.state,
  };
};

// ============================================================================
// Default Export (Singleton Instance)
// ============================================================================

export const websocket = createWebSocketConnection();
