/**
 * WebSocket Connection Manager
 * Handles real-time communication with FastAPI backend for chat pipeline
 * (STT → LLM → TTS streaming)
 */

// WebSocket URL configuration
const WS_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'ws://localhost:8000'
  : `wss://${window.location.host}`;

const WS_ENDPOINT = '/ws';

/**
 * WebSocket Manager Class
 */
export class WebSocketManager {
  constructor() {
    this.ws = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000; // Start with 1 second

    // Message handlers
    this.messageHandlers = new Map();

    // Event callbacks
    this.onOpen = null;
    this.onClose = null;
    this.onError = null;
  }

  /**
   * Connect to WebSocket server
   */
  connect() {
    const url = `${WS_BASE_URL}${WS_ENDPOINT}`;
    console.log('Connecting to WebSocket:', url);

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = (event) => {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;

        if (this.onOpen) {
          this.onOpen(event);
        }
      };

      this.ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        this.isConnected = false;

        if (this.onClose) {
          this.onClose(event);
        }

        // Attempt reconnection if not a normal closure
        if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.attemptReconnect();
        }
      };

      this.ws.onerror = (event) => {
        console.error('WebSocket error:', event);

        if (this.onError) {
          this.onError(event);
        }
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event);
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
    }
  }

  /**
   * Attempt to reconnect with exponential backoff
   */
  attemptReconnect() {
    this.reconnectAttempts++;
    console.log(`Reconnecting... (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, this.reconnectDelay);

    // Exponential backoff
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000); // Max 30 seconds
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect() {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnecting');
      this.ws = null;
      this.isConnected = false;
    }
  }

  /**
   * Handle incoming messages
   */
  handleMessage(event) {
    // Check if it's binary data (audio)
    if (event.data instanceof Blob) {
      this.handleBinaryMessage(event.data);
      return;
    }

    // Otherwise it's text (JSON)
    try {
      const message = JSON.parse(event.data);
      const messageType = message.type;

      // Call registered handler for this message type
      if (this.messageHandlers.has(messageType)) {
        const handler = this.messageHandlers.get(messageType);
        handler(message.data || message);
      } else {
        console.warn('No handler registered for message type:', messageType);
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }

  /**
   * Handle binary messages (audio data)
   */
  handleBinaryMessage(blob) {
    if (this.messageHandlers.has('audio')) {
      const handler = this.messageHandlers.get('audio');
      handler(blob);
    }
  }

  /**
   * Register a message handler for a specific message type
   * @param {string} messageType - Type of message to handle
   * @param {Function} handler - Callback function
   */
  on(messageType, handler) {
    this.messageHandlers.set(messageType, handler);
  }

  /**
   * Unregister a message handler
   * @param {string} messageType - Type of message
   */
  off(messageType) {
    this.messageHandlers.delete(messageType);
  }

  /**
   * Send a text message to the server
   * @param {string} type - Message type
   * @param {object} data - Message payload
   */
  sendMessage(type, data = {}) {
    if (!this.isConnected || !this.ws) {
      console.error('Cannot send message: WebSocket not connected');
      return;
    }

    const message = JSON.stringify({ type, data });
    this.ws.send(message);
  }

  /**
   * Send audio data to the server
   * @param {ArrayBuffer|Blob} audioData - Audio data to send
   */
  sendAudio(audioData) {
    if (!this.isConnected || !this.ws) {
      console.error('Cannot send audio: WebSocket not connected');
      return;
    }

    this.ws.send(audioData);
  }

  /**
   * Send user message
   * @param {string} text - User message text
   */
  sendUserMessage(text) {
    this.sendMessage('user_message', { text });
  }

  /**
   * Send model settings
   * @param {object} settings - Model configuration
   */
  sendModelSettings(settings) {
    this.sendMessage('model_settings', settings);
  }

  /**
   * Start listening (STT)
   */
  startListening() {
    this.sendMessage('start_listening');
  }

  /**
   * Stop listening (STT)
   */
  stopListening() {
    this.sendMessage('stop_listening');
  }
}

// Export singleton instance
export const websocket = new WebSocketManager();

// Export message type constants for convenience
export const MESSAGE_TYPES = {
  // Outgoing (client → server)
  USER_MESSAGE: 'user_message',
  MODEL_SETTINGS: 'model_settings',
  START_LISTENING: 'start_listening',
  STOP_LISTENING: 'stop_listening',

  // Incoming (server → client)
  STT_UPDATE: 'stt_update',           // Real-time STT updates
  STT_STABILIZED: 'stt_stabilized',   // Stabilized STT text
  STT_FINAL: 'stt_final',             // Final transcription
  TEXT_CHUNK: 'text_chunk',           // LLM text chunk
  AUDIO: 'audio',                     // TTS audio data (binary)
};
