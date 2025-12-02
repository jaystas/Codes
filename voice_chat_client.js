/**
 * Browser-side Voice Chat Client
 * 
 * Captures microphone audio, converts to PCM16 @ 16kHz,
 * and sends via WebSocket to the FastAPI backend.
 * 
 * Also handles TTS interrupt signals from the server.
 */

class VoiceChatClient {
    constructor(websocketUrl) {
        this.websocketUrl = websocketUrl;
        this.ws = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.audioWorklet = null;
        this.isListening = false;
        this.isRecording = false;
        
        // TTS playback state
        this.ttsAudioQueue = [];
        this.isPlayingTTS = false;
        this.currentTTSSource = null;
        
        // Callbacks
        this.onTranscriptionUpdate = null;
        this.onTranscriptionFinal = null;
        this.onVADStart = null;
        this.onVADStop = null;
        this.onRecordingStart = null;
        this.onRecordingStop = null;
        this.onTTSInterrupt = null;
        this.onConnectionChange = null;
    }
    
    /**
     * Connect to WebSocket and initialize audio
     */
    async connect() {
        try {
            // Initialize WebSocket
            this.ws = new WebSocket(this.websocketUrl);
            this.ws.binaryType = 'arraybuffer';
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.onConnectionChange?.('connected');
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.onConnectionChange?.('disconnected');
                this.cleanup();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.onConnectionChange?.('error');
            };
            
            this.ws.onmessage = (event) => this.handleMessage(event);
            
            // Initialize audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000  // Target sample rate
            });
            
            // Get microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            console.log('Audio initialized');
            return true;
            
        } catch (error) {
            console.error('Failed to connect:', error);
            return false;
        }
    }
    
    /**
     * Handle incoming WebSocket messages
     */
    handleMessage(event) {
        if (typeof event.data === 'string') {
            const data = JSON.parse(event.data);
            
            switch (data.type) {
                case 'stt_update':
                    this.onTranscriptionUpdate?.(data.text, false);
                    break;
                    
                case 'stt_stabilized':
                    this.onTranscriptionUpdate?.(data.text, true);
                    break;
                    
                case 'stt_final':
                    this.onTranscriptionFinal?.(data.text);
                    break;
                    
                case 'vad_start':
                    this.onVADStart?.();
                    break;
                    
                case 'vad_stop':
                    this.onVADStop?.();
                    break;
                    
                case 'recording_start':
                    this.isRecording = true;
                    this.onRecordingStart?.();
                    break;
                    
                case 'recording_stop':
                    this.isRecording = false;
                    this.onRecordingStop?.();
                    break;
                    
                case 'tts_interrupt':
                    this.handleTTSInterrupt();
                    break;
                    
                case 'listening_started':
                    this.isListening = true;
                    console.log('Server listening for voice');
                    break;
                    
                case 'listening_stopped':
                    this.isListening = false;
                    console.log('Server stopped listening');
                    break;
                    
                case 'pong':
                    // Heartbeat response
                    break;
                    
                default:
                    console.log('Unknown message type:', data.type);
            }
        } else if (event.data instanceof ArrayBuffer) {
            // Binary data - TTS audio
            this.handleTTSAudio(event.data);
        }
    }
    
    /**
     * Start capturing and sending audio
     */
    async startListening() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected');
            return;
        }
        
        // Tell server to start listening
        this.sendControl({ type: 'start_listening' });
        
        // Start audio capture
        await this.startAudioCapture();
    }
    
    /**
     * Stop capturing audio
     */
    stopListening() {
        this.stopAudioCapture();
        this.sendControl({ type: 'stop_listening' });
    }
    
    /**
     * Start capturing microphone audio
     */
    async startAudioCapture() {
        if (!this.audioContext || !this.mediaStream) {
            console.error('Audio not initialized');
            return;
        }
        
        // Resume audio context if suspended
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
        
        // Create source from microphone
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        
        // Use ScriptProcessor for audio processing
        // (AudioWorklet is preferred but ScriptProcessor is simpler for demo)
        const bufferSize = 4096;
        const processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
        
        processor.onaudioprocess = (event) => {
            if (!this.isListening) return;
            
            const inputData = event.inputBuffer.getChannelData(0);
            
            // Convert float32 [-1, 1] to int16
            const pcm16 = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                const s = Math.max(-1, Math.min(1, inputData[i]));
                pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            
            // Send as binary
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(pcm16.buffer);
            }
        };
        
        source.connect(processor);
        processor.connect(this.audioContext.destination);
        
        this.audioProcessor = processor;
        this.audioSource = source;
        this.isListening = true;
        
        console.log('Audio capture started');
    }
    
    /**
     * Stop audio capture
     */
    stopAudioCapture() {
        this.isListening = false;
        
        if (this.audioProcessor) {
            this.audioProcessor.disconnect();
            this.audioProcessor = null;
        }
        
        if (this.audioSource) {
            this.audioSource.disconnect();
            this.audioSource = null;
        }
        
        console.log('Audio capture stopped');
    }
    
    /**
     * Handle incoming TTS audio
     */
    handleTTSAudio(audioData) {
        this.ttsAudioQueue.push(audioData);
        
        if (!this.isPlayingTTS) {
            this.playNextTTSChunk();
        }
    }
    
    /**
     * Play next TTS audio chunk
     */
    async playNextTTSChunk() {
        if (this.ttsAudioQueue.length === 0) {
            this.isPlayingTTS = false;
            this.sendControl({ type: 'tts_ended' });
            return;
        }
        
        this.isPlayingTTS = true;
        this.sendControl({ type: 'tts_started' });
        
        const audioData = this.ttsAudioQueue.shift();
        
        try {
            // Decode audio data
            const audioBuffer = await this.audioContext.decodeAudioData(audioData.slice(0));
            
            // Create source and play
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            
            source.onended = () => {
                this.currentTTSSource = null;
                this.playNextTTSChunk();
            };
            
            this.currentTTSSource = source;
            source.start(0);
            
        } catch (error) {
            console.error('Error playing TTS:', error);
            this.playNextTTSChunk();
        }
    }
    
    /**
     * Handle TTS interrupt signal from server
     */
    handleTTSInterrupt() {
        console.log('TTS interrupt received');
        
        // Stop current playback
        if (this.currentTTSSource) {
            try {
                this.currentTTSSource.stop();
            } catch (e) {
                // May already be stopped
            }
            this.currentTTSSource = null;
        }
        
        // Clear queue
        this.ttsAudioQueue = [];
        this.isPlayingTTS = false;
        
        // Notify callback
        this.onTTSInterrupt?.();
        
        // Notify server
        this.sendControl({ type: 'tts_ended' });
    }
    
    /**
     * Send control message to server
     */
    sendControl(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }
    
    /**
     * Request final transcription
     */
    requestTranscription() {
        this.sendControl({ type: 'get_transcription' });
    }
    
    /**
     * Cleanup resources
     */
    cleanup() {
        this.stopAudioCapture();
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }
    
    /**
     * Disconnect and cleanup
     */
    disconnect() {
        this.cleanup();
        
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}


// =============================================================================
// Usage Example
// =============================================================================

/*
// Initialize client
const client = new VoiceChatClient('ws://localhost:8000/ws/voice');

// Set up callbacks
client.onTranscriptionUpdate = (text, isStabilized) => {
    console.log(`Transcription ${isStabilized ? '(stabilized)' : '(updating)'}: ${text}`);
    document.getElementById('transcription').textContent = text;
};

client.onTranscriptionFinal = (text) => {
    console.log('Final transcription:', text);
    document.getElementById('final-transcription').textContent = text;
};

client.onVADStart = () => {
    console.log('Voice detected');
    document.getElementById('status').textContent = 'Voice detected...';
};

client.onRecordingStart = () => {
    document.getElementById('status').textContent = 'Recording...';
};

client.onRecordingStop = () => {
    document.getElementById('status').textContent = 'Processing...';
};

client.onTTSInterrupt = () => {
    console.log('TTS was interrupted by user');
};

client.onConnectionChange = (status) => {
    document.getElementById('connection').textContent = status;
};

// Connect and start
async function start() {
    const connected = await client.connect();
    if (connected) {
        await client.startListening();
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    client.disconnect();
});

// Start on button click (required for audio context)
document.getElementById('start-btn').addEventListener('click', start);
*/


// =============================================================================
// HTML Example
// =============================================================================

/*
<!DOCTYPE html>
<html>
<head>
    <title>Voice Chat</title>
    <style>
        body { font-family: sans-serif; padding: 20px; }
        .status { margin: 10px 0; padding: 10px; background: #f0f0f0; }
        #transcription { 
            min-height: 100px; 
            border: 1px solid #ccc; 
            padding: 10px; 
            margin: 10px 0;
            color: #666;
        }
        #final-transcription {
            min-height: 50px;
            border: 1px solid #4CAF50;
            padding: 10px;
            margin: 10px 0;
            background: #e8f5e9;
        }
        button { padding: 10px 20px; font-size: 16px; margin: 5px; }
    </style>
</head>
<body>
    <h1>Voice Chat Demo</h1>
    
    <div class="status">
        Connection: <span id="connection">disconnected</span>
    </div>
    
    <div class="status">
        Status: <span id="status">Ready</span>
    </div>
    
    <button id="start-btn">Start Voice Chat</button>
    <button id="stop-btn">Stop</button>
    
    <h3>Live Transcription:</h3>
    <div id="transcription"></div>
    
    <h3>Final Transcription:</h3>
    <div id="final-transcription"></div>
    
    <script src="voice_chat_client.js"></script>
    <script>
        const client = new VoiceChatClient('ws://localhost:8000/ws/voice');
        
        client.onTranscriptionUpdate = (text, isStabilized) => {
            document.getElementById('transcription').textContent = text;
            document.getElementById('transcription').style.color = isStabilized ? '#333' : '#666';
        };
        
        client.onTranscriptionFinal = (text) => {
            document.getElementById('final-transcription').textContent = text;
            document.getElementById('transcription').textContent = '';
        };
        
        client.onVADStart = () => {
            document.getElementById('status').textContent = 'Voice detected...';
        };
        
        client.onRecordingStart = () => {
            document.getElementById('status').textContent = 'Recording...';
        };
        
        client.onRecordingStop = () => {
            document.getElementById('status').textContent = 'Processing...';
        };
        
        client.onConnectionChange = (status) => {
            document.getElementById('connection').textContent = status;
        };
        
        document.getElementById('start-btn').addEventListener('click', async () => {
            const connected = await client.connect();
            if (connected) {
                await client.startListening();
                document.getElementById('status').textContent = 'Listening...';
            }
        });
        
        document.getElementById('stop-btn').addEventListener('click', () => {
            client.stopListening();
            document.getElementById('status').textContent = 'Stopped';
        });
        
        window.addEventListener('beforeunload', () => client.disconnect());
    </script>
</body>
</html>
*/
