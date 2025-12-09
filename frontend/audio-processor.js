/**
 * Audio Worklet Processor for Microphone Capture
 * Handles real-time audio processing in a separate thread
 */

class MicrophoneProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Buffer for accumulating samples before resampling
    this.buffer = [];

    // Target sample rate (16kHz for STT)
    this.targetSampleRate = 16000;

    // Chunk size (320 samples @ 16kHz = 20ms)
    this.chunkSize = 320;

    // Listen for control messages
    this.port.onmessage = (event) => {
      if (event.data.type === 'setSampleRate') {
        this.targetSampleRate = event.data.sampleRate;
      } else if (event.data.type === 'setChunkSize') {
        this.chunkSize = event.data.chunkSize;
      }
    };
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];

    if (!input || !input[0]) {
      return true; // Keep processor alive
    }

    // Get mono audio (first channel)
    const samples = input[0];

    // Accumulate samples
    for (let i = 0; i < samples.length; i++) {
      this.buffer.push(samples[i]);
    }

    // Calculate expected buffer size for resampling
    // We need enough samples to produce at least one chunk at target rate
    const inputRate = sampleRate; // AudioWorklet global
    const resampleRatio = inputRate / this.targetSampleRate;
    const requiredInputSamples = Math.ceil(this.chunkSize * resampleRatio);

    // Process buffer when we have enough samples
    while (this.buffer.length >= requiredInputSamples) {
      // Extract samples for one chunk
      const inputSamples = this.buffer.splice(0, requiredInputSamples);

      // Resample to target rate
      const resampled = this.resample(inputSamples, inputRate, this.targetSampleRate);

      // Send resampled chunk to main thread
      this.port.postMessage({
        type: 'audioData',
        samples: resampled,
      });
    }

    return true; // Keep processor alive
  }

  /**
   * Simple linear resampling
   * More efficient than high-quality resampling for real-time use
   */
  resample(inputSamples, inputRate, outputRate) {
    const ratio = inputRate / outputRate;
    const outputLength = Math.floor(inputSamples.length / ratio);
    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const srcIndex = i * ratio;
      const srcIndexFloor = Math.floor(srcIndex);
      const srcIndexCeil = Math.min(srcIndexFloor + 1, inputSamples.length - 1);
      const fraction = srcIndex - srcIndexFloor;

      // Linear interpolation
      output[i] = inputSamples[srcIndexFloor] * (1 - fraction) +
                  inputSamples[srcIndexCeil] * fraction;
    }

    return output;
  }
}

registerProcessor('microphone-processor', MicrophoneProcessor);
