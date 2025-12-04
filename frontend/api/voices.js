/**
 * Voices API Client
 * Handles all voice-related API operations
 */

import { get, post, put, del, APIError, handleAPIError, logAPIError } from './config.js';

/**
 * Get all voices
 * @returns {Promise<Array>} Array of voice objects
 */
export async function getAllVoices() {
  try {
    const voices = await get('/api/voices');
    return voices;
  } catch (error) {
    logAPIError('getAllVoices', error);
    throw error;
  }
}

/**
 * Get a specific voice by name
 * @param {string} voiceName - Voice name/identifier
 * @returns {Promise<Object>} Voice object
 */
export async function getVoice(voiceName) {
  try {
    const voice = await get(`/api/voices/${voiceName}`);
    return voice;
  } catch (error) {
    logAPIError('getVoice', error);
    throw error;
  }
}

/**
 * Create a new voice
 * @param {Object} voiceData - Voice configuration data
 * @param {string} voiceData.voice - Voice name/identifier (required, serves as PK)
 * @param {string} voiceData.method - Voice method: "description" or "clone"
 * @param {string} voiceData.speaker_desc - Speaker description (for description method)
 * @param {string} voiceData.scene_prompt - Scene prompt (for description method)
 * @param {string} voiceData.audio_path - Audio file path (for clone method)
 * @param {string} voiceData.text_path - Text file path (for clone method)
 * @returns {Promise<Object>} Created voice object
 */
export async function createVoice(voiceData) {
  try {
    // Map frontend fields to backend schema
    const backendData = {
      voice: voiceData.voice || voiceData.name, // Voice name serves as PK
      method: voiceData.method || 'description',
      speaker_desc: voiceData.speakerDescription || voiceData.speaker_desc || '',
      scene_prompt: voiceData.scenePrompt || voiceData.scene_prompt || '',
      audio_path: voiceData.audioPath || voiceData.audio_path || '',
      text_path: voiceData.textPath || voiceData.text_path || '',
    };

    const voice = await post('/api/voices', backendData);
    return voice;
  } catch (error) {
    logAPIError('createVoice', error);
    throw error;
  }
}

/**
 * Update an existing voice
 * @param {string} voiceName - Voice name/identifier
 * @param {Object} voiceData - Updated voice data (partial)
 * @returns {Promise<Object>} Updated voice object
 */
export async function updateVoice(voiceName, voiceData) {
  try {
    // Map frontend fields to backend schema (only include provided fields)
    const backendData = {};

    // Note: Cannot update voice name as it's the primary key
    if (voiceData.method !== undefined) {
      backendData.method = voiceData.method;
    }
    if (voiceData.speakerDescription !== undefined || voiceData.speaker_desc !== undefined) {
      backendData.speaker_desc = voiceData.speakerDescription || voiceData.speaker_desc;
    }
    if (voiceData.scenePrompt !== undefined || voiceData.scene_prompt !== undefined) {
      backendData.scene_prompt = voiceData.scenePrompt || voiceData.scene_prompt;
    }
    if (voiceData.audioPath !== undefined || voiceData.audio_path !== undefined) {
      backendData.audio_path = voiceData.audioPath || voiceData.audio_path;
    }
    if (voiceData.textPath !== undefined || voiceData.text_path !== undefined) {
      backendData.text_path = voiceData.textPath || voiceData.text_path;
    }
    if (voiceData.audio_tokens !== undefined) {
      backendData.audio_tokens = voiceData.audio_tokens;
    }

    const voice = await put(`/api/voices/${voiceName}`, backendData);
    return voice;
  } catch (error) {
    logAPIError('updateVoice', error);
    throw error;
  }
}

/**
 * Delete a voice
 * @param {string} voiceName - Voice name/identifier
 * @returns {Promise<Object>} Success message
 */
export async function deleteVoice(voiceName) {
  try {
    const result = await del(`/api/voices/${voiceName}`);
    return result;
  } catch (error) {
    logAPIError('deleteVoice', error);
    throw error;
  }
}

/**
 * Map backend voice object to frontend format
 * @param {Object} backendVoice - Voice from backend API
 * @returns {Object} Frontend-formatted voice
 */
export function mapVoiceToFrontend(backendVoice) {
  return {
    voice: backendVoice.voice,
    name: backendVoice.voice, // Use voice as display name
    method: backendVoice.method || 'description',
    speakerDescription: backendVoice.speaker_desc || '',
    scenePrompt: backendVoice.scene_prompt || '',
    audioPath: backendVoice.audio_path || '',
    textPath: backendVoice.text_path || '',
    audioTokens: backendVoice.audio_tokens,
    created_at: backendVoice.created_at,
    updated_at: backendVoice.updated_at,
  };
}

/**
 * Map frontend voice object to backend format
 * @param {Object} frontendVoice - Voice from frontend
 * @returns {Object} Backend-formatted voice
 */
export function mapVoiceToBackend(frontendVoice) {
  return {
    voice: frontendVoice.voice || frontendVoice.name,
    method: frontendVoice.method || 'description',
    speaker_desc: frontendVoice.speakerDescription || '',
    scene_prompt: frontendVoice.scenePrompt || '',
    audio_path: frontendVoice.audioPath || '',
    text_path: frontendVoice.textPath || '',
  };
}

/**
 * Generate a voice name from character name
 * @param {string} characterName - Character name
 * @returns {string} Generated voice name
 */
export function generateVoiceName(characterName) {
  // Convert to lowercase, replace spaces with dashes, remove special chars
  const baseName = characterName
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9\s-]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');

  // Add timestamp to ensure uniqueness
  const timestamp = Date.now();
  return `${baseName}-voice-${timestamp}`;
}
