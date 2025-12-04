/**
 * Characters API Client
 * Handles all character-related API operations
 */

import { get, post, put, del, APIError, handleAPIError, logAPIError } from './config.js';

/**
 * Get all characters
 * @returns {Promise<Array>} Array of character objects
 */
export async function getAllCharacters() {
  try {
    const characters = await get('/api/characters');
    return characters;
  } catch (error) {
    logAPIError('getAllCharacters', error);
    throw error;
  }
}

/**
 * Get all active characters
 * @returns {Promise<Array>} Array of active character objects
 */
export async function getActiveCharacters() {
  try {
    const characters = await get('/api/characters/active');
    return characters;
  } catch (error) {
    logAPIError('getActiveCharacters', error);
    throw error;
  }
}

/**
 * Search characters by name
 * @param {string} query - Search query
 * @returns {Promise<Array>} Array of matching character objects
 */
export async function searchCharacters(query) {
  try {
    const characters = await get('/api/characters/search', { query });
    return characters;
  } catch (error) {
    logAPIError('searchCharacters', error);
    throw error;
  }
}

/**
 * Get a specific character by ID
 * @param {string} characterId - Character ID
 * @returns {Promise<Object>} Character object
 */
export async function getCharacter(characterId) {
  try {
    const character = await get(`/api/characters/${characterId}`);
    return character;
  } catch (error) {
    logAPIError('getCharacter', error);
    throw error;
  }
}

/**
 * Create a new character
 * @param {Object} characterData - Character data
 * @param {string} characterData.name - Character name (required)
 * @param {string} characterData.voice - Voice identifier
 * @param {string} characterData.system_prompt - System prompt
 * @param {string} characterData.image_url - Image URL
 * @param {Array<string>} characterData.images - Array of image URLs
 * @param {boolean} characterData.is_active - Active status
 * @returns {Promise<Object>} Created character object
 */
export async function createCharacter(characterData) {
  try {
    // Map frontend fields to backend schema
    const backendData = {
      name: characterData.name,
      voice: characterData.voice || '',
      system_prompt: characterData.systemPrompt || characterData.system_prompt || '',
      image_url: characterData.avatar || characterData.image_url || '',
      images: characterData.images || [],
      is_active: characterData.is_active !== undefined ? characterData.is_active : false,
    };

    const character = await post('/api/characters', backendData);
    return character;
  } catch (error) {
    logAPIError('createCharacter', error);
    throw error;
  }
}

/**
 * Update an existing character
 * @param {string} characterId - Character ID
 * @param {Object} characterData - Updated character data (partial)
 * @returns {Promise<Object>} Updated character object
 */
export async function updateCharacter(characterId, characterData) {
  try {
    // Map frontend fields to backend schema (only include provided fields)
    const backendData = {};

    if (characterData.name !== undefined) {
      backendData.name = characterData.name;
    }
    if (characterData.voice !== undefined) {
      backendData.voice = characterData.voice;
    }
    if (characterData.systemPrompt !== undefined || characterData.system_prompt !== undefined) {
      backendData.system_prompt = characterData.systemPrompt || characterData.system_prompt;
    }
    if (characterData.avatar !== undefined || characterData.image_url !== undefined) {
      backendData.image_url = characterData.avatar || characterData.image_url;
    }
    if (characterData.images !== undefined) {
      backendData.images = characterData.images;
    }
    if (characterData.is_active !== undefined) {
      backendData.is_active = characterData.is_active;
    }

    const character = await put(`/api/characters/${characterId}`, backendData);
    return character;
  } catch (error) {
    logAPIError('updateCharacter', error);
    throw error;
  }
}

/**
 * Set character active status
 * @param {string} characterId - Character ID
 * @param {boolean} isActive - Active status
 * @returns {Promise<Object>} Updated character object
 */
export async function setCharacterActive(characterId, isActive) {
  try {
    const character = await put(`/api/characters/${characterId}/active`, null, {
      is_active: isActive,
    });
    return character;
  } catch (error) {
    logAPIError('setCharacterActive', error);
    throw error;
  }
}

/**
 * Delete a character
 * @param {string} characterId - Character ID
 * @returns {Promise<Object>} Success message
 */
export async function deleteCharacter(characterId) {
  try {
    const result = await del(`/api/characters/${characterId}`);
    return result;
  } catch (error) {
    logAPIError('deleteCharacter', error);
    throw error;
  }
}

/**
 * Map backend character object to frontend format
 * @param {Object} backendCharacter - Character from backend API
 * @returns {Object} Frontend-formatted character
 */
export function mapCharacterToFrontend(backendCharacter) {
  return {
    id: backendCharacter.id,
    name: backendCharacter.name,
    voice: backendCharacter.voice || '',
    systemPrompt: backendCharacter.system_prompt || '',
    avatar: backendCharacter.image_url || null,
    images: backendCharacter.images || [],
    is_active: backendCharacter.is_active || false,
    created_at: backendCharacter.created_at,
    updated_at: backendCharacter.updated_at,
    // Keep voice data structure for the voice tab
    voiceData: {
      method: 'description', // Default, will be populated from voice if exists
      speakerDescription: '',
      scenePrompt: '',
      audioPath: '',
      textPath: '',
    },
  };
}

/**
 * Map frontend character object to backend format
 * @param {Object} frontendCharacter - Character from frontend
 * @returns {Object} Backend-formatted character
 */
export function mapCharacterToBackend(frontendCharacter) {
  return {
    name: frontendCharacter.name,
    voice: frontendCharacter.voice || '',
    system_prompt: frontendCharacter.systemPrompt || '',
    image_url: frontendCharacter.avatar || '',
    images: frontendCharacter.images || [],
    is_active: frontendCharacter.is_active !== undefined ? frontendCharacter.is_active : false,
  };
}
