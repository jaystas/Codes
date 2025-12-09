/**
 * characters.js - Character Management Functionality
 * Handles character creation, editing, deletion, and display
 * Now using Supabase JS client directly for low-latency access
 */

// Import Supabase client
import { supabase, TABLES, handleSupabaseError, logError } from './supabase.js';

// Character data storage
let characters = [];
let voices = [];
let selectedCharacterId = null;
let currentCharacter = null;
let isLoading = false;

/**
 * Initialize the characters page
 */
export async function initCharacters() {
  // Create notification container
  createNotificationContainer();

  // Setup event listeners
  setupEventListeners();

  // Load data from Supabase
  await loadData();

  console.log('Characters page initialized');
}

/**
 * Load all data from Supabase (characters and voices)
 */
async function loadData() {
  isLoading = true;
  showLoadingState();

  try {
    // Load characters and voices in parallel using Supabase
    const [charactersResult, voicesResult] = await Promise.all([
      supabase.from(TABLES.CHARACTERS).select('*').order('created_at', { ascending: false }),
      supabase.from(TABLES.VOICES).select('*').order('created_at', { ascending: false }),
    ]);

    // Handle characters result
    if (charactersResult.error) {
      throw charactersResult.error;
    }
    characters = charactersResult.data || [];

    // Handle voices result
    if (voicesResult.error) {
      throw voicesResult.error;
    }
    voices = voicesResult.data || [];

    console.log(`Loaded ${characters.length} characters and ${voices.length} voices from Supabase`);

    // Render the character list
    renderCharacterList();

    // Populate voice dropdown if card is open
    populateVoiceDropdown();
  } catch (error) {
    console.error('Error loading data:', error);
    logError('loadData', error);
    const errorMessage = handleSupabaseError(error);
    showNotification('Error Loading Data', errorMessage, 'error');

    // Render empty state
    characters = [];
    voices = [];
    renderCharacterList();
  } finally {
    isLoading = false;
    hideLoadingState();
  }
}

/**
 * Create notification container
 */
function createNotificationContainer() {
  // Check if container already exists
  if (document.getElementById('notification-container')) {
    return;
  }

  const container = document.createElement('div');
  container.id = 'notification-container';
  container.className = 'notification-container';
  document.body.appendChild(container);
}

/**
 * Show loading state
 */
function showLoadingState() {
  const listContainer = document.getElementById('character-list');
  if (listContainer) {
    listContainer.innerHTML = `
      <div class="character-list-loading">
        <div class="loading-spinner"></div>
        <p>Loading characters...</p>
      </div>
    `;
  }
}

/**
 * Hide loading state
 */
function hideLoadingState() {
  // Loading state will be replaced by renderCharacterList()
}

/**
 * Populate voice dropdown
 */
function populateVoiceDropdown() {
  const voiceSelect = document.getElementById('character-voice');
  if (!voiceSelect) return;

  // Clear existing options except the first placeholder
  voiceSelect.innerHTML = '<option value="">Select voice</option>';

  // Add voices from Supabase
  voices.forEach(voice => {
    const option = document.createElement('option');
    option.value = voice.voice;
    option.textContent = voice.voice;
    voiceSelect.appendChild(option);
  });

  // Set current character's voice if exists
  if (currentCharacter && currentCharacter.voice) {
    voiceSelect.value = currentCharacter.voice;
  }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
  // Add character button
  const addBtn = document.getElementById('add-character-btn');
  if (addBtn) {
    addBtn.addEventListener('click', () => showCharacterCard(true));
  }

  // Character search
  const searchInput = document.getElementById('character-search-input');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => filterCharacters(e.target.value));
  }

  // Close card button
  const closeBtn = document.getElementById('character-card-close-btn');
  if (closeBtn) {
    closeBtn.addEventListener('click', () => hideCharacterCard());
  }

  // Tab buttons
  const tabButtons = document.querySelectorAll('.character-tab-button');
  tabButtons.forEach(button => {
    button.addEventListener('click', () => switchTab(button.dataset.tab));
  });

  // Image upload
  const imageSection = document.getElementById('image-section');
  const imageInput = document.getElementById('character-image-input');
  const avatarEditBtn = document.getElementById('avatar-edit-btn');

  if (imageSection && imageInput) {
    imageSection.addEventListener('click', () => imageInput.click());
  }

  if (avatarEditBtn && imageInput) {
    avatarEditBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      imageInput.click();
    });
  }

  if (imageInput) {
    imageInput.addEventListener('change', handleImageUpload);
  }

  // Character name input sync
  const characterNameInput = document.getElementById('character-name-input');
  if (characterNameInput) {
    characterNameInput.addEventListener('input', (e) => {
      const name = e.target.value || 'Character name';
      const characterName = document.getElementById('character-name-display');
      if (characterName) {
        characterName.textContent = name;
      }
    });
  }

  // Save button
  const saveBtn = document.getElementById('save-character-btn');
  if (saveBtn) {
    saveBtn.addEventListener('click', saveCharacter);
  }

  // Delete button
  const deleteBtn = document.getElementById('delete-character-btn');
  if (deleteBtn) {
    deleteBtn.addEventListener('click', deleteCharacter);
  }

  // Chat button
  const chatBtn = document.getElementById('chat-character-btn');
  if (chatBtn) {
    chatBtn.addEventListener('click', () => {
      console.log('Opening chat with character...');
      // TODO: Navigate to chat page with this character
      alert('Chat functionality will be implemented in the next phase');
    });
  }

  // Voice tab - Method radio buttons
  const cloneRadio = document.getElementById('voice-method-clone');
  const profileRadio = document.getElementById('voice-method-profile');
  if (cloneRadio && profileRadio) {
    cloneRadio.addEventListener('change', handleVoiceMethodChange);
    profileRadio.addEventListener('change', handleVoiceMethodChange);
  }

  // Create Voice button
  const createVoiceBtn = document.getElementById('create-voice-btn');
  if (createVoiceBtn) {
    createVoiceBtn.addEventListener('click', handleCreateVoice);
  }
}

/**
 * Render the character list
 */
function renderCharacterList() {
  const listContainer = document.getElementById('character-list');

  if (!listContainer) {
    console.warn('Character list container not found');
    return;
  }

  // Clear existing list
  listContainer.innerHTML = '';

  if (characters.length === 0) {
    listContainer.innerHTML = `
      <div class="character-list-empty">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
        </svg>
        <p>No characters yet.<br>Click "Add Character" to create one.</p>
      </div>
    `;
    return;
  }

  // Render character items
  characters.forEach(character => {
    const item = createCharacterItem(character);
    listContainer.appendChild(item);
  });
}

/**
 * Create a character list item element
 */
function createCharacterItem(character) {
  const item = document.createElement('div');
  item.className = 'character-item';
  if (character.id === selectedCharacterId) {
    item.classList.add('active');
  }

  const avatar = character.image_url
    ? `<img src="${character.image_url}" alt="${character.name}" />`
    : `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
         <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
       </svg>`;

  const description = character.system_prompt
    ? character.system_prompt.substring(0, 40) + '...'
    : 'No description';

  item.innerHTML = `
    <div class="character-item-avatar">
      ${avatar}
    </div>
    <div class="character-item-info">
      <div class="character-item-name">${character.name}</div>
      <div class="character-item-desc">${description}</div>
    </div>
  `;

  item.addEventListener('click', () => selectCharacter(character.id));

  return item;
}

/**
 * Filter characters based on search query
 */
function filterCharacters(query) {
  const items = document.querySelectorAll('.character-item');
  const lowerQuery = query.toLowerCase();

  items.forEach(item => {
    const name = item.querySelector('.character-item-name').textContent.toLowerCase();
    const desc = item.querySelector('.character-item-desc').textContent.toLowerCase();

    if (name.includes(lowerQuery) || desc.includes(lowerQuery)) {
      item.style.display = 'flex';
    } else {
      item.style.display = 'none';
    }
  });
}

/**
 * Select a character and show their card
 */
function selectCharacter(characterId) {
  selectedCharacterId = characterId;
  currentCharacter = characters.find(c => c.id === characterId);

  if (!currentCharacter) {
    console.error('Character not found:', characterId);
    return;
  }

  // Update active state in list
  document.querySelectorAll('.character-item').forEach(item => {
    item.classList.remove('active');
  });

  event.currentTarget?.classList.add('active');

  const card = document.getElementById('character-card');
  const isCardVisible = card?.classList.contains('show');

  // If card is already visible, animate the transition
  if (isCardVisible) {
    // Add switching class for animation
    card.classList.add('switching');

    setTimeout(() => {
      // Load new character data
      loadCharacterData(currentCharacter);

      // Remove switching class to fade back in
      card.classList.remove('switching');
    }, 150);
  } else {
    // Load character data into card
    loadCharacterData(currentCharacter);

    // Show the character card
    showCharacterCard();
  }
}

/**
 * Show the character card
 */
function showCharacterCard(isNew = false) {
  const card = document.getElementById('character-card');
  const welcome = document.getElementById('character-welcome');

  if (!card || !welcome) {
    console.warn('Character card or welcome element not found');
    return;
  }

  if (isNew || !currentCharacter) {
    // Create a new blank character
    currentCharacter = {
      id: null,
      name: 'Character name',
      image_url: null,
      voice: '',
      system_prompt: '',
      images: [],
      is_active: false,
      voiceData: {
        method: 'clone',
        speaker_desc: '',
        scene_prompt: '',
        audio_path: '',
        text_path: ''
      }
    };
    loadCharacterData(currentCharacter);
  }

  // Hide welcome message and show card with animation
  welcome.classList.add('hidden');

  // Small delay to ensure smooth transition
  setTimeout(() => {
    card.classList.add('show');
  }, 50);
}

/**
 * Hide the character card
 */
function hideCharacterCard() {
  const card = document.getElementById('character-card');
  const welcome = document.getElementById('character-welcome');

  if (!card || !welcome) {
    return;
  }

  // Hide card and show welcome message
  card.classList.remove('show');

  setTimeout(() => {
    welcome.classList.remove('hidden');
  }, 300);

  // Reset current character and selection
  currentCharacter = null;
  selectedCharacterId = null;

  // Update active state in list
  document.querySelectorAll('.character-item').forEach(item => {
    item.classList.remove('active');
  });
}

/**
 * Load character data into the card form
 */
function loadCharacterData(character) {
  // Character name in header
  const nameDisplay = document.getElementById('character-name-display');
  if (nameDisplay) {
    nameDisplay.textContent = character.name;
  }

  // Character name input
  const nameInput = document.getElementById('character-name-input');
  if (nameInput) {
    nameInput.value = character.name;
  }

  // Avatar
  const headerAvatar = document.getElementById('header-avatar');
  const imageUploadArea = document.getElementById('image-upload-area');

  if (character.image_url) {
    if (headerAvatar) {
      headerAvatar.innerHTML = `<img src="${character.image_url}" alt="${character.name}" />`;
    }
    if (imageUploadArea) {
      imageUploadArea.innerHTML = `<img src="${character.image_url}" class="image-preview" alt="${character.name}" />`;
    }
  } else {
    if (headerAvatar) {
      headerAvatar.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
        </svg>
      `;
    }
    if (imageUploadArea) {
      imageUploadArea.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
        </svg>
        <span>Click to upload image</span>
      `;
    }
  }

  // Global prompt (not in database schema - skip for now)
  const globalPromptInput = document.getElementById('character-global-prompt');
  if (globalPromptInput) {
    globalPromptInput.value = '';
  }

  // System prompt
  const systemPromptInput = document.getElementById('character-system-prompt');
  if (systemPromptInput) {
    systemPromptInput.value = character.system_prompt || '';
  }

  // Voice
  populateVoiceDropdown();
  const voiceSelect = document.getElementById('character-voice');
  if (voiceSelect) {
    voiceSelect.value = character.voice || '';
  }

  // Voice tab data
  if (character.voiceData) {
    const cloneRadio = document.getElementById('voice-method-clone');
    const profileRadio = document.getElementById('voice-method-profile');
    const speakerDesc = document.getElementById('voice-speaker-description');
    const scenePrompt = document.getElementById('voice-scene-prompt');
    const audioPath = document.getElementById('voice-audio-path');
    const textPath = document.getElementById('voice-text-path');

    if (cloneRadio && profileRadio) {
      if (character.voiceData.method === 'clone') {
        cloneRadio.checked = true;
      } else {
        profileRadio.checked = true;
      }
    }

    if (speakerDesc) {
      speakerDesc.value = character.voiceData.speaker_desc || '';
    }
    if (scenePrompt) {
      scenePrompt.value = character.voiceData.scene_prompt || '';
    }
    if (audioPath) {
      audioPath.value = character.voiceData.audio_path || '';
    }
    if (textPath) {
      textPath.value = character.voiceData.text_path || '';
    }

    // Update disabled states based on method
    handleVoiceMethodChange();
  }
}

/**
 * Handle image upload
 */
function handleImageUpload(e) {
  const file = e.target.files[0];
  if (file && file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = (event) => {
      const imgUrl = event.target.result;

      // Update image upload area
      const imageUploadArea = document.getElementById('image-upload-area');
      if (imageUploadArea) {
        imageUploadArea.innerHTML = `<img src="${imgUrl}" class="image-preview" alt="Character image">`;
      }

      // Update header avatar
      const headerAvatar = document.getElementById('header-avatar');
      if (headerAvatar) {
        headerAvatar.innerHTML = `<img src="${imgUrl}" alt="Character avatar">`;
      }

      // Store in current character
      if (currentCharacter) {
        currentCharacter.image_url = imgUrl;
      }
    };
    reader.readAsDataURL(file);
  }
}

/**
 * Switch between tabs
 */
function switchTab(tabName) {
  // Remove active class from all buttons and panels
  document.querySelectorAll('.character-tab-button').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));

  // Add active class to clicked button and corresponding panel
  const button = document.querySelector(`[data-tab="${tabName}"]`);
  const panel = document.getElementById(`${tabName}-panel`);

  if (button) button.classList.add('active');
  if (panel) panel.classList.add('active');

  // Show/hide image section based on tab
  const imageSection = document.getElementById('image-section');
  const contentSection = document.getElementById('content-section');

  if (tabName === 'profile') {
    imageSection?.classList.remove('hidden');
    contentSection?.classList.remove('full-width');
  } else {
    imageSection?.classList.add('hidden');
    contentSection?.classList.add('full-width');
  }

  // Initialize voice method state if switching to voice tab
  if (tabName === 'voice') {
    handleVoiceMethodChange();
  }
}

/**
 * Handle voice method change (Clone vs Profile)
 */
function handleVoiceMethodChange() {
  const cloneRadio = document.getElementById('voice-method-clone');
  const profileRadio = document.getElementById('voice-method-profile');
  const speakerDesc = document.getElementById('voice-speaker-description');
  const audioPath = document.getElementById('voice-audio-path');
  const textPath = document.getElementById('voice-text-path');

  if (!cloneRadio || !profileRadio || !speakerDesc || !audioPath || !textPath) {
    return;
  }

  if (cloneRadio.checked) {
    // Clone method: disable speaker description, enable audio and text paths
    speakerDesc.disabled = true;
    speakerDesc.classList.add('disabled');
    audioPath.disabled = false;
    audioPath.classList.remove('disabled');
    textPath.disabled = false;
    textPath.classList.remove('disabled');
  } else if (profileRadio.checked) {
    // Profile method: enable speaker description, disable audio and text paths
    speakerDesc.disabled = false;
    speakerDesc.classList.remove('disabled');
    audioPath.disabled = true;
    audioPath.classList.add('disabled');
    textPath.disabled = true;
    textPath.classList.add('disabled');
  }
}

/**
 * Generate a voice name from character name
 */
function generateVoiceName(characterName) {
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

/**
 * Handle Create Voice button click
 */
async function handleCreateVoice() {
  const cloneRadio = document.getElementById('voice-method-clone');
  const speakerDesc = document.getElementById('voice-speaker-description');
  const scenePrompt = document.getElementById('voice-scene-prompt');
  const audioPath = document.getElementById('voice-audio-path');
  const textPath = document.getElementById('voice-text-path');

  if (!cloneRadio || !speakerDesc || !scenePrompt || !audioPath || !textPath) {
    return;
  }

  // Validate: need character name for voice name
  if (!currentCharacter || !currentCharacter.name) {
    showNotification('Validation Error', 'Please enter a character name first', 'error');
    return;
  }

  const method = cloneRadio.checked ? 'clone' : 'description';

  // Validate based on method
  if (method === 'description' && !speakerDesc.value.trim()) {
    showNotification('Validation Error', 'Speaker description is required for description method', 'error');
    return;
  }

  if (method === 'clone' && (!audioPath.value.trim() || !textPath.value.trim())) {
    showNotification('Validation Error', 'Audio path and text path are required for clone method', 'error');
    return;
  }

  // Generate voice name from character name
  const voiceName = generateVoiceName(currentCharacter.name);

  const voiceData = {
    voice: voiceName,
    method: method,
    speaker_desc: speakerDesc.value,
    scene_prompt: scenePrompt.value,
    audio_path: audioPath.value,
    text_path: textPath.value
  };

  // Disable button
  const createBtn = document.getElementById('create-voice-btn');
  if (createBtn) {
    createBtn.disabled = true;
    createBtn.textContent = 'Creating...';
  }

  try {
    // Create voice via Supabase
    const { data, error } = await supabase
      .from(TABLES.VOICES)
      .insert([voiceData])
      .select()
      .single();

    if (error) throw error;

    console.log('Voice created:', data);

    // Add to local voices array
    voices.push(data);

    // Update voice dropdown
    populateVoiceDropdown();

    // Auto-select the newly created voice
    const voiceSelect = document.getElementById('character-voice');
    if (voiceSelect) {
      voiceSelect.value = data.voice;
      currentCharacter.voice = data.voice;
    }

    showNotification(
      'Voice Created',
      `Voice "${data.voice}" created successfully and assigned to ${currentCharacter.name}`,
      'success'
    );
  } catch (error) {
    console.error('Error creating voice:', error);
    logError('handleCreateVoice', error);
    const errorMessage = handleSupabaseError(error);
    showNotification('Error Creating Voice', errorMessage, 'error');
  } finally {
    // Re-enable button
    if (createBtn) {
      createBtn.disabled = false;
      createBtn.textContent = 'Create Voice';
    }
  }
}

/**
 * Save character
 */
async function saveCharacter() {
  if (!currentCharacter) {
    console.error('No character to save');
    return;
  }

  // Get form values
  const nameInput = document.getElementById('character-name-input');
  const systemPromptInput = document.getElementById('character-system-prompt');
  const voiceSelect = document.getElementById('character-voice');

  // Validate required fields
  const characterName = nameInput?.value?.trim();
  if (!characterName) {
    showNotification('Validation Error', 'Character name is required', 'error');
    return;
  }

  // Prepare character data for Supabase
  const characterData = {
    name: characterName,
    system_prompt: systemPromptInput?.value || '',
    voice: voiceSelect?.value || '',
    image_url: currentCharacter.image_url || '',
    images: currentCharacter.images || [],
    is_active: currentCharacter.is_active || false,
  };

  const isNewCharacter = !currentCharacter.id;

  // Disable save button to prevent double-clicks
  const saveBtn = document.getElementById('save-character-btn');
  if (saveBtn) {
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';
  }

  try {
    let savedCharacter;

    if (isNewCharacter) {
      // Create new character via Supabase
      const { data, error } = await supabase
        .from(TABLES.CHARACTERS)
        .insert([characterData])
        .select()
        .single();

      if (error) throw error;
      savedCharacter = data;
      console.log('Character created:', savedCharacter);

      // Add to local array
      characters.push(savedCharacter);
    } else {
      // Update existing character via Supabase
      const { data, error } = await supabase
        .from(TABLES.CHARACTERS)
        .update(characterData)
        .eq('id', currentCharacter.id)
        .select()
        .single();

      if (error) throw error;
      savedCharacter = data;
      console.log('Character updated:', savedCharacter);

      // Update in local array
      const index = characters.findIndex(c => c.id === currentCharacter.id);
      if (index !== -1) {
        characters[index] = savedCharacter;
      }
    }

    // Re-render the character list
    renderCharacterList();

    // Show success notification
    showNotification(
      isNewCharacter ? 'Character Created' : 'Character Saved',
      `${characterName} has been ${isNewCharacter ? 'created' : 'updated'} successfully`,
      'success'
    );

    // Close the card after saving
    hideCharacterCard();
  } catch (error) {
    console.error('Error saving character:', error);
    logError('saveCharacter', error);
    const errorMessage = handleSupabaseError(error);
    showNotification(
      'Error Saving Character',
      errorMessage,
      'error'
    );
  } finally {
    // Re-enable save button
    if (saveBtn) {
      saveBtn.disabled = false;
      saveBtn.textContent = 'Save Character';
    }
  }
}

/**
 * Delete character
 */
async function deleteCharacter() {
  if (!currentCharacter || !currentCharacter.id) {
    console.error('No character to delete');
    return;
  }

  if (!confirm(`Are you sure you want to delete ${currentCharacter.name}? This action cannot be undone.`)) {
    return;
  }

  const characterName = currentCharacter.name;
  const characterId = currentCharacter.id;

  // Disable delete button to prevent double-clicks
  const deleteBtn = document.getElementById('delete-character-btn');
  if (deleteBtn) {
    deleteBtn.disabled = true;
    deleteBtn.textContent = 'Deleting...';
  }

  try {
    // Delete via Supabase
    const { error } = await supabase
      .from(TABLES.CHARACTERS)
      .delete()
      .eq('id', characterId);

    if (error) throw error;

    console.log('Character deleted:', characterId);

    // Remove from local array
    characters = characters.filter(c => c.id !== characterId);

    // Re-render the character list
    renderCharacterList();

    // Show success notification
    showNotification(
      'Character Deleted',
      `${characterName} has been deleted successfully`,
      'success'
    );

    // Hide the card
    hideCharacterCard();
  } catch (error) {
    console.error('Error deleting character:', error);
    logError('deleteCharacter', error);
    const errorMessage = handleSupabaseError(error);
    showNotification(
      'Error Deleting Character',
      errorMessage,
      'error'
    );

    // Re-enable delete button
    if (deleteBtn) {
      deleteBtn.disabled = false;
      deleteBtn.textContent = 'Delete';
    }
  }
}

/**
 * Get all characters
 */
export function getCharacters() {
  return characters;
}

/**
 * Get selected character
 */
export function getSelectedCharacter() {
  return currentCharacter;
}

/**
 * Show notification
 */
function showNotification(title, message, type = 'success') {
  const container = document.getElementById('notification-container');

  if (!container) {
    console.warn('Notification container not found');
    return;
  }

  // Create notification element
  const notification = document.createElement('div');
  notification.className = `notification ${type}`;

  // Icon based on type
  let iconSvg = '';
  if (type === 'success') {
    iconSvg = `
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
        <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
      </svg>
    `;
  } else if (type === 'error') {
    iconSvg = `
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
        <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
      </svg>
    `;
  } else if (type === 'warning') {
    iconSvg = `
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
        <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    `;
  }

  notification.innerHTML = `
    <div class="notification-icon ${type}">
      ${iconSvg}
    </div>
    <div class="notification-content">
      <div class="notification-title">${title}</div>
      ${message ? `<div class="notification-message">${message}</div>` : ''}
    </div>
    <button class="notification-close">
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
        <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
      </svg>
    </button>
  `;

  // Add to container
  container.appendChild(notification);

  // Setup close button
  const closeBtn = notification.querySelector('.notification-close');
  closeBtn.addEventListener('click', () => {
    removeNotification(notification);
  });

  // Show with animation
  setTimeout(() => {
    notification.classList.add('show');
  }, 10);

  // Auto-remove after 3 seconds
  setTimeout(() => {
    removeNotification(notification);
  }, 3000);
}

/**
 * Remove notification
 */
function removeNotification(notification) {
  notification.classList.remove('show');

  setTimeout(() => {
    notification.remove();
  }, 300);
}
