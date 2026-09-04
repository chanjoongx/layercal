/**
 * Safe localStorage wrapper utilities
 * Works gracefully in Private mode or when cookies are blocked
 */

/**
 * No storage global at all (server rendering, tests) is an expected
 * environment, not a fault. Only a storage that *throws* is worth warning
 * about, since that means the browser is actively blocking it.
 */
const hasStorage = () => {
  try {
    return typeof localStorage !== 'undefined' && localStorage !== null;
  } catch {
    return false;
  }
};

export const safeLocalStorage = {
  /**
   * Get a value from localStorage
   * @param {string} key - Storage key
   * @param {any} defaultValue - Fallback if localStorage is inaccessible
   * @returns {any} Stored value or defaultValue
   */
  getItem: (key, defaultValue = null) => {
    if (!hasStorage()) return defaultValue;
    try {
      const item = localStorage.getItem(key);
      return item !== null ? item : defaultValue;
    } catch (error) {
      console.warn(`localStorage getItem failed for key "${key}":`, error);
      return defaultValue;
    }
  },

  /**
   * Save a value to localStorage
   * @param {string} key - Storage key
   * @param {string} value - Value to store
   * @returns {boolean} Whether the operation succeeded
   */
  setItem: (key, value) => {
    if (!hasStorage()) return false;
    try {
      localStorage.setItem(key, value);
      return true;
    } catch (error) {
      console.warn(`localStorage setItem failed for key "${key}":`, error);
      return false;
    }
  },

  /**
   * Remove a value from localStorage
   * @param {string} key - Storage key to remove
   * @returns {boolean} Whether the operation succeeded
   */
  removeItem: (key) => {
    if (!hasStorage()) return false;
    try {
      localStorage.removeItem(key);
      return true;
    } catch (error) {
      console.warn(`localStorage removeItem failed for key "${key}":`, error);
      return false;
    }
  },

  /**
   * Read and JSON.parse a value. Corrupt entries return the default
   * rather than throwing on startup.
   */
  getJSON: (key, defaultValue = null) => {
    if (!hasStorage()) return defaultValue;
    try {
      const raw = localStorage.getItem(key);
      if (raw === null) return defaultValue;
      const parsed = JSON.parse(raw);
      return parsed ?? defaultValue;
    } catch (error) {
      console.warn(`localStorage getJSON failed for key "${key}":`, error);
      return defaultValue;
    }
  },

  /**
   * JSON.stringify and store a value.
   * @returns {boolean} Whether the operation succeeded
   */
  setJSON: (key, value) => {
    if (!hasStorage()) return false;
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (error) {
      console.warn(`localStorage setJSON failed for key "${key}":`, error);
      return false;
    }
  },

  /**
   * Check if localStorage is available
   * @returns {boolean} Availability status
   */
  isAvailable: () => {
    if (!hasStorage()) return false;
    try {
      const testKey = '__localStorage_test__';
      localStorage.setItem(testKey, 'test');
      localStorage.removeItem(testKey);
      return true;
    } catch {
      return false;
    }
  }
};

/**
 * Detect system dark mode preference
 * @returns {boolean} Whether dark mode is active
 */
export const detectSystemDarkMode = () => {
  if (typeof window !== 'undefined' && window.matchMedia) {
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  return false;
};

/**
 * Resolve the initial UI language: saved choice, otherwise English.
 * @param {string} storageKey - localStorage key holding the saved choice
 * @param {string[]} supported - Supported language codes
 */
export const resolveInitialLanguage = (storageKey, supported) => {
  const saved = safeLocalStorage.getItem(storageKey);
  if (saved && supported.includes(saved)) return saved;

  // English is the default for everyone. The browser's language used to win
  // here, which meant a Korean or Japanese visitor landed on a translated page
  // they never asked for, and the shared screenshots and links in the docs did
  // not match what they saw. An explicit choice still persists.
  return supported.includes('en') ? 'en' : supported[0];
};
