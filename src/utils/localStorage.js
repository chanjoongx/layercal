/**
 * Safe localStorage wrapper utilities
 * Works gracefully in Private mode or when cookies are blocked
 */

export const safeLocalStorage = {
  /**
   * Get a value from localStorage
   * @param {string} key - Storage key
   * @param {any} defaultValue - Fallback if localStorage is inaccessible
   * @returns {any} Stored value or defaultValue
   */
  getItem: (key, defaultValue = null) => {
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
    try {
      localStorage.removeItem(key);
      return true;
    } catch (error) {
      console.warn(`localStorage removeItem failed for key "${key}":`, error);
      return false;
    }
  },

  /**
   * Check if localStorage is available
   * @returns {boolean} Availability status
   */
  isAvailable: () => {
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
 * Detect browser language
 * @returns {string} Language code (e.g. 'en', 'ko', 'ja')
 */
export const detectBrowserLanguage = () => {
  if (typeof window !== 'undefined' && window.navigator) {
    const lang = window.navigator.language || window.navigator.userLanguage;
    return lang.split('-')[0];
  }
  return 'en';
};