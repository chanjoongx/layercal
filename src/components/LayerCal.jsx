import React, { useState, useRef, useMemo, useCallback, useEffect } from 'react';
import { Trash2, GripVertical, Plus, Info, Layers, Moon, Sun, Globe, ChevronDown, ChevronUp, Camera, X, Mail, Code, Sparkles, Eraser, Check, TriangleAlert } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import Modal from '@/components/ui/modal';
import { TRANSLATIONS, LANGUAGE_OPTIONS, SUPPORTED_LANGUAGES } from '@/config/translations';
import { getLayerTypes, formatNumber, calculateMemory, formatBytes, LAYER_TYPE_IDS, DEFAULT_LAYER_PARAMS } from '@/config/layerTypes';
import { safeLocalStorage, detectSystemDarkMode, resolveInitialLanguage } from '@/utils/localStorage';
import { exportToImage, validateExportElement } from '@/utils/imageExport';
import { validateModelDimensions } from '@/utils/modelValidation';
import { generatePyTorchCode, generateTensorFlowCode, generateJAXCode } from '@/utils/codeGenerator';
import AIAdvisor from '@/components/AIAdvisor';

/**
 * LayerCal: deep learning parameter calculator and AI architecture advisor.
 *
 * State that survives a reload: the model itself, language, dark mode, and the
 * AI Advisor's provider/key (the last of those lives in AIAdvisor).
 */

/**
 * GitHub mark. lucide dropped brand icons in v1, and GitHub's brand guidelines
 * permit the mark when it links to GitHub, so it lives here rather than
 * pinning the whole icon set to a 2023 release.
 */
const GithubMark = ({ className }) => (
  <svg viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden="true" focusable="false">
    <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
  </svg>
);

const STORAGE_KEYS = {
  model: 'layercal-model-v1',
  darkMode: 'layercal-darkmode',
  language: 'layercal-language',
};

/** Year of first publication. Rendered as a range once the year rolls over. */
const COPYRIGHT_START_YEAR = 2026;

const VALID_LAYER_TYPES = new Set(LAYER_TYPE_IDS);

/** Monotonic id source. Date.now() collides when several layers are added in one tick. */
let layerIdCounter = 0;
const nextLayerId = () => `layer-${++layerIdCounter}`;

/**
 * Drop anything that isn't a layer we can still render, and fill in defaults
 * for params a previous version didn't store. A missing param would otherwise
 * flip its input from controlled to uncontrolled.
 */
const sanitizeRestoredLayers = (raw) => {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter(l => l && typeof l === 'object' && VALID_LAYER_TYPES.has(l.type))
    .slice(0, 200)
    .map(l => ({
      id: nextLayerId(),
      type: l.type,
      params: {
        ...DEFAULT_LAYER_PARAMS[l.type],
        ...(l.params && typeof l.params === 'object' ? l.params : {}),
      },
    }));
};

const copyrightYears = () => {
  const now = new Date().getFullYear();
  return now > COPYRIGHT_START_YEAR ? `${COPYRIGHT_START_YEAR}-${now}` : `${COPYRIGHT_START_YEAR}`;
};

export default function LayerCal() {
  // Model state, restored from the previous session
  const [modelLayers, setModelLayers] = useState(() =>
    sanitizeRestoredLayers(safeLocalStorage.getJSON(STORAGE_KEYS.model, []))
  );
  const [draggedType, setDraggedType] = useState(null);
  const [draggedIndex, setDraggedIndex] = useState(null);
  const [showLanguageMenu, setShowLanguageMenu] = useState(false);
  const [showDonationModal, setShowDonationModal] = useState(false);

  const [showCodeModal, setShowCodeModal] = useState(false);
  const [showAdvisorModal, setShowAdvisorModal] = useState(false);
  const [selectedFramework, setSelectedFramework] = useState('pytorch');
  const [codeCopied, setCodeCopied] = useState(false);
  const [memoryMode, setMemoryMode] = useState('inference');
  const [precision, setPrecision] = useState('fp32');
  const [toast, setToast] = useState(null);

  // Dark mode init: localStorage → system preference
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = safeLocalStorage.getItem(STORAGE_KEYS.darkMode);
    if (saved !== null) return saved === 'true';
    return detectSystemDarkMode();
  });

  // Language init: localStorage → browser language → 'en'
  const [language, setLanguage] = useState(() =>
    resolveInitialLanguage(STORAGE_KEYS.language, SUPPORTED_LANGUAGES)
  );

  const languageMenuRef = useRef(null);
  const captureAreaRef = useRef(null);
  const toastTimerRef = useRef(null);

  const t = TRANSLATIONS[language] || TRANSLATIONS.en;

  // LAYER_TYPES recalculates only on language / dark-mode change
  const LAYER_TYPES = useMemo(() => getLayerTypes(t, isDarkMode), [t, isDarkMode]);

  /**
   * @param {string} message
   * @param {{ action?: { label: string, onClick: () => void }, duration?: number }} [options]
   */
  const showToast = useCallback((message, options = {}) => {
    clearTimeout(toastTimerRef.current);
    setToast({ message, action: options.action || null });
    toastTimerRef.current = setTimeout(() => setToast(null), options.duration || 5000);
  }, []);

  useEffect(() => () => clearTimeout(toastTimerRef.current), []);

  // Persist the model so a refresh doesn't discard the user's work
  useEffect(() => {
    safeLocalStorage.setJSON(
      STORAGE_KEYS.model,
      modelLayers.map(({ type, params }) => ({ type, params }))
    );
  }, [modelLayers]);

  // Keep the document language in sync for screen readers and search engines
  useEffect(() => {
    document.documentElement.lang = language;
  }, [language]);

  // The page background lives on <body>, outside this component's tree, so it
  // stays light on overscroll unless the theme is mirrored onto the root. The
  // `dark` class activates the CSS variables already declared in index.css.
  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDarkMode);
    document.documentElement.style.colorScheme = isDarkMode ? 'dark' : 'light';
  }, [isDarkMode]);

  // Detect clicks outside language menu
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (languageMenuRef.current && !languageMenuRef.current.contains(event.target)) {
        setShowLanguageMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Escape closes the language menu. Dialogs handle their own via <Modal>.
  useEffect(() => {
    if (!showLanguageMenu) return;
    const handleKey = (e) => {
      if (e.key === 'Escape') setShowLanguageMenu(false);
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [showLanguageMenu]);

  const handleLanguageChange = useCallback((newLang) => {
    setLanguage(newLang);
    safeLocalStorage.setItem(STORAGE_KEYS.language, newLang);
    setShowLanguageMenu(false);
  }, []);

  const handleDarkModeToggle = useCallback(() => {
    setIsDarkMode(prev => {
      const newMode = !prev;
      safeLocalStorage.setItem(STORAGE_KEYS.darkMode, String(newMode));
      return newMode;
    });
  }, []);

  const addLayer = useCallback((type) => {
    const layerConfig = LAYER_TYPES[type];
    if (!layerConfig) return;
    setModelLayers(prev => [...prev, {
      id: nextLayerId(),
      type,
      params: { ...layerConfig.defaultParams },
    }]);
  }, [LAYER_TYPES]);

  const addLayersFromAdvisor = useCallback((layerConfigs, mode = 'append') => {
    const newLayers = layerConfigs
      .filter(config => LAYER_TYPES[config.type])
      .map(config => ({
        id: nextLayerId(),
        type: config.type,
        params: { ...LAYER_TYPES[config.type].defaultParams, ...config.params },
      }));
    setModelLayers(prev => mode === 'replace' ? newLayers : [...prev, ...newLayers]);
    setShowAdvisorModal(false);
  }, [LAYER_TYPES]);

  const deleteLayer = useCallback((id) => {
    setModelLayers(prev => prev.filter(layer => layer.id !== id));
  }, []);

  /**
   * Clears immediately and offers an undo, rather than blocking on a native
   * confirm. Reversible beats interruptive for an action this cheap to redo.
   */
  const clearAllLayers = useCallback(() => {
    if (modelLayers.length === 0) return;
    const snapshot = modelLayers;
    setModelLayers([]);
    showToast(t.layersCleared || 'All layers removed.', {
      duration: 8000,
      action: {
        label: t.undo || 'Undo',
        onClick: () => setModelLayers(snapshot),
      },
    });
  }, [modelLayers, showToast, t]);

  /**
   * Keyboard and touch reordering. HTML5 drag and drop fires no events on
   * touch devices and cannot be driven from the keyboard at all, so without
   * these the layer order is unreachable for a large share of users.
   */
  const moveLayer = useCallback((index, delta) => {
    setModelLayers(prev => {
      const target = index + delta;
      if (target < 0 || target >= prev.length) return prev;
      const next = [...prev];
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  }, []);

  /**
   * Accepts the raw input string so a field can be cleared mid-edit; the value
   * is coerced back into range on blur. Numeric coercion happens here rather
   * than at render time so a half-typed field never reaches the formulas.
   */
  const updateLayerParam = useCallback((id, field, rawValue) => {
    setModelLayers(prev => prev.map(layer => {
      if (layer.id !== id) return layer;
      return { ...layer, params: { ...layer.params, [field.key]: rawValue } };
    }));
  }, []);

  const normalizeLayerParam = useCallback((id, field) => {
    setModelLayers(prev => prev.map(layer => {
      if (layer.id !== id) return layer;

      const current = layer.params[field.key];
      const num = Number(current);
      const min = Number.isFinite(field.min) ? field.min : 1;
      const max = Number.isFinite(field.max) ? field.max : Infinity;

      let next;
      if (!Number.isFinite(num)) {
        next = min;
      } else if (field.step && field.step < 1) {
        // fractional field (dropout rate): keep one decimal place
        next = Math.min(max, Math.max(min, Math.round(num * 10) / 10));
      } else {
        next = Math.min(max, Math.max(min, Math.floor(num)));
      }

      if (next === current) return layer;
      return { ...layer, params: { ...layer.params, [field.key]: next } };
    }));
  }, []);

  const safeCalculate = useCallback((config, params) => {
    const value = config.calculate(params);
    return Number.isFinite(value) ? value : 0;
  }, []);

  const totalParams = useMemo(() => {
    return modelLayers.reduce((total, layer) => {
      const config = LAYER_TYPES[layer.type];
      return config ? total + safeCalculate(config, layer.params) : total;
    }, 0);
  }, [modelLayers, LAYER_TYPES, safeCalculate]);

  const totalFLOPs = useMemo(() => {
    return modelLayers.reduce((total, layer) => {
      const config = LAYER_TYPES[layer.type];
      if (!config?.calculateFLOPs) return total;
      const value = config.calculateFLOPs(layer.params);
      return Number.isFinite(value) ? total + value : total;
    }, 0);
  }, [modelLayers, LAYER_TYPES]);

  const memoryEstimate = useMemo(() => {
    return calculateMemory(totalParams, memoryMode, precision);
  }, [totalParams, memoryMode, precision]);

  // Stock defaults alone produce a stack that cannot run (Conv2D emits 64
  // channels, BatchNorm defaults to 128), so mismatches are surfaced on the
  // canvas instead of silently reaching the exported code.
  const dimensionIssues = useMemo(() => validateModelDimensions(modelLayers), [modelLayers]);

  const generatedCode = useMemo(() => {
    switch (selectedFramework) {
      case 'pytorch': return generatePyTorchCode(modelLayers);
      case 'tensorflow': return generateTensorFlowCode(modelLayers);
      case 'jax': return generateJAXCode(modelLayers);
      default: return '';
    }
  }, [modelLayers, selectedFramework]);

  // ── Drag & drop ──────────────────────────────────
  const handleDragStart = useCallback((e, type) => {
    setDraggedType(type);
    e.dataTransfer.effectAllowed = 'copy';
    e.dataTransfer.setData('text/plain', type);
  }, []);

  const handleDragEnd = useCallback(() => setDraggedType(null), []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = draggedType ? 'copy' : 'move';
  }, [draggedType]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    if (draggedType) addLayer(draggedType);
    setDraggedType(null);
  }, [draggedType, addLayer]);

  const handleLayerDragStart = useCallback((e, index) => {
    setDraggedIndex(index);
    e.dataTransfer.effectAllowed = 'move';
  }, []);

  const handleLayerDragOver = useCallback((e, index) => {
    e.preventDefault();
    if (draggedIndex === null || draggedIndex === index) return;

    setModelLayers(prev => {
      const newLayers = [...prev];
      const [draggedLayer] = newLayers.splice(draggedIndex, 1);
      newLayers.splice(index, 0, draggedLayer);
      return newLayers;
    });
    setDraggedIndex(index);
  }, [draggedIndex]);

  const handleLayerDragEnd = useCallback(() => setDraggedIndex(null), []);

  // ── Export ───────────────────────────────────────
  const handleExportImageClick = useCallback(() => setShowDonationModal(true), []);

  const exportErrorMessage = useCallback((code) => {
    if (code === 'empty-element') return t.exportErrEmpty || 'Nothing to export yet.';
    if (code === 'load-failed') return t.exportErrLoad || 'Could not load the image exporter.';
    return t.exportFailed || 'Failed to export image. Please try again.';
  }, [t]);

  const handleExportImage = useCallback(async () => {
    setShowDonationModal(false);

    const element = captureAreaRef.current;
    const validation = validateExportElement(element);
    if (!validation.valid) {
      showToast(exportErrorMessage(validation.error));
      return;
    }

    const result = await exportToImage(element, { isDarkMode });
    if (!result.ok) {
      showToast(exportErrorMessage(result.error));
    }
  }, [isDarkMode, showToast, exportErrorMessage]);

  const handleCopyCode = useCallback(async () => {
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard API unavailable');
      await navigator.clipboard.writeText(generatedCode);
      setCodeCopied(true);
      setTimeout(() => setCodeCopied(false), 2000);
    } catch {
      // Non-secure origins and older browsers have no clipboard access.
      showToast(t.copyFailed || 'Copy failed. Select the code and copy it manually.');
    }
  }, [generatedCode, showToast, t]);

  const currentLanguageOption = useMemo(() =>
    LANGUAGE_OPTIONS.find(opt => opt.code === language) || LANGUAGE_OPTIONS[0],
    [language]
  );

  const modelSizeBytes = totalParams * 4; // FP32 reference size

  return (
    <div className={`min-h-screen transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-900' : 'bg-gradient-to-br from-purple-50 via-white to-blue-50'
    }`}>
      <a href="#main-content" className="skip-link">
        {t.skipToContent || 'Skip to content'}
      </a>

      <div className="container mx-auto px-3 sm:px-4 py-4 sm:py-8 max-w-7xl">
        {/* Header
          ┌─ 3단계 브레이크포인트 전략 ──────────────────────────────────┐
          │ ~767px (mobile)  : 2행 레이아웃                              │
          │   Row1 - 로고+타이틀 / 아이콘 유틸 버튼 4개                  │
          │   Row2 - 액션 버튼 3개 (flex-1 균등 분할, 텍스트 포함)        │
          │                                                              │
          │ 768~1023px (tablet/md) : 1행, 액션 버튼은 아이콘만           │
          │ 1024px~ (desktop/lg) : 1행, 액션 버튼에 텍스트까지           │
          └──────────────────────────────────────────────────────────────┘
        */}
        <header className="mb-4 sm:mb-6">
          <div className={`px-3 sm:px-4 py-3 rounded-xl ${
            isDarkMode
              ? 'bg-gray-800/90 border border-gray-700'
              : 'bg-white/80 backdrop-blur-sm border border-gray-200/80 shadow-sm'
          }`}>

            {/* Row 1: 항상 표시 */}
            <div className="flex items-center justify-between gap-2">

              {/* 왼쪽: 로고 + 타이틀. min-w-0 필수 */}
              <div className="flex items-center gap-2 min-w-0">
                <img
                  src="/calculator-icon.svg"
                  alt=""
                  width="44"
                  height="44"
                  className="w-8 h-8 md:w-11 md:h-11 flex-shrink-0"
                />
                <h1 className={`text-xl md:text-2xl lg:text-3xl font-bold truncate ${
                  isDarkMode ? 'text-white' : 'text-gray-900'
                }`}>
                  {t.title}
                </h1>
              </div>

              {/* 오른쪽: 버튼 묶음. flex-shrink-0, 타이틀이 truncate로 양보 */}
              <div className="flex items-center gap-1 flex-shrink-0">

                <a
                  href="mailto:contact@layercal.com"
                  className={`p-1.5 sm:p-2 rounded-lg transition-colors ${
                    isDarkMode
                      ? 'bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-gray-100'
                      : 'bg-white hover:bg-gray-50 text-gray-600 hover:text-gray-900 shadow-sm'
                  }`}
                  aria-label="Contact us via email"
                  title="contact@layercal.com"
                >
                  <Mail className="w-4 h-4 sm:w-5 sm:h-5" />
                </a>

                <a
                  href="https://github.com/chanjoongx/layercal"
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`p-1.5 sm:p-2 rounded-lg transition-colors ${
                    isDarkMode
                      ? 'bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white'
                      : 'bg-white hover:bg-gray-50 text-gray-600 hover:text-gray-900 shadow-sm'
                  }`}
                  aria-label="View source on GitHub"
                  title="Star on GitHub"
                >
                  <GithubMark className="w-4 h-4 sm:w-5 sm:h-5" />
                </a>

                <button
                  onClick={handleDarkModeToggle}
                  className={`p-1.5 sm:p-2 rounded-lg transition-colors ${
                    isDarkMode
                      ? 'bg-gray-700 hover:bg-gray-600 text-yellow-400'
                      : 'bg-white hover:bg-gray-50 text-gray-700 shadow-sm'
                  }`}
                  aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                  {isDarkMode
                    ? <Sun className="w-4 h-4 sm:w-5 sm:h-5" />
                    : <Moon className="w-4 h-4 sm:w-5 sm:h-5" />}
                </button>

                {/* 언어 버튼: md(768px)부터 국기+코드 텍스트 표시 */}
                <div className="relative" ref={languageMenuRef}>
                  <button
                    onClick={() => setShowLanguageMenu(!showLanguageMenu)}
                    className={`flex items-center gap-1 md:gap-2 px-2 md:px-3 py-1.5 sm:py-2 rounded-lg transition-colors ${
                      isDarkMode
                        ? 'bg-gray-700 hover:bg-gray-600 text-gray-200'
                        : 'bg-white hover:bg-gray-50 text-gray-700 shadow-sm'
                    }`}
                    aria-label={`Current language: ${currentLanguageOption.name}. Click to change language`}
                    aria-expanded={showLanguageMenu}
                    aria-haspopup="true"
                  >
                    <Globe className="w-4 h-4 sm:w-5 sm:h-5" />
                    <span className="hidden md:inline text-sm font-medium">
                      {currentLanguageOption.flag} {currentLanguageOption.code.toUpperCase()}
                    </span>
                    <ChevronDown className="w-3 h-3 sm:w-4 sm:h-4" />
                  </button>

                  {showLanguageMenu && (
                    <div className={`absolute right-0 mt-2 w-44 rounded-lg shadow-lg z-50 ${
                      isDarkMode ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-200'
                    }`}>
                      {LANGUAGE_OPTIONS.map(option => (
                        <button
                          key={option.code}
                          onClick={() => handleLanguageChange(option.code)}
                          lang={option.code}
                          aria-current={language === option.code ? 'true' : undefined}
                          className={`w-full flex items-center gap-2 px-4 py-2 text-sm transition-colors first:rounded-t-lg last:rounded-b-lg ${
                            language === option.code
                              ? (isDarkMode ? 'bg-purple-900/30 text-purple-300' : 'bg-purple-50 text-purple-700')
                              : (isDarkMode ? 'text-gray-300 hover:bg-gray-700' : 'text-gray-700 hover:bg-gray-50')
                          }`}
                        >
                          <span aria-hidden="true">{option.flag}</span>
                          <span className="flex-1 text-left">{option.name}</span>
                          {/* Colour alone must not be the only cue for the active item. */}
                          {language === option.code && (
                            <Check className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
                          )}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {/* ── 액션 버튼 3개: md부터 1행 합류, lg부터 텍스트 표시 ── */}
                <div className="hidden md:flex items-center gap-1.5">
                  <button
                    onClick={() => setShowAdvisorModal(true)}
                    className={`flex items-center gap-1.5 px-2 lg:px-3 py-2 rounded-lg transition-colors ${
                      isDarkMode
                        ? 'bg-amber-900/30 hover:bg-amber-900/50 text-amber-400 border border-amber-700'
                        : 'bg-amber-100 hover:bg-amber-200 text-amber-700 border border-amber-300'
                    }`}
                    aria-label={t.aiAdvisor || 'AI Architecture Advisor'}
                  >
                    <Sparkles className="w-4 h-4 flex-shrink-0" />
                    <span className="hidden lg:inline text-sm whitespace-nowrap">
                      {t.aiAdvisor || 'AI Advisor'}
                    </span>
                  </button>

                  <button
                    onClick={() => setShowCodeModal(true)}
                    className={`flex items-center gap-1.5 px-2 lg:px-3 py-2 rounded-lg transition-colors ${
                      isDarkMode
                        ? 'bg-green-900/30 hover:bg-green-900/50 text-green-400 border border-green-700'
                        : 'bg-green-100 hover:bg-green-200 text-green-700 border border-green-300'
                    }`}
                    aria-label={t.exportCode || 'Export code'}
                  >
                    <Code className="w-4 h-4 flex-shrink-0" />
                    <span className="hidden lg:inline text-sm whitespace-nowrap">
                      {t.exportCode || 'Export Code'}
                    </span>
                  </button>

                  <button
                    onClick={handleExportImageClick}
                    className={`flex items-center gap-1.5 px-2 lg:px-3 py-2 rounded-lg transition-colors ${
                      isDarkMode
                        ? 'bg-purple-900/30 hover:bg-purple-900/50 text-purple-400 border border-purple-700'
                        : 'bg-purple-100 hover:bg-purple-200 text-purple-700 border border-purple-300'
                    }`}
                    aria-label={t.exportImage}
                  >
                    <Camera className="w-4 h-4 flex-shrink-0" />
                    <span className="hidden lg:inline text-sm whitespace-nowrap">
                      {t.exportImage}
                    </span>
                  </button>
                </div>

              </div>
            </div>

            {/* Row 2: 모바일 전용 액션 버튼 바 (md 이상에서 숨김) */}
            <div className="flex md:hidden gap-2 mt-2 pt-2 border-t border-gray-400/20">
              <button
                onClick={() => setShowAdvisorModal(true)}
                className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-2 rounded-lg transition-colors text-xs font-medium min-w-0 ${
                  isDarkMode
                    ? 'bg-amber-900/30 hover:bg-amber-900/50 text-amber-400 border border-amber-700'
                    : 'bg-amber-100 hover:bg-amber-200 text-amber-700 border border-amber-300'
                }`}
                aria-label={t.aiAdvisor || 'AI Architecture Advisor'}
              >
                <Sparkles className="w-3.5 h-3.5 flex-shrink-0" />
                <span className="truncate">{t.aiAdvisor || 'AI Advisor'}</span>
              </button>

              <button
                onClick={() => setShowCodeModal(true)}
                className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-2 rounded-lg transition-colors text-xs font-medium min-w-0 ${
                  isDarkMode
                    ? 'bg-green-900/30 hover:bg-green-900/50 text-green-400 border border-green-700'
                    : 'bg-green-100 hover:bg-green-200 text-green-700 border border-green-300'
                }`}
                aria-label={t.exportCode || 'Export code'}
              >
                <Code className="w-3.5 h-3.5 flex-shrink-0" />
                <span className="truncate">{t.exportCode || 'Export Code'}</span>
              </button>

              <button
                onClick={handleExportImageClick}
                className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-2 rounded-lg transition-colors text-xs font-medium min-w-0 ${
                  isDarkMode
                    ? 'bg-purple-900/30 hover:bg-purple-900/50 text-purple-400 border border-purple-700'
                    : 'bg-purple-100 hover:bg-purple-200 text-purple-700 border border-purple-300'
                }`}
                aria-label={t.exportImage}
              >
                <Camera className="w-3.5 h-3.5 flex-shrink-0" />
                <span className="truncate">{t.exportImage}</span>
              </button>
            </div>

          </div>
        </header>

        <main id="main-content" tabIndex={-1}>
          {/* Info message */}
          <Alert className={`mb-4 sm:mb-6 text-xs sm:text-sm ${
            isDarkMode
              ? 'bg-blue-900/30 border-blue-700 text-blue-300'
              : 'bg-blue-50 border-blue-200'
          }`}>
            <Info className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`} />
            <AlertDescription className={isDarkMode ? 'text-blue-300' : 'text-blue-800'}>
              {t.alertMessage}
            </AlertDescription>
          </Alert>

          {/* Main content */}
          <div ref={captureAreaRef} data-capture-area>
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 sm:gap-6">
              {/* Left: Layer palette */}
              <div className="lg:col-span-1">
                <Card className={`border ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
                  <CardHeader>
                    <CardTitle className={`flex items-center gap-2 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                      <Layers className="w-5 h-5" />
                      {t.layerPalette}
                    </CardTitle>
                    <CardDescription className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>
                      {t.dragToAdd}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2 max-h-[60vh] lg:max-h-none overflow-y-auto">
                    {Object.entries(LAYER_TYPES).map(([type, config]) => (
                      <div
                        key={type}
                        draggable
                        onDragStart={(e) => handleDragStart(e, type)}
                        onDragEnd={handleDragEnd}
                        onClick={() => addLayer(type)}
                        className={`p-2.5 sm:p-3 rounded-lg border cursor-move hover:cursor-pointer transition-all hover:shadow-md active:scale-95 ${config.color} ${
                          draggedType === type ? 'opacity-50' : ''
                        }`}
                        role="button"
                        tabIndex={0}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            addLayer(type);
                          }
                        }}
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-xl sm:text-2xl" aria-hidden="true">{config.icon}</span>
                          <div className="flex-1 min-w-0">
                            <div className={`font-medium text-xs sm:text-sm ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                              {config.name}
                            </div>
                            <div className={`text-[11px] sm:text-xs truncate ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                              {config.description}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </div>

              {/* Center: Model builder */}
              <div className="lg:col-span-2">
                <Card className={`border ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
                  <CardHeader>
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <CardTitle className={isDarkMode ? 'text-white' : 'text-gray-900'}>
                          {t.modelArchitecture}
                        </CardTitle>
                        <CardDescription className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>
                          {modelLayers.length} {t.layers} • {totalParams.toLocaleString()} {t.parameters}
                        </CardDescription>
                      </div>
                      {modelLayers.length > 0 && (
                        <button
                          onClick={clearAllLayers}
                          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors flex-shrink-0 ${
                            isDarkMode
                              ? 'bg-gray-700 hover:bg-red-900/40 text-gray-300 hover:text-red-300'
                              : 'bg-gray-100 hover:bg-red-50 text-gray-600 hover:text-red-600'
                          }`}
                        >
                          <Eraser className="w-3.5 h-3.5" />
                          {t.clearAll || 'Clear all'}
                        </button>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div
                      onDragOver={handleDragOver}
                      onDrop={handleDrop}
                      id="model-drop-zone"
                      className={`min-h-[400px] rounded-lg border-2 border-dashed p-4 transition-colors ${
                        draggedType
                          ? (isDarkMode ? 'border-purple-500 bg-purple-900/20' : 'border-purple-400 bg-purple-50')
                          : (isDarkMode ? 'border-gray-600 bg-gray-900/50' : 'border-gray-300 bg-gray-50/50')
                      }`}
                    >
                      {modelLayers.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-full min-h-[300px] text-center">
                          <Plus className={`w-16 h-16 mb-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`} />
                          <p className={`text-lg font-medium ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                            {t.dropHere}
                          </p>
                          <p className={`text-sm mt-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                            {t.yourModel}
                          </p>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          {modelLayers.map((layer, index) => {
                            const config = LAYER_TYPES[layer.type];
                            if (!config) return null;
                            const layerParams = safeCalculate(config, layer.params);

                            return (
                              <div
                                key={layer.id}
                                draggable
                                onDragStart={(e) => handleLayerDragStart(e, index)}
                                onDragOver={(e) => handleLayerDragOver(e, index)}
                                onDragEnd={handleLayerDragEnd}
                                className={`p-3 sm:p-4 rounded-lg border ${config.color} ${
                                  draggedIndex === index ? 'opacity-50' : ''
                                }`}
                              >
                                <div className="flex items-start gap-2 sm:gap-3">
                                  {/* Reorder controls. Dragging is unavailable on
                                      touch and from the keyboard, so these are the
                                      only reachable way to change the order. */}
                                  <div className="flex flex-col items-center flex-shrink-0 pt-0.5">
                                    <button
                                      type="button"
                                      onClick={() => moveLayer(index, -1)}
                                      disabled={index === 0}
                                      aria-label={`${t.moveUp}: ${config.name}`}
                                      className={`rounded p-0.5 transition-colors disabled:opacity-25 ${
                                        isDarkMode
                                          ? 'text-gray-400 hover:bg-gray-700/70 hover:text-gray-100'
                                          : 'text-gray-500 hover:bg-white/70 hover:text-gray-900'
                                      }`}
                                    >
                                      <ChevronUp className="w-3.5 h-3.5" />
                                    </button>

                                    <GripVertical
                                      className={`w-4 h-4 cursor-move ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}
                                      aria-hidden="true"
                                    />

                                    <button
                                      type="button"
                                      onClick={() => moveLayer(index, 1)}
                                      disabled={index === modelLayers.length - 1}
                                      aria-label={`${t.moveDown}: ${config.name}`}
                                      className={`rounded p-0.5 transition-colors disabled:opacity-25 ${
                                        isDarkMode
                                          ? 'text-gray-400 hover:bg-gray-700/70 hover:text-gray-100'
                                          : 'text-gray-500 hover:bg-white/70 hover:text-gray-900'
                                      }`}
                                    >
                                      <ChevronDown className="w-3.5 h-3.5" />
                                    </button>
                                  </div>
                                  <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between mb-2">
                                      <div className="flex items-center gap-2 flex-1 min-w-0">
                                        <span className="text-lg sm:text-xl flex-shrink-0" aria-hidden="true">{config.icon}</span>
                                        <div className="min-w-0">
                                          <div className={`font-semibold text-sm sm:text-base truncate ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                                            {config.name}
                                          </div>
                                          <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                                            {layerParams.toLocaleString()} {t.parameters}
                                          </div>
                                        </div>
                                      </div>
                                      <button
                                        onClick={() => deleteLayer(layer.id)}
                                        className={`p-1.5 rounded transition-colors ${
                                          isDarkMode
                                            ? 'hover:bg-red-900/30 text-red-400'
                                            : 'hover:bg-red-100 text-red-600'
                                        }`}
                                        aria-label={`Delete ${config.name} layer ${index + 1}`}
                                      >
                                        <Trash2 className="w-4 h-4" />
                                      </button>
                                    </div>

                                    {/* Dimension mismatch against the preceding layer */}
                                    {(() => {
                                      const issue = dimensionIssues.get(index);
                                      if (!issue) return null;
                                      const field = config.fields.find(f => f.key === issue.field);
                                      const text = (t.dimMismatch || 'Set {field} to {n} to match the previous layer')
                                        .replace('{field}', field?.label || issue.field)
                                        .replace('{n}', String(issue.expected));
                                      return (
                                        // Deliberately not a live region: it sits beside the field
                                        // it describes, and one announcement per keystroke per
                                        // mismatched layer would drown out everything else.
                                        <p
                                          className={`mt-2 flex items-start gap-1.5 rounded px-2 py-1 text-[11px] leading-snug ${
                                            isDarkMode
                                              ? 'bg-amber-950/60 text-amber-200'
                                              : 'bg-amber-100 text-amber-900'
                                          }`}
                                        >
                                          <TriangleAlert className="w-3 h-3 mt-0.5 shrink-0" aria-hidden="true" />
                                          <span>{text}</span>
                                        </p>
                                      );
                                    })()}

                                    {/* Parameter controls */}
                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-3">
                                      {config.fields.map((field) => {
                                        const inputId = `${layer.id}-${field.key}`;
                                        return (
                                          <div key={field.key} className="space-y-1">
                                            <label
                                              htmlFor={inputId}
                                              className={`text-xs font-medium block ${
                                                isDarkMode ? 'text-gray-300' : 'text-gray-700'
                                              }`}
                                            >
                                              {field.label}
                                            </label>
                                            {field.type === 'select' ? (
                                              <select
                                                id={inputId}
                                                value={layer.params[field.key]}
                                                onChange={(e) => updateLayerParam(layer.id, field, Number(e.target.value))}
                                                className={`w-full px-3 py-2 text-sm border rounded-lg ${
                                                  isDarkMode
                                                    ? 'bg-gray-700 border-gray-600 text-white'
                                                    : 'border-gray-300 bg-white'
                                                }`}
                                              >
                                                {field.options.map(opt => (
                                                  <option key={opt} value={opt}>{opt}</option>
                                                ))}
                                              </select>
                                            ) : field.type === 'checkbox' ? (
                                              <div className="flex items-center h-10">
                                                <input
                                                  id={inputId}
                                                  type="checkbox"
                                                  checked={!!layer.params[field.key]}
                                                  onChange={(e) => updateLayerParam(layer.id, field, e.target.checked)}
                                                  className="w-5 h-5"
                                                />
                                              </div>
                                            ) : (
                                              <input
                                                id={inputId}
                                                type="number"
                                                value={layer.params[field.key]}
                                                onChange={(e) => updateLayerParam(layer.id, field, e.target.value)}
                                                onBlur={() => normalizeLayerParam(layer.id, field)}
                                                step={field.step || 1}
                                                min={field.min}
                                                max={field.max}
                                                inputMode={field.step && field.step < 1 ? 'decimal' : 'numeric'}
                                                className={`w-full px-3 py-2 text-sm border rounded-lg ${
                                                  isDarkMode
                                                    ? 'bg-gray-700 border-gray-600 text-white'
                                                    : 'border-gray-300 bg-white'
                                                }`}
                                              />
                                            )}
                                          </div>
                                        );
                                      })}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Right: Calculation results */}
              <div className="col-span-1">
                <Card className={`md:sticky md:top-8 border ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
                  <CardHeader>
                    <CardTitle className={isDarkMode ? 'text-white' : 'text-gray-900'}>{t.modelSummary}</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 sm:space-y-4">
                    {/* Total Parameters */}
                    <div className={`rounded-lg p-3 sm:p-4 border ${
                      isDarkMode ? 'bg-purple-900/30 border-purple-700' : 'bg-purple-50 border-purple-200'
                    }`}>
                      <p className={`text-xs sm:text-sm mb-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{t.totalParameters}</p>
                      <p className={`text-2xl sm:text-3xl font-bold ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`}>
                        {totalParams.toLocaleString()}
                      </p>
                    </div>

                    {/* Model Size */}
                    <div className={`rounded-lg p-3 sm:p-4 border ${
                      isDarkMode ? 'bg-blue-900/30 border-blue-700' : 'bg-blue-50 border-blue-200'
                    }`}>
                      <p className={`text-xs sm:text-sm mb-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{t.modelSize}</p>
                      <p className={`text-xl sm:text-2xl font-bold ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>
                        {formatBytes(modelSizeBytes)}
                      </p>
                    </div>

                    {/* Total FLOPs */}
                    <div className={`rounded-lg p-3 sm:p-4 border ${
                      isDarkMode ? 'bg-orange-900/30 border-orange-700' : 'bg-orange-50 border-orange-200'
                    }`}>
                      <p className={`text-xs sm:text-sm mb-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{t.totalFLOPs || 'Total FLOPs'}</p>
                      <p className={`text-xl sm:text-2xl font-bold ${isDarkMode ? 'text-orange-400' : 'text-orange-600'}`}>
                        {formatNumber(totalFLOPs)}
                      </p>
                      <p className={`mt-1.5 text-[11px] leading-snug ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        {t.flopsNote}
                      </p>
                    </div>

                    {/* Memory Estimation */}
                    <div className={`rounded-lg p-3 sm:p-4 border ${
                      isDarkMode ? 'bg-cyan-900/30 border-cyan-700' : 'bg-cyan-50 border-cyan-200'
                    }`}>
                      <div className="flex items-center justify-between mb-2">
                        <p className={`text-xs sm:text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{t.memoryEstimation || 'Memory'}</p>
                        <div className="flex gap-1" role="group" aria-label={t.memoryEstimation || 'Memory mode'}>
                          <button
                            onClick={() => setMemoryMode('inference')}
                            aria-pressed={memoryMode === 'inference'}
                            className={`px-2 py-0.5 text-xs rounded transition-colors ${
                              memoryMode === 'inference'
                                ? (isDarkMode ? 'bg-cyan-700 text-white' : 'bg-cyan-600 text-white')
                                : (isDarkMode ? 'bg-gray-700 text-gray-400' : 'bg-gray-200 text-gray-600')
                            }`}
                          >
                            {t.inferenceMode || 'Inference'}
                          </button>
                          <button
                            onClick={() => setMemoryMode('training')}
                            aria-pressed={memoryMode === 'training'}
                            className={`px-2 py-0.5 text-xs rounded transition-colors ${
                              memoryMode === 'training'
                                ? (isDarkMode ? 'bg-cyan-700 text-white' : 'bg-cyan-600 text-white')
                                : (isDarkMode ? 'bg-gray-700 text-gray-400' : 'bg-gray-200 text-gray-600')
                            }`}
                          >
                            {t.trainingMode || 'Training'}
                          </button>
                        </div>
                      </div>
                      <p className={`text-xl sm:text-2xl font-bold ${isDarkMode ? 'text-cyan-400' : 'text-cyan-600'}`}>
                        {formatBytes(memoryEstimate)}
                      </p>
                      <label htmlFor="precision-select" className="sr-only">{t.precision || 'Precision'}</label>
                      <select
                        id="precision-select"
                        value={precision}
                        onChange={(e) => setPrecision(e.target.value)}
                        disabled={memoryMode === 'training'}
                        className={`mt-2 w-full px-2 py-1 text-xs rounded border disabled:opacity-50 ${
                          isDarkMode
                            ? 'bg-gray-700 border-gray-600 text-gray-200'
                            : 'bg-white border-gray-300 text-gray-700'
                        }`}
                      >
                        <option value="fp32">{t.fp32 || 'FP32 (32-bit)'}</option>
                        <option value="fp16">{t.fp16 || 'FP16 (16-bit)'}</option>
                        <option value="bf16">{t.bf16 || 'BF16 (16-bit)'}</option>
                        <option value="int8">{t.int8 || 'INT8 (8-bit)'}</option>
                      </select>
                      <p className={`mt-1.5 text-[11px] leading-snug ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        {memoryMode === 'training' ? t.memoryNoteTraining : t.memoryNoteInference}
                      </p>
                    </div>

                    {/* Number of Layers */}
                    <div className={`rounded-lg p-3 sm:p-4 border ${
                      isDarkMode ? 'bg-green-900/30 border-green-700' : 'bg-green-50 border-green-200'
                    }`}>
                      <p className={`text-xs sm:text-sm mb-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{t.numberOfLayers}</p>
                      <p className={`text-xl sm:text-2xl font-bold ${isDarkMode ? 'text-green-400' : 'text-green-600'}`}>
                        {modelLayers.length}
                      </p>
                    </div>

                    {modelLayers.length > 0 && (
                      <div className={`border-t pt-4 ${isDarkMode ? 'border-gray-600' : 'border-gray-200'}`}>
                        <p className={`text-sm font-semibold mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{t.layerDistribution}</p>
                        <div className="space-y-2">
                          {modelLayers.map((layer, idx) => {
                            const config = LAYER_TYPES[layer.type];
                            if (!config) return null;
                            const layerParams = safeCalculate(config, layer.params);
                            const percentage = totalParams > 0 ? ((layerParams / totalParams) * 100).toFixed(1) : '0.0';

                            return (
                              <div key={layer.id} className="text-xs">
                                <div className="flex justify-between mb-1">
                                  <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>
                                    <span aria-hidden="true">{config.icon}</span> {t.layer} {idx + 1}
                                  </span>
                                  <span className={`font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{percentage}%</span>
                                </div>
                                <div className={`w-full rounded-full h-2 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>
                                  <div
                                    className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full transition-all"
                                    style={{ width: `${percentage}%` }}
                                  />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </main>

        {/* Donation modal */}
        {showDonationModal && (
          <Modal
            isDarkMode={isDarkMode}
            labelledBy="donation-title"
            onClose={() => setShowDonationModal(false)}
            className="w-full max-w-md p-5 sm:p-6"
          >
            <>
              <button
                onClick={() => setShowDonationModal(false)}
                className={`absolute top-3 right-3 sm:top-4 sm:right-4 transition-colors ${
                  isDarkMode ? 'text-gray-400 hover:text-gray-200' : 'text-gray-500 hover:text-gray-700'
                }`}
                aria-label={t.closeModal || 'Close'}
              >
                <X className="w-5 h-5 sm:w-6 sm:h-6" />
              </button>

              <div className="text-center mb-5 sm:mb-6">
                <h2 id="donation-title" className={`text-xl sm:text-2xl font-bold mb-2 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                  {t.enjoyingLayerCal}
                </h2>
                <p className={`text-sm sm:text-base ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  {t.supportMessage}
                </p>
              </div>

              <div className="space-y-3">
                <a
                  href="https://buymeacoffee.com/layercal"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block w-full px-5 sm:px-6 py-3 bg-gradient-to-r from-yellow-400 to-orange-400 hover:from-yellow-500 hover:to-orange-500 text-white font-semibold rounded-lg transition-all shadow-md hover:shadow-lg text-center text-sm sm:text-base active:scale-95"
                >
                  {t.buyMeCoffee}
                </a>

                <button
                  onClick={handleExportImage}
                  className={`block w-full px-5 sm:px-6 py-3 font-semibold rounded-lg transition-all text-center text-sm sm:text-base active:scale-95 ${
                    isDarkMode
                      ? 'bg-gray-700 hover:bg-gray-600 text-gray-200'
                      : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                  }`}
                >
                  {t.noThanksDownload}
                </button>
              </div>
            </>
          </Modal>
        )}

        {/* Code Export Modal */}
        {showCodeModal && (
          <Modal
            isDarkMode={isDarkMode}
            labelledBy="code-modal-title"
            onClose={() => setShowCodeModal(false)}
            className="w-full max-w-2xl max-h-[80vh] flex flex-col"
          >
            <>
              {/* Header */}
              <div className={`flex items-center justify-between p-4 border-b ${
                isDarkMode ? 'border-gray-700' : 'border-gray-200'
              }`}>
                <div>
                  <h2 id="code-modal-title" className={`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    {t.codeExportTitle || 'Export Code'}
                  </h2>
                  <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    {t.codeExportDesc || 'Copy the generated code for your framework'}
                  </p>
                </div>
                <button
                  onClick={() => setShowCodeModal(false)}
                  aria-label={t.closeModal || 'Close'}
                  className={`p-2 rounded-lg transition-colors ${
                    isDarkMode
                      ? 'text-gray-400 hover:text-gray-200 hover:bg-gray-700'
                      : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Framework tabs */}
              <div className={`flex border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`} role="tablist">
                {['pytorch', 'tensorflow', 'jax'].map((fw) => (
                  <button
                    key={fw}
                    role="tab"
                    aria-selected={selectedFramework === fw}
                    onClick={() => setSelectedFramework(fw)}
                    className={`flex-1 py-3 text-sm font-medium transition-colors ${
                      selectedFramework === fw
                        ? (isDarkMode
                            ? 'text-green-400 border-b-2 border-green-400'
                            : 'text-green-600 border-b-2 border-green-600')
                        : (isDarkMode
                            ? 'text-gray-400 hover:text-gray-200'
                            : 'text-gray-600 hover:text-gray-900')
                    }`}
                  >
                    {fw === 'pytorch' ? 'PyTorch' : fw === 'tensorflow' ? 'TensorFlow' : 'JAX'}
                  </button>
                ))}
              </div>

              {/* Code display */}
              <div className="flex-1 overflow-auto p-4">
                <pre className={`text-xs sm:text-sm p-4 rounded-lg overflow-x-auto ${
                  isDarkMode ? 'bg-gray-900 text-gray-300' : 'bg-gray-50 text-gray-800'
                }`}>
                  <code>{generatedCode}</code>
                </pre>
              </div>

              {/* Buttons with donation */}
              <div className={`p-4 border-t space-y-3 ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <div className="text-center mb-2">
                  <p className={`text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                    {t.supportMessage}
                  </p>
                </div>

                <a
                  href="https://buymeacoffee.com/layercal"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block w-full px-5 py-3 bg-gradient-to-r from-yellow-400 to-orange-400 hover:from-yellow-500 hover:to-orange-500 text-white font-semibold rounded-lg transition-all shadow-md hover:shadow-lg text-center text-sm active:scale-95"
                >
                  {t.buyMeCoffee}
                </a>

                <button
                  onClick={handleCopyCode}
                  className={`w-full py-3 rounded-lg font-semibold transition-all text-sm ${
                    codeCopied
                      ? (isDarkMode ? 'bg-green-700 text-white' : 'bg-green-500 text-white')
                      : (isDarkMode
                          ? 'bg-gray-700 hover:bg-gray-600 text-gray-200'
                          : 'bg-gray-200 hover:bg-gray-300 text-gray-800')
                  }`}
                >
                  {codeCopied ? (t.codeCopied || 'Copied!') : (t.copyCode || 'Copy Code')}
                </button>
              </div>
            </>
          </Modal>
        )}

        {/* AI Architecture Advisor Modal */}
        {showAdvisorModal && (
          <AIAdvisor
            isDarkMode={isDarkMode}
            t={t}
            canvasHasLayers={modelLayers.length > 0}
            onApply={addLayersFromAdvisor}
            onClose={() => setShowAdvisorModal(false)}
          />
        )}

        {/* Toast */}
        {toast && (
          <div
            role="status"
            aria-live="polite"
            className="fixed inset-x-0 bottom-4 z-[60] flex justify-center px-4 pointer-events-none"
          >
            <div className={`pointer-events-auto flex items-center gap-3 max-w-md rounded-lg px-4 py-3 text-sm shadow-lg border ${
              isDarkMode
                ? 'bg-gray-800 border-gray-700 text-gray-100'
                : 'bg-white border-gray-200 text-gray-800'
            }`}>
              <span className="flex-1">{toast.message}</span>

              {toast.action && (
                <button
                  onClick={() => { toast.action.onClick(); setToast(null); }}
                  className={`font-semibold whitespace-nowrap rounded px-2 py-1 transition-colors ${
                    isDarkMode
                      ? 'text-purple-300 hover:bg-purple-900/40'
                      : 'text-purple-700 hover:bg-purple-50'
                  }`}
                >
                  {toast.action.label}
                </button>
              )}

              <button
                onClick={() => setToast(null)}
                aria-label={t.closeModal || 'Close'}
                className={isDarkMode ? 'text-gray-400 hover:text-gray-200' : 'text-gray-500 hover:text-gray-700'}
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className={`mt-6 sm:mt-8 text-center text-xs space-y-1.5 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          <p>{t.calculationNote}</p>
          <p>
            © {copyrightYears()} LayerCal
            <span aria-hidden="true"> · </span>
            <a
              href="https://github.com/chanjoongx/layercal/blob/main/LICENSE"
              target="_blank"
              rel="noopener noreferrer"
              className="underline hover:no-underline"
            >
              MIT License
            </a>
            <span aria-hidden="true"> · </span>
            layercal.com
          </p>
        </footer>
      </div>
    </div>
  );
}
