import React, { useState, useRef, useMemo, useCallback, useEffect } from 'react';
import {
  Trash2, GripVertical, Info, Layers, Moon, Sun, Globe, ChevronDown, ChevronUp,
  Camera, X, Mail, Code, Sparkles, Eraser, Check, TriangleAlert,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import Modal from '@/components/ui/modal';
import { TRANSLATIONS, LANGUAGE_OPTIONS, SUPPORTED_LANGUAGES } from '@/config/translations';
import { getLayerTypes, formatNumber, calculateMemory, formatBytes, LAYER_TYPE_IDS, DEFAULT_LAYER_PARAMS } from '@/config/layerTypes';
import { safeLocalStorage, detectSystemDarkMode, resolveInitialLanguage } from '@/utils/localStorage';
import { exportToImage, validateExportElement } from '@/utils/imageExport';
import { validateModelDimensions } from '@/utils/modelValidation';
import { generatePyTorchCode, generateTensorFlowCode, generateJAXCode } from '@/utils/codeGenerator';
import { useAnimatedNumber } from '@/utils/useAnimatedNumber';
import { paletteStyle, paintFor } from '@/viz/palette';
import { wireAppendedLayer } from '@/utils/layerWiring';
import AIAdvisor from '@/components/AIAdvisor';
import ModelViewer from '@/components/ModelViewer';

/**
 * LayerCal: deep learning parameter calculator, live 3D architecture viewer
 * and AI architecture advisor.
 *
 * State that survives a reload: the model itself, language, dark mode, and the
 * AI Advisor's provider/key (the last of those lives in AIAdvisor).
 *
 * The 3D panel and the layer list are one interface, not two views of the same
 * data: selecting in either reflects in the other, and both read the same
 * scene graph.
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
  const [selectedLayerId, setSelectedLayerId] = useState(null);

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
  const layerCardRefs = useRef(new Map());
  // The viewer parks a synchronous render function here, so PNG export can
  // force a frame before html2canvas reads the canvas back.
  const captureRenderRef = useRef(null);

  const t = TRANSLATIONS[language] || TRANSLATIONS.en;

  // Language only. The table carries no colour, so a theme toggle no longer
  // rebuilds it - and therefore no longer rebuilds the whole 3D scene.
  const LAYER_TYPES = useMemo(() => getLayerTypes(t), [t]);

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

  // Escape closes the language menu, then clears the 3D selection. Dialogs
  // handle their own Escape via <Modal>.
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key !== 'Escape') return;
      if (showLanguageMenu) setShowLanguageMenu(false);
      else if (selectedLayerId) setSelectedLayerId(null);
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [showLanguageMenu, selectedLayerId]);

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

  /**
   * Appending wires the new layer's input to whatever the stack currently
   * emits. Pasting fixed defaults meant the second layer anyone added was
   * already mismatched, so building a model by clicking produced warnings and
   * code that would not run.
   */
  const addLayer = useCallback((type) => {
    if (!LAYER_TYPES[type]) return;
    const id = nextLayerId();
    setModelLayers(prev => [...prev, {
      id,
      type,
      params: wireAppendedLayer(type, prev, LAYER_TYPES),
    }]);
    setSelectedLayerId(id);
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
    setSelectedLayerId(null);
    setShowAdvisorModal(false);
  }, [LAYER_TYPES]);

  const deleteLayer = useCallback((id) => {
    setModelLayers(prev => prev.filter(layer => layer.id !== id));
    setSelectedLayerId(prev => (prev === id ? null : prev));
  }, []);

  /**
   * Clears immediately and offers an undo, rather than blocking on a native
   * confirm. Reversible beats interruptive for an action this cheap to redo.
   */
  const clearAllLayers = useCallback(() => {
    if (modelLayers.length === 0) return;
    const snapshot = modelLayers;
    setModelLayers([]);
    setSelectedLayerId(null);
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

  const modelSizeBytes = totalParams * 4; // FP32 reference size

  const animatedParams = useAnimatedNumber(totalParams);
  const animatedFlops = useAnimatedNumber(totalFLOPs);
  const animatedSize = useAnimatedNumber(modelSizeBytes);
  const animatedMemory = useAnimatedNumber(memoryEstimate);

  // ── Selection ────────────────────────────────────
  const focusLayerCard = useCallback((id) => {
    const el = layerCardRefs.current.get(id);
    el?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, []);

  const handleSelectFromCanvas = useCallback((id) => {
    setSelectedLayerId(id);
  }, []);

  const handleSelectCard = useCallback((id) => {
    setSelectedLayerId(prev => (prev === id ? null : id));
  }, []);

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

    // html2canvas copies the canvas' current pixels rather than re-running the
    // GPU, so a paused or offscreen viewer would export as an empty rectangle.
    captureRenderRef.current?.();

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

  // The summary accents come from the one colour table rather than being three
  // more hexes that would drift away from the layers they describe.
  const accentFor = (type) => {
    const paint = paintFor(type);
    return isDarkMode ? paint.hexDark : paint.hex;
  };

  const iconButton = 'press rounded-lg border border-border bg-surface p-2 text-muted-foreground shadow-sm hover:border-border-strong hover:text-foreground';
  const fieldClass = 'w-full rounded-md border border-input bg-surface px-3 py-2 text-sm text-foreground transition-colors focus:border-accent';

  return (
    <div className="min-h-screen">
      <a href="#main-content" className="skip-link">
        {t.skipToContent || 'Skip to content'}
      </a>

      <div className="container mx-auto max-w-[92rem] px-3 py-4 sm:px-5 sm:py-6">
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
        <header className="mb-4 sm:mb-5">
          <div className="panel px-3 py-2.5 sm:px-4">

            {/* Row 1: 항상 표시 */}
            <div className="flex items-center justify-between gap-2">

              {/* 왼쪽: 로고 + 타이틀. min-w-0 필수 */}
              <div className="flex min-w-0 items-center gap-2.5">
                <img
                  src="/calculator-icon.svg"
                  alt=""
                  width="44"
                  height="44"
                  className="h-9 w-9 flex-shrink-0 md:h-11 md:w-11"
                />
                <div className="min-w-0">
                  <h1 className="truncate text-xl font-bold tracking-tight text-foreground md:text-2xl">
                    {t.title}
                  </h1>
                  <p className="hidden truncate text-xs text-muted-foreground lg:block">
                    {t.subtitle}
                  </p>
                </div>
              </div>

              {/* 오른쪽: 버튼 묶음. flex-shrink-0, 타이틀이 truncate로 양보 */}
              <div className="flex flex-shrink-0 items-center gap-1.5">

                <a
                  href="mailto:contact@layercal.com"
                  className={iconButton}
                  aria-label="Contact us via email"
                  title="contact@layercal.com"
                >
                  <Mail className="h-4 w-4" />
                </a>

                <a
                  href="https://github.com/chanjoongx/layercal"
                  target="_blank"
                  rel="noopener noreferrer"
                  className={iconButton}
                  aria-label="View source on GitHub"
                  title="Star on GitHub"
                >
                  <GithubMark className="h-4 w-4" />
                </a>

                <button
                  onClick={handleDarkModeToggle}
                  className={iconButton}
                  aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                  {isDarkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                </button>

                {/* 언어 버튼: md(768px)부터 국기+코드 텍스트 표시 */}
                <div className="relative" ref={languageMenuRef}>
                  <button
                    onClick={() => setShowLanguageMenu(!showLanguageMenu)}
                    className="press flex items-center gap-1 rounded-lg border border-border bg-surface px-2 py-2 text-muted-foreground shadow-sm hover:border-border-strong hover:text-foreground md:gap-2 md:px-3"
                    aria-label={`Current language: ${currentLanguageOption.name}. Click to change language`}
                    aria-expanded={showLanguageMenu}
                    aria-haspopup="true"
                  >
                    <Globe className="h-4 w-4" />
                    <span className="hidden text-sm font-medium md:inline">
                      {currentLanguageOption.flag} {currentLanguageOption.code.toUpperCase()}
                    </span>
                    <ChevronDown className="h-3 w-3" />
                  </button>

                  {showLanguageMenu && (
                    <div className="absolute right-0 z-50 mt-2 w-44 animate-slide-up overflow-hidden rounded-lg border border-border bg-surface shadow-lg">
                      {LANGUAGE_OPTIONS.map(option => (
                        <button
                          key={option.code}
                          onClick={() => handleLanguageChange(option.code)}
                          lang={option.code}
                          aria-current={language === option.code ? 'true' : undefined}
                          className={`flex w-full items-center gap-2 px-4 py-2 text-sm transition-colors ${
                            language === option.code
                              ? 'bg-accent-soft text-accent'
                              : 'text-foreground hover:bg-muted'
                          }`}
                        >
                          <span aria-hidden="true">{option.flag}</span>
                          <span className="flex-1 text-left">{option.name}</span>
                          {/* Colour alone must not be the only cue for the active item. */}
                          {language === option.code && (
                            <Check className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                          )}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {/* ── 액션 버튼 3개: md부터 1행 합류, lg부터 텍스트 표시 ── */}
                <div className="hidden items-center gap-1.5 md:flex">
                  <ActionButton
                    onClick={() => setShowAdvisorModal(true)}
                    icon={Sparkles}
                    label={t.aiAdvisor || 'AI Advisor'}
                    primary
                  />
                  <ActionButton
                    onClick={() => setShowCodeModal(true)}
                    icon={Code}
                    label={t.exportCode || 'Export Code'}
                  />
                  <ActionButton
                    onClick={handleExportImageClick}
                    icon={Camera}
                    label={t.exportImage}
                  />
                </div>

              </div>
            </div>

            {/* Row 2: 모바일 전용 액션 버튼 바 (md 이상에서 숨김) */}
            <div className="mt-2 flex gap-2 border-t border-border pt-2 md:hidden">
              <ActionButton
                onClick={() => setShowAdvisorModal(true)}
                icon={Sparkles}
                label={t.aiAdvisor || 'AI Advisor'}
                primary
                compact
              />
              <ActionButton
                onClick={() => setShowCodeModal(true)}
                icon={Code}
                label={t.exportCode || 'Export Code'}
                compact
              />
              <ActionButton
                onClick={handleExportImageClick}
                icon={Camera}
                label={t.exportImage}
                compact
              />
            </div>

          </div>
        </header>

        <main id="main-content" tabIndex={-1}>
          <div ref={captureAreaRef} data-capture-area className="space-y-4 sm:space-y-5">

            {/* Hero: the live 3D architecture */}
            <ModelViewer
              layers={modelLayers}
              layerTypes={LAYER_TYPES}
              issues={dimensionIssues}
              isDarkMode={isDarkMode}
              t={t}
              selectedId={selectedLayerId}
              onSelect={handleSelectFromCanvas}
              onFocusLayer={focusLayerCard}
              captureRef={captureRenderRef}
            />

            <div className="grid grid-cols-1 gap-4 sm:gap-5 lg:grid-cols-4">
              {/* Left: Layer palette */}
              <div className="lg:col-span-1">
                <Card className="panel border-0">
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center gap-2 text-lg text-foreground">
                      <Layers className="h-4 w-4 text-accent" />
                      {t.layerPalette}
                    </CardTitle>
                    <CardDescription>{t.dragToAdd}</CardDescription>
                  </CardHeader>
                  <CardContent className="scroll-slim max-h-[52vh] space-y-1.5 overflow-y-auto lg:max-h-none">
                    {Object.entries(LAYER_TYPES).map(([type, config]) => (
                      <div
                        key={type}
                        draggable
                        onDragStart={(e) => handleDragStart(e, type)}
                        onDragEnd={handleDragEnd}
                        onClick={() => addLayer(type)}
                        style={paletteStyle(type, isDarkMode)}
                        className={`layer-surface press cursor-grab rounded-md border p-2.5 active:cursor-grabbing ${
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
                        <div className="flex items-center gap-2.5">
                          <span className="layer-dot h-2.5 w-2.5 shrink-0 rounded-[3px]" aria-hidden="true" />
                          <div className="min-w-0 flex-1">
                            <div className="text-xs font-semibold text-foreground sm:text-sm">
                              {config.name}
                            </div>
                            <div className="truncate text-[11px] text-muted-foreground">
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
                <Card className="panel border-0">
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <CardTitle className="text-lg text-foreground">{t.modelArchitecture}</CardTitle>
                        <CardDescription data-numeric>
                          {modelLayers.length} {t.layers} · {totalParams.toLocaleString()} {t.parameters}
                        </CardDescription>
                      </div>
                      {modelLayers.length > 0 && (
                        <button
                          onClick={clearAllLayers}
                          className="press flex flex-shrink-0 items-center gap-1.5 rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs font-medium text-muted-foreground hover:border-danger/40 hover:text-danger"
                        >
                          <Eraser className="h-3.5 w-3.5" />
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
                      className={`scroll-slim min-h-[420px] rounded-lg border-2 border-dashed p-3 transition-colors ${
                        draggedType
                          ? 'dropzone-active border-accent'
                          : 'border-border bg-muted/40'
                      }`}
                    >
                      {modelLayers.length === 0 ? (
                        <div className="flex min-h-[380px] flex-col items-center justify-center text-center">
                          <Layers className="mb-3 h-10 w-10 text-muted-foreground/40" aria-hidden="true" />
                          <p className="text-base font-medium text-foreground">{t.dropHere}</p>
                          <p className="mt-1 text-sm text-muted-foreground">{t.yourModel}</p>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          {modelLayers.map((layer, index) => {
                            const config = LAYER_TYPES[layer.type];
                            if (!config) return null;
                            const layerParams = safeCalculate(config, layer.params);
                            const issue = dimensionIssues.get(index);
                            const selected = selectedLayerId === layer.id;

                            return (
                              <div
                                key={layer.id}
                                ref={(el) => {
                                  if (el) layerCardRefs.current.set(layer.id, el);
                                  else layerCardRefs.current.delete(layer.id);
                                }}
                                draggable
                                onDragStart={(e) => handleLayerDragStart(e, index)}
                                onDragOver={(e) => handleLayerDragOver(e, index)}
                                onDragEnd={handleLayerDragEnd}
                                onClick={() => handleSelectCard(layer.id)}
                                style={paletteStyle(layer.type, isDarkMode)}
                                className={`layer-surface enter-rise rounded-lg border p-3 transition-shadow sm:p-3.5 ${
                                  draggedIndex === index ? 'opacity-50' : ''
                                } ${selected ? 'shadow-md ring-2 ring-accent/70' : ''}`}
                              >
                                <div className="flex items-start gap-2 sm:gap-3">
                                  {/* Reorder controls. Dragging is unavailable on
                                      touch and from the keyboard, so these are the
                                      only reachable way to change the order. */}
                                  <div className="flex flex-shrink-0 flex-col items-center pt-0.5">
                                    <button
                                      type="button"
                                      onClick={(e) => { e.stopPropagation(); moveLayer(index, -1); }}
                                      disabled={index === 0}
                                      aria-label={`${t.moveUp}: ${config.name}`}
                                      className="rounded p-0.5 text-muted-foreground transition-colors hover:bg-surface hover:text-foreground disabled:opacity-25"
                                    >
                                      <ChevronUp className="h-3.5 w-3.5" />
                                    </button>

                                    <GripVertical
                                      className="h-4 w-4 cursor-move text-muted-foreground"
                                      aria-hidden="true"
                                    />

                                    <button
                                      type="button"
                                      onClick={(e) => { e.stopPropagation(); moveLayer(index, 1); }}
                                      disabled={index === modelLayers.length - 1}
                                      aria-label={`${t.moveDown}: ${config.name}`}
                                      className="rounded p-0.5 text-muted-foreground transition-colors hover:bg-surface hover:text-foreground disabled:opacity-25"
                                    >
                                      <ChevronDown className="h-3.5 w-3.5" />
                                    </button>
                                  </div>

                                  <div className="min-w-0 flex-1">
                                    <div className="mb-2 flex items-center justify-between gap-2">
                                      <div className="flex min-w-0 flex-1 items-center gap-2">
                                        <span className="layer-dot h-3 w-3 shrink-0 rounded-[3px]" aria-hidden="true" />
                                        <div className="min-w-0">
                                          <div className="truncate text-sm font-semibold text-foreground sm:text-base">
                                            <span className="mr-1.5 font-mono text-xs text-muted-foreground">{index + 1}</span>
                                            {config.name}
                                          </div>
                                          <div className="text-xs text-muted-foreground" data-numeric>
                                            {layerParams.toLocaleString()} {t.parameters}
                                          </div>
                                        </div>
                                      </div>
                                      <button
                                        onClick={(e) => { e.stopPropagation(); deleteLayer(layer.id); }}
                                        className="press rounded p-1.5 text-muted-foreground transition-colors hover:bg-danger/10 hover:text-danger"
                                        aria-label={`Delete ${config.name} layer ${index + 1}`}
                                      >
                                        <Trash2 className="h-4 w-4" />
                                      </button>
                                    </div>

                                    {/* Dimension mismatch against the preceding layer.
                                        Deliberately not a live region: it sits beside the
                                        field it describes, and one announcement per
                                        keystroke per mismatched layer would drown out
                                        everything else. */}
                                    {issue && (
                                      <p className="mt-2 flex items-start gap-1.5 rounded px-2 py-1 text-[11px] leading-snug text-warning ring-1 ring-inset ring-warning/40">
                                        <TriangleAlert className="mt-0.5 h-3 w-3 shrink-0" aria-hidden="true" />
                                        <span>
                                          {(t.dimMismatch || 'Set {field} to {n} to match the previous layer')
                                            .replace('{field}', config.fields.find(f => f.key === issue.field)?.label || issue.field)
                                            .replace('{n}', String(issue.expected))}
                                        </span>
                                      </p>
                                    )}

                                    {/* Parameter controls */}
                                    <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
                                      {config.fields.map((field) => {
                                        const inputId = `${layer.id}-${field.key}`;
                                        return (
                                          <div key={field.key} className="space-y-1">
                                            <label
                                              htmlFor={inputId}
                                              className="block text-xs font-medium text-muted-foreground"
                                            >
                                              {field.label}
                                            </label>
                                            {field.type === 'select' ? (
                                              <select
                                                id={inputId}
                                                value={layer.params[field.key]}
                                                onClick={(e) => e.stopPropagation()}
                                                onChange={(e) => updateLayerParam(layer.id, field, Number(e.target.value))}
                                                className={fieldClass}
                                              >
                                                {field.options.map(opt => (
                                                  <option key={opt} value={opt}>{opt}</option>
                                                ))}
                                              </select>
                                            ) : field.type === 'checkbox' ? (
                                              <div className="flex h-10 items-center">
                                                <input
                                                  id={inputId}
                                                  type="checkbox"
                                                  checked={!!layer.params[field.key]}
                                                  onClick={(e) => e.stopPropagation()}
                                                  onChange={(e) => updateLayerParam(layer.id, field, e.target.checked)}
                                                  className="h-5 w-5 accent-[hsl(var(--accent))]"
                                                />
                                              </div>
                                            ) : (
                                              <input
                                                id={inputId}
                                                type="number"
                                                value={layer.params[field.key]}
                                                onClick={(e) => e.stopPropagation()}
                                                onChange={(e) => updateLayerParam(layer.id, field, e.target.value)}
                                                onBlur={() => normalizeLayerParam(layer.id, field)}
                                                step={field.step || 1}
                                                min={field.min}
                                                max={field.max}
                                                inputMode={field.step && field.step < 1 ? 'decimal' : 'numeric'}
                                                className={fieldClass}
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
                <Card className="panel border-0 lg:sticky lg:top-5">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg text-foreground">{t.modelSummary}</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <Metric
                      label={t.totalParameters}
                      value={Math.round(animatedParams).toLocaleString()}
                      accent="hsl(var(--accent))"
                      large
                    />

                    <Metric
                      label={t.modelSize}
                      value={formatBytes(animatedSize)}
                      accent={accentFor('linear')}
                    />

                    <Metric
                      label={t.totalFLOPs || 'Total FLOPs'}
                      value={formatNumber(animatedFlops)}
                      accent={accentFor('lstm')}
                      note={t.flopsNote}
                    />

                    {/* Memory */}
                    <div className="metric p-3 pl-4" style={{ '--metric-accent': accentFor('batchnorm') }}>
                      <div className="mb-2 flex items-center justify-between gap-2">
                        <p className="text-xs text-muted-foreground">{t.memoryEstimation || 'Memory'}</p>
                        <div className="flex gap-1" role="group" aria-label={t.memoryEstimation || 'Memory mode'}>
                          {['inference', 'training'].map(mode => (
                            <button
                              key={mode}
                              onClick={() => setMemoryMode(mode)}
                              aria-pressed={memoryMode === mode}
                              className={`rounded px-2 py-0.5 text-xs transition-colors ${
                                memoryMode === mode
                                  ? 'bg-accent text-primary-foreground'
                                  : 'bg-muted text-muted-foreground hover:text-foreground'
                              }`}
                            >
                              {mode === 'inference'
                                ? (t.inferenceMode || 'Inference')
                                : (t.trainingMode || 'Training')}
                            </button>
                          ))}
                        </div>
                      </div>
                      <p className="text-xl font-bold text-foreground sm:text-2xl" data-numeric>
                        {formatBytes(animatedMemory)}
                      </p>
                      <label htmlFor="precision-select" className="sr-only">{t.precision || 'Precision'}</label>
                      <select
                        id="precision-select"
                        value={precision}
                        onChange={(e) => setPrecision(e.target.value)}
                        disabled={memoryMode === 'training'}
                        className="mt-2 w-full rounded border border-input bg-surface px-2 py-1 text-xs text-foreground disabled:opacity-50"
                      >
                        <option value="fp32">{t.fp32 || 'FP32 (32-bit)'}</option>
                        <option value="fp16">{t.fp16 || 'FP16 (16-bit)'}</option>
                        <option value="bf16">{t.bf16 || 'BF16 (16-bit)'}</option>
                        <option value="int8">{t.int8 || 'INT8 (8-bit)'}</option>
                      </select>
                      <p className="mt-1.5 text-[11px] leading-snug text-muted-foreground">
                        {memoryMode === 'training' ? t.memoryNoteTraining : t.memoryNoteInference}
                      </p>
                    </div>

                    <Metric
                      label={t.numberOfLayers}
                      value={String(modelLayers.length)}
                      accent="hsl(var(--positive))"
                    />

                    {modelLayers.length > 0 && (
                      <div className="border-t border-border pt-4">
                        <p className="mb-2 text-sm font-semibold text-foreground">{t.layerDistribution}</p>
                        <div className="space-y-2">
                          {modelLayers.map((layer, idx) => {
                            const config = LAYER_TYPES[layer.type];
                            if (!config) return null;
                            const layerParams = safeCalculate(config, layer.params);
                            const percentage = totalParams > 0 ? (layerParams / totalParams) * 100 : 0;
                            const paint = paintFor(layer.type);

                            return (
                              <button
                                key={layer.id}
                                onClick={() => handleSelectCard(layer.id)}
                                className="block w-full text-left text-xs"
                              >
                                <div className="mb-1 flex justify-between">
                                  <span className="flex items-center gap-1.5 text-muted-foreground">
                                    <span
                                      className="h-2 w-2 rounded-[2px]"
                                      style={{ background: isDarkMode ? paint.hexDark : paint.hex }}
                                      aria-hidden="true"
                                    />
                                    {t.layer} {idx + 1}
                                  </span>
                                  <span className="font-medium text-foreground" data-numeric>
                                    {percentage.toFixed(1)}%
                                  </span>
                                </div>
                                <div className="h-1.5 w-full rounded-full bg-muted">
                                  <div
                                    className="h-1.5 rounded-full transition-[width] duration-slow ease-out"
                                    style={{
                                      width: `${percentage}%`,
                                      background: isDarkMode ? paint.hexDark : paint.hex,
                                    }}
                                  />
                                </div>
                              </button>
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

          {/* Info message */}
          <Alert className="mt-4 border-border bg-surface/70 text-xs text-muted-foreground sm:mt-5 sm:text-sm">
            <Info className="h-4 w-4 text-accent" />
            <AlertDescription>{t.alertMessage}</AlertDescription>
          </Alert>
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
                className="absolute right-3 top-3 text-muted-foreground transition-colors hover:text-foreground sm:right-4 sm:top-4"
                aria-label={t.closeModal || 'Close'}
              >
                <X className="h-5 w-5 sm:h-6 sm:w-6" />
              </button>

              <div className="mb-5 text-center sm:mb-6">
                <h2 id="donation-title" className="mb-2 text-xl font-bold text-foreground sm:text-2xl">
                  {t.enjoyingLayerCal}
                </h2>
                <p className="text-sm text-muted-foreground sm:text-base">
                  {t.supportMessage}
                </p>
              </div>

              <div className="space-y-3">
                <a
                  href="https://buymeacoffee.com/layercal"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="press block w-full rounded-lg bg-gradient-to-r from-amber-400 to-orange-400 px-5 py-3 text-center text-sm font-semibold text-white shadow-md hover:shadow-lg sm:px-6 sm:text-base"
                >
                  {t.buyMeCoffee}
                </a>

                <button
                  onClick={handleExportImage}
                  className="press block w-full rounded-lg bg-muted px-5 py-3 text-center text-sm font-semibold text-foreground hover:bg-border sm:px-6 sm:text-base"
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
            className="flex max-h-[80vh] w-full max-w-2xl flex-col"
          >
            <>
              {/* Header */}
              <div className="flex items-center justify-between border-b border-border p-4">
                <div>
                  <h2 id="code-modal-title" className="text-lg font-bold text-foreground">
                    {t.codeExportTitle || 'Export Code'}
                  </h2>
                  <p className="text-sm text-muted-foreground">
                    {t.codeExportDesc || 'Copy the generated code for your framework'}
                  </p>
                </div>
                <button
                  onClick={() => setShowCodeModal(false)}
                  aria-label={t.closeModal || 'Close'}
                  className="rounded-lg p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              {/* Framework tabs */}
              <div className="flex border-b border-border" role="tablist">
                {['pytorch', 'tensorflow', 'jax'].map((fw) => (
                  <button
                    key={fw}
                    role="tab"
                    aria-selected={selectedFramework === fw}
                    onClick={() => setSelectedFramework(fw)}
                    className={`flex-1 py-3 text-sm font-medium transition-colors ${
                      selectedFramework === fw
                        ? 'border-b-2 border-accent text-accent'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    {fw === 'pytorch' ? 'PyTorch' : fw === 'tensorflow' ? 'TensorFlow' : 'JAX'}
                  </button>
                ))}
              </div>

              {/* Code display */}
              <div className="scroll-slim flex-1 overflow-auto p-4">
                <pre className="overflow-x-auto rounded-lg bg-muted p-4 font-mono text-xs text-foreground sm:text-sm">
                  <code>{generatedCode}</code>
                </pre>
              </div>

              {/* Buttons with donation */}
              <div className="space-y-3 border-t border-border p-4">
                <p className="text-center text-sm text-muted-foreground">
                  {t.supportMessage}
                </p>

                <a
                  href="https://buymeacoffee.com/layercal"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="press block w-full rounded-lg bg-gradient-to-r from-amber-400 to-orange-400 px-5 py-3 text-center text-sm font-semibold text-white shadow-md hover:shadow-lg"
                >
                  {t.buyMeCoffee}
                </a>

                <button
                  onClick={handleCopyCode}
                  className={`press w-full rounded-lg py-3 text-sm font-semibold transition-all ${
                    codeCopied
                      ? 'bg-positive text-white'
                      : 'bg-muted text-foreground hover:bg-border'
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
            layerTypes={LAYER_TYPES}
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
            className="pointer-events-none fixed inset-x-0 bottom-4 z-[60] flex justify-center px-4"
          >
            <div className="pointer-events-auto flex max-w-md animate-slide-up items-center gap-3 rounded-lg border border-border bg-surface px-4 py-3 text-sm text-foreground shadow-lg">
              <span className="flex-1">{toast.message}</span>

              {toast.action && (
                <button
                  onClick={() => { toast.action.onClick(); setToast(null); }}
                  className="whitespace-nowrap rounded px-2 py-1 font-semibold text-accent transition-colors hover:bg-accent-soft"
                >
                  {toast.action.label}
                </button>
              )}

              <button
                onClick={() => setToast(null)}
                aria-label={t.closeModal || 'Close'}
                className="text-muted-foreground hover:text-foreground"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="mt-6 space-y-1.5 text-center text-xs text-muted-foreground sm:mt-8">
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

/** Header action: icon-only from md, icon plus label from lg. */
function ActionButton({ onClick, icon: Icon, label, primary, compact }) {
  const base = primary
    ? 'border-accent/40 bg-accent-soft text-accent hover:bg-accent hover:text-primary-foreground'
    : 'border-border bg-surface text-muted-foreground hover:border-border-strong hover:text-foreground';

  if (compact) {
    return (
      <button
        onClick={onClick}
        aria-label={label}
        className={`press flex min-w-0 flex-1 items-center justify-center gap-1.5 rounded-lg border px-2 py-2 text-xs font-medium ${base}`}
      >
        <Icon className="h-3.5 w-3.5 flex-shrink-0" />
        <span className="truncate">{label}</span>
      </button>
    );
  }

  return (
    <button
      onClick={onClick}
      aria-label={label}
      className={`press flex items-center gap-1.5 rounded-lg border px-2 py-2 shadow-sm lg:px-3 ${base}`}
    >
      <Icon className="h-4 w-4 flex-shrink-0" />
      <span className="hidden whitespace-nowrap text-sm lg:inline">{label}</span>
    </button>
  );
}

/** One readout in the summary column. */
function Metric({ label, value, accent, note, large }) {
  return (
    <div className="metric p-3 pl-4" style={{ '--metric-accent': accent }}>
      <p className="mb-1 text-xs text-muted-foreground">{label}</p>
      <p
        className={`font-bold text-foreground ${large ? 'text-2xl sm:text-3xl' : 'text-xl sm:text-2xl'}`}
        data-numeric
      >
        {value}
      </p>
      {note && <p className="mt-1.5 text-[11px] leading-snug text-muted-foreground">{note}</p>}
    </div>
  );
}
