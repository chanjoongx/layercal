import React, { useState, useRef, useMemo, useCallback } from 'react';
import { Trash2, GripVertical, Plus, Info, Layers, Moon, Sun, Globe, ChevronDown, Camera, X, Mail, Code } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { TRANSLATIONS, LANGUAGE_OPTIONS } from '@/config/translations';
import { getLayerTypes, formatNumber, calculateMemory, formatBytes } from '@/config/layerTypes';
import { safeLocalStorage, detectSystemDarkMode } from '@/utils/localStorage';
import { exportToImage, showExportError, validateExportElement } from '@/utils/imageExport';
import { generatePyTorchCode, generateTensorFlowCode, generateJAXCode } from '@/utils/codeGenerator';

/**
 * LayerCal - Deep Learning Model Parameter Calculator
 * 
 * 🚀 Performance Optimizations:
 * - useMemo: LAYER_TYPES, totalParams, modelSizeMB, totalFLOPs
 * - useCallback: All event handlers and state updaters
 * - Functional setState: Avoids stale closure issues
 * 
 * 📊 14 Layer Types Supported:
 * Embedding, Linear, Conv2D, LSTM, Transformer, BatchNorm, Dropout,
 * MaxPool2D, AvgPool2D, LayerNorm, GRU, Attention, ReLU, Softmax
 */

export default function LayerCal() {
  // Initialize state (restore from localStorage)
  const [modelLayers, setModelLayers] = useState([]);
  const [draggedType, setDraggedType] = useState(null);
  const [draggedIndex, setDraggedIndex] = useState(null);
  const [showLanguageMenu, setShowLanguageMenu] = useState(false);
  const [showDonationModal, setShowDonationModal] = useState(false);
  
  // NEW: Phase 1 states
  const [showCodeModal, setShowCodeModal] = useState(false);
  const [selectedFramework, setSelectedFramework] = useState('pytorch');
  const [codeCopied, setCodeCopied] = useState(false);
  const [memoryMode, setMemoryMode] = useState('inference');
  const [precision, setPrecision] = useState('fp32');
  
  // Dark mode init: localStorage → system preference
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = safeLocalStorage.getItem('layercal-darkmode');
    if (saved !== null) return saved === 'true';
    return detectSystemDarkMode();
  });
  
  // Language init: localStorage → default 'en'
  const [language, setLanguage] = useState(() => 
    safeLocalStorage.getItem('layercal-language', 'en')
  );

  const dragCounter = useRef(0);
  const languageMenuRef = useRef(null);
  const captureAreaRef = useRef(null);

  const t = TRANSLATIONS[language];
  
  // 🚀 Optimization: LAYER_TYPES recalculates only on language/darkMode change
  const LAYER_TYPES = useMemo(() => getLayerTypes(t, isDarkMode), [t, isDarkMode]);

  // Detect clicks outside language menu
  React.useEffect(() => {
    const handleClickOutside = (event) => {
      if (languageMenuRef.current && !languageMenuRef.current.contains(event.target)) {
        setShowLanguageMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Language change handler
  const handleLanguageChange = useCallback((newLang) => {
    setLanguage(newLang);
    safeLocalStorage.setItem('layercal-language', newLang);
    setShowLanguageMenu(false);
  }, []);

  // Dark mode toggle
  const handleDarkModeToggle = useCallback(() => {
    setIsDarkMode(prev => {
      const newMode = !prev;
      safeLocalStorage.setItem('layercal-darkmode', String(newMode));
      return newMode;
    });
  }, []);

  // 🚀 Optimization: Memoize functions with useCallback
  const addLayer = useCallback((type) => {
    const layerConfig = LAYER_TYPES[type];
    const newLayer = {
      id: Date.now() + Math.random(),
      type,
      params: { ...layerConfig.defaultParams }
    };
    setModelLayers(prev => [...prev, newLayer]);
  }, [LAYER_TYPES]);

  const deleteLayer = useCallback((id) => {
    setModelLayers(prev => prev.filter(layer => layer.id !== id));
  }, []);

  const updateLayerParam = useCallback((id, paramKey, value) => {
    setModelLayers(prev => prev.map(layer => {
      if (layer.id !== id) return layer;
      
      // Validate input (min 1, integers only for non-rate params)
      let validatedValue = value;
      if (typeof value === 'number' && paramKey !== 'rate') {
        validatedValue = Math.max(1, Math.floor(value));
      }
      
      return {
        ...layer,
        params: { ...layer.params, [paramKey]: validatedValue }
      };
    }));
  }, []);

  // 🚀 Optimization: Cache calculations with useMemo
  const totalParams = useMemo(() => {
    return modelLayers.reduce((total, layer) => {
      const config = LAYER_TYPES[layer.type];
      return total + config.calculate(layer.params);
    }, 0);
  }, [modelLayers, LAYER_TYPES]);

  const modelSizeMB = useMemo(() => {
    return (totalParams * 4) / (1024 * 1024);
  }, [totalParams]);

  // NEW: FLOPs calculation
  const totalFLOPs = useMemo(() => {
    return modelLayers.reduce((total, layer) => {
      const config = LAYER_TYPES[layer.type];
      if (config.calculateFLOPs) {
        return total + config.calculateFLOPs(layer.params);
      }
      return total;
    }, 0);
  }, [modelLayers, LAYER_TYPES]);

  // NEW: Memory estimation
  const memoryEstimate = useMemo(() => {
    return calculateMemory(totalParams, memoryMode, precision);
  }, [totalParams, memoryMode, precision]);

  // NEW: Code generation
  const generatedCode = useMemo(() => {
    switch (selectedFramework) {
      case 'pytorch':
        return generatePyTorchCode(modelLayers);
      case 'tensorflow':
        return generateTensorFlowCode(modelLayers);
      case 'jax':
        return generateJAXCode(modelLayers);
      default:
        return '';
    }
  }, [modelLayers, selectedFramework]);

  const handleDragStart = useCallback((e, type) => {
    setDraggedType(type);
    e.dataTransfer.effectAllowed = 'copy';
    e.dataTransfer.setData('text/plain', type);
  }, []);

  const handleDragEnd = useCallback(() => {
    setDraggedType(null);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    if (draggedType) {
      addLayer(draggedType);
    }
    setDraggedType(null);
    dragCounter.current = 0;
  }, [draggedType, addLayer]);

  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    dragCounter.current += 1;
  }, []);

  const handleDragLeave = useCallback(() => {
    dragCounter.current -= 1;
  }, []);

  // Reorder layers via drag & drop
  const handleLayerDragStart = useCallback((e, index) => {
    setDraggedIndex(index);
    e.dataTransfer.effectAllowed = 'move';
  }, []);

  const handleLayerDragOver = useCallback((e, index) => {
    e.preventDefault();
    if (draggedIndex === null || draggedIndex === index) return;

    setModelLayers(prev => {
      const newLayers = [...prev];
      const draggedLayer = newLayers[draggedIndex];
      newLayers.splice(draggedIndex, 1);
      newLayers.splice(index, 0, draggedLayer);
      return newLayers;
    });
    setDraggedIndex(index);
  }, [draggedIndex]);

  const handleLayerDragEnd = useCallback(() => {
    setDraggedIndex(null);
  }, []);

  // Image export handler (improved version)
  const handleExportImageClick = useCallback(() => {
    setShowDonationModal(true);
  }, []);

  const handleExportImage = useCallback(async () => {
    setShowDonationModal(false);
    
    const element = captureAreaRef.current;
    
    // Pre-validation check
    const validation = validateExportElement(element);
    if (!validation.valid) {
      showExportError(validation.error);
      return;
    }

    // Execute export
    const success = await exportToImage(element, { isDarkMode });
    
    if (!success) {
      showExportError('unknown');
    }
  }, [isDarkMode]);

  // NEW: Copy code handler
  const handleCopyCode = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(generatedCode);
      setCodeCopied(true);
      setTimeout(() => setCodeCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, [generatedCode]);

  const currentLanguageOption = useMemo(() => 
    LANGUAGE_OPTIONS.find(opt => opt.code === language) || LANGUAGE_OPTIONS[0],
    [language]
  );

  return (
    <div className={`min-h-screen transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-900' : 'bg-gradient-to-br from-purple-50 via-white to-blue-50'
    }`}>
      <div className="container mx-auto px-3 sm:px-4 py-4 sm:py-8 max-w-7xl">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-center gap-3 sm:gap-4 mb-4 sm:mb-6">
          <div className="flex items-center gap-3">
            <img
              src="/square-logo.svg"
              alt="LayerCal logo"
              className="w-10 h-10 sm:w-12 sm:h-12"
            />
            <div>
              <h1 className={`text-2xl sm:text-3xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                {t.title}
              </h1>
              <p className={`text-xs sm:text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                {t.subtitle}
              </p>
          </div>
        </div>


          <div className="flex items-center gap-1.5 sm:gap-2">
            {/* Contact button */}
            <a
              href="mailto:me@chanjoongx.com"
              className={`p-1.5 sm:p-2 rounded-lg transition-colors ${
                isDarkMode 
                  ? 'bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-gray-100' 
                  : 'bg-white hover:bg-gray-50 text-gray-600 hover:text-gray-900 shadow-sm'
              }`}
              aria-label="Contact us via email"
              title="me@chanjoongx.com"
            >
              <Mail className="w-4 h-4 sm:w-5 sm:h-5" />
            </a>

            {/* Dark mode toggle */}
            <button
              onClick={handleDarkModeToggle}
              className={`p-1.5 sm:p-2 rounded-lg transition-colors ${
                isDarkMode 
                  ? 'bg-gray-800 hover:bg-gray-700 text-yellow-400' 
                  : 'bg-white hover:bg-gray-50 text-gray-700 shadow-sm'
              }`}
              aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {isDarkMode ? <Sun className="w-4 h-4 sm:w-5 sm:h-5" /> : <Moon className="w-4 h-4 sm:w-5 sm:h-5" />}
            </button>

            {/* Language selector */}
            <div className="relative" ref={languageMenuRef}>
              <button
                onClick={() => setShowLanguageMenu(!showLanguageMenu)}
                className={`flex items-center gap-1 sm:gap-2 px-2 sm:px-4 py-1.5 sm:py-2 rounded-lg transition-colors ${
                  isDarkMode 
                    ? 'bg-gray-800 hover:bg-gray-700 text-gray-200' 
                    : 'bg-white hover:bg-gray-50 text-gray-700 shadow-sm'
                }`}
                aria-label={`Current language: ${language}. Click to change language`}
                aria-expanded={showLanguageMenu}
                aria-haspopup="true"
              >
                <Globe className="w-4 h-4 sm:w-5 sm:h-5" />
                <span className="hidden sm:inline text-sm font-medium">{currentLanguageOption.flag} {currentLanguageOption.code.toUpperCase()}</span>
                <ChevronDown className="w-3 h-3 sm:w-4 sm:h-4" />
              </button>

              {showLanguageMenu && (
                <div className={`absolute right-0 mt-2 w-40 sm:w-48 rounded-lg shadow-lg z-50 ${
                  isDarkMode ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-200'
                }`}>
                  {LANGUAGE_OPTIONS.map(option => (
                    <button
                      key={option.code}
                      onClick={() => handleLanguageChange(option.code)}
                      className={`w-full text-left px-3 sm:px-4 py-2 text-sm transition-colors first:rounded-t-lg last:rounded-b-lg ${
                        language === option.code
                          ? (isDarkMode ? 'bg-purple-900/30 text-purple-400' : 'bg-purple-50 text-purple-600')
                          : (isDarkMode ? 'text-gray-300 hover:bg-gray-700' : 'text-gray-700 hover:bg-gray-50')
                      }`}
                    >
                      <span className="hidden sm:inline mr-2">{option.flag}</span>
                      {option.name}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Export Code button - NEW */}
            <button
              onClick={() => setShowCodeModal(true)}
              className={`flex items-center gap-1 sm:gap-2 px-2 sm:px-4 py-1.5 sm:py-2 rounded-lg transition-colors ${
                isDarkMode 
                  ? 'bg-green-900/30 hover:bg-green-900/50 text-green-400 border border-green-700' 
                  : 'bg-green-100 hover:bg-green-200 text-green-700 border border-green-300'
              }`}
              aria-label="Export code"
            >
              <Code className="w-4 h-4 sm:w-5 sm:h-5" />
              <span className="hidden sm:inline text-sm">{t.exportCode || 'Export Code'}</span>
            </button>

            {/* Export Image button */}
            <button
              onClick={handleExportImageClick}
              className={`flex items-center gap-1 sm:gap-2 px-2 sm:px-4 py-1.5 sm:py-2 rounded-lg transition-colors ${
                isDarkMode 
                  ? 'bg-purple-900/30 hover:bg-purple-900/50 text-purple-400 border border-purple-700' 
                  : 'bg-purple-100 hover:bg-purple-200 text-purple-700 border border-purple-300'
              }`}
              aria-label="Export model as image"
            >
              <Camera className="w-4 h-4 sm:w-5 sm:h-5" />
              <span className="hidden sm:inline text-sm">{t.exportImage}</span>
            </button>
          </div>
        </div>

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
                        <span className="text-xl sm:text-2xl">{config.icon}</span>
                        <div className="flex-1 min-w-0">
                          <div className={`font-medium text-xs sm:text-sm ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                            {config.name}
                          </div>
                          <div className={`text-[10px] sm:text-xs truncate ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
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
                  <CardTitle className={isDarkMode ? 'text-white' : 'text-gray-900'}>
                    {t.modelArchitecture}
                  </CardTitle>
                  <CardDescription className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>
                    {modelLayers.length} {t.layers} • {totalParams.toLocaleString()} {t.parameters}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    onDragEnter={handleDragEnter}
                    onDragLeave={handleDragLeave}
                    id="model-drop-zone"
                    className={`min-h-[400px] rounded-lg border-2 border-dashed p-4 transition-colors ${
                      draggedType
                        ? (isDarkMode ? 'border-purple-500 bg-purple-900/20' : 'border-purple-400 bg-purple-50')
                        : (isDarkMode ? 'border-gray-600 bg-gray-900/50' : 'border-gray-300 bg-gray-50/50')
                    }`}
                  >
                    {modelLayers.length === 0 ? (
                      <div className="flex flex-col items-center justify-center h-full min-h-[300px] text-center">
                        <Plus className={`w-16 h-16 mb-4 ${isDarkMode ? 'text-gray-600' : 'text-gray-400'}`} />
                        <p className={`text-lg font-medium ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                          {t.dropHere}
                        </p>
                        <p className={`text-sm mt-2 ${isDarkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                          {t.yourModel}
                        </p>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {modelLayers.map((layer, index) => {
                          const config = LAYER_TYPES[layer.type];
                          const layerParams = config.calculate(layer.params);

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
                                <GripVertical className={`w-4 h-4 sm:w-5 sm:h-5 mt-1 cursor-move flex-shrink-0 ${
                                  isDarkMode ? 'text-gray-500' : 'text-gray-400'
                                }`} />
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center justify-between mb-2">
                                    <div className="flex items-center gap-2 flex-1 min-w-0">
                                      <span className="text-lg sm:text-xl flex-shrink-0">{config.icon}</span>
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
                                      onKeyDown={(e) => {
                                        if (e.key === 'Enter' || e.key === ' ') {
                                          e.preventDefault();
                                          deleteLayer(layer.id);
                                        }
                                      }}
                                      className={`p-1.5 rounded transition-colors ${
                                        isDarkMode 
                                          ? 'hover:bg-red-900/30 text-red-400' 
                                          : 'hover:bg-red-100 text-red-600'
                                      }`}
                                      aria-label={`Delete ${layer.type} layer`}
                                      tabIndex={0}
                                    >
                                      <Trash2 className="w-4 h-4" />
                                    </button>
                                  </div>

                                  {/* Parameter controls */}
                                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-2 mt-3">
                                    {config.fields.map((field) => (
                                      <div key={field.key} className="space-y-1">
                                        <label className={`text-xs font-medium block ${
                                          isDarkMode ? 'text-gray-300' : 'text-gray-700'
                                        }`}>
                                          {field.label}
                                        </label>
                                        {field.type === 'select' ? (
                                          <select
                                            value={layer.params[field.key]}
                                            onChange={(e) => updateLayerParam(layer.id, field.key, Number(e.target.value))}
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
                                              type="checkbox"
                                              checked={layer.params[field.key]}
                                              onChange={(e) => updateLayerParam(layer.id, field.key, e.target.checked)}
                                              className="w-5 h-5"
                                            />
                                          </div>
                                        ) : (
                                          <input
                                            type="number"
                                            value={layer.params[field.key]}
                                            onChange={(e) => updateLayerParam(layer.id, field.key, Number(e.target.value))}
                                            step={field.step || 1}
                                            min={field.min}
                                            max={field.max}
                                            className={`w-full px-3 py-2 text-sm border rounded-lg ${
                                              isDarkMode 
                                                ? 'bg-gray-700 border-gray-600 text-white' 
                                                : 'border-gray-300 bg-white'
                                            }`}
                                          />
                                        )}
                                      </div>
                                    ))}
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
                    isDarkMode 
                      ? 'bg-purple-900/30 border-purple-700' 
                      : 'bg-purple-50 border-purple-200'
                  }`}>
                    <p className={`text-xs sm:text-sm mb-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{t.totalParameters}</p>
                    <p className={`text-2xl sm:text-3xl font-bold ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`}>
                      {totalParams.toLocaleString()}
                    </p>
                  </div>

                  {/* Model Size */}
                  <div className={`rounded-lg p-3 sm:p-4 border ${
                    isDarkMode 
                      ? 'bg-blue-900/30 border-blue-700' 
                      : 'bg-blue-50 border-blue-200'
                  }`}>
                    <p className={`text-xs sm:text-sm mb-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{t.modelSize}</p>
                    <p className={`text-xl sm:text-2xl font-bold ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>
                      {modelSizeMB < 1 
                        ? `${(modelSizeMB * 1024).toFixed(2)} KB`
                        : `${modelSizeMB.toFixed(2)} MB`
                      }
                    </p>
                  </div>

                  {/* Total FLOPs - NEW */}
                  <div className={`rounded-lg p-3 sm:p-4 border ${
                    isDarkMode 
                      ? 'bg-orange-900/30 border-orange-700' 
                      : 'bg-orange-50 border-orange-200'
                  }`}>
                    <p className={`text-xs sm:text-sm mb-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{t.totalFLOPs || 'Total FLOPs'}</p>
                    <p className={`text-xl sm:text-2xl font-bold ${isDarkMode ? 'text-orange-400' : 'text-orange-600'}`}>
                      {formatNumber(totalFLOPs)}
                    </p>
                  </div>

                  {/* Memory Estimation - NEW */}
                  <div className={`rounded-lg p-3 sm:p-4 border ${
                    isDarkMode 
                      ? 'bg-cyan-900/30 border-cyan-700' 
                      : 'bg-cyan-50 border-cyan-200'
                  }`}>
                    <div className="flex items-center justify-between mb-2">
                      <p className={`text-xs sm:text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{t.memoryEstimation || 'Memory'}</p>
                      <div className="flex gap-1">
                        <button
                          onClick={() => setMemoryMode('inference')}
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
                    <select
                      value={precision}
                      onChange={(e) => setPrecision(e.target.value)}
                      className={`mt-2 w-full px-2 py-1 text-xs rounded border ${
                        isDarkMode 
                          ? 'bg-gray-700 border-gray-600 text-gray-200' 
                          : 'bg-white border-gray-300 text-gray-700'
                      }`}
                    >
                      <option value="fp32">{t.fp32 || 'FP32 (32-bit)'}</option>
                      <option value="fp16">{t.fp16 || 'FP16 (16-bit)'}</option>
                      <option value="int8">{t.int8 || 'INT8 (8-bit)'}</option>
                    </select>
                  </div>

                  {/* Number of Layers */}
                  <div className={`rounded-lg p-3 sm:p-4 border ${
                    isDarkMode 
                      ? 'bg-green-900/30 border-green-700' 
                      : 'bg-green-50 border-green-200'
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
                          const layerParams = config.calculate(layer.params);
                          const percentage = totalParams > 0 ? ((layerParams / totalParams) * 100).toFixed(1) : '0.0';
                          
                          return (
                            <div key={layer.id} className="text-xs">
                              <div className="flex justify-between mb-1">
                                <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>
                                  {config.icon} {t.layer} {idx + 1}
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

        {/* Donation modal */}
        {showDonationModal && (
          <div className={`fixed inset-0 flex items-center justify-center z-50 p-4 ${
            isDarkMode ? 'bg-black bg-opacity-70' : 'bg-black bg-opacity-50'
          }`} onClick={() => setShowDonationModal(false)}>
            <div 
              className={`rounded-2xl shadow-2xl max-w-md w-full p-5 sm:p-6 relative transition-all duration-200 ${
                isDarkMode ? 'bg-gray-800' : 'bg-white'
              }`}
              onClick={(e) => e.stopPropagation()}
            >
              <button
                onClick={() => setShowDonationModal(false)}
                className={`absolute top-3 right-3 sm:top-4 sm:right-4 transition-colors ${
                  isDarkMode 
                    ? 'text-gray-400 hover:text-gray-200' 
                    : 'text-gray-400 hover:text-gray-600'
                }`}
                aria-label="Close modal"
              >
                <X className="w-5 h-5 sm:w-6 sm:h-6" />
              </button>
              
              <div className="text-center mb-5 sm:mb-6">
                <h2 className={`text-xl sm:text-2xl font-bold mb-2 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
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
            </div>
          </div>
        )}

        {/* Code Export Modal - NEW */}
        {showCodeModal && (
          <div className={`fixed inset-0 flex items-center justify-center z-50 p-4 ${
            isDarkMode ? 'bg-black bg-opacity-70' : 'bg-black bg-opacity-50'
          }`} onClick={() => setShowCodeModal(false)}>
            <div 
              className={`rounded-2xl shadow-2xl max-w-2xl w-full max-h-[80vh] flex flex-col relative transition-all duration-200 ${
                isDarkMode ? 'bg-gray-800' : 'bg-white'
              }`}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className={`flex items-center justify-between p-4 border-b ${
                isDarkMode ? 'border-gray-700' : 'border-gray-200'
              }`}>
                <div>
                  <h2 className={`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    {t.codeExportTitle || 'Export Code'}
                  </h2>
                  <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    {t.codeExportDesc || 'Copy the generated code for your framework'}
                  </p>
                </div>
                <button
                  onClick={() => setShowCodeModal(false)}
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
              <div className={`flex border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                {['pytorch', 'tensorflow', 'jax'].map((fw) => (
                  <button
                    key={fw}
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
                      ? (isDarkMode 
                          ? 'bg-green-700 text-white' 
                          : 'bg-green-500 text-white')
                      : (isDarkMode 
                          ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' 
                          : 'bg-gray-200 hover:bg-gray-300 text-gray-800')
                  }`}
                >
                  {codeCopied ? (t.codeCopied || 'Copied!') : (t.copyCode || '⬇️ No thanks, just copy code')}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className={`mt-6 sm:mt-8 text-center text-xs sm:text-sm space-y-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          <p className="text-[10px] sm:text-xs">{t.calculationNote}</p>
          <p className="text-[10px] sm:text-xs">© {new Date().getFullYear()} LayerCal. All rights reserved.</p>
        </div>
      </div>
    </div>
  );
}
