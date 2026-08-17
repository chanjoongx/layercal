import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { X, Eye, EyeOff, Sparkles, Loader2, AlertCircle, ChevronDown, ChevronRight, Lock, Zap, ExternalLink } from 'lucide-react';
import Modal from '@/components/ui/modal';
import { safeLocalStorage } from '@/utils/localStorage';
import { generateArchitecture } from '@/utils/ragPipeline';
import { DEFAULT_MODELS } from '@/utils/llmClient';
import { LLM_ERROR_KEYS } from '@/config/translations';
import { getLayerTypes, formatNumber } from '@/config/layerTypes';

const PROVIDERS = [
  {
    id: 'gemini',
    name: 'Google Gemini',
    hint: 'Free tier available',
    keyUrl: 'https://aistudio.google.com/apikey',
  },
  {
    id: 'openai',
    name: 'OpenAI',
    hint: 'Requires billing setup',
    keyUrl: 'https://platform.openai.com/api-keys',
  },
  {
    id: 'claude',
    name: 'Anthropic Claude',
    hint: 'Requires billing setup',
    keyUrl: 'https://console.anthropic.com/settings/keys',
  },
];

const STORAGE_KEYS = {
  provider: 'layercal-ai-provider',
  apiKeyPrefix: 'layercal-ai-apikey-',   // per-provider: layercal-ai-apikey-gemini, etc.
  modelPrefix: 'layercal-ai-model-',     // per-provider model override
  rememberKey: 'layercal-ai-remember',
};

const apiKeyStorageKey = (providerId) => STORAGE_KEYS.apiKeyPrefix + providerId;
const modelStorageKey = (providerId) => STORAGE_KEYS.modelPrefix + providerId;

const EXAMPLE_PROMPTS = [
  'Image classifier under 5M params for edge deployment',
  'BERT-style text encoder with 4 transformer blocks',
  'Lightweight sentiment analysis model using LSTM',
  'Simple autoencoder for image reconstruction',
  'Tabular data classifier with 3 hidden layers',
];

export default function AIAdvisor({ isDarkMode, t, onApply, onClose, canvasHasLayers }) {
  // ── Persisted state ────────────────────────
  const [provider, setProvider] = useState(() =>
    safeLocalStorage.getItem(STORAGE_KEYS.provider, 'gemini')
  );
  const [rememberKey, setRememberKey] = useState(() =>
    safeLocalStorage.getItem(STORAGE_KEYS.rememberKey, 'true') !== 'false'
  );
  const [apiKey, setApiKey] = useState(() => {
    const savedProvider = safeLocalStorage.getItem(STORAGE_KEYS.provider, 'gemini');
    return safeLocalStorage.getItem(apiKeyStorageKey(savedProvider), '');
  });
  const [model, setModel] = useState(() => {
    const savedProvider = safeLocalStorage.getItem(STORAGE_KEYS.provider, 'gemini');
    return safeLocalStorage.getItem(modelStorageKey(savedProvider), '');
  });
  const [showKey, setShowKey] = useState(false);
  const [showProviderMenu, setShowProviderMenu] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // ── Transient state ────────────────────────
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);     // { layers, references, warnings, usage, model }
  const [error, setError] = useState(null);       // { key?, detail }

  const textareaRef = useRef(null);
  const providerMenuRef = useRef(null);
  const abortRef = useRef(null);

  const LAYER_TYPES = useMemo(() => getLayerTypes(t, isDarkMode), [t, isDarkMode]);

  // Close provider dropdown on outside click
  useEffect(() => {
    const handleClick = (e) => {
      if (providerMenuRef.current && !providerMenuRef.current.contains(e.target)) {
        setShowProviderMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  // Escape should dismiss the provider dropdown before the dialog itself.
  const handleEscape = useCallback(() => {
    if (!showProviderMenu) return false;
    setShowProviderMenu(false);
    return true;
  }, [showProviderMenu]);

  // Abandon any in-flight request if the dialog is dismissed mid-generation.
  useEffect(() => () => abortRef.current?.abort(), []);

  // ── Settings handlers ──────────────────────
  const handleProviderChange = useCallback((newProvider) => {
    setProvider(newProvider);
    safeLocalStorage.setItem(STORAGE_KEYS.provider, newProvider);
    setShowProviderMenu(false);
    // load previously saved key/model for this provider (if any)
    setApiKey(safeLocalStorage.getItem(apiKeyStorageKey(newProvider), ''));
    setModel(safeLocalStorage.getItem(modelStorageKey(newProvider), ''));
    setError(null);
  }, []);

  const handleApiKeyChange = useCallback((e) => {
    const val = e.target.value;
    setApiKey(val);
    if (rememberKey) {
      safeLocalStorage.setItem(apiKeyStorageKey(provider), val);
    }
  }, [provider, rememberKey]);

  const handleRememberToggle = useCallback((e) => {
    const next = e.target.checked;
    setRememberKey(next);
    safeLocalStorage.setItem(STORAGE_KEYS.rememberKey, String(next));
    if (next) {
      safeLocalStorage.setItem(apiKeyStorageKey(provider), apiKey);
    } else {
      // Opting out must also remove what was already written to disk.
      PROVIDERS.forEach(p => safeLocalStorage.removeItem(apiKeyStorageKey(p.id)));
    }
  }, [provider, apiKey]);

  const handleClearKey = useCallback(() => {
    setApiKey('');
    safeLocalStorage.removeItem(apiKeyStorageKey(provider));
  }, [provider]);

  const handleModelChange = useCallback((e) => {
    const val = e.target.value;
    setModel(val);
    safeLocalStorage.setItem(modelStorageKey(provider), val);
  }, [provider]);

  // ── Generate ───────────────────────────────
  const handleGenerate = useCallback(async () => {
    if (!apiKey.trim() || !query.trim() || isLoading) return;

    const controller = new AbortController();
    abortRef.current = controller;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const output = await generateArchitecture({
        query: query.trim(),
        provider,
        apiKey: apiKey.trim(),
        model: model.trim() || undefined,
        signal: controller.signal,
      });
      setResult(output);
    } catch (err) {
      if (err?.code !== 'CANCELLED') {
        setError({
          key: LLM_ERROR_KEYS[err?.code],
          detail: err?.message || 'An unexpected error occurred.',
        });
      }
    } finally {
      setIsLoading(false);
      abortRef.current = null;
    }
  }, [apiKey, query, provider, model, isLoading]);

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const handleKeyDown = useCallback((e) => {
    // Ctrl/Cmd + Enter to generate
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleGenerate();
    }
  }, [handleGenerate]);

  const handleApply = useCallback((mode) => {
    if (result?.layers) {
      onApply(result.layers, mode);
    }
  }, [result, onApply]);

  const handleExampleClick = useCallback((example) => {
    setQuery(example);
    setResult(null);
    setError(null);
    textareaRef.current?.focus();
  }, []);

  // ── Derived ────────────────────────────────
  const selectedProvider = PROVIDERS.find(p => p.id === provider) || PROVIDERS[0];
  const canGenerate = apiKey.trim().length > 0 && query.trim().length > 0 && !isLoading;

  // Running the same formulas the canvas uses lets the user judge the
  // proposal against their budget before applying it.
  const estimatedParams = useMemo(() => {
    if (!result?.layers) return 0;
    return result.layers.reduce((total, layer) => {
      const config = LAYER_TYPES[layer.type];
      return total + (config ? config.calculate(layer.params) : 0);
    }, 0);
  }, [result, LAYER_TYPES]);

  const errorText = error
    ? (error.key && t[error.key]) || error.detail
    : null;
  const errorDetail = error && error.key && t[error.key] && error.detail !== t[error.key]
    ? error.detail
    : null;

  // Layer preview: compact text representation
  const formatLayerPreview = (layer) => {
    const p = layer.params || {};
    switch (layer.type) {
      case 'conv2d': return `Conv2D(${p.in_channels}→${p.out_channels}, k=${p.kernel_size})`;
      case 'linear': return `Linear(${p.input_dim}→${p.output_dim})`;
      case 'embedding': return `Embedding(${Number(p.vocab_size || 0).toLocaleString()}×${p.embedding_dim})`;
      case 'lstm': return `LSTM(in=${p.input_size}, h=${p.hidden_size}${p.bidirectional ? ', bi' : ''})`;
      case 'gru': return `GRU(in=${p.input_size}, h=${p.hidden_size}${p.bidirectional ? ', bi' : ''})`;
      case 'transformer': return `Transformer(d=${p.d_model}, h=${p.num_heads}, ff=${p.d_ff})`;
      case 'attention': return `Attention(d=${p.d_model}, h=${p.num_heads})`;
      case 'batchnorm': return `BatchNorm(${p.num_features})`;
      case 'layernorm': return `LayerNorm(${p.normalized_shape})`;
      case 'dropout': return `Dropout(${p.rate})`;
      case 'maxpool2d': return `MaxPool2D(k=${p.kernel_size})`;
      case 'avgpool2d': return `AvgPool2D(k=${p.kernel_size})`;
      default: return layer.type.charAt(0).toUpperCase() + layer.type.slice(1);
    }
  };

  const inputClass = isDarkMode
    ? 'bg-gray-700 border-gray-600 text-gray-200 placeholder-gray-400'
    : 'bg-gray-50 border-gray-200 text-gray-800 placeholder-gray-400';


  // ── Render ─────────────────────────────────
  return (
    <Modal
      isDarkMode={isDarkMode}
      labelledBy="advisor-title"
      onClose={onClose}
      onEscape={handleEscape}
      initialFocusRef={textareaRef}
      className="w-full max-w-lg max-h-[85vh] flex flex-col"
    >
      <>
        {/* ── Header ────────────────────────── */}
        <div className={`flex items-center justify-between p-4 border-b shrink-0 ${
          isDarkMode ? 'border-gray-700' : 'border-gray-200'
        }`}>
          <div className="flex items-center gap-2">
            <Sparkles className={`w-5 h-5 ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`} />
            <div>
              <h2 id="advisor-title" className={`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                {t.aiAdvisor || 'AI Architecture Advisor'}
              </h2>
              <p className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                {t.aiAdvisorDesc || 'Describe your model and AI designs the layers'}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
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

        {/* ── Body (scrollable) ─────────────── */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">

          {/* API Settings */}
          <div className="space-y-2">
            <label htmlFor="advisor-api-key" className={`text-xs font-medium block ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              {t.apiProvider || 'API Provider'}
            </label>
            <div className="flex gap-2">
              {/* Provider dropdown */}
              <div className="relative" ref={providerMenuRef}>
                <button
                  onClick={() => setShowProviderMenu(!showProviderMenu)}
                  aria-haspopup="listbox"
                  aria-expanded={showProviderMenu}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm border transition-colors ${
                    isDarkMode
                      ? 'bg-gray-700 border-gray-600 text-gray-200 hover:bg-gray-600'
                      : 'bg-gray-50 border-gray-200 text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  {selectedProvider.name}
                  <ChevronDown className="w-3 h-3" />
                </button>
                {showProviderMenu && (
                  <div role="listbox" className={`absolute left-0 top-full mt-1 w-48 rounded-lg shadow-lg z-10 ${
                    isDarkMode ? 'bg-gray-700 border border-gray-600' : 'bg-white border border-gray-200'
                  }`}>
                    {PROVIDERS.map(p => (
                      <button
                        key={p.id}
                        role="option"
                        aria-selected={provider === p.id}
                        onClick={() => handleProviderChange(p.id)}
                        className={`w-full text-left px-3 py-2 text-sm first:rounded-t-lg last:rounded-b-lg ${
                          provider === p.id
                            ? (isDarkMode ? 'bg-purple-900/30 text-purple-400' : 'bg-purple-50 text-purple-600')
                            : (isDarkMode ? 'text-gray-300 hover:bg-gray-600' : 'text-gray-700 hover:bg-gray-50')
                        }`}
                      >
                        <div>{p.name}</div>
                        <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>{p.hint}</div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* API Key input */}
              <div className="flex-1 relative">
                <input
                  id="advisor-api-key"
                  name="advisor-api-key"
                  type={showKey ? 'text' : 'password'}
                  value={apiKey}
                  onChange={handleApiKeyChange}
                  placeholder={t.apiKeyPlaceholder || 'Paste your API key'}
                  className={`w-full px-3 py-2 pr-16 rounded-lg text-sm border transition-colors ${inputClass}`}
                  autoComplete="off"
                  spellCheck="false"
                />
                <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
                  {apiKey && (
                    <button
                      onClick={handleClearKey}
                      aria-label={t.clearKey || 'Clear'}
                      title={t.clearKey || 'Clear'}
                      className={isDarkMode ? 'text-gray-400 hover:text-gray-200' : 'text-gray-500 hover:text-gray-700'}
                      tabIndex={-1}
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                  <button
                    onClick={() => setShowKey(!showKey)}
                    aria-label={showKey ? 'Hide API key' : 'Show API key'}
                    className={isDarkMode ? 'text-gray-400 hover:text-gray-200' : 'text-gray-500 hover:text-gray-700'}
                    tabIndex={-1}
                  >
                    {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            </div>

            {/* Security note + key link */}
            <div className={`flex items-center justify-between gap-2 text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              <span className="flex items-center gap-1.5">
                <Lock className="w-3 h-3 shrink-0" />
                {t.apiKeySecurityNote || 'Stored locally. Never sent to our servers.'}
              </span>
              <a
                href={selectedProvider.keyUrl}
                target="_blank"
                rel="noopener noreferrer"
                className={`flex items-center gap-1 shrink-0 underline ${
                  isDarkMode ? 'hover:text-gray-300' : 'hover:text-gray-600'
                }`}
              >
                {t.apiProvider || 'API key'}
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>

            {/* Advanced: model override + key retention */}
            <button
              onClick={() => setShowAdvanced(v => !v)}
              aria-expanded={showAdvanced}
              className={`flex items-center gap-1 text-xs transition-colors ${
                isDarkMode ? 'text-gray-400 hover:text-gray-200' : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              {showAdvanced ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              {t.advanced || 'Advanced'}
            </button>

            {showAdvanced && (
              <div className={`space-y-2 rounded-lg border p-3 ${
                isDarkMode ? 'border-gray-700 bg-gray-900/40' : 'border-gray-200 bg-gray-50'
              }`}>
                <label htmlFor="advisor-model" className={`text-xs font-medium block ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  {t.modelLabel || 'Model'}
                </label>
                <input
                  id="advisor-model"
                  name="advisor-model"
                  type="text"
                  value={model}
                  onChange={handleModelChange}
                  placeholder={DEFAULT_MODELS[provider]}
                  className={`w-full px-3 py-2 rounded-lg text-sm border font-mono transition-colors ${inputClass}`}
                  autoComplete="off"
                  spellCheck="false"
                />
                <label className={`flex items-center gap-2 text-xs cursor-pointer ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  <input
                    type="checkbox"
                    checked={rememberKey}
                    onChange={handleRememberToggle}
                    className="w-4 h-4"
                  />
                  {t.rememberKey || 'Remember key in this browser'}
                </label>
              </div>
            )}
          </div>

          {/* Query input */}
          <div className="space-y-2">
            <label htmlFor="advisor-query" className={`text-xs font-medium block ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              {t.queryLabel || 'What kind of model do you need?'}
            </label>
            <textarea
              id="advisor-query"
              name="advisor-query"
              ref={textareaRef}
              value={query}
              onChange={(e) => { setQuery(e.target.value); setResult(null); setError(null); }}
              onKeyDown={handleKeyDown}
              placeholder={t.queryPlaceholder || 'e.g., Image classifier under 5M params for edge deployment'}
              rows={3}
              className={`w-full px-3 py-2 rounded-lg text-sm border resize-none transition-colors ${inputClass}`}
            />
            <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Ctrl+Enter {t.toGenerate || 'to generate'}
            </div>
          </div>

          {/* Example prompts */}
          <div className="space-y-1.5">
            <span className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              {t.examplePrompts || 'Try:'}
            </span>
            <div className="flex flex-wrap gap-1.5">
              {EXAMPLE_PROMPTS.map((ex, i) => (
                <button
                  key={i}
                  onClick={() => handleExampleClick(ex)}
                  className={`px-2 py-1 text-xs rounded-md transition-colors ${
                    isDarkMode
                      ? 'bg-gray-700 text-gray-400 hover:bg-gray-600 hover:text-gray-200'
                      : 'bg-gray-100 text-gray-500 hover:bg-gray-200 hover:text-gray-700'
                  }`}
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>

          {/* Generate / Cancel */}
          {isLoading ? (
            <div className="flex gap-2">
              <div className={`flex-1 py-2.5 rounded-lg font-medium text-sm flex items-center justify-center gap-2 ${
                isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-600'
              }`}>
                <Loader2 className="w-4 h-4 animate-spin" />
                {t.generating || 'Generating...'}
              </div>
              <button
                onClick={handleCancel}
                className={`px-4 py-2.5 rounded-lg font-medium text-sm transition-colors ${
                  isDarkMode
                    ? 'bg-gray-700 hover:bg-gray-600 text-gray-200'
                    : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                }`}
              >
                {t.cancel || 'Cancel'}
              </button>
            </div>
          ) : (
            <button
              onClick={handleGenerate}
              disabled={!canGenerate}
              className={`w-full py-2.5 rounded-lg font-medium text-sm transition-all flex items-center justify-center gap-2 ${
                canGenerate
                  ? 'bg-gradient-to-r from-purple-500 to-indigo-500 hover:from-purple-600 hover:to-indigo-600 text-white shadow-md hover:shadow-lg active:scale-[0.98]'
                  : (isDarkMode
                      ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                      : 'bg-gray-100 text-gray-400 cursor-not-allowed')
              }`}
            >
              <Zap className="w-4 h-4" />
              {t.generateBtn || 'Generate Architecture'}
            </button>
          )}

          {/* Error message */}
          {errorText && (
            <div
              role="alert"
              className={`flex items-start gap-2 p-3 rounded-lg text-sm ${
                isDarkMode ? 'bg-red-900/20 text-red-300 border border-red-800' : 'bg-red-50 text-red-700 border border-red-200'
              }`}
            >
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              <div className="min-w-0">
                <div>{errorText}</div>
                {errorDetail && (
                  <div className={`mt-1 text-xs break-words ${isDarkMode ? 'text-red-400/70' : 'text-red-500/80'}`}>
                    {errorDetail}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Result preview */}
          {result && (
            <div className={`rounded-lg border p-3 space-y-3 ${
              isDarkMode ? 'bg-gray-900/50 border-gray-700' : 'bg-gray-50 border-gray-200'
            }`}>
              <div className="flex items-baseline justify-between gap-2">
                <div className={`text-sm font-medium ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                  {t.recommendedArch || 'Recommended Architecture'} ({result.layers.length} {t.layers || 'layers'})
                </div>
                <div className={`text-xs font-mono shrink-0 ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`}>
                  {formatNumber(estimatedParams)}
                </div>
              </div>

              {/* Layer list */}
              <div className="space-y-1">
                {result.layers.map((layer, idx) => (
                  <div
                    key={idx}
                    className={`flex items-center gap-2 px-2 py-1 rounded text-xs font-mono ${
                      isDarkMode ? 'text-gray-300' : 'text-gray-700'
                    }`}
                  >
                    <span className={`w-5 text-right ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                      {idx + 1}
                    </span>
                    <span className={`${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`}>
                      {formatLayerPreview(layer)}
                    </span>
                  </div>
                ))}
              </div>

              {/* Warnings */}
              {result.warnings?.length > 0 && (
                <div className={`text-xs space-y-0.5 ${isDarkMode ? 'text-yellow-400' : 'text-yellow-600'}`}>
                  {result.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
                </div>
              )}

              {/* Provenance */}
              <div className={`text-xs space-y-0.5 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                {result.references?.length > 0 && (
                  <div>{t.referencedModels || 'Referenced'}: {result.references.map(r => r.name).join(', ')}</div>
                )}
                <div className="flex flex-wrap gap-x-3">
                  {result.model && <span className="font-mono">{result.model}</span>}
                  {result.usage && (
                    <span>
                      {t.tokensUsed || 'Tokens'}: {result.usage.promptTokens.toLocaleString()} + {result.usage.completionTokens.toLocaleString()}
                    </span>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ── Footer ────────────────────────── */}
        {result && (
          <div className={`p-4 border-t shrink-0 flex gap-2 ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
            <button
              onClick={() => handleApply('replace')}
              className="flex-1 py-2.5 rounded-lg font-medium text-sm bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white shadow-md hover:shadow-lg transition-all active:scale-[0.98]"
            >
              ✅ {canvasHasLayers ? (t.applyReplace || 'Replace canvas') : (t.applyToCanvas || 'Apply to Canvas')}
            </button>
            {canvasHasLayers && (
              <button
                onClick={() => handleApply('append')}
                className={`px-4 py-2.5 rounded-lg font-medium text-sm transition-colors ${
                  isDarkMode
                    ? 'bg-gray-700 hover:bg-gray-600 text-gray-200'
                    : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                }`}
              >
                {t.applyAppend || 'Append'}
              </button>
            )}
          </div>
        )}
      </>
    </Modal>
  );
}
