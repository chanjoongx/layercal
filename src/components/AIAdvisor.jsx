import React, { useState, useCallback, useRef, useEffect } from 'react';
import { X, Eye, EyeOff, Sparkles, Loader2, AlertCircle, ChevronDown, Lock, Zap } from 'lucide-react';
import { safeLocalStorage } from '@/utils/localStorage';
import { generateArchitecture } from '@/utils/ragPipeline';

const PROVIDERS = [
  { id: 'gemini', name: 'Google Gemini', hint: 'Free tier: 10 req/min' },
  { id: 'openai', name: 'OpenAI', hint: 'Requires billing setup' },
  { id: 'claude', name: 'Anthropic Claude', hint: 'Requires billing setup' },
];

const STORAGE_KEYS = {
  provider: 'layercal-ai-provider',
  apiKeyPrefix: 'layercal-ai-apikey-',    // per-provider: layercal-ai-apikey-gemini, etc.
};

function getApiKeyStorageKey(providerId) {
  return STORAGE_KEYS.apiKeyPrefix + providerId;
}

const EXAMPLE_PROMPTS = [
  'Image classifier under 5M params for edge deployment',
  'BERT-style text encoder with 4 transformer blocks',
  'Lightweight sentiment analysis model using LSTM',
  'Simple autoencoder for image reconstruction',
  'Tabular data classifier with 3 hidden layers',
];

export default function AIAdvisor({ isDarkMode, t, onApply, onClose }) {
  // ── Persisted state ────────────────────────
  const [provider, setProvider] = useState(() =>
    safeLocalStorage.getItem(STORAGE_KEYS.provider, 'gemini')
  );
  const [apiKey, setApiKey] = useState(() => {
    const savedProvider = safeLocalStorage.getItem(STORAGE_KEYS.provider, 'gemini');
    return safeLocalStorage.getItem(getApiKeyStorageKey(savedProvider), '');
  });
  const [showKey, setShowKey] = useState(false);
  const [showProviderMenu, setShowProviderMenu] = useState(false);

  // ── Transient state ────────────────────────
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);     // { layers, references, warnings }
  const [error, setError] = useState(null);

  const textareaRef = useRef(null);
  const providerMenuRef = useRef(null);

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

  // Persist provider & key on change
  const handleProviderChange = useCallback((newProvider) => {
    setProvider(newProvider);
    safeLocalStorage.setItem(STORAGE_KEYS.provider, newProvider);
    setShowProviderMenu(false);
    // load previously saved key for this provider (if any)
    setApiKey(safeLocalStorage.getItem(getApiKeyStorageKey(newProvider), ''));
  }, []);

  const handleApiKeyChange = useCallback((e) => {
    const val = e.target.value;
    setApiKey(val);
    safeLocalStorage.setItem(getApiKeyStorageKey(provider), val);
  }, [provider]);

  // ── Generate ───────────────────────────────
  const handleGenerate = useCallback(async () => {
    if (!apiKey.trim() || !query.trim()) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const output = await generateArchitecture({
        query: query.trim(),
        provider,
        apiKey: apiKey.trim(),
      });
      setResult(output);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setIsLoading(false);
    }
  }, [apiKey, query, provider]);

  const handleKeyDown = useCallback((e) => {
    // Ctrl/Cmd + Enter to generate
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleGenerate();
    }
  }, [handleGenerate]);

  const handleApply = useCallback(() => {
    if (result?.layers) {
      onApply(result.layers);
    }
  }, [result, onApply]);

  const handleExampleClick = useCallback((example) => {
    setQuery(example);
    setResult(null);
    setError(null);
    textareaRef.current?.focus();
  }, []);

  // ── Helpers ────────────────────────────────
  const selectedProvider = PROVIDERS.find(p => p.id === provider);
  const canGenerate = apiKey.trim().length > 0 && query.trim().length > 0 && !isLoading;

  // Layer preview: compact text representation
  const formatLayerPreview = (layer) => {
    const p = layer.params;
    switch (layer.type) {
      case 'conv2d': return `Conv2D(${p.in_channels}→${p.out_channels}, k=${p.kernel_size})`;
      case 'linear': return `Linear(${p.input_dim}→${p.output_dim})`;
      case 'embedding': return `Embedding(${p.vocab_size.toLocaleString()}×${p.embedding_dim})`;
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


  // ── Render ─────────────────────────────────
  return (
    <div
      className={`fixed inset-0 flex items-center justify-center z-50 p-4 ${
        isDarkMode ? 'bg-black bg-opacity-70' : 'bg-black bg-opacity-50'
      }`}
      onClick={onClose}
    >
      <div
        className={`rounded-2xl shadow-2xl max-w-lg w-full max-h-[85vh] flex flex-col relative transition-all duration-200 ${
          isDarkMode ? 'bg-gray-800' : 'bg-white'
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Header ────────────────────────── */}
        <div className={`flex items-center justify-between p-4 border-b shrink-0 ${
          isDarkMode ? 'border-gray-700' : 'border-gray-200'
        }`}>
          <div className="flex items-center gap-2">
            <Sparkles className={`w-5 h-5 ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`} />
            <div>
              <h2 className={`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                {t.aiAdvisor || 'AI Architecture Advisor'}
              </h2>
              <p className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                {t.aiAdvisorDesc || 'Describe your model — AI designs the layers'}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
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
            <label className={`text-xs font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              {t.apiProvider || 'API Provider'}
            </label>
            <div className="flex gap-2">
              {/* Provider dropdown */}
              <div className="relative" ref={providerMenuRef}>
                <button
                  onClick={() => setShowProviderMenu(!showProviderMenu)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm border transition-colors ${
                    isDarkMode
                      ? 'bg-gray-700 border-gray-600 text-gray-200 hover:bg-gray-600'
                      : 'bg-gray-50 border-gray-200 text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  {selectedProvider?.name}
                  <ChevronDown className="w-3 h-3" />
                </button>
                {showProviderMenu && (
                  <div className={`absolute left-0 top-full mt-1 w-48 rounded-lg shadow-lg z-10 ${
                    isDarkMode ? 'bg-gray-700 border border-gray-600' : 'bg-white border border-gray-200'
                  }`}>
                    {PROVIDERS.map(p => (
                      <button
                        key={p.id}
                        onClick={() => handleProviderChange(p.id)}
                        className={`w-full text-left px-3 py-2 text-sm first:rounded-t-lg last:rounded-b-lg ${
                          provider === p.id
                            ? (isDarkMode ? 'bg-purple-900/30 text-purple-400' : 'bg-purple-50 text-purple-600')
                            : (isDarkMode ? 'text-gray-300 hover:bg-gray-600' : 'text-gray-700 hover:bg-gray-50')
                        }`}
                      >
                        <div>{p.name}</div>
                        <div className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{p.hint}</div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* API Key input */}
              <div className="flex-1 relative">
                <input
                  type={showKey ? 'text' : 'password'}
                  value={apiKey}
                  onChange={handleApiKeyChange}
                  placeholder={t.apiKeyPlaceholder || 'Paste your API key'}
                  className={`w-full px-3 py-2 pr-9 rounded-lg text-sm border transition-colors ${
                    isDarkMode
                      ? 'bg-gray-700 border-gray-600 text-gray-200 placeholder-gray-500'
                      : 'bg-gray-50 border-gray-200 text-gray-800 placeholder-gray-400'
                  }`}
                  autoComplete="off"
                  spellCheck="false"
                />
                <button
                  onClick={() => setShowKey(!showKey)}
                  className={`absolute right-2 top-1/2 -translate-y-1/2 ${
                    isDarkMode ? 'text-gray-500 hover:text-gray-300' : 'text-gray-400 hover:text-gray-600'
                  }`}
                  tabIndex={-1}
                >
                  {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Security note */}
            <div className={`flex items-center gap-1.5 text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              <Lock className="w-3 h-3" />
              {t.apiKeySecurityNote || 'Stored locally. Never sent to our servers.'}
            </div>
          </div>

          {/* Query input */}
          <div className="space-y-2">
            <label className={`text-xs font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              {t.queryLabel || 'What kind of model do you need?'}
            </label>
            <textarea
              ref={textareaRef}
              value={query}
              onChange={(e) => { setQuery(e.target.value); setResult(null); setError(null); }}
              onKeyDown={handleKeyDown}
              placeholder={t.queryPlaceholder || 'e.g., Image classifier under 5M params for edge deployment'}
              rows={3}
              className={`w-full px-3 py-2 rounded-lg text-sm border resize-none transition-colors ${
                isDarkMode
                  ? 'bg-gray-700 border-gray-600 text-gray-200 placeholder-gray-500'
                  : 'bg-gray-50 border-gray-200 text-gray-800 placeholder-gray-400'
              }`}
            />
            <div className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              Ctrl+Enter {t.toGenerate || 'to generate'}
            </div>
          </div>

          {/* Example prompts */}
          <div className="space-y-1.5">
            <span className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
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

          {/* Generate button */}
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
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                {t.generating || 'Generating...'}
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                {t.generateBtn || 'Generate Architecture'}
              </>
            )}
          </button>

          {/* Error message */}
          {error && (
            <div className={`flex items-start gap-2 p-3 rounded-lg text-sm ${
              isDarkMode ? 'bg-red-900/20 text-red-300 border border-red-800' : 'bg-red-50 text-red-700 border border-red-200'
            }`}>
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Result preview */}
          {result && (
            <div className={`rounded-lg border p-3 space-y-3 ${
              isDarkMode ? 'bg-gray-900/50 border-gray-700' : 'bg-gray-50 border-gray-200'
            }`}>
              <div className={`text-sm font-medium ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                {t.recommendedArch || 'Recommended Architecture'} ({result.layers.length} {t.layers || 'layers'})
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
                    <span className={`w-5 text-right ${isDarkMode ? 'text-gray-600' : 'text-gray-400'}`}>
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

              {/* References */}
              {result.references?.length > 0 && (
                <div className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  {t.referencedModels || 'Referenced'}: {result.references.map(r => r.name).join(', ')}
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Footer ────────────────────────── */}
        {result && (
          <div className={`p-4 border-t shrink-0 ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
            <button
              onClick={handleApply}
              className="w-full py-2.5 rounded-lg font-medium text-sm bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white shadow-md hover:shadow-lg transition-all active:scale-[0.98]"
            >
              ✅ {t.applyToCanvas || 'Apply to Canvas'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}