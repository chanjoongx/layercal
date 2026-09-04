import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { X, Eye, EyeOff, Sparkles, Loader2, AlertCircle, ChevronDown, ChevronRight, Lock, Zap, ExternalLink } from 'lucide-react';
import Modal from '@/components/ui/modal';
import { safeLocalStorage } from '@/utils/localStorage';
import { generateArchitecture } from '@/utils/ragPipeline';
import { DEFAULT_MODELS } from '@/utils/llmClient';
import { LLM_ERROR_KEYS } from '@/config/translations';
import { getLayerTypes, formatNumber } from '@/config/layerTypes';
import { modelsFor, findModel, priceLabel, CATALOG_VERIFIED_ON } from '@/config/modelCatalog';
import { buildScene } from '@/viz/sceneGraph';
import ModelDiagram2D from '@/components/ModelDiagram2D';

/** Sentinel for the "type an id yourself" option in the model picker. */
const CUSTOM_MODEL = '__custom__';

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

export default function AIAdvisor({ isDarkMode, t, onApply, onClose, canvasHasLayers, layerTypes }) {
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
  // The picker only falls back to a free-text field when the saved id is not
  // one the catalogue knows, so a retired model the user pinned by hand stays
  // editable instead of being silently swapped for a catalogue entry.
  const [useCustomModel, setUseCustomModel] = useState(() => {
    const savedProvider = safeLocalStorage.getItem(STORAGE_KEYS.provider, 'gemini');
    const saved = safeLocalStorage.getItem(modelStorageKey(savedProvider), '');
    return Boolean(saved) && !findModel(savedProvider, saved);
  });
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

  // The parent already built this for the canvas; recomputing it here would
  // produce a second, equal table on every render of the dialog.
  const fallbackLayerTypes = useMemo(() => getLayerTypes(t), [t]);
  const LAYER_TYPES = layerTypes || fallbackLayerTypes;

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
    const savedModel = safeLocalStorage.getItem(modelStorageKey(newProvider), '');
    setModel(savedModel);
    setUseCustomModel(Boolean(savedModel) && !findModel(newProvider, savedModel));
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

  const handleModelSelect = useCallback((e) => {
    const val = e.target.value;
    if (val === CUSTOM_MODEL) {
      setUseCustomModel(true);
      return;
    }
    setUseCustomModel(false);
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

  const catalogModels = useMemo(() => modelsFor(provider), [provider]);
  const activeModelId = model.trim() || DEFAULT_MODELS[provider];
  const activeModel = findModel(provider, activeModelId);

  // The proposal is drawn with the same scene graph the canvas uses, so what
  // the user approves is exactly what lands.
  const previewScene = useMemo(
    () => (result?.layers ? buildScene(result.layers, LAYER_TYPES) : null),
    [result, LAYER_TYPES]
  );

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

  const inputClass = 'bg-surface-raised border-input text-foreground placeholder-muted-foreground';


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
        <div className="flex items-center justify-between p-4 border-b shrink-0 border-border">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-accent" />
            <div>
              <h2 id="advisor-title" className="text-lg font-bold text-foreground">
                {t.aiAdvisor || 'AI Architecture Advisor'}
              </h2>
              <p className="text-xs text-muted-foreground">
                {t.aiAdvisorDesc || 'Describe your model and AI designs the layers'}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label={t.closeModal || 'Close'}
            className="p-2 rounded-lg transition-colors text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* ── Body (scrollable) ─────────────── */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">

          {/* API Settings */}
          <div className="space-y-2">
            <label htmlFor="advisor-api-key" className="text-xs font-medium block text-muted-foreground">
              {t.apiProvider || 'API Provider'}
            </label>
            <div className="flex gap-2">
              {/* Provider dropdown */}
              <div className="relative" ref={providerMenuRef}>
                <button
                  onClick={() => setShowProviderMenu(!showProviderMenu)}
                  aria-haspopup="listbox"
                  aria-expanded={showProviderMenu}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm border transition-colors bg-surface-raised border-input text-foreground hover:bg-muted"
                >
                  {selectedProvider.name}
                  <ChevronDown className="w-3 h-3" />
                </button>
                {showProviderMenu && (
                  <div role="listbox" className="absolute left-0 top-full mt-1 w-48 rounded-lg shadow-lg z-10 bg-surface border border-border">
                    {PROVIDERS.map(p => (
                      <button
                        key={p.id}
                        role="option"
                        aria-selected={provider === p.id}
                        onClick={() => handleProviderChange(p.id)}
                        className={`w-full text-left px-3 py-2 text-sm first:rounded-t-lg last:rounded-b-lg ${
                          provider === p.id
                            ? ('bg-accent-soft text-accent')
                            : ('text-foreground hover:bg-muted')
                        }`}
                      >
                        <div>{p.name}</div>
                        <div className="text-xs text-muted-foreground">{p.hint}</div>
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
                      className={'text-muted-foreground hover:text-foreground'}
                      tabIndex={-1}
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                  <button
                    onClick={() => setShowKey(!showKey)}
                    aria-label={showKey ? 'Hide API key' : 'Show API key'}
                    className={'text-muted-foreground hover:text-foreground'}
                    tabIndex={-1}
                  >
                    {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            </div>

            {/* Security note + key link */}
            <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
              <span className="flex items-center gap-1.5">
                <Lock className="w-3 h-3 shrink-0" />
                {t.apiKeySecurityNote || 'Stored locally. Never sent to our servers.'}
              </span>
              <a
                href={selectedProvider.keyUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 shrink-0 underline hover:text-foreground"
              >
                {t.apiProvider || 'API key'}
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>

            {/* Advanced: model override + key retention */}
            <button
              onClick={() => setShowAdvanced(v => !v)}
              aria-expanded={showAdvanced}
              className="flex items-center gap-1 text-xs transition-colors text-muted-foreground hover:text-foreground"
            >
              {showAdvanced ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              {t.advanced || 'Advanced'}
            </button>

            {showAdvanced && (
              <div className="space-y-2 rounded-lg border p-3 border-border bg-surface-raised">
                <label htmlFor="advisor-model" className="text-xs font-medium block text-muted-foreground">
                  {t.modelLabel || 'Model'}
                </label>

                <select
                  id="advisor-model"
                  name="advisor-model"
                  value={useCustomModel ? CUSTOM_MODEL : activeModelId}
                  onChange={handleModelSelect}
                  className={`w-full px-3 py-2 rounded-lg text-sm border transition-colors ${inputClass}`}
                >
                  {catalogModels.map(entry => {
                    const price = priceLabel(entry);
                    const tier = t[`tier${entry.tier.charAt(0).toUpperCase()}${entry.tier.slice(1)}`] || entry.tier;
                    const detail = price || (entry.noteKey && t[entry.noteKey]) || entry.note || '';
                    return (
                      <option key={entry.id} value={entry.id}>
                        {`${entry.label} · ${tier}${detail ? ` · ${detail}` : ''}`}
                      </option>
                    );
                  })}
                  <option value={CUSTOM_MODEL}>{t.modelCustom || 'Custom model ID'}</option>
                </select>

                {useCustomModel && (
                  <input
                    id="advisor-model-custom"
                    name="advisor-model-custom"
                    type="text"
                    value={model}
                    onChange={handleModelChange}
                    placeholder={DEFAULT_MODELS[provider]}
                    aria-label={t.modelCustom || 'Custom model ID'}
                    className={`w-full px-3 py-2 rounded-lg text-sm border font-mono transition-colors ${inputClass}`}
                    autoComplete="off"
                    spellCheck="false"
                  />
                )}

                <p className="text-[11px] leading-snug text-muted-foreground">
                  <span className="font-mono">{activeModelId}</span>
                  {activeModel ? ` · verified ${CATALOG_VERIFIED_ON}` : ''}
                </p>
                <label className="flex items-center gap-2 text-xs cursor-pointer text-muted-foreground">
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
            <label htmlFor="advisor-query" className="text-xs font-medium block text-muted-foreground">
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
            <div className="text-xs text-muted-foreground">
              Ctrl+Enter {t.toGenerate || 'to generate'}
            </div>
          </div>

          {/* Example prompts */}
          <div className="space-y-1.5">
            <span className="text-xs text-muted-foreground">
              {t.examplePrompts || 'Try:'}
            </span>
            <div className="flex flex-wrap gap-1.5">
              {EXAMPLE_PROMPTS.map((ex, i) => (
                <button
                  key={i}
                  onClick={() => handleExampleClick(ex)}
                  className="px-2 py-1 text-xs rounded-md transition-colors bg-muted text-muted-foreground hover:bg-border hover:text-foreground"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>

          {/* Generate / Cancel */}
          {isLoading ? (
            <div className="flex gap-2">
              <div className="flex-1 py-2.5 rounded-lg font-medium text-sm flex items-center justify-center gap-2 bg-muted text-muted-foreground">
                <Loader2 className="w-4 h-4 animate-spin" />
                {t.generating || 'Generating...'}
              </div>
              <button
                onClick={handleCancel}
                className="px-4 py-2.5 rounded-lg font-medium text-sm transition-colors bg-muted text-foreground hover:bg-border"
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
                  : ('bg-muted text-muted-foreground cursor-not-allowed')
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
              className="flex items-start gap-2 p-3 rounded-lg text-sm border border-danger/40 bg-danger/10 text-danger"
            >
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              <div className="min-w-0">
                <div>{errorText}</div>
                {errorDetail && (
                  <div className="mt-1 text-xs break-words text-danger/80">
                    {errorDetail}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Result preview */}
          {result && (
            <div className="rounded-lg border p-3 space-y-3 bg-surface-raised border-border">
              <div className="flex items-baseline justify-between gap-2">
                <div className="text-sm font-medium text-foreground">
                  {t.recommendedArch || 'Recommended Architecture'} ({result.layers.length} {t.layers || 'layers'})
                </div>
                <div className="text-xs font-mono shrink-0 text-accent">
                  {formatNumber(estimatedParams)}
                </div>
              </div>

              {/* What the proposal actually looks like. A list of layer names
                  does not tell you the shape of the model; this does. */}
              {previewScene && (
                <div className="rounded-md border p-1 border-border bg-surface-raised">
                  <ModelDiagram2D
                    scene={previewScene}
                    isDarkMode={isDarkMode}
                    className="h-28 w-full"
                    label={t.recommendedArch || 'Recommended architecture'}
                  />
                </div>
              )}

              {/* Layer list */}
              <div className="space-y-1">
                {result.layers.map((layer, idx) => (
                  <div
                    key={idx}
                    className="flex items-center gap-2 px-2 py-1 rounded text-xs font-mono text-muted-foreground"
                  >
                    <span className="w-5 text-right text-muted-foreground">
                      {idx + 1}
                    </span>
                    <span className="text-accent">
                      {formatLayerPreview(layer)}
                    </span>
                  </div>
                ))}
              </div>

              {/* Warnings */}
              {result.warnings?.length > 0 && (
                <div className="text-xs space-y-0.5 text-warning">
                  {result.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
                </div>
              )}

              {/* Provenance */}
              <div className="text-xs space-y-0.5 text-muted-foreground">
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
          <div className="p-4 border-t shrink-0 flex gap-2 border-border">
            <button
              onClick={() => handleApply('replace')}
              className="flex-1 py-2.5 rounded-lg font-medium text-sm bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white shadow-md hover:shadow-lg transition-all active:scale-[0.98]"
            >
              ✅ {canvasHasLayers ? (t.applyReplace || 'Replace canvas') : (t.applyToCanvas || 'Apply to Canvas')}
            </button>
            {canvasHasLayers && (
              <button
                onClick={() => handleApply('append')}
                className="px-4 py-2.5 rounded-lg font-medium text-sm transition-colors bg-muted text-foreground hover:bg-border"
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
