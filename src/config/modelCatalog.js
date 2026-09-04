/**
 * Curated model catalogue for the AI Architecture Advisor.
 *
 * A BYOK tool with no backend cannot hot-patch itself when a provider retires
 * a model, so this file is a convenience rather than a gate: the free-text
 * override in Advanced still accepts any id, and `MODEL_NOT_FOUND` in
 * llmClient.js remains the safety net. What the catalogue buys is that the
 * common case — "which model should I pick, and what will it cost me?" — is
 * answered in the UI instead of in another tab.
 *
 * Prices are USD per million tokens, taken from the providers' own pricing
 * pages on the date below. They are shown to help choose a tier, not to
 * reconcile a bill.
 */

/** Date the ids and prices in this file were last checked against provider docs. */
export const CATALOG_VERIFIED_ON = '2026-09-04';

/**
 * @typedef {'fast'|'balanced'|'frontier'} ModelTier
 *
 * @typedef {{
 *   id: string,
 *   label: string,
 *   tier: ModelTier,
 *   inputPrice: number|null,   // USD / 1M input tokens, null when free-tier only
 *   outputPrice: number|null,
 *   note?: string,        // shown when there is no per-token price
 *   noteKey?: string,     // translation key for `note`
 * }} CatalogModel
 */

/** @type {Record<string, CatalogModel[]>} */
export const MODEL_CATALOG = {
  gemini: [
    {
      id: 'gemini-3.5-flash-lite',
      label: 'Gemini 3.5 Flash-Lite',
      tier: 'fast',
      inputPrice: null,
      outputPrice: null,
      note: 'Free tier',
      noteKey: 'freeTier',
    },
    {
      id: 'gemini-3.1-flash-lite',
      label: 'Gemini 3.1 Flash-Lite',
      tier: 'fast',
      inputPrice: null,
      outputPrice: null,
      note: 'Free tier',
      noteKey: 'freeTier',
    },
    // No note: these are not on the free tier, and an interface that falls back
    // to "Free tier" whenever a price is missing would say so untruthfully.
    {
      id: 'gemini-3.6-flash',
      label: 'Gemini 3.6 Flash',
      tier: 'balanced',
      inputPrice: null,
      outputPrice: null,
    },
    {
      id: 'gemini-3.8-flash',
      label: 'Gemini 3.8 Flash',
      tier: 'balanced',
      inputPrice: null,
      outputPrice: null,
    },
  ],

  openai: [
    {
      id: 'gpt-5.6-luna',
      label: 'GPT-5.6 Luna',
      tier: 'fast',
      inputPrice: 0.2,
      outputPrice: 1.2,
    },
    {
      id: 'gpt-5.6-terra',
      label: 'GPT-5.6 Terra',
      tier: 'balanced',
      inputPrice: 2,
      outputPrice: 12,
    },
    {
      id: 'gpt-5.6-sol',
      label: 'GPT-5.6 Sol',
      tier: 'frontier',
      inputPrice: 4,
      outputPrice: 20,
    },
    {
      id: 'gpt-6-astra',
      label: 'GPT-6 Astra',
      tier: 'frontier',
      inputPrice: 10,
      outputPrice: 50,
    },
  ],

  claude: [
    {
      id: 'claude-haiku-4-5',
      label: 'Claude Haiku 4.5',
      tier: 'fast',
      inputPrice: 1,
      outputPrice: 5,
    },
    {
      id: 'claude-sonnet-5',
      label: 'Claude Sonnet 5',
      tier: 'balanced',
      inputPrice: 2,
      outputPrice: 10,
    },
    {
      id: 'claude-opus-5',
      label: 'Claude Opus 5',
      tier: 'frontier',
      inputPrice: 5,
      outputPrice: 25,
    },
  ],
};

/** Every provider id the catalogue covers, in the order the UI lists them. */
export const CATALOG_PROVIDERS = Object.keys(MODEL_CATALOG);

/**
 * @param {string} provider
 * @returns {CatalogModel[]}
 */
export function modelsFor(provider) {
  return MODEL_CATALOG[provider] || [];
}

/**
 * @param {string} provider
 * @param {string} id
 * @returns {CatalogModel|null}
 */
export function findModel(provider, id) {
  return modelsFor(provider).find(m => m.id === id) || null;
}

/**
 * A one-line price summary, or null when the model has no published per-token
 * price (the Gemini free tier). Callers render `note` instead in that case.
 *
 * @param {CatalogModel} model
 */
export function priceLabel(model) {
  if (!model || model.inputPrice == null || model.outputPrice == null) return null;
  // Two decimals unless the figure is whole, so a column of prices lines up
  // and $1.20 does not render as $1.2.
  const fmt = (n) => (Number.isInteger(n) ? `$${n}` : `$${n.toFixed(2)}`);
  return `${fmt(model.inputPrice)} / ${fmt(model.outputPrice)} per 1M`;
}
