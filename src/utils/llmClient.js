/**
 * LLM API client: a unified interface for OpenAI, Google Gemini and Anthropic Claude.
 *
 * All calls happen directly from the browser (no backend proxy).
 * The user's API key never leaves their device except to the
 * provider's own endpoint.
 *
 * Model IDs are verified against provider docs as of 2026-08-17. Because a
 * BYOK tool has no way to hot-patch a retired model, every call accepts a
 * caller-supplied `model` override and a retired model surfaces as a distinct
 * MODEL_NOT_FOUND error telling the user exactly what to do.
 */

const OPENAI_URL = 'https://api.openai.com/v1/chat/completions';
const GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta/models';
const ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages';

/**
 * Default models: the current, cheapest capable tier per provider.
 *
 * openai  gpt-5.6-luna         GA 2026-07-09, cost-optimised GPT-5.6 tier
 * gemini  gemini-3.5-flash-lite stable GA, free tier, fastest 3.5 tier
 * claude  claude-haiku-4-5     replaces claude-3-5-haiku, retired 2026-02-19
 */
export const DEFAULT_MODELS = {
  openai: 'gpt-5.6-luna',
  gemini: 'gemini-3.5-flash-lite',
  claude: 'claude-haiku-4-5',
};

// Reasoning models spend output tokens on hidden reasoning before writing a
// single visible character, so the ceiling has to clear both.
const MAX_OUTPUT_TOKENS = 8192;

const REQUEST_TIMEOUT = 60_000; // 60s, because reasoning models routinely exceed 30s

/**
 * GPT-5.x and the o-series renamed `max_tokens` to `max_completion_tokens`
 * and reject `temperature` outright. Everything older keeps the classic
 * parameters. Exported for tests and so the UI can explain the difference.
 */
export function isOpenAIReasoningModel(model) {
  return /^(gpt-5|o[1-9])/i.test(String(model || ''));
}

/**
 * Build the OpenAI request body for a given model family.
 * Split out from the request so it can be unit-tested without a network call.
 */
export function buildOpenAIBody(model, systemPrompt, userPrompt, { jsonMode = true } = {}) {
  const body = {
    model,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ],
  };

  if (isOpenAIReasoningModel(model)) {
    body.max_completion_tokens = MAX_OUTPUT_TOKENS;
    // temperature / top_p are rejected by this family, so omit them entirely.
  } else {
    body.max_tokens = MAX_OUTPUT_TOKENS;
    body.temperature = 0.3; // low temp → more deterministic architecture output
  }

  // The prompt already says "JSON", which json_object mode requires.
  if (jsonMode) {
    body.response_format = { type: 'json_object' };
  }

  return body;
}

/**
 * Unified LLM call.
 * Returns { content, usage, model } on success, throws LLMError on failure.
 */
export async function callLLM({ provider, apiKey, systemPrompt, userPrompt, model, signal }) {
  if (!apiKey || !apiKey.trim()) {
    throw new LLMError('NO_API_KEY', 'API key is required.');
  }

  const resolvedModel = (model && model.trim()) || DEFAULT_MODELS[provider];
  const key = apiKey.trim();

  if (provider === 'openai') {
    return callOpenAI(key, resolvedModel, systemPrompt, userPrompt, signal);
  }
  if (provider === 'gemini') {
    return callGemini(key, resolvedModel, systemPrompt, userPrompt, signal);
  }
  if (provider === 'claude') {
    return callClaude(key, resolvedModel, systemPrompt, userPrompt, signal);
  }

  throw new LLMError('UNKNOWN_PROVIDER', `Unknown provider: ${provider}`);
}


// ─── OpenAI ───────────────────────────────────────────

async function callOpenAI(apiKey, model, systemPrompt, userPrompt, signal, jsonMode = true) {
  const res = await fetchWithTimeout(OPENAI_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify(buildOpenAIBody(model, systemPrompt, userPrompt, { jsonMode })),
  }, signal);

  if (!res.ok) {
    const err = await parseOpenAIError(res, model);
    // Some models reject json_object mode. Retry once in plain-text mode
    // rather than failing a request the parser could have handled anyway.
    if (jsonMode && err.code === 'UNSUPPORTED_PARAM') {
      return callOpenAI(apiKey, model, systemPrompt, userPrompt, signal, false);
    }
    throw err;
  }

  const data = await res.json();
  const choice = data.choices?.[0];

  if (choice?.finish_reason === 'length' && !choice?.message?.content) {
    throw new LLMError(
      'TRUNCATED',
      'The model used its entire output budget on reasoning. Try a simpler request or a non-reasoning model.'
    );
  }
  if (!choice?.message?.content) {
    throw new LLMError('EMPTY_RESPONSE', 'The model returned an empty response.');
  }

  return {
    content: choice.message.content,
    model: data.model || model,
    usage: {
      promptTokens: data.usage?.prompt_tokens ?? 0,
      completionTokens: data.usage?.completion_tokens ?? 0,
    },
  };
}

async function parseOpenAIError(res, model) {
  const status = res.status;
  try {
    const body = await res.json();
    const msg = body?.error?.message || '';
    const param = body?.error?.param || '';

    if (status === 400 && /response_format|json/i.test(msg + param)) {
      return new LLMError('UNSUPPORTED_PARAM', msg);
    }
    if (status === 400 && /max_tokens|max_completion_tokens|temperature/i.test(msg + param)) {
      return new LLMError('UNSUPPORTED_PARAM', msg);
    }
    if (status === 401) return new LLMError('INVALID_KEY', 'Invalid API key. Please check your OpenAI key.');
    if (status === 403) return new LLMError('INVALID_KEY', msg || 'This API key is not permitted to use this model.');
    if (status === 404) return modelNotFound(model, msg);
    if (status === 429) {
      return new LLMError('RATE_LIMIT', /quota|billing/i.test(msg)
        ? 'API quota exceeded. Check your OpenAI billing.'
        : 'Rate limit hit. Please wait a moment.');
    }
    if (status >= 500) return new LLMError('SERVER_ERROR', 'OpenAI servers are temporarily unavailable.');
    return new LLMError('API_ERROR', msg || `Request failed (HTTP ${status})`);
  } catch {
    if (status === 404) return modelNotFound(model);
    return new LLMError('API_ERROR', `Request failed (HTTP ${status})`);
  }
}


// ─── Gemini ───────────────────────────────────────────

async function callGemini(apiKey, model, systemPrompt, userPrompt, signal) {
  // The key goes in a header, not the query string: URLs leak into browser
  // history, referrers and proxy logs in a way headers do not.
  const url = `${GEMINI_BASE}/${encodeURIComponent(model)}:generateContent`;

  const body = {
    system_instruction: { parts: [{ text: systemPrompt }] },
    contents: [{ role: 'user', parts: [{ text: userPrompt }] }],
    generationConfig: {
      temperature: 0.3,
      maxOutputTokens: MAX_OUTPUT_TOKENS, // thinking models need headroom
      responseMimeType: 'application/json',
    },
  };

  const res = await fetchWithTimeout(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-goog-api-key': apiKey,
    },
    body: JSON.stringify(body),
  }, signal);

  if (!res.ok) {
    throw await parseGeminiError(res, model);
  }

  const data = await res.json();

  // A prompt rejected by safety filters returns 200 with no candidates.
  const blockReason = data.promptFeedback?.blockReason;
  if (blockReason) {
    throw new LLMError('BLOCKED', `The request was blocked by Gemini's safety filters (${blockReason}).`);
  }

  const candidate = data.candidates?.[0];
  const finishReason = candidate?.finishReason;

  // Gemini 2.5+ emits "thinking" parts, so filter them out and keep only text.
  const parts = candidate?.content?.parts || [];
  const text = parts.filter(p => !p.thought && p.text).map(p => p.text).join('');

  if (!text) {
    if (finishReason === 'MAX_TOKENS') {
      throw new LLMError(
        'TRUNCATED',
        'The model used its entire output budget on reasoning. Try a simpler request.'
      );
    }
    if (finishReason === 'SAFETY' || finishReason === 'RECITATION') {
      throw new LLMError('BLOCKED', `Response withheld by Gemini's safety filters (${finishReason}).`);
    }
    throw new LLMError('EMPTY_RESPONSE', 'The model returned an empty response.');
  }

  return {
    content: text,
    model: data.modelVersion || model,
    usage: {
      promptTokens: data.usageMetadata?.promptTokenCount ?? 0,
      completionTokens: data.usageMetadata?.candidatesTokenCount ?? 0,
    },
  };
}

async function parseGeminiError(res, model) {
  const status = res.status;
  try {
    const body = await res.json();
    const msg = body?.error?.message || '';

    if (status === 400 && /API[_ ]?KEY/i.test(msg)) {
      return new LLMError('INVALID_KEY', 'Invalid API key. Please check your Gemini key.');
    }
    if (status === 400) return new LLMError('API_ERROR', msg || 'Bad request. The model may not be available.');
    if (status === 401 || status === 403) {
      return new LLMError('INVALID_KEY', msg || 'API key is not authorized. Check your Google AI Studio settings.');
    }
    if (status === 404) return modelNotFound(model, msg);
    if (status === 429) return new LLMError('RATE_LIMIT', msg || 'Rate limit exceeded. Please wait and try again.');
    if (status >= 500) return new LLMError('SERVER_ERROR', 'Gemini servers are temporarily unavailable.');
    return new LLMError('API_ERROR', msg || `Request failed (HTTP ${status})`);
  } catch {
    if (status === 404) return modelNotFound(model);
    return new LLMError('API_ERROR', `Request failed (HTTP ${status})`);
  }
}


// ─── Claude (Anthropic) ──────────────────────────────

async function callClaude(apiKey, model, systemPrompt, userPrompt, signal) {
  const body = {
    model,
    max_tokens: MAX_OUTPUT_TOKENS,
    system: systemPrompt,
    messages: [
      { role: 'user', content: userPrompt },
    ],
    temperature: 0.3,
  };

  const res = await fetchWithTimeout(ANTHROPIC_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      // Required for browser-originated calls; without it Anthropic rejects
      // the request rather than sending CORS headers.
      'anthropic-dangerous-direct-browser-access': 'true',
    },
    body: JSON.stringify(body),
  }, signal);

  if (!res.ok) {
    throw await parseClaudeError(res, model);
  }

  const data = await res.json();

  if (data.stop_reason === 'refusal') {
    throw new LLMError('BLOCKED', 'Claude declined this request. Try rephrasing it.');
  }

  const textBlock = data.content?.find(b => b.type === 'text');
  if (!textBlock?.text) {
    if (data.stop_reason === 'max_tokens') {
      throw new LLMError('TRUNCATED', 'The response was cut off before any text was produced.');
    }
    throw new LLMError('EMPTY_RESPONSE', 'The model returned an empty response.');
  }

  return {
    content: textBlock.text,
    model: data.model || model,
    usage: {
      promptTokens: data.usage?.input_tokens ?? 0,
      completionTokens: data.usage?.output_tokens ?? 0,
    },
  };
}

async function parseClaudeError(res, model) {
  const status = res.status;
  try {
    const body = await res.json();
    const msg = body?.error?.message || '';

    if (status === 400 && /model/i.test(msg)) return modelNotFound(model, msg);
    if (status === 401) return new LLMError('INVALID_KEY', 'Invalid API key. Please check your Anthropic key.');
    if (status === 403) return new LLMError('INVALID_KEY', 'API key lacks permission. Check your Anthropic console.');
    if (status === 404) return modelNotFound(model, msg);
    if (status === 429) return new LLMError('RATE_LIMIT', 'Rate limit exceeded. Please wait a moment.');
    if (status === 500 || status === 529) return new LLMError('SERVER_ERROR', 'Anthropic servers are temporarily overloaded.');
    return new LLMError('API_ERROR', msg || `Request failed (HTTP ${status})`);
  } catch {
    if (status === 404) return modelNotFound(model);
    return new LLMError('API_ERROR', `Request failed (HTTP ${status})`);
  }
}


// ─── Helpers ──────────────────────────────────────────

function modelNotFound(model, detail = '') {
  return new LLMError(
    'MODEL_NOT_FOUND',
    `Model "${model}" is unavailable. It may have been retired, or your key may not have access to it. ` +
    `Set a different model under Advanced.${detail ? ` (${detail})` : ''}`
  );
}

async function fetchWithTimeout(url, options, externalSignal) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort('timeout'), REQUEST_TIMEOUT);

  // Let the caller cancel too, without losing the timeout.
  const onExternalAbort = () => controller.abort('cancelled');
  if (externalSignal) {
    if (externalSignal.aborted) onExternalAbort();
    else externalSignal.addEventListener('abort', onExternalAbort, { once: true });
  }

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (err) {
    if (externalSignal?.aborted) {
      throw new LLMError('CANCELLED', 'Request cancelled.');
    }
    if (err?.name === 'AbortError') {
      throw new LLMError('TIMEOUT', 'Request timed out. Please try again.');
    }
    throw new LLMError('NETWORK_ERROR', 'Network error. Check your internet connection.');
  } finally {
    clearTimeout(timer);
    externalSignal?.removeEventListener?.('abort', onExternalAbort);
  }
}

/**
 * Custom error class for structured error handling in the UI.
 * `code` maps to i18n keys; `message` is the English fallback.
 */
export class LLMError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'LLMError';
    this.code = code;
  }
}
