/**
 * LLM API Client — unified interface for OpenAI, Google Gemini, and Anthropic Claude
 *
 * All calls happen directly from the browser (no backend proxy).
 * The user's API key never leaves their device except to the
 * provider's own endpoint.
 */

const OPENAI_URL = 'https://api.openai.com/v1/chat/completions';
const GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta/models';
const ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages';

// Default models — picked for cost-efficiency and quality balance
const DEFAULT_MODELS = {
  openai: 'gpt-4o-mini',
  gemini: 'gemini-2.5-flash-lite',
  claude: 'claude-3-5-haiku-20241022',
};

const REQUEST_TIMEOUT = 30_000; // 30 seconds

/**
 * Unified LLM call.
 * Returns { content, usage } on success, throws on failure.
 */
export async function callLLM({ provider, apiKey, systemPrompt, userPrompt, model }) {
  if (!apiKey || !apiKey.trim()) {
    throw new LLMError('NO_API_KEY', 'API key is required.');
  }

  const resolvedModel = model || DEFAULT_MODELS[provider];

  if (provider === 'openai') {
    return callOpenAI(apiKey.trim(), resolvedModel, systemPrompt, userPrompt);
  }
  if (provider === 'gemini') {
    return callGemini(apiKey.trim(), resolvedModel, systemPrompt, userPrompt);
  }
  if (provider === 'claude') {
    return callClaude(apiKey.trim(), resolvedModel, systemPrompt, userPrompt);
  }

  throw new LLMError('UNKNOWN_PROVIDER', `Unknown provider: ${provider}`);
}


// ─── OpenAI ───────────────────────────────────────────

async function callOpenAI(apiKey, model, systemPrompt, userPrompt) {
  const body = {
    model,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ],
    temperature: 0.3,    // low temp → more deterministic architecture output
    max_tokens: 2048,
  };

  const res = await fetchWithTimeout(OPENAI_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    throw await parseOpenAIError(res);
  }

  const data = await res.json();
  const choice = data.choices?.[0];
  if (!choice?.message?.content) {
    throw new LLMError('EMPTY_RESPONSE', 'The model returned an empty response.');
  }

  return {
    content: choice.message.content,
    usage: {
      promptTokens: data.usage?.prompt_tokens ?? 0,
      completionTokens: data.usage?.completion_tokens ?? 0,
    },
  };
}

async function parseOpenAIError(res) {
  const status = res.status;
  try {
    const body = await res.json();
    const msg = body?.error?.message || '';

    if (status === 401) return new LLMError('INVALID_KEY', 'Invalid API key. Please check your OpenAI key.');
    if (status === 429) return new LLMError('RATE_LIMIT', msg.includes('quota') ? 'API quota exceeded. Check your billing.' : 'Rate limit hit. Please wait a moment.');
    if (status === 500 || status === 503) return new LLMError('SERVER_ERROR', 'OpenAI servers are temporarily unavailable.');
    return new LLMError('API_ERROR', msg || `Request failed (HTTP ${status})`);
  } catch {
    return new LLMError('API_ERROR', `Request failed (HTTP ${status})`);
  }
}


// ─── Gemini ───────────────────────────────────────────

async function callGemini(apiKey, model, systemPrompt, userPrompt) {
  const url = `${GEMINI_BASE}/${model}:generateContent?key=${apiKey}`;

  const body = {
    system_instruction: { parts: [{ text: systemPrompt }] },
    contents: [{ parts: [{ text: userPrompt }] }],
    generationConfig: {
      temperature: 0.3,
      maxOutputTokens: 8192,    // thinking models use tokens for reasoning, need headroom
    },
  };

  const res = await fetchWithTimeout(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    throw await parseGeminiError(res);
  }

  const data = await res.json();

  // Gemini 2.5 Flash has "thinking" parts — filter them out, keep only text
  const parts = data.candidates?.[0]?.content?.parts || [];
  const textParts = parts.filter(p => !p.thought && p.text);
  const text = textParts.map(p => p.text).join('');

  if (!text) {
    throw new LLMError('EMPTY_RESPONSE', 'The model returned an empty response.');
  }

  return {
    content: text,
    usage: {
      promptTokens: data.usageMetadata?.promptTokenCount ?? 0,
      completionTokens: data.usageMetadata?.candidatesTokenCount ?? 0,
    },
  };
}

async function parseGeminiError(res) {
  const status = res.status;
  try {
    const body = await res.json();
    const msg = body?.error?.message || '';

    if (status === 400 && msg.includes('API_KEY')) return new LLMError('INVALID_KEY', 'Invalid API key. Please check your Gemini key.');
    if (status === 400) return new LLMError('API_ERROR', msg || 'Bad request. The model may not be available.');
    if (status === 403) return new LLMError('INVALID_KEY', msg || 'API key is not authorized. Check your Google AI Studio settings.');
    if (status === 404) return new LLMError('API_ERROR', 'Model not found. It may have been deprecated.');
    if (status === 429) return new LLMError('RATE_LIMIT', msg || 'Rate limit exceeded. Please wait and try again.');
    if (status === 500 || status === 503) return new LLMError('SERVER_ERROR', 'Gemini servers are temporarily unavailable.');
    return new LLMError('API_ERROR', msg || `Request failed (HTTP ${status})`);
  } catch {
    return new LLMError('API_ERROR', `Request failed (HTTP ${status})`);
  }
}


// ─── Claude (Anthropic) ──────────────────────────────

async function callClaude(apiKey, model, systemPrompt, userPrompt) {
  const body = {
    model,
    max_tokens: 2048,
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
      'anthropic-dangerous-direct-browser-access': 'true',
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    throw await parseClaudeError(res);
  }

  const data = await res.json();
  const textBlock = data.content?.find(b => b.type === 'text');
  if (!textBlock?.text) {
    throw new LLMError('EMPTY_RESPONSE', 'The model returned an empty response.');
  }

  return {
    content: textBlock.text,
    usage: {
      promptTokens: data.usage?.input_tokens ?? 0,
      completionTokens: data.usage?.output_tokens ?? 0,
    },
  };
}

async function parseClaudeError(res) {
  const status = res.status;
  try {
    const body = await res.json();
    const msg = body?.error?.message || '';

    if (status === 401) return new LLMError('INVALID_KEY', 'Invalid API key. Please check your Anthropic key.');
    if (status === 403) return new LLMError('INVALID_KEY', 'API key lacks permission. Check your Anthropic console.');
    if (status === 429) return new LLMError('RATE_LIMIT', 'Rate limit exceeded. Please wait a moment.');
    if (status === 500 || status === 529) return new LLMError('SERVER_ERROR', 'Anthropic servers are temporarily overloaded.');
    return new LLMError('API_ERROR', msg || `Request failed (HTTP ${status})`);
  } catch {
    return new LLMError('API_ERROR', `Request failed (HTTP ${status})`);
  }
}


// ─── Helpers ──────────────────────────────────────────

async function fetchWithTimeout(url, options) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new LLMError('TIMEOUT', 'Request timed out. Please try again.');
    }
    throw new LLMError('NETWORK_ERROR', 'Network error. Check your internet connection.');
  } finally {
    clearTimeout(timer);
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