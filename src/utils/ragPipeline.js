/**
 * RAG Pipeline for AI Architecture Advisor
 *
 * Implements a Retrieval-Augmented Generation pattern:
 *   1. Retrieve  — keyword-match relevant architectures from the KB
 *   2. Augment   — build a system prompt with layer schema + references
 *   3. Generate  — call LLM via llmClient
 *   4. Validate  — parse JSON, snap invalid params, filter bad layers
 */

import { callLLM } from './llmClient';
import ARCHITECTURE_KB from '../config/architectureKB';

// ─────────────────────────────────────────────────────
// 1. RETRIEVAL — keyword search over the knowledge base
// ─────────────────────────────────────────────────────

/**
 * Score each KB entry against the user query by counting
 * tag matches. Returns top-K entries sorted by relevance.
 */
export function retrieveArchitectures(query, topK = 3) {
  const tokens = tokenize(query);
  if (tokens.length === 0) return ARCHITECTURE_KB.slice(0, topK);

  const scored = ARCHITECTURE_KB.map(arch => {
    let score = 0;
    const tagSet = arch.tags.join(' ') + ' ' + arch.category + ' ' + arch.name.toLowerCase();

    for (const token of tokens) {
      // exact tag hit → strong signal
      if (arch.tags.includes(token)) {
        score += 3;
      }
      // partial match in tag string or description
      else if (tagSet.includes(token)) {
        score += 1.5;
      }
      // substring of description
      else if (arch.description.toLowerCase().includes(token)) {
        score += 1;
      }
    }
    return { arch, score };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).map(s => s.arch);
}

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')    // strip punctuation but keep hyphens
    .split(/\s+/)
    .filter(t => t.length > 1)
    // drop stop words that add noise to scoring
    .filter(t => !STOP_WORDS.has(t));
}

const STOP_WORDS = new Set([
  'a', 'an', 'the', 'for', 'and', 'or', 'but', 'is', 'are', 'was',
  'with', 'that', 'this', 'to', 'of', 'in', 'on', 'it', 'my', 'me',
  'want', 'need', 'make', 'create', 'build', 'design', 'please',
  'like', 'using', 'use', 'can', 'do', 'would', 'should', 'be', 'have',
  'model', 'network', 'neural', 'deep', 'learning', 'architecture',
]);


// ─────────────────────────────────────────────────────
// 2. PROMPT BUILDER — construct system & user prompts
// ─────────────────────────────────────────────────────

/**
 * Layer type schema for the LLM prompt.
 *
 * Derived from layerTypes.js field definitions.
 * Tells the LLM exactly which types and param values are valid.
 */
const LAYER_SCHEMA = `Available layer types and their valid parameter values:

embedding:
  - vocab_size: positive integer (common values: 5000, 10000, 30000, 50000)
  - embedding_dim: one of [64, 128, 256, 512, 768, 1024]

linear:
  - input_dim: one of [64, 128, 256, 512, 768, 1024, 2048]
  - output_dim: one of [64, 128, 256, 512, 768, 1024, 2048]
  - use_bias: true or false (default: true)

conv2d:
  - in_channels: one of [1, 3, 16, 32, 64, 128, 256]
  - out_channels: one of [16, 32, 64, 128, 256, 512]
  - kernel_size: one of [1, 3, 5, 7]
  - use_bias: true or false (default: true)

lstm:
  - input_size: one of [64, 128, 256, 512, 768]
  - hidden_size: one of [128, 256, 512, 768, 1024]
  - num_layers: one of [1, 2, 3, 4]
  - bidirectional: true or false (default: false)

gru:
  - input_size: one of [64, 128, 256, 512, 768]
  - hidden_size: one of [128, 256, 512, 768, 1024]
  - num_layers: one of [1, 2, 3, 4]
  - bidirectional: true or false (default: false)

transformer:
  - d_model: one of [256, 512, 768, 1024]
  - num_heads: one of [4, 8, 12, 16]
  - d_ff: one of [1024, 2048, 3072, 4096]

attention:
  - d_model: one of [256, 512, 768, 1024]
  - num_heads: one of [4, 8, 12, 16]

batchnorm:
  - num_features: one of [16, 32, 64, 128, 256, 512, 768, 1024]

layernorm:
  - normalized_shape: one of [128, 256, 512, 768, 1024]

dropout:
  - rate: number between 0.0 and 1.0 (step 0.1)

maxpool2d:
  - kernel_size: one of [2, 3, 4]

avgpool2d:
  - kernel_size: one of [2, 3, 4]

relu:
  (no parameters)

softmax:
  (no parameters)`;


export function buildPrompt(userQuery, retrievedArchs) {
  const refSection = retrievedArchs.length > 0
    ? formatReferences(retrievedArchs)
    : 'No closely matching references found. Design from first principles.';

  const systemPrompt = `You are an expert deep learning architect. The user will describe what kind of neural network they need. Design a model using ONLY the layer types below.

${LAYER_SCHEMA}

REFERENCE ARCHITECTURES (retrieved from knowledge base):
${refSection}

RULES:
1. Respond with ONLY a JSON array. No text before or after. No explanation. No markdown.
2. Each object must have "type" (string) and "params" (object) fields.
3. Use ONLY the layer types and param values listed above.
4. Ensure layers connect logically:
   - Conv2D out_channels should match next BatchNorm num_features
   - LSTM/GRU input_size should match the previous layer's output dimension
   - Linear input_dim should match the preceding layer's output
5. If the user specifies a parameter budget, design accordingly.
6. Include activation functions (relu) and regularization (dropout, batchnorm) where appropriate.
7. Keep architectures practical — between 4 and 25 layers.

CRITICAL: Your entire response must be parseable as JSON. Do not include any reasoning, thinking, or explanation. Start with [ and end with ].`;

  const userPrompt = `Design a neural network for: ${userQuery}

Respond with ONLY a JSON array. Start with [ and end with ].`;

  return { systemPrompt, userPrompt };
}

function formatReferences(archs) {
  return archs.map(a => {
    const layerSummary = a.layers
      .map(l => `  ${l.type}(${JSON.stringify(l.params)})`)
      .join('\n');
    return `[${a.name}] ${a.description} (${a.approxParams} params)\n${layerSummary}`;
  }).join('\n\n');
}


// ─────────────────────────────────────────────────────
// 3. OUTPUT PARSER — LLM response → validated layers
// ─────────────────────────────────────────────────────

// Valid select-field options, mirroring layerTypes.js exactly
const VALID_OPTIONS = {
  embedding: { embedding_dim: [64, 128, 256, 512, 768, 1024] },
  linear: { input_dim: [64, 128, 256, 512, 768, 1024, 2048], output_dim: [64, 128, 256, 512, 768, 1024, 2048] },
  conv2d: { in_channels: [1, 3, 16, 32, 64, 128, 256], out_channels: [16, 32, 64, 128, 256, 512], kernel_size: [1, 3, 5, 7] },
  lstm: { input_size: [64, 128, 256, 512, 768], hidden_size: [128, 256, 512, 768, 1024], num_layers: [1, 2, 3, 4] },
  gru: { input_size: [64, 128, 256, 512, 768], hidden_size: [128, 256, 512, 768, 1024], num_layers: [1, 2, 3, 4] },
  transformer: { d_model: [256, 512, 768, 1024], num_heads: [4, 8, 12, 16], d_ff: [1024, 2048, 3072, 4096] },
  attention: { d_model: [256, 512, 768, 1024], num_heads: [4, 8, 12, 16] },
  batchnorm: { num_features: [16, 32, 64, 128, 256, 512, 768, 1024] },
  layernorm: { normalized_shape: [128, 256, 512, 768, 1024] },
  maxpool2d: { kernel_size: [2, 3, 4] },
  avgpool2d: { kernel_size: [2, 3, 4] },
};

const VALID_LAYER_TYPES = new Set([
  'embedding', 'linear', 'conv2d', 'lstm', 'gru',
  'transformer', 'attention', 'batchnorm', 'layernorm',
  'dropout', 'maxpool2d', 'avgpool2d', 'relu', 'softmax',
]);

// Default params for each layer type (mirrors layerTypes.js)
const DEFAULT_PARAMS = {
  embedding: { vocab_size: 10000, embedding_dim: 128 },
  linear: { input_dim: 512, output_dim: 256, use_bias: true },
  conv2d: { in_channels: 3, out_channels: 64, kernel_size: 3, use_bias: true },
  lstm: { input_size: 128, hidden_size: 256, num_layers: 1, bidirectional: false },
  gru: { input_size: 128, hidden_size: 256, num_layers: 1, bidirectional: false },
  transformer: { d_model: 512, num_heads: 8, d_ff: 2048, dropout: 0.1 },
  attention: { d_model: 512, num_heads: 8 },
  batchnorm: { num_features: 128 },
  layernorm: { normalized_shape: 512 },
  dropout: { rate: 0.1 },
  maxpool2d: { kernel_size: 2 },
  avgpool2d: { kernel_size: 2 },
  relu: {},
  softmax: {},
};


/**
 * Parse the LLM's raw text output into a validated layer array.
 * Handles markdown code blocks, invalid types, and out-of-range params.
 */
export function parseAndValidateLayers(rawText) {
  const jsonStr = extractJSON(rawText);
  if (!jsonStr) {
    throw new Error('Could not extract a JSON array from the response.');
  }

  let parsed;
  try {
    parsed = JSON.parse(jsonStr);
  } catch {
    throw new Error('Response contained invalid JSON.');
  }

  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new Error('Expected a non-empty JSON array of layers.');
  }

  const validated = [];
  const warnings = [];

  for (let i = 0; i < parsed.length; i++) {
    const raw = parsed[i];
    if (!raw || typeof raw !== 'object') continue;

    const type = String(raw.type || '').toLowerCase().trim();
    if (!VALID_LAYER_TYPES.has(type)) {
      warnings.push(`Layer ${i + 1}: unknown type "${raw.type}" — skipped.`);
      continue;
    }

    const defaults = DEFAULT_PARAMS[type];
    const rawParams = (raw.params && typeof raw.params === 'object') ? raw.params : {};

    // merge: LLM output → default fill for missing keys
    const merged = { ...defaults };
    for (const [key, val] of Object.entries(rawParams)) {
      if (key in defaults) {
        merged[key] = val;
      }
      // silently drop unknown param keys
    }

    // snap select-field values to nearest valid option
    const selectOpts = VALID_OPTIONS[type];
    if (selectOpts) {
      for (const [field, options] of Object.entries(selectOpts)) {
        if (field in merged) {
          merged[field] = snapToNearest(merged[field], options);
        }
      }
    }

    // validate special cases
    if (type === 'dropout') {
      merged.rate = clampDropoutRate(merged.rate);
    }
    if ('use_bias' in merged) {
      merged.use_bias = merged.use_bias === true || merged.use_bias === 'true' || merged.use_bias === 1;
    }
    if ('bidirectional' in merged) {
      merged.bidirectional = merged.bidirectional === true || merged.bidirectional === 'true' || merged.bidirectional === 1;
    }
    if (type === 'embedding' && 'vocab_size' in merged) {
      merged.vocab_size = Math.max(1, Math.round(Number(merged.vocab_size) || 10000));
    }

    validated.push({ type, params: merged });
  }

  if (validated.length === 0) {
    throw new Error('No valid layers found after parsing.');
  }

  // Cross-layer consistency fix
  fixCrossLayerDimensions(validated, warnings);

  return { layers: validated, warnings };
}


/**
 * Track output dimensions through the layer stack and fix mismatches.
 * When a "must-match" layer (BN, LN) can't accept the upstream dim,
 * we snap to the nearest valid value and ALSO fix the upstream source.
 */
function fixCrossLayerDimensions(layers, warnings) {
  let dim = null;  // current output dimension (channels or features)

  for (let i = 0; i < layers.length; i++) {
    const layer = layers[i];
    const p = layer.params;

    switch (layer.type) {
      case 'conv2d':
        if (dim !== null && p.in_channels !== dim) {
          const fixed = snapToNearest(dim, VALID_OPTIONS.conv2d.in_channels);
          warnings.push(`Layer ${i + 1}: fixed Conv2D in_channels ${p.in_channels}→${fixed}`);
          p.in_channels = fixed;
        }
        dim = p.out_channels;
        break;

      case 'batchnorm': {
        if (dim !== null) {
          const target = snapToNearest(dim, VALID_OPTIONS.batchnorm.num_features);
          if (p.num_features !== target) {
            warnings.push(`Layer ${i + 1}: fixed BatchNorm num_features ${p.num_features}→${target}`);
            p.num_features = target;
          }
          if (target !== dim) {
            // dim wasn't valid for BN — also fix the upstream layer that produced it
            reconcileUpstream(layers, i - 1, target, warnings);
            dim = target;
          }
        }
        break;
      }

      case 'layernorm': {
        if (dim !== null) {
          const target = snapToNearest(dim, VALID_OPTIONS.layernorm.normalized_shape);
          if (p.normalized_shape !== target) {
            warnings.push(`Layer ${i + 1}: fixed LayerNorm normalized_shape ${p.normalized_shape}→${target}`);
            p.normalized_shape = target;
          }
          if (target !== dim) {
            reconcileUpstream(layers, i - 1, target, warnings);
            dim = target;
          }
        }
        break;
      }

      case 'linear':
        if (dim !== null && p.input_dim !== dim) {
          const fixed = snapToNearest(dim, VALID_OPTIONS.linear.input_dim);
          warnings.push(`Layer ${i + 1}: fixed Linear input_dim ${p.input_dim}→${fixed}`);
          p.input_dim = fixed;
        }
        dim = p.output_dim;
        break;

      case 'embedding':
        dim = p.embedding_dim;
        break;

      case 'lstm':
        if (dim !== null && p.input_size !== dim) {
          const fixed = snapToNearest(dim, VALID_OPTIONS.lstm.input_size);
          warnings.push(`Layer ${i + 1}: fixed LSTM input_size ${p.input_size}→${fixed}`);
          p.input_size = fixed;
        }
        dim = p.bidirectional ? p.hidden_size * 2 : p.hidden_size;
        break;

      case 'gru':
        if (dim !== null && p.input_size !== dim) {
          const fixed = snapToNearest(dim, VALID_OPTIONS.gru.input_size);
          warnings.push(`Layer ${i + 1}: fixed GRU input_size ${p.input_size}→${fixed}`);
          p.input_size = fixed;
        }
        dim = p.bidirectional ? p.hidden_size * 2 : p.hidden_size;
        break;

      case 'transformer':
        if (dim !== null && p.d_model !== dim) {
          const fixed = snapToNearest(dim, VALID_OPTIONS.transformer.d_model);
          warnings.push(`Layer ${i + 1}: fixed Transformer d_model ${p.d_model}→${fixed}`);
          p.d_model = fixed;
        }
        dim = p.d_model;
        break;

      case 'attention':
        if (dim !== null && p.d_model !== dim) {
          const fixed = snapToNearest(dim, VALID_OPTIONS.attention.d_model);
          warnings.push(`Layer ${i + 1}: fixed Attention d_model ${p.d_model}→${fixed}`);
          p.d_model = fixed;
        }
        dim = p.d_model;
        break;

      // passthrough — don't change dim
      case 'relu':
      case 'softmax':
      case 'dropout':
      case 'maxpool2d':
      case 'avgpool2d':
        break;
    }
  }
}

/**
 * Walk backward through passthrough layers to find the source layer
 * and fix its output dimension to `target`.
 */
function reconcileUpstream(layers, fromIdx, target, warnings) {
  const PASSTHROUGH = new Set(['relu', 'softmax', 'dropout', 'maxpool2d', 'avgpool2d']);

  for (let j = fromIdx; j >= 0; j--) {
    const prev = layers[j];
    const pp = prev.params;

    if (PASSTHROUGH.has(prev.type)) continue;

    switch (prev.type) {
      case 'conv2d': {
        const fixed = snapToNearest(target, VALID_OPTIONS.conv2d.out_channels);
        if (pp.out_channels !== fixed) {
          warnings.push(`Layer ${j + 1}: adjusted Conv2D out_channels ${pp.out_channels}→${fixed}`);
          pp.out_channels = fixed;
        }
        return;
      }
      case 'linear': {
        const fixed = snapToNearest(target, VALID_OPTIONS.linear.output_dim);
        if (pp.output_dim !== fixed) {
          warnings.push(`Layer ${j + 1}: adjusted Linear output_dim ${pp.output_dim}→${fixed}`);
          pp.output_dim = fixed;
        }
        return;
      }
      case 'lstm': case 'gru': {
        const fixed = snapToNearest(target, VALID_OPTIONS[prev.type].hidden_size);
        if (pp.hidden_size !== fixed) {
          warnings.push(`Layer ${j + 1}: adjusted ${prev.type.toUpperCase()} hidden_size ${pp.hidden_size}→${fixed}`);
          pp.hidden_size = fixed;
        }
        return;
      }
      case 'embedding': {
        const fixed = snapToNearest(target, VALID_OPTIONS.embedding.embedding_dim);
        if (pp.embedding_dim !== fixed) {
          warnings.push(`Layer ${j + 1}: adjusted Embedding embedding_dim ${pp.embedding_dim}→${fixed}`);
          pp.embedding_dim = fixed;
        }
        return;
      }
      default:
        return;  // unknown source — can't fix, stop
    }
  }
}


function extractJSON(text) {
  let cleaned = text;

  // 1. strip thinking/reasoning blocks some models emit
  cleaned = cleaned.replace(/<think>[\s\S]*?<\/think>/gi, '');
  cleaned = cleaned.replace(/<thinking>[\s\S]*?<\/thinking>/gi, '');
  cleaned = cleaned.replace(/<reasoning>[\s\S]*?<\/reasoning>/gi, '');

  // 2. strip markdown fences
  cleaned = cleaned.replace(/```(?:json|JSON)?\s*([\s\S]*?)```/g, '$1');
  cleaned = cleaned.replace(/^`([\s\S]*)`$/, '$1');
  cleaned = cleaned.trim();

  // 3. try bracket-depth matching for outermost array
  const start = cleaned.indexOf('[');
  if (start !== -1) {
    let depth = 0;
    for (let i = start; i < cleaned.length; i++) {
      if (cleaned[i] === '[') depth++;
      else if (cleaned[i] === ']') depth--;
      if (depth === 0) {
        let slice = cleaned.slice(start, i + 1);
        // fix trailing commas before ] (common LLM mistake)
        slice = slice.replace(/,\s*]/g, ']');
        return slice;
      }
    }
    // unclosed bracket — try best effort
    const end = cleaned.lastIndexOf(']');
    if (end > start) return cleaned.slice(start, end + 1);
  }

  // 4. no array found — maybe the LLM returned a single object or comma-separated objects
  const firstBrace = cleaned.indexOf('{');
  const lastBrace = cleaned.lastIndexOf('}');
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    return '[' + cleaned.slice(firstBrace, lastBrace + 1) + ']';
  }

  return null;
}

function snapToNearest(value, options) {
  const num = Number(value);
  if (isNaN(num)) return options[0];

  let closest = options[0];
  let minDist = Math.abs(num - closest);

  for (let i = 1; i < options.length; i++) {
    const dist = Math.abs(num - options[i]);
    if (dist < minDist) {
      minDist = dist;
      closest = options[i];
    }
  }
  return closest;
}

function clampDropoutRate(val) {
  const num = Number(val);
  if (isNaN(num)) return 0.1;
  // round to nearest 0.1 and clamp
  return Math.min(1.0, Math.max(0.0, Math.round(num * 10) / 10));
}


// ─────────────────────────────────────────────────────
// 4. ORCHESTRATOR — tie it all together
// ─────────────────────────────────────────────────────

/**
 * Full RAG pipeline:
 *   query → retrieve → prompt → LLM → parse → validated layers
 *
 * @returns {{ layers, references, warnings, usage }}
 */
export async function generateArchitecture({ query, provider, apiKey, model }) {
  // Step 1: Retrieve
  const references = retrieveArchitectures(query, 3);

  // Step 2: Build prompt
  const { systemPrompt, userPrompt } = buildPrompt(query, references);

  // Step 3: Call LLM
  const llmResult = await callLLM({
    provider,
    apiKey,
    model,
    systemPrompt,
    userPrompt,
  });

  // Step 4: Parse & validate
  const { layers, warnings } = parseAndValidateLayers(llmResult.content);

  return {
    layers,
    references: references.map(r => ({ id: r.id, name: r.name })),
    warnings,
    usage: llmResult.usage,
  };
}
