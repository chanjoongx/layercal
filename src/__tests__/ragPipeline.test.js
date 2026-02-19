import { describe, it, expect } from 'vitest';
import { retrieveArchitectures, parseAndValidateLayers } from '../utils/ragPipeline';

// ──────────────────────────────────────────────────
// retrieveArchitectures
// ──────────────────────────────────────────────────

describe('retrieveArchitectures', () => {

  it('returns CNN architectures for image-related queries', () => {
    const results = retrieveArchitectures('image classification');
    const ids = results.map(r => r.id);
    // at least one CNN model should rank in top 3
    const hasCNN = ids.some(id =>
      ['lenet-5', 'simple-cnn', 'vgg-style', 'mobilenet-lite'].includes(id)
    );
    expect(hasCNN).toBe(true);
  });

  it('returns NLP architectures for text-related queries', () => {
    const results = retrieveArchitectures('text classification sentiment');
    const ids = results.map(r => r.id);
    const hasNLP = ids.some(id =>
      ['lstm-text-clf', 'gru-seq-model', 'bert-encoder', 'self-attention-clf'].includes(id)
    );
    expect(hasNLP).toBe(true);
  });

  it('returns transformer models for "transformer encoder" query', () => {
    const results = retrieveArchitectures('transformer encoder');
    const ids = results.map(r => r.id);
    const hasTransformer = ids.some(id =>
      ['bert-encoder', 'small-transformer'].includes(id)
    );
    expect(hasTransformer).toBe(true);
  });

  it('favors lightweight models for edge/mobile queries', () => {
    const results = retrieveArchitectures('edge mobile lightweight');
    const ids = results.map(r => r.id);
    expect(ids[0]).toBe('mobilenet-lite');
  });

  it('returns top-K results (default 3)', () => {
    const results = retrieveArchitectures('deep learning');
    expect(results.length).toBeLessThanOrEqual(3);
  });

  it('handles empty query gracefully', () => {
    const results = retrieveArchitectures('');
    expect(results.length).toBeGreaterThan(0);    // falls back to first K entries
  });

  it('handles query with only stop words', () => {
    const results = retrieveArchitectures('make a model for me please');
    expect(results.length).toBeGreaterThan(0);
  });
});


// ──────────────────────────────────────────────────
// parseAndValidateLayers
// ──────────────────────────────────────────────────

describe('parseAndValidateLayers', () => {

  // --- Successful parsing ---

  it('parses a clean JSON array', () => {
    const input = JSON.stringify([
      { type: 'conv2d', params: { in_channels: 3, out_channels: 64, kernel_size: 3, use_bias: true } },
      { type: 'relu', params: {} },
    ]);
    const { layers } = parseAndValidateLayers(input);
    expect(layers).toHaveLength(2);
    expect(layers[0].type).toBe('conv2d');
    expect(layers[1].type).toBe('relu');
  });

  it('handles markdown code fences', () => {
    const input = '```json\n[{"type":"linear","params":{"input_dim":256,"output_dim":128}}]\n```';
    const { layers } = parseAndValidateLayers(input);
    expect(layers).toHaveLength(1);
    expect(layers[0].type).toBe('linear');
  });

  it('handles triple backticks without json label', () => {
    const input = '```\n[{"type":"relu","params":{}}]\n```';
    const { layers } = parseAndValidateLayers(input);
    expect(layers).toHaveLength(1);
  });

  it('handles extra text around JSON', () => {
    const input = 'Here is the architecture:\n[{"type":"softmax","params":{}}]\nHope this helps!';
    const { layers } = parseAndValidateLayers(input);
    expect(layers).toHaveLength(1);
    expect(layers[0].type).toBe('softmax');
  });

  it('handles single backtick wrapping', () => {
    const input = '`[{"type":"relu","params":{}}]`';
    const { layers } = parseAndValidateLayers(input);
    expect(layers).toHaveLength(1);
  });

  it('handles LLM prefixing explanation before JSON', () => {
    const input = 'Based on your requirements, here is a suitable architecture for sentiment analysis:\n\n[{"type":"embedding","params":{"vocab_size":10000,"embedding_dim":128}},{"type":"lstm","params":{"input_size":128,"hidden_size":256,"num_layers":2,"bidirectional":false}},{"type":"linear","params":{"input_dim":256,"output_dim":64}},{"type":"relu","params":{}},{"type":"linear","params":{"input_dim":64,"output_dim":128}}]\n\nThis architecture uses an LSTM-based approach.';
    const { layers } = parseAndValidateLayers(input);
    expect(layers).toHaveLength(5);
    expect(layers[0].type).toBe('embedding');
  });

  it('strips <think> blocks from thinking models', () => {
    const input = '<think>I need to design a CNN for image classification. Let me consider the requirements...</think>\n[{"type":"conv2d","params":{"in_channels":3,"out_channels":64,"kernel_size":3}},{"type":"relu","params":{}}]';
    const { layers } = parseAndValidateLayers(input);
    expect(layers).toHaveLength(2);
    expect(layers[0].type).toBe('conv2d');
  });

  it('handles trailing comma before closing bracket', () => {
    const input = '[{"type":"relu","params":{}},{"type":"softmax","params":{}},]';
    const { layers } = parseAndValidateLayers(input);
    expect(layers).toHaveLength(2);
  });

  it('wraps a single object response into an array', () => {
    const input = '{"type":"linear","params":{"input_dim":512,"output_dim":256}}';
    const { layers } = parseAndValidateLayers(input);
    expect(layers).toHaveLength(1);
    expect(layers[0].type).toBe('linear');
  });


  // --- Parameter snapping ---

  it('snaps embedding_dim to nearest valid option', () => {
    const input = JSON.stringify([
      { type: 'embedding', params: { vocab_size: 10000, embedding_dim: 384 } },
    ]);
    const { layers } = parseAndValidateLayers(input);
    // 384 is between 256 and 512; nearest is 256 (distance 128) vs 512 (distance 128) — tie goes to 256 (earlier in iteration)
    expect([256, 512]).toContain(layers[0].params.embedding_dim);
  });

  it('snaps out-of-range conv kernel to nearest valid option', () => {
    const input = JSON.stringify([
      { type: 'conv2d', params: { in_channels: 3, out_channels: 64, kernel_size: 4, use_bias: true } },
    ]);
    const { layers } = parseAndValidateLayers(input);
    // 4 → nearest is 3 or 5 (both distance 1); 3 wins (earlier)
    expect([3, 5]).toContain(layers[0].params.kernel_size);
  });

  it('fills missing params with defaults', () => {
    const input = JSON.stringify([
      { type: 'lstm', params: { hidden_size: 512 } },
    ]);
    const { layers } = parseAndValidateLayers(input);
    expect(layers[0].params.input_size).toBe(128);       // default
    expect(layers[0].params.num_layers).toBe(1);          // default
    expect(layers[0].params.bidirectional).toBe(false);   // default
    expect(layers[0].params.hidden_size).toBe(512);       // provided
  });

  it('clamps dropout rate to [0, 1] range', () => {
    const input = JSON.stringify([
      { type: 'dropout', params: { rate: 1.5 } },
    ]);
    const { layers } = parseAndValidateLayers(input);
    expect(layers[0].params.rate).toBe(1.0);
  });

  it('rounds dropout rate to nearest 0.1', () => {
    const input = JSON.stringify([
      { type: 'dropout', params: { rate: 0.27 } },
    ]);
    const { layers } = parseAndValidateLayers(input);
    expect(layers[0].params.rate).toBe(0.3);
  });

  it('coerces use_bias to boolean', () => {
    const input = JSON.stringify([
      { type: 'linear', params: { input_dim: 256, output_dim: 128, use_bias: 1 } },
    ]);
    const { layers } = parseAndValidateLayers(input);
    expect(layers[0].params.use_bias).toBe(true);
  });

  it('coerces bidirectional to boolean', () => {
    const input = JSON.stringify([
      { type: 'lstm', params: { input_size: 128, hidden_size: 256, num_layers: 1, bidirectional: 'true' } },
    ]);
    const { layers } = parseAndValidateLayers(input);
    expect(layers[0].params.bidirectional).toBe(true);
  });

  it('correctly handles string "false" for boolean fields', () => {
    const input = JSON.stringify([
      { type: 'lstm', params: { input_size: 128, hidden_size: 256, num_layers: 1, bidirectional: 'false' } },
    ]);
    const { layers } = parseAndValidateLayers(input);
    expect(layers[0].params.bidirectional).toBe(false);
  });

  it('correctly handles string "false" for use_bias', () => {
    const input = JSON.stringify([
      { type: 'linear', params: { input_dim: 256, output_dim: 128, use_bias: 'false' } },
    ]);
    const { layers } = parseAndValidateLayers(input);
    expect(layers[0].params.use_bias).toBe(false);
  });

  it('validates vocab_size as positive integer', () => {
    const input = JSON.stringify([
      { type: 'embedding', params: { vocab_size: -100, embedding_dim: 256 } },
    ]);
    const { layers } = parseAndValidateLayers(input);
    expect(layers[0].params.vocab_size).toBeGreaterThan(0);
  });


  // --- Filtering invalid layers ---

  it('skips unknown layer types and produces warnings', () => {
    const input = JSON.stringify([
      { type: 'conv2d', params: { in_channels: 3, out_channels: 64, kernel_size: 3, use_bias: true } },
      { type: 'superduper_layer', params: {} },
      { type: 'relu', params: {} },
    ]);
    const { layers, warnings } = parseAndValidateLayers(input);
    expect(layers).toHaveLength(2);
    expect(warnings.length).toBeGreaterThan(0);
    expect(warnings[0]).toContain('superduper_layer');
  });

  it('handles layer type with wrong casing', () => {
    const input = JSON.stringify([
      { type: 'Conv2D', params: { in_channels: 3, out_channels: 64, kernel_size: 3 } },
    ]);
    const { layers } = parseAndValidateLayers(input);
    expect(layers).toHaveLength(1);
    expect(layers[0].type).toBe('conv2d');
  });


  // --- Error cases ---

  it('throws on completely non-JSON text', () => {
    expect(() => parseAndValidateLayers('This is not JSON at all'))
      .toThrow();
  });

  it('throws on empty array', () => {
    expect(() => parseAndValidateLayers('[]'))
      .toThrow('non-empty');
  });

  it('throws when no valid layers survive filtering', () => {
    const input = JSON.stringify([
      { type: 'nonexistent', params: {} },
      { type: 'also_fake', params: {} },
    ]);
    expect(() => parseAndValidateLayers(input))
      .toThrow('No valid layers');
  });

  it('throws on invalid JSON syntax', () => {
    expect(() => parseAndValidateLayers('[{type: broken}]'))
      .toThrow();
  });

  it('skips null entries in the array', () => {
    const input = JSON.stringify([
      null,
      { type: 'relu', params: {} },
      undefined,
    ]);
    // JSON.stringify turns undefined to null in arrays
    const { layers } = parseAndValidateLayers(input);
    expect(layers).toHaveLength(1);
  });

  it('handles layer with no params object', () => {
    const input = JSON.stringify([
      { type: 'relu' },    // params key missing entirely
    ]);
    const { layers } = parseAndValidateLayers(input);
    expect(layers).toHaveLength(1);
    expect(layers[0].params).toEqual({});
  });


  // --- Cross-layer consistency ---

  it('fixes Conv2D + BatchNorm — BN matches conv output when valid', () => {
    const input = JSON.stringify([
      { type: 'conv2d', params: { in_channels: 3, out_channels: 32, kernel_size: 3 } },
      { type: 'batchnorm', params: { num_features: 64 } },
    ]);
    const { layers, warnings } = parseAndValidateLayers(input);
    // Conv out=32 is now valid for BN [16,32,64,...] → BN fixed to 32, Conv stays 32
    expect(layers[0].params.out_channels).toBe(32);
    expect(layers[1].params.num_features).toBe(32);
    expect(warnings.some(w => w.includes('BatchNorm'))).toBe(true);
  });

  it('fixes BatchNorm to match Conv2D(128) output without upstream change', () => {
    const input = JSON.stringify([
      { type: 'conv2d', params: { in_channels: 64, out_channels: 128, kernel_size: 3 } },
      { type: 'batchnorm', params: { num_features: 64 } },
    ]);
    const { layers, warnings } = parseAndValidateLayers(input);
    // Conv out=128 is valid for BN → BN fixed to 128, no upstream change needed
    expect(layers[0].params.out_channels).toBe(128);
    expect(layers[1].params.num_features).toBe(128);
    expect(warnings.some(w => w.includes('BatchNorm'))).toBe(true);
  });

  it('fixes Linear input_dim to match preceding Linear output_dim', () => {
    const input = JSON.stringify([
      { type: 'linear', params: { input_dim: 512, output_dim: 128 } },
      { type: 'relu', params: {} },
      { type: 'linear', params: { input_dim: 256, output_dim: 64 } },
    ]);
    const { layers, warnings } = parseAndValidateLayers(input);
    expect(layers[2].params.input_dim).toBe(128);
  });

  it('fixes LSTM input_size to match Embedding embedding_dim', () => {
    const input = JSON.stringify([
      { type: 'embedding', params: { vocab_size: 10000, embedding_dim: 256 } },
      { type: 'lstm', params: { input_size: 128, hidden_size: 512, num_layers: 1, bidirectional: false } },
    ]);
    const { layers, warnings } = parseAndValidateLayers(input);
    expect(layers[1].params.input_size).toBe(256);
  });

  it('handles bidirectional LSTM output to Linear', () => {
    const input = JSON.stringify([
      { type: 'embedding', params: { vocab_size: 10000, embedding_dim: 128 } },
      { type: 'lstm', params: { input_size: 128, hidden_size: 256, num_layers: 1, bidirectional: true } },
      { type: 'linear', params: { input_dim: 256, output_dim: 128 } },
    ]);
    const { layers } = parseAndValidateLayers(input);
    expect(layers[2].params.input_dim).toBe(512);
  });

  it('reconciles upstream through passthrough layers', () => {
    const input = JSON.stringify([
      { type: 'conv2d', params: { in_channels: 3, out_channels: 32, kernel_size: 3 } },
      { type: 'relu', params: {} },
      { type: 'dropout', params: { rate: 0.5 } },
      { type: 'batchnorm', params: { num_features: 256 } },
    ]);
    const { layers, warnings } = parseAndValidateLayers(input);
    // Conv out=32 → relu → dropout → BN. 32 is now valid for BN → BN fixed to 32, Conv stays 32.
    expect(layers[0].params.out_channels).toBe(32);
    expect(layers[3].params.num_features).toBe(32);
  });

  it('no fix needed when dims already match', () => {
    const input = JSON.stringify([
      { type: 'conv2d', params: { in_channels: 3, out_channels: 128, kernel_size: 3 } },
      { type: 'batchnorm', params: { num_features: 128 } },
    ]);
    const { layers, warnings } = parseAndValidateLayers(input);
    expect(layers[0].params.out_channels).toBe(128);
    expect(layers[1].params.num_features).toBe(128);
    // no cross-layer warnings
    expect(warnings.filter(w => w.includes('fixed') || w.includes('adjusted'))).toHaveLength(0);
  });
});