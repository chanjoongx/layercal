/**
 * Cross-layer dimension checking for hand-built models.
 *
 * The RAG pipeline already repairs dimensions in LLM output, but a model the
 * user assembles by clicking gets no such pass. Stock defaults alone produce a
 * broken stack (Conv2D emits 64 channels, BatchNorm defaults to 128), so the
 * canvas needs to say so rather than exporting code that cannot run.
 *
 * Read-only by design: nothing is auto-corrected, because a half-edited model
 * is a normal intermediate state and silently rewriting the user's numbers is
 * worse than pointing at them.
 */

/** Layers that leave the feature dimension untouched. */
const PASSTHROUGH = new Set(['relu', 'softmax', 'dropout', 'maxpool2d', 'avgpool2d']);

/**
 * Walk the stack tracking the running feature dimension and report the first
 * mismatch on each layer.
 *
 * A Conv2D feeding a Linear is compared against the channel count, because the
 * generated code inserts a global average pool at that transition, which
 * collapses (N, C, H, W) to (N, C).
 *
 * @param {Array<{type: string, params: object}>} layers
 * @returns {Map<number, { field: string, expected: number }>} keyed by layer index
 */
export function validateModelDimensions(layers) {
  const issues = new Map();
  if (!Array.isArray(layers)) return issues;

  let dim = null;

  layers.forEach((layer, index) => {
    const p = (layer && layer.params) || {};

    const expect = (field, value) => {
      const actual = Number(value);
      if (dim === null || !Number.isFinite(actual) || !Number.isFinite(dim)) return;
      if (actual !== dim) issues.set(index, { field, expected: dim });
    };

    const emit = (value) => {
      const next = Number(value);
      dim = Number.isFinite(next) && next > 0 ? next : dim;
    };

    switch (layer?.type) {
      case 'conv2d':
        expect('in_channels', p.in_channels);
        emit(p.out_channels);
        break;

      case 'linear':
        expect('input_dim', p.input_dim);
        emit(p.output_dim);
        break;

      case 'embedding':
        emit(p.embedding_dim);
        break;

      case 'batchnorm':
        expect('num_features', p.num_features);
        break;

      case 'layernorm':
        expect('normalized_shape', p.normalized_shape);
        break;

      case 'lstm':
      case 'gru':
        expect('input_size', p.input_size);
        emit(Number(p.hidden_size) * (p.bidirectional ? 2 : 1));
        break;

      case 'transformer':
      case 'attention':
        expect('d_model', p.d_model);
        emit(p.d_model);
        break;

      default:
        // Passthrough layers and unknown types leave `dim` alone.
        if (layer?.type && !PASSTHROUGH.has(layer.type)) dim = null;
        break;
    }
  });

  return issues;
}
