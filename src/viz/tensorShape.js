/**
 * Tensor shape propagation and the mapping from a shape to a drawable extent.
 *
 * This walks the same rules as `modelValidation.js` — the two must agree, or
 * the canvas would flag a mismatch the 3D view happily draws as connected.
 * Where validation asks "does this layer's declared input match what arrived?",
 * this asks "what shape leaves this layer?", and both answer from the same
 * table.
 *
 * The assumed input sizes (224x224 images, 512-token sequences, 128 RNN
 * timesteps) are imported from layerTypes.js rather than restated, because a
 * FLOPs figure and a drawn tensor that disagree about the input resolution
 * would be two different models on one screen.
 */

import { FLOPS_ASSUMPTIONS } from '@/config/layerTypes';

const { imageSize, seqLen, rnnSeqLen } = FLOPS_ASSUMPTIONS;

/**
 * @typedef {'spatial'|'sequence'|'vector'|'passthrough'} TensorKind
 *
 * @typedef {{
 *   kind: TensorKind,
 *   dims: number[],
 *   label: string,
 *   channels: number|null,
 * }} TensorShape
 */

/** Layers that change nothing about the tensor they receive. */
const PASSTHROUGH = new Set(['relu', 'softmax', 'dropout']);

/** Layers that normalise in place: shape and kind are inherited untouched. */
const NORMALISING = new Set(['batchnorm', 'layernorm']);

const num = (value, fallback) => {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : fallback;
};

const shape = (kind, dims, channels) => ({
  kind,
  dims,
  label: dims.map(d => Math.max(1, Math.round(d)).toLocaleString()).join(' x '),
  channels: Number.isFinite(channels) ? channels : null,
});

/** The tensor a first layer is assumed to receive, when nothing precedes it. */
export function seedShape(layer) {
  const p = (layer && layer.params) || {};
  switch (layer && layer.type) {
    case 'conv2d':
    case 'maxpool2d':
    case 'avgpool2d':
      return shape('spatial', [num(p.in_channels, 3), imageSize, imageSize], num(p.in_channels, 3));
    case 'linear':
      return shape('vector', [num(p.input_dim, 512)], num(p.input_dim, 512));
    case 'lstm':
    case 'gru':
      return shape('sequence', [rnnSeqLen, num(p.input_size, 128)], num(p.input_size, 128));
    case 'transformer':
    case 'attention':
      return shape('sequence', [seqLen, num(p.d_model, 512)], num(p.d_model, 512));
    case 'embedding':
      return shape('sequence', [seqLen], null);
    case 'batchnorm':
      return shape('vector', [num(p.num_features, 128)], num(p.num_features, 128));
    case 'layernorm':
      return shape('vector', [num(p.normalized_shape, 512)], num(p.normalized_shape, 512));
    default:
      return shape('vector', [num(p.input_dim, 256)], num(p.input_dim, 256));
  }
}

/**
 * The shape leaving `layer`, given the shape that arrived.
 *
 * @param {{type: string, params: object}} layer
 * @param {TensorShape|null} incoming
 * @returns {TensorShape}
 */
export function outputShape(layer, incoming) {
  const type = layer && layer.type;
  const p = (layer && layer.params) || {};
  const prev = incoming || seedShape(layer);

  if (PASSTHROUGH.has(type)) {
    return { ...prev, kind: prev.kind };
  }

  if (NORMALISING.has(type)) {
    // Normalisation rescales, it does not reshape. Keeping the upstream shape
    // is what makes a BatchNorm read as an annotation on the flow rather than
    // as a tensor of its own.
    return { ...prev };
  }

  switch (type) {
    case 'embedding': {
      const dim = num(p.embedding_dim, 128);
      return shape('sequence', [seqLen, dim], dim);
    }

    case 'linear': {
      const out = num(p.output_dim, 256);
      return shape('vector', [out], out);
    }

    case 'conv2d': {
      const out = num(p.out_channels, 64);
      // 'same' padding, matching the FLOPs estimate in layerTypes.js.
      const h = prev.kind === 'spatial' ? num(prev.dims[1], imageSize) : imageSize;
      const w = prev.kind === 'spatial' ? num(prev.dims[2], imageSize) : imageSize;
      return shape('spatial', [out, h, w], out);
    }

    case 'maxpool2d':
    case 'avgpool2d': {
      const k = Math.max(1, Math.floor(num(p.kernel_size, 2)));
      const c = prev.kind === 'spatial' ? num(prev.dims[0], 64) : num(prev.channels, 64);
      const h = prev.kind === 'spatial' ? num(prev.dims[1], imageSize) : imageSize;
      const w = prev.kind === 'spatial' ? num(prev.dims[2], imageSize) : imageSize;
      return shape('spatial', [c, Math.max(1, Math.floor(h / k)), Math.max(1, Math.floor(w / k))], c);
    }

    case 'lstm':
    case 'gru': {
      const dir = p.bidirectional ? 2 : 1;
      const width = num(p.hidden_size, 256) * dir;
      const steps = prev.kind === 'sequence' ? num(prev.dims[0], rnnSeqLen) : rnnSeqLen;
      return shape('sequence', [steps, width], width);
    }

    case 'transformer':
    case 'attention': {
      const d = num(p.d_model, 512);
      const steps = prev.kind === 'sequence' ? num(prev.dims[0], seqLen) : seqLen;
      return shape('sequence', [steps, d], d);
    }

    default:
      return { ...prev };
  }
}

/**
 * Walk a whole stack, returning one shape per layer.
 *
 * @param {Array<{type: string, params: object}>} layers
 * @returns {TensorShape[]}
 */
export function propagateShapes(layers) {
  const out = [];
  let current = null;
  for (const layer of Array.isArray(layers) ? layers : []) {
    current = outputShape(layer, current);
    out.push(current);
  }
  return out;
}

/**
 * Layers drawn as a thin plate across the flow rather than as a tensor of
 * their own. They either change nothing (activations, dropout) or rescale in
 * place (normalisation), so giving them a full-depth box would claim the model
 * is deeper than it is.
 */
export const ANNOTATION_TYPES = new Set(['relu', 'softmax', 'dropout', 'batchnorm', 'layernorm']);

/** @param {{w:number,h:number,d:number}|null} previous */
export function annotationExtent(previous) {
  const base = previous || { w: 0.5, h: 0.5, d: 0.2 };
  return { w: base.w * 0.86, h: base.h * 0.86, d: 0.16 };
}

/** Drawn size for a tensor axis of length `n`, in world units. */
export function sizeFor(n) {
  const v = Number(n);
  if (!Number.isFinite(v) || v <= 0) return 0.34;
  const s = 0.34 + 0.42 * Math.log10(1 + v);
  return Math.min(2.6, Math.max(0.34, s));
}

/**
 * Map a shape to a box extent.
 *
 * The log compression in `sizeFor` is the whole point: a 50,000-word embedding
 * table beside a 64-unit dense layer is a 780x ratio, and drawn to scale the
 * dense layer would be a third of a pixel. Log scale keeps every layer visible
 * while preserving the ordering, which is the property the eye actually reads.
 *
 * @param {TensorShape} s
 * @param {{w:number,h:number,d:number}|null} previous extent, for passthroughs
 * @returns {{w:number,h:number,d:number}}
 */
export function extentFor(s, previous = null) {
  if (!s) return { w: 0.5, h: 0.5, d: 0.2 };

  if (s.kind === 'spatial') {
    const [c, h, w] = s.dims;
    return { w: sizeFor(w), h: sizeFor(h), d: sizeFor(c) * 0.75 };
  }

  if (s.kind === 'sequence') {
    const steps = s.dims[0];
    const width = s.channels != null ? s.channels : s.dims[1];
    const side = sizeFor(width) * 0.9;
    return { w: side, h: side, d: sizeFor(steps) * 0.5 };
  }

  if (s.kind === 'vector') {
    const side = sizeFor(s.channels != null ? s.channels : s.dims[0]);
    return { w: side, h: side, d: 0.22 };
  }

  const base = previous || { w: 0.5, h: 0.5, d: 0.2 };
  return { w: base.w * 0.86, h: base.h * 0.86, d: 0.16 };
}
