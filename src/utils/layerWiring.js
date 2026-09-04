import { propagateShapes } from '@/viz/tensorShape';
import { DEFAULT_LAYER_PARAMS } from '@/config/layerTypes';

/**
 * Wiring for a layer that is about to be appended to the stack.
 *
 * Adding a layer used to paste fixed defaults, which meant the *second* layer
 * anyone added was already mismatched: Conv2D emits 64 channels and BatchNorm
 * defaulted to 128. Building a model by clicking through the palette produced a
 * canvas covered in warnings and code that could not run, which is not a
 * reasonable thing to hand someone on their first click.
 *
 * So the incoming feature width is measured and the new layer's input parameter
 * is set to it. Only the *input* is wired: how wide the layer should be on the
 * way out is a design decision that belongs to the person building the model.
 */

/** The parameter of each type that has to equal the incoming feature width. */
const INPUT_FIELD = {
  conv2d: 'in_channels',
  linear: 'input_dim',
  lstm: 'input_size',
  gru: 'input_size',
  // d_model is both the input and the output of these, so matching it wires
  // the layer in and keeps the stack's width unchanged through it.
  transformer: 'd_model',
  attention: 'd_model',
  batchnorm: 'num_features',
  layernorm: 'normalized_shape',
};

/**
 * The feature width currently leaving the stack, or null for an empty model.
 *
 * Walks backwards rather than reading only the last shape, so a trailing run of
 * layers that carry no width of their own cannot erase the answer.
 *
 * @param {Array<{type: string, params: object}>} layers
 * @returns {number|null}
 */
export function outgoingWidth(layers) {
  const shapes = propagateShapes(layers);
  for (let i = shapes.length - 1; i >= 0; i--) {
    const width = shapes[i] && shapes[i].channels;
    if (Number.isFinite(width) && width > 0) return width;
  }
  return null;
}

/** Nearest legal option, preferring an exact match. */
function snapToOption(value, options) {
  if (!Array.isArray(options) || options.length === 0) return value;
  if (options.includes(value)) return value;

  let best = options[0];
  let bestDistance = Math.abs(value - best);
  for (let i = 1; i < options.length; i++) {
    const distance = Math.abs(value - options[i]);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = options[i];
    }
  }
  return best;
}

/**
 * Parameters for a new layer of `type` appended to `layers`.
 *
 * @param {string} type
 * @param {Array<{type: string, params: object}>} layers  the stack it joins
 * @param {Record<string, {fields: Array<object>}>} layerTypes
 * @returns {object} a fresh params object, never a shared reference
 */
export function wireAppendedLayer(type, layers, layerTypes) {
  const params = { ...(DEFAULT_LAYER_PARAMS[type] || {}) };

  const field = INPUT_FIELD[type];
  if (!field) return params;

  const width = outgoingWidth(Array.isArray(layers) ? layers : []);
  if (width == null) return params;

  const spec = layerTypes && layerTypes[type]
    && (layerTypes[type].fields || []).find(f => f.key === field);

  params[field] = snapToOption(width, spec && spec.options);
  return params;
}
