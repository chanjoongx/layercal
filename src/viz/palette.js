/**
 * Layer palette: the single source of truth for what colour a layer is.
 *
 * Three consumers read this table and they must never disagree, because a
 * Conv2D that is green on the canvas and blue in the 3D view is two tools, not
 * one:
 *
 *   1. the 2D layer cards (via `paletteStyle`, as CSS custom properties)
 *   2. the SVG diagram (via `hex` / `hexDark`)
 *   3. the WebGL instance buffer (via `base` / `glow`, converted to linear)
 *
 * `base` and `glow` are sRGB in 0..1. The GPU needs linear light, so the
 * renderer runs them through `srgbToLinear` when it packs instances rather
 * than storing two copies that could drift apart.
 */

/**
 * @typedef {'input'|'dense'|'conv'|'recurrent'|'attention'|'norm'|'pool'|'reg'|'act'} LayerFamily
 *
 * @typedef {{
 *   family: LayerFamily,
 *   base: [number, number, number],
 *   glow: [number, number, number],
 *   hex: string,
 *   hexDark: string,
 * }} LayerPaint
 */

/** Every family in the fixed order used by the legend. */
export const LAYER_FAMILIES = /** @type {const} */ ([
  'input', 'dense', 'conv', 'recurrent', 'attention', 'norm', 'pool', 'reg', 'act',
]);

/** Family display names, keyed for the legend. English fallback; `t` overrides. */
export const FAMILY_LABELS = {
  input: 'Input',
  dense: 'Dense',
  conv: 'Convolution',
  recurrent: 'Recurrent',
  attention: 'Attention',
  norm: 'Normalisation',
  pool: 'Pooling',
  reg: 'Regularisation',
  act: 'Activation',
};

const rgb = (hex) => [
  parseInt(hex.slice(1, 3), 16) / 255,
  parseInt(hex.slice(3, 5), 16) / 255,
  parseInt(hex.slice(5, 7), 16) / 255,
];

/**
 * Build one entry. `body` is the surface colour the GPU lights, `light` the
 * tint the rim and the activation pulse take.
 *
 * The values below are deliberately muted. An earlier version used the Tailwind
 * 500/600 ramp - spring green, lime, sky, fuchsia - which under an emissive
 * material and bloom rendered as neon candy: saturated, glowing, and closer to
 * a game than to a technical drawing. These sit around a common lightness with
 * roughly half the chroma, so the eye reads shape and depth first and colour
 * second, which is the right order for a diagram.
 *
 * There is a hierarchy in the hues, not just nine of them. Layers that change
 * the tensor - embedding, convolution, dense, recurrent, attention, pooling -
 * carry the colour. Layers that annotate one without reshaping it -
 * normalisation, dropout, activations - are close to neutral, so a stack reads
 * as a few coloured tensors with markers on them rather than a rainbow.
 */
const paint = (family, body, light, hex, hexDark) => ({
  family,
  base: rgb(body),
  glow: rgb(light),
  hex,
  hexDark,
});

/** @type {Record<string, LayerPaint>} */
export const LAYER_PALETTE = {
  // Structural: these carry the colour.
  embedding: paint('input', '#6b5cc4', '#a79ae8', '#7b6cd9', '#9385e9'),
  linear: paint('dense', '#3468ac', '#86aedf', '#3e7bc8', '#5f9be0'),
  conv2d: paint('conv', '#2a8578', '#7cc9bc', '#2f9e8f', '#45b8a6'),
  lstm: paint('recurrent', '#a9743a', '#e0b278', '#c1873f', '#d89f58'),
  gru: paint('recurrent', '#9c6544', '#d9a886', '#b37850', '#cb9270'),
  transformer: paint('attention', '#85508f', '#c99bd3', '#9d5fa8', '#b87cc3'),
  attention: paint('attention', '#73456f', '#be8fb4', '#8a5484', '#a96ea1'),
  maxpool2d: paint('pool', '#33738c', '#85bdd1', '#3c8aa8', '#58a8c4'),
  avgpool2d: paint('pool', '#2c6076', '#7baabe', '#356f88', '#4e90a8'),

  // Annotations: near-neutral, so they read as markers on the flow.
  batchnorm: paint('norm', '#5d6878', '#a3aebd', '#6e7a8a', '#8d9bad'),
  layernorm: paint('norm', '#52606c', '#98a6b2', '#63717e', '#8291a0'),
  dropout: paint('reg', '#676770', '#a9a9b2', '#77777f', '#94949e'),
  relu: paint('act', '#6c7f49', '#b2c58a', '#7e9455', '#9ab16b'),
  softmax: paint('act', '#8a7a3e', '#cbbe86', '#9c8b4b', '#b6a566'),
};

/** Neutral paint for a type the palette has never heard of. */
export const FALLBACK_PAINT = paint('reg', '#676770', '#a9a9b2', '#77777f', '#94949e');

/**
 * @param {string} type
 * @returns {LayerPaint}
 */
export function paintFor(type) {
  return LAYER_PALETTE[type] || FALLBACK_PAINT;
}

/**
 * sRGB transfer function. The piecewise form matters at the dark end: the
 * naive `pow(c, 2.2)` puts the near-black rim colours roughly 15% too dark,
 * which is exactly where the fresnel edge lives.
 *
 * @param {number} c channel in 0..1
 */
export function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** @param {[number,number,number]} c */
export function toLinear(c) {
  return [srgbToLinear(c[0]), srgbToLinear(c[1]), srgbToLinear(c[2])];
}

/**
 * Inline style for a 2D card: finished colour strings, one per role.
 *
 * These are deliberately plain `rgba()` rather than `color-mix()`. Chrome
 * serialises `color-mix(in oklab, ...)` into an `oklab()` colour in
 * getComputedStyle, and html2canvas - which the PNG export runs over the whole
 * page - throws on any colour function it does not know, taking the entire
 * export down with it. Doing the mix here keeps the stylesheet on colour syntax
 * that every consumer of the DOM can read.
 *
 * @param {string} type
 * @param {boolean} isDarkMode
 */
export function paletteStyle(type, isDarkMode) {
  const p = paintFor(type);
  const hex = isDarkMode ? p.hexDark : p.hex;
  const [r, g, b] = rgb(hex).map(v => Math.round(v * 255));
  const tint = (alpha) => `rgba(${r}, ${g}, ${b}, ${alpha})`;
  return {
    '--layer': hex,
    '--layer-rgb': `${r} ${g} ${b}`,
    '--layer-fill': tint(isDarkMode ? 0.14 : 0.11),
    '--layer-fill-hover': tint(isDarkMode ? 0.2 : 0.17),
    '--layer-line': tint(isDarkMode ? 0.42 : 0.34),
  };
}
