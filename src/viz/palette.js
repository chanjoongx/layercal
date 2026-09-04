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
 * Build one entry. `body` is the solid colour, `light` the emissive tint that
 * bloom picks up — a lighter, more saturated relative of the body so the glow
 * reads as the same material lit from inside rather than as a second colour.
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
  embedding: paint('input', '#7c3aed', '#c4b5fd', '#8b5cf6', '#a78bfa'),
  linear: paint('dense', '#2563eb', '#93c5fd', '#3b82f6', '#60a5fa'),
  conv2d: paint('conv', '#059669', '#6ee7b7', '#10b981', '#34d399'),
  lstm: paint('recurrent', '#d97706', '#fcd34d', '#f59e0b', '#fbbf24'),
  gru: paint('recurrent', '#ea580c', '#fdba74', '#f97316', '#fb923c'),
  transformer: paint('attention', '#c026d3', '#f0abfc', '#d946ef', '#e879f9'),
  attention: paint('attention', '#a21caf', '#e9a8fb', '#c026d3', '#d946ef'),
  batchnorm: paint('norm', '#0891b2', '#67e8f9', '#06b6d4', '#22d3ee'),
  layernorm: paint('norm', '#0e7490', '#5eead4', '#0d9488', '#2dd4bf'),
  maxpool2d: paint('pool', '#0284c7', '#7dd3fc', '#0ea5e9', '#38bdf8'),
  avgpool2d: paint('pool', '#0369a1', '#93c5fd', '#0284c7', '#38bdf8'),
  dropout: paint('reg', '#475569', '#cbd5e1', '#64748b', '#94a3b8'),
  relu: paint('act', '#65a30d', '#bef264', '#84cc16', '#a3e635'),
  softmax: paint('act', '#ca8a04', '#fde047', '#eab308', '#facc15'),
};

/** Neutral paint for a type the palette has never heard of. */
export const FALLBACK_PAINT = paint('reg', '#475569', '#cbd5e1', '#64748b', '#94a3b8');

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
