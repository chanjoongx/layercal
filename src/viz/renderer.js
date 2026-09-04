/**
 * WebGL2 renderer for the model viewer.
 *
 * Nine passes, about a dozen draw calls, independent of how many layers the
 * model has: every per-layer thing is instanced and every per-frame motion is a
 * uniform. A 60-layer transformer costs the same as a 3-layer MLP.
 *
 *   grid -> contact shadows -> slabs -> halo shell -> ribbons -> particles
 *        -> resolve -> bright pass -> blur chain -> composite
 *
 * The module is loaded dynamically by ModelViewer, so none of it ships in the
 * initial bundle and none of it runs during server rendering.
 */

import {
  probe, createProgram, createBuffer, createVAO, createTarget,
  drawFramebuffer, resolveTarget, unitBoxGeometry, QUAD_CORNERS,
} from './glCore';
import {
  SLAB_VERT, SLAB_FRAG, SHELL_FRAG,
  RIBBON_VERT, RIBBON_FRAG,
  PARTICLE_VERT, PARTICLE_FRAG,
  GRID_VERT, GRID_FRAG, SHADOW_VERT, SHADOW_FRAG, SKY_FRAG,
  FULLSCREEN_VERT, BRIGHT_FRAG, BLUR_FRAG, COMPOSITE_FRAG,
} from './glsl';
import { toLinear, srgbToLinear } from './palette';
import { mat4, vec3, rayBox, clamp } from './math';
import {
  createCamera, updateCamera, frameBounds, settle, orbit, dolly, pan,
  focusOn, rayFromNDC,
} from './camera';
import { interleaveByLink } from './sceneGraph';

const SLAB_STRIDE = 80;          // 20 floats
const PARTICLE_STRIDE = 88;      // 22 floats
const RIBBON_STRIDE = 36;        // 9 floats
const RIBBON_SEGMENTS = 24;
const PARTICLE_BUDGET = 1400;

/** dpr cap, bloom iterations and particle fraction per quality level. */
const QUALITY = [
  { dpr: 1.0, bloom: 1, particles: 0.25 },
  { dpr: 1.5, bloom: 2, particles: 0.5 },
  { dpr: 2.0, bloom: 3, particles: 1.0 },
];

/**
 * `skyTop`/`skyBottom`/`skyGlow` are display-referred sRGB: they are what the
 * background should look like on screen, and the renderer runs them backwards
 * through the tone map so that is exactly what lands.
 *
 * `sky`/`ground` are the hemispheric ambient for the slab lighting, which is a
 * different thing entirely and lives in linear light.
 */
const THEMES = {
  light: {
    // Not paper-white. Muted layers need something to sit against, and a panel
    // that is pure white makes every one of them look washed out no matter how
    // the material is lit.
    skyTop: [0.906, 0.914, 0.933],
    skyBottom: [0.812, 0.826, 0.859],
    skyGlow: [0.014, 0.014, 0.020],
    // Held well below the dark theme's: on a near-white ground the ambient is
    // already most of the light, and at the old level every layer washed out to
    // pastel before the fog even touched it.
    sky: [0.46, 0.50, 0.60],
    ground: [0.18, 0.20, 0.26],
    key: [1.0, 0.99, 0.96],
    grid: [0.30, 0.33, 0.41],
    accent: [0.40, 0.43, 0.52],
    warn: [0.78, 0.38, 0.06],
    exposure: 0.95,
    bloom: 0.16,
    gridOpacity: 0.70,
    shadow: 0.72,
    ribbon: 0.62,
    particle: 0.42,
    shell: 0.0,
    grain: 0.006,
    saturation: 1.18,
    // A quarter of the dark theme's, and the asymmetry is the whole point.
    // Fog mixes toward the background in *scene* space, and inverse-ACES sends
    // a near-white background to ~1.2 there - so on a light ground even a tenth
    // of fog adds more absolute light than the layer's own darkest channel
    // carries, and a teal lands as pale mint. The depth cue is worth keeping;
    // it just has to be a whisper here and can be a voice on a dark ground.
    fogDensity: 0.045,
    // Additive light on a near-white ground clips to white and loses the hue
    // that carries the meaning, so the light theme composites over instead.
    additive: false,
  },
  dark: {
    skyTop: [0.072, 0.082, 0.108],
    skyBottom: [0.024, 0.027, 0.038],
    skyGlow: [0.018, 0.021, 0.034],
    sky: [0.12, 0.14, 0.20],
    ground: [0.030, 0.032, 0.042],
    key: [0.95, 0.95, 1.0],
    grid: [0.17, 0.19, 0.25],
    accent: [0.30, 0.34, 0.44],
    warn: [1.0, 0.62, 0.20],
    exposure: 1.10,
    bloom: 0.32,
    gridOpacity: 0.34,
    shadow: 0.75,
    ribbon: 0.42,
    particle: 0.30,
    shell: 0.30,
    grain: 0.010,
    saturation: 1.08,
    fogDensity: 0.52,
    additive: true,
  },
};

/**
 * Inverse of the Narkowicz ACES fit.
 *
 * The composite pass tone-maps everything, so clearing the scene buffer to the
 * panel's background colour would come out visibly lighter than the CSS around
 * it. Solving the fit backwards gives the linear value that tone-maps *to* the
 * colour we want, so the canvas and the page agree exactly.
 */
function acesInverse(y) {
  const a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  const t = clamp(y, 0, 0.999);
  const A = t * c - a;
  const B = t * d - b;
  const C = t * e;
  const disc = B * B - 4 * A * C;
  if (disc <= 0 || A === 0) return t;
  return (-B - Math.sqrt(disc)) / (2 * A);
}

/** Display-referred sRGB -> the linear value that tone-maps back to it. */
function toSceneColor(srgb, exposure) {
  return [
    acesInverse(srgbToLinear(srgb[0])) / exposure,
    acesInverse(srgbToLinear(srgb[1])) / exposure,
    acesInverse(srgbToLinear(srgb[2])) / exposure,
  ];
}

// The sky colours are fixed per theme, so solve them backwards through the
// tone map once at module load rather than six pow() and six sqrt() calls
// every frame.
for (const theme of Object.values(THEMES)) {
  theme.skyTopScene = toSceneColor(theme.skyTop, theme.exposure);
  theme.skyBottomScene = toSceneColor(theme.skyBottom, theme.exposure);
  // Layers recede into the same colour the background is painted with, so the
  // fog term needs it in scene space too.
  theme.fogScene = theme.skyTopScene.map((v, i) => (v + theme.skyBottomScene[i]) * 0.5);
}

/**
 * @param {HTMLCanvasElement} canvas
 * @param {{
 *   isDarkMode?: boolean,
 *   motion?: boolean,
 *   onHover?: (id: string|null) => void,
 *   onSelect?: (id: string|null) => void,
 *   onFrame?: (labels: Array<object>) => void,
 * }} options
 * @returns {object|null} null when WebGL2 is unavailable
 */
export function createRenderer(canvas, options = {}) {
  const caps = probe(canvas);
  if (!caps.gl) return null;

  const gl = caps.gl;
  const reduced = caps.level === 'reduced';
  const software = caps.reason === 'software-renderer';

  // A software rasteriser is fill-rate bound, not geometry bound: the nine
  // passes cost the same whether the model has three layers or sixty. So start
  // it at the cheapest quality and render below native resolution, where the
  // softness is a far better trade than the frame rate.
  let quality = software ? 0 : (reduced ? 1 : 2);
  let motion = options.motion !== false;
  let playing = true;
  let theme = options.isDarkMode ? THEMES.dark : THEMES.light;

  // A complete Scene, not a partial one: the framing solve reads bounds.min and
  // bounds.max, and a placeholder missing them threw during engine construction,
  // which the caller then caught as "no WebGL" and quietly fell back to the SVG.
  let scene = {
    nodes: [],
    links: [],
    bounds: { min: [-1, 0, -1], max: [1, 1.8, 1], center: [0, 0.9, 0], radius: 1 },
    totals: {},
  };
  let selectedId = null;
  let hoveredId = null;

  const camera = createCamera();
  let width = 1;
  let height = 1;
  let dpr = 1;
  // The CSS box, refreshed by resize() from the ResizeObserver. Reading it with
  // getBoundingClientRect() inside the frame forced a layout flush twice per
  // frame, which is the classic way to make a render loop slower than the GPU.
  let cssWidth = 1;
  let cssHeight = 1;
  const aspect = () => cssWidth / cssHeight;

  // ── static GPU resources ────────────────────────────
  const box = unitBoxGeometry();
  const boxVertexBuffer = createBuffer(gl, gl.ARRAY_BUFFER, box.vertices, gl.STATIC_DRAW);
  const boxIndexBuffer = createBuffer(gl, gl.ELEMENT_ARRAY_BUFFER, box.indices, gl.STATIC_DRAW);
  const quadBuffer = createBuffer(gl, gl.ARRAY_BUFFER, QUAD_CORNERS, gl.STATIC_DRAW);

  const sliceInstanceBuffer = gl.createBuffer();
  const nodeInstanceBuffer = gl.createBuffer();
  const particleBuffer = gl.createBuffer();
  const ribbonVertexBuffer = gl.createBuffer();
  const ribbonIndexBuffer = gl.createBuffer();

  const boxStreams = (instanceBuffer) => ([
    {
      buffer: boxVertexBuffer,
      stride: box.stride,
      attribs: [
        { location: 0, size: 3, offset: 0 },
        { location: 1, size: 3, offset: 12 },
        { location: 2, size: 2, offset: 24 },
      ],
    },
    {
      buffer: instanceBuffer,
      stride: SLAB_STRIDE,
      attribs: [
        { location: 3, size: 4, offset: 0, divisor: 1 },
        { location: 4, size: 4, offset: 16, divisor: 1 },
        { location: 5, size: 4, offset: 32, divisor: 1 },
        { location: 6, size: 4, offset: 48, divisor: 1 },
        { location: 7, size: 4, offset: 64, divisor: 1 },
      ],
    },
  ]);

  // Slabs are drawn per slice; the halo and the contact shadow are drawn once
  // per layer, because N overlapping glows or shadows stack into a smear.
  const slabVAO = createVAO(gl, boxStreams(sliceInstanceBuffer), boxIndexBuffer);
  const shellVAO = createVAO(gl, boxStreams(nodeInstanceBuffer), boxIndexBuffer);

  const shadowVAO = createVAO(gl, [
    { buffer: quadBuffer, stride: 8, attribs: [{ location: 0, size: 2, offset: 0 }] },
    {
      buffer: nodeInstanceBuffer,
      stride: SLAB_STRIDE,
      attribs: [
        { location: 1, size: 4, offset: 0, divisor: 1 },
        { location: 2, size: 4, offset: 16, divisor: 1 },
      ],
    },
  ]);

  const particleVAO = createVAO(gl, [
    { buffer: quadBuffer, stride: 8, attribs: [{ location: 0, size: 2, offset: 0 }] },
    {
      buffer: particleBuffer,
      stride: PARTICLE_STRIDE,
      attribs: [
        { location: 1, size: 3, offset: 0, divisor: 1 },
        { location: 2, size: 3, offset: 12, divisor: 1 },
        { location: 3, size: 3, offset: 24, divisor: 1 },
        { location: 4, size: 3, offset: 36, divisor: 1 },
        { location: 5, size: 4, offset: 48, divisor: 1 },
        { location: 6, size: 3, offset: 64, divisor: 1 },
        { location: 7, size: 3, offset: 76, divisor: 1 },
      ],
    },
  ]);

  const ribbonVAO = createVAO(gl, [{
    buffer: ribbonVertexBuffer,
    stride: RIBBON_STRIDE,
    attribs: [
      { location: 0, size: 3, offset: 0 },
      { location: 1, size: 3, offset: 12 },
      { location: 2, size: 3, offset: 24 },
    ],
  }], ribbonIndexBuffer);

  const gridVAO = createVAO(gl, [{
    buffer: quadBuffer, stride: 8, attribs: [{ location: 0, size: 2, offset: 0 }],
  }]);

  const programs = {
    slab: createProgram(gl, SLAB_VERT, SLAB_FRAG, 'slab'),
    shell: createProgram(gl, SLAB_VERT, SHELL_FRAG, 'shell'),
    ribbon: createProgram(gl, RIBBON_VERT, RIBBON_FRAG, 'ribbon'),
    particle: createProgram(gl, PARTICLE_VERT, PARTICLE_FRAG, 'particle'),
    sky: createProgram(gl, FULLSCREEN_VERT, SKY_FRAG, 'sky'),
    grid: createProgram(gl, GRID_VERT, GRID_FRAG, 'grid'),
    shadow: createProgram(gl, SHADOW_VERT, SHADOW_FRAG, 'shadow'),
    bright: createProgram(gl, FULLSCREEN_VERT, BRIGHT_FRAG, 'bright'),
    blur: createProgram(gl, FULLSCREEN_VERT, BLUR_FRAG, 'blur'),
    composite: createProgram(gl, FULLSCREEN_VERT, COMPOSITE_FRAG, 'composite'),
  };

  /** @type {{scene: object|null, bloom: object[], temp: object[]}} */
  const targets = { scene: null, bloom: [], temp: [] };

  let nodeCount = 0;
  let ribbonIndexCount = 0;
  let particleCount = 0;
  /** Which layer each slice instance belongs to, for selection highlighting. */
  const sliceOwner = [];

  // ── scene upload ────────────────────────────────────

  /**
   * How many planes a layer is drawn as. Derived from the channel count, so a
   * 512-channel convolution is visibly a deeper stack than a 32-channel one,
   * and capped because past about nine the planes stop being countable and
   * start being noise.
   */
  function sliceCount(node) {
    if (node.annotation) return 1;
    const kind = node.shape && node.shape.kind;
    // A dense layer emits a vector, not a stack of feature maps. Slicing its
    // thin plate turned it into a comb and claimed a structure it does not have.
    if (kind === 'vector') return 1;
    const channels = (node.shape && node.shape.channels) || 32;
    const cap = kind === 'sequence' ? 7 : 9;
    return Math.round(clamp(Math.log2(Math.max(2, channels)) - 1, 2, cap));
  }

  /** Cells across a feature-map face, for the faint grid on spatial layers. */
  function faceCells(node) {
    if (!node.shape || node.shape.kind !== 'spatial') return 0;
    const side = node.shape.dims[1] || 0;
    return Math.round(clamp(Math.log2(Math.max(2, side)) - 1, 2, 8));
  }

  function writeInstance(data, o, node, center, extent, meta) {
    const base = toLinear(node.paint.base);
    const glow = toLinear(node.paint.glow);
    data[o] = center.x; data[o + 1] = center.y; data[o + 2] = center.z; data[o + 3] = node.phase;
    data[o + 4] = extent.w; data[o + 5] = extent.h; data[o + 6] = extent.d; data[o + 7] = meta;
    data[o + 8] = base[0]; data[o + 9] = base[1]; data[o + 10] = base[2]; data[o + 11] = node.paramShare;
    data[o + 12] = glow[0]; data[o + 13] = glow[1]; data[o + 14] = glow[2]; data[o + 15] = node.warning ? 1 : 0;
    data[o + 16] = 0; data[o + 17] = 0; data[o + 18] = node.annotation ? 1 : 0; data[o + 19] = node.flopShare;
  }

  function uploadScene() {
    const nodes = scene.nodes;
    nodeCount = nodes.length;
    sliceOwner.length = 0;

    if (nodeCount > 0) {
      const nodeData = new Float32Array(nodeCount * 20);
      for (let i = 0; i < nodeCount; i++) {
        const node = nodes[i];
        writeInstance(nodeData, i * 20, node, node.center, node.extent, 0);
        for (let k = 0; k < sliceCount(node); k++) sliceOwner.push(i);
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, nodeInstanceBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, nodeData, gl.DYNAMIC_DRAW);

      const sliceData = new Float32Array(sliceOwner.length * 20);
      let s = 0;
      for (let i = 0; i < nodeCount; i++) {
        const node = nodes[i];
        const n = sliceCount(node);
        const cells = faceCells(node);

        // n plates and n-1 gaps fill the layer's depth. The gap is a fraction
        // of a plate, so a deep stack stays a stack rather than becoming a comb.
        const gapRatio = n > 1 ? 0.55 : 0;
        const thickness = node.extent.d / (n + (n - 1) * gapRatio);
        const step = thickness * (1 + gapRatio);
        const start = -node.extent.d / 2 + thickness / 2;

        for (let k = 0; k < n; k++) {
          writeInstance(
            sliceData, s * 20, node,
            { x: node.center.x, y: node.center.y, z: node.center.z + start + k * step },
            { w: node.extent.w, h: node.extent.h, d: thickness },
            cells
          );
          s++;
        }
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, sliceInstanceBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, sliceData, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
    }

    buildRibbons();
    buildParticles();
    applySelection();
  }

  /** Bezier control points for a link, exiting one slab and entering the next. */
  function linkCurve(link) {
    const a = scene.nodes[link.from];
    const b = scene.nodes[link.to];
    const p0 = [a.center.x, a.center.y, a.center.z + a.extent.d / 2];
    const p3 = [b.center.x, b.center.y, b.center.z - b.extent.d / 2];
    const span = Math.max(0.0001, p3[2] - p0[2]);
    const p1 = [p0[0], p0[1], p0[2] + span * 0.42];
    const p2 = [p3[0], p3[1], p3[2] - span * 0.42];
    return { p0, p1, p2, p3 };
  }

  function bezierAt(c, t) {
    const u = 1 - t;
    const w0 = u * u * u, w1 = 3 * u * u * t, w2 = 3 * u * t * t, w3 = t * t * t;
    return [
      c.p0[0] * w0 + c.p1[0] * w1 + c.p2[0] * w2 + c.p3[0] * w3,
      c.p0[1] * w0 + c.p1[1] * w1 + c.p2[1] * w2 + c.p3[1] * w3,
      c.p0[2] * w0 + c.p1[2] * w1 + c.p2[2] * w2 + c.p3[2] * w3,
    ];
  }

  /**
   * A ribbon is two perpendicular strips forming a cross-section, so the
   * connection never disappears when the camera looks along its plane.
   */
  function buildRibbons() {
    const links = scene.links;
    if (links.length === 0) {
      ribbonIndexCount = 0;
      return;
    }

    const rings = RIBBON_SEGMENTS + 1;
    const vertsPerLink = rings * 4;
    const indicesPerLink = RIBBON_SEGMENTS * 12;
    const vertices = new Float32Array(links.length * vertsPerLink * 9);
    const indices = new Uint32Array(links.length * indicesPerLink);

    let v = 0;
    let i = 0;

    for (let l = 0; l < links.length; l++) {
      const link = links[l];
      const curve = linkCurve(link);
      const from = scene.nodes[link.from];
      const to = scene.nodes[link.to];
      const colorA = theme.additive ? from.paint.glow : from.paint.base;
      const colorB = theme.additive ? to.paint.glow : to.paint.base;
      const broken = link.broken ? 1 : 0;
      const vertexBase = l * vertsPerLink;

      for (let s = 0; s < rings; s++) {
        const t = s / RIBBON_SEGMENTS;
        const p = bezierAt(curve, t);
        const ease = t * t * (3 - 2 * t);
        const halfWidth = link.width0 + (link.width1 - link.width0) * ease;
        const r = colorA[0] + (colorB[0] - colorA[0]) * t;
        const g = colorA[1] + (colorB[1] - colorA[1]) * t;
        const b = colorA[2] + (colorB[2] - colorA[2]) * t;
        const lr = srgbToLinear(r), lg = srgbToLinear(g), lb = srgbToLinear(b);

        // horizontal pair, then vertical pair
        const offsets = [
          [-halfWidth, 0, -1], [halfWidth, 0, 1],
          [0, -halfWidth, -1], [0, halfWidth, 1],
        ];
        for (const [ox, oy, side] of offsets) {
          vertices[v++] = p[0] + ox;
          vertices[v++] = p[1] + oy;
          vertices[v++] = p[2];
          vertices[v++] = t;
          vertices[v++] = side;
          vertices[v++] = broken;
          vertices[v++] = lr;
          vertices[v++] = lg;
          vertices[v++] = lb;
        }
      }

      for (let s = 0; s < RIBBON_SEGMENTS; s++) {
        const a = vertexBase + s * 4;
        const b = vertexBase + (s + 1) * 4;
        // horizontal strip
        indices[i++] = a; indices[i++] = a + 1; indices[i++] = b + 1;
        indices[i++] = a; indices[i++] = b + 1; indices[i++] = b;
        // vertical strip
        indices[i++] = a + 2; indices[i++] = a + 3; indices[i++] = b + 3;
        indices[i++] = a + 2; indices[i++] = b + 3; indices[i++] = b + 2;
      }
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, ribbonVertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ribbonIndexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    ribbonIndexCount = i;
  }

  /**
   * Particles carry their whole path in their instance data, so the vertex
   * shader can place them from `uTime` alone and the CPU does nothing per frame.
   * A deterministic hash stands in for Math.random: the same model always
   * produces the same flow, which matters for screenshot comparisons.
   */
  function buildParticles() {
    const links = scene.links;
    if (links.length === 0) {
      particleCount = 0;
      return;
    }

    const weights = links.map((l) => {
      const a = scene.nodes[l.from];
      const b = scene.nodes[l.to];
      const length = Math.max(0.2, (b.center.z - b.extent.d / 2) - (a.center.z + a.extent.d / 2));
      return (0.3 + 0.7 * (Number.isFinite(l.flow) ? l.flow : 0)) * length;
    });
    const totalWeight = weights.reduce((a, b) => a + b, 0) || 1;

    const counts = weights.map(w => Math.max(6, Math.round((w / totalWeight) * PARTICLE_BUDGET)));
    const total = counts.reduce((a, b) => a + b, 0);

    // The camera pulls back in proportion to the scene, so a fixed world size
    // renders sub-pixel on a large model and as blobs on a small one.
    const sizeScale = Math.max(0.35, scene.bounds.radius * 0.11);

    const info = links.map((link, l) => {
      const from = scene.nodes[link.from];
      const to = scene.nodes[link.to];
      return {
        curve: linkCurve(link),
        a: toLinear(theme.additive ? from.paint.glow : from.paint.base),
        b: toLinear(theme.additive ? to.paint.glow : to.paint.base),
        swirl: Math.max(link.width0, link.width1) * 0.85,
        speed: 0.55 + 1.25 * (Number.isFinite(link.flow) ? link.flow : 0),
        count: counts[l],
      };
    });

    const data = new Float32Array(total * 22);
    let o = 0;
    let seed = 1;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    };

    // Interleaved across links, not link by link: reduced quality draws a
    // prefix of this buffer, and a link-major layout would drop the flow off
    // the tail of the model rather than thinning it evenly. The ordering is a
    // pure function so the property can be asserted in a unit test.
    for (const l of interleaveByLink(counts)) {
      const { curve, a, b, swirl, speed } = info[l];
      data[o++] = curve.p0[0]; data[o++] = curve.p0[1]; data[o++] = curve.p0[2];
      data[o++] = curve.p1[0]; data[o++] = curve.p1[1]; data[o++] = curve.p1[2];
      data[o++] = curve.p2[0]; data[o++] = curve.p2[1]; data[o++] = curve.p2[2];
      data[o++] = curve.p3[0]; data[o++] = curve.p3[1]; data[o++] = curve.p3[2];
      data[o++] = rand();                       // phase
      data[o++] = speed;
      data[o++] = sizeScale * (0.028 + rand() * 0.042);  // billboard size
      data[o++] = swirl * (0.35 + rand() * 0.9);
      data[o++] = a[0]; data[o++] = a[1]; data[o++] = a[2];
      data[o++] = b[0]; data[o++] = b[1]; data[o++] = b[2];
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, particleBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    particleCount = total;
  }

  /** Patch the selection flags in place rather than rebuilding the buffers. */
  function applySelection() {
    if (nodeCount === 0) return;
    const patch = new Float32Array(2);
    const flags = scene.nodes.map(node => [
      node.id === selectedId ? 1 : 0,
      node.id === hoveredId ? 1 : 0,
    ]);

    gl.bindBuffer(gl.ARRAY_BUFFER, nodeInstanceBuffer);
    for (let i = 0; i < flags.length; i++) {
      patch[0] = flags[i][0]; patch[1] = flags[i][1];
      gl.bufferSubData(gl.ARRAY_BUFFER, i * SLAB_STRIDE + 64, patch);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, sliceInstanceBuffer);
    for (let i = 0; i < sliceOwner.length; i++) {
      const owner = flags[sliceOwner[i]];
      if (!owner) continue;
      patch[0] = owner[0]; patch[1] = owner[1];
      gl.bufferSubData(gl.ARRAY_BUFFER, i * SLAB_STRIDE + 64, patch);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  // ── render targets ──────────────────────────────────

  function allocateTargets() {
    disposeTargets();
    targets.scene = createTarget(gl, {
      width, height, hdr: caps.hdr, depth: true, samples: caps.samples,
    });
    for (let i = 0; i < 3; i++) {
      const w = Math.max(1, width >> (i + 1));
      const h = Math.max(1, height >> (i + 1));
      targets.bloom.push(createTarget(gl, { width: w, height: h, hdr: caps.hdr }));
      targets.temp.push(createTarget(gl, { width: w, height: h, hdr: caps.hdr }));
    }
  }

  function disposeTargets() {
    if (targets.scene) targets.scene.dispose();
    targets.scene = null;
    targets.bloom.forEach(t => t.dispose());
    targets.temp.forEach(t => t.dispose());
    targets.bloom = [];
    targets.temp = [];
  }

  const RESOLUTION_SCALE = software ? 0.62 : 1;

  function resize() {
    const rect = canvas.getBoundingClientRect();
    cssWidth = rect.width || 1;
    cssHeight = rect.height || 1;
    const cap = QUALITY[quality].dpr;
    const nextDpr = Math.min(window.devicePixelRatio || 1, cap) * RESOLUTION_SCALE;
    const w = Math.max(1, Math.round((rect.width || 1) * nextDpr));
    const h = Math.max(1, Math.round((rect.height || 1) * nextDpr));
    if (w === width && h === height && nextDpr === dpr) return;
    width = w;
    height = h;
    dpr = nextDpr;
    canvas.width = w;
    canvas.height = h;
    allocateTargets();

    // Re-frame for the new panel. Everything the fit was solved against has just
    // changed: the aspect it was fitted to, the angle composed for that aspect,
    // and the safe-area insets, which are a fraction of a height that no longer
    // holds. Only the projection followed the resize on its own, so narrowing
    // the window used to cut the model off at both edges outright.
    //
    // A camera the viewer has posed is theirs and is left alone; Reset re-fits.
    if (!camera.userPosed) frameAll();
  }

  // ── uniform helpers ─────────────────────────────────

  const use = (entry) => {
    gl.useProgram(entry.program);
    return entry.uniforms;
  };
  const u1f = (u, name, v) => { if (u[name]) gl.uniform1f(u[name], v); };
  const u2f = (u, name, a, b) => { if (u[name]) gl.uniform2f(u[name], a, b); };
  const u3v = (u, name, v) => { if (u[name]) gl.uniform3f(u[name], v[0], v[1], v[2]); };
  const u1i = (u, name, v) => { if (u[name]) gl.uniform1i(u[name], v); };
  const uMat = (u, name, m) => { if (u[name]) gl.uniformMatrix4fv(u[name], false, m); };

  function bindTexture(unit, texture, uniforms, name) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    u1i(uniforms, name, unit);
  }

  // ── the frame ───────────────────────────────────────

  let clock = 0;

  /** Additive in the dark theme, premultiplied "over" in the light one. */
  function flowBlend() {
    if (theme.additive) gl.blendFunc(gl.ONE, gl.ONE);
    else gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  function renderFrame() {
    if (!targets.scene) return;
    const q = QUALITY[quality];
    const time = clock;
    const motionFlag = motion && playing ? 1 : 0;

    // -- geometry into the scene target ---------------
    gl.bindFramebuffer(gl.FRAMEBUFFER, drawFramebuffer(targets.scene));
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 1);
    gl.clearDepth(1);
    gl.depthMask(true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // 0. sky ----------------------------------------
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);
    gl.depthMask(false);
    {
      const u = use(programs.sky);
      u3v(u, 'uSkyTop', theme.skyTopScene);
      u3v(u, 'uSkyBottom', theme.skyBottomScene);
      u3v(u, 'uSkyGlow', theme.skyGlow);
      gl.bindVertexArray(gridVAO);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.BLEND);

    // 1. grid ---------------------------------------
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    {
      const u = use(programs.grid);
      uMat(u, 'uViewProj', camera.viewProjection);
      const radius = Math.max(1, scene.bounds.radius);
      const size = radius * 6;
      u1f(u, 'uGridSize', size);
      u3v(u, 'uGridColor', theme.grid);
      u3v(u, 'uAccentColor', theme.accent);
      // One grid cell per ~12% of the model, so the ground reads at the same
      // density whether the stack is three layers or sixty.
      u1f(u, 'uCellScale', 6 / radius);
      u1f(u, 'uFadeStart', radius * 1.1);
      u1f(u, 'uFadeEnd', radius * 3.4);
      u1f(u, 'uOpacity', theme.gridOpacity);
      gl.bindVertexArray(gridVAO);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    // 2. contact shadows ----------------------------
    if (nodeCount > 0) {
      gl.blendFunc(gl.ZERO, gl.ONE_MINUS_SRC_ALPHA);
      const u = use(programs.shadow);
      uMat(u, 'uViewProj', camera.viewProjection);
      u1f(u, 'uStrength', theme.shadow);
      gl.bindVertexArray(shadowVAO);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, nodeCount);
    }

    // 3. slabs --------------------------------------
    if (nodeCount > 0) {
      gl.disable(gl.BLEND);
      gl.depthMask(true);
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.BACK);

      const u = use(programs.slab);
      uMat(u, 'uViewProj', camera.viewProjection);
      u3v(u, 'uCameraPos', camera.eye);
      u1f(u, 'uTime', time);
      u1f(u, 'uMotion', motionFlag);
      u1f(u, 'uPulseRate', 0.28);
      u1f(u, 'uPulseWidth', 6.0);
      u1f(u, 'uShellScale', 1.0);
      u3v(u, 'uSkyColor', theme.sky);
      u3v(u, 'uGroundColor', theme.ground);
      u3v(u, 'uKeyColor', theme.key);
      u3v(u, 'uWarnColor', theme.warn);
      u3v(u, 'uFogColor', theme.fogScene);
      // Tied to the model's own size, so the depth cue is the same whether the
      // stack is three layers or sixty.
      const modelRadius = Math.max(1, scene.bounds.radius);
      u1f(u, 'uFogNear', modelRadius * 0.7);
      u1f(u, 'uFogDensity', theme.fogDensity / modelRadius);
      gl.bindVertexArray(slabVAO);
      gl.drawElementsInstanced(gl.TRIANGLES, box.indices.length, gl.UNSIGNED_SHORT, 0, sliceOwner.length);

      // 4. halo shell -------------------------------
      // Sized to the whole layer rather than a plate, because a halo cut to a
      // plate would strobe with the gaps between them. It is off entirely in
      // the light theme: it renders behind the plates, so it shows through
      // those same gaps, and on a light ground that reads as a pale film over
      // every layer rather than as a glow.
      if (theme.shell > 0) {
        gl.enable(gl.BLEND);
        flowBlend();
        gl.depthMask(false);
        gl.cullFace(gl.FRONT);
        const s = use(programs.shell);
        uMat(s, 'uViewProj', camera.viewProjection);
        u3v(s, 'uCameraPos', camera.eye);
        u1f(s, 'uTime', time);
        u1f(s, 'uMotion', motionFlag);
        u1f(s, 'uPulseRate', 0.28);
        u1f(s, 'uPulseWidth', 6.0);
        u1f(s, 'uShellScale', 1.18);
        u1f(s, 'uShellOpacity', theme.shell);
        gl.bindVertexArray(shellVAO);
        gl.drawElementsInstanced(gl.TRIANGLES, box.indices.length, gl.UNSIGNED_SHORT, 0, nodeCount);
      }

      // Leave the state the flow passes expect, whether or not the shell drew.
      gl.disable(gl.CULL_FACE);
      gl.depthMask(false);
    }

    // 5. ribbons ------------------------------------
    if (ribbonIndexCount > 0) {
      gl.enable(gl.BLEND);
      flowBlend();
      gl.depthMask(false);
      const u = use(programs.ribbon);
      uMat(u, 'uViewProj', camera.viewProjection);
      u1f(u, 'uTime', time);
      u1f(u, 'uMotion', motionFlag);
      u1f(u, 'uBandDensity', 2.2);
      u1f(u, 'uBandSpeed', 0.55);
      u1f(u, 'uOpacity', theme.ribbon);
      u3v(u, 'uWarnColor', theme.warn);
      gl.bindVertexArray(ribbonVAO);
      gl.drawElements(gl.TRIANGLES, ribbonIndexCount, gl.UNSIGNED_INT, 0);
    }

    // 6. particles ----------------------------------
    if (particleCount > 0) {
      flowBlend();
      const drawn = Math.max(1, Math.floor(particleCount * q.particles));
      const u = use(programs.particle);
      uMat(u, 'uViewProj', camera.viewProjection);
      uMat(u, 'uView', camera.view);
      u1f(u, 'uTime', time);
      u1f(u, 'uMotion', motionFlag);
      u1f(u, 'uSpeed', 0.16);
      u1f(u, 'uOpacity', theme.particle);
      gl.bindVertexArray(particleVAO);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, drawn);
    }

    gl.bindVertexArray(null);
    gl.depthMask(true);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);

    resolveTarget(gl, targets.scene);

    // -- post ----------------------------------------
    const iterations = Math.min(q.bloom, targets.bloom.length);

    {
      const u = use(programs.bright);
      const target = targets.bloom[0];
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
      gl.viewport(0, 0, target.width, target.height);
      bindTexture(0, targets.scene.texture, u, 'uScene');
      u1f(u, 'uThreshold', caps.hdr ? 1.05 : 0.78);
      u1f(u, 'uKnee', 0.4);
      gl.bindVertexArray(gridVAO);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    {
      const u = use(programs.blur);
      for (let i = 0; i < iterations; i++) {
        const source = i === 0 ? targets.bloom[0] : targets.bloom[i - 1];
        const bloom = targets.bloom[i];
        const temp = targets.temp[i];

        // Downsample into the temp target on the horizontal pass, so the chain
        // gets progressively wider blur for the cost of progressively fewer pixels.
        gl.bindFramebuffer(gl.FRAMEBUFFER, temp.framebuffer);
        gl.viewport(0, 0, temp.width, temp.height);
        bindTexture(0, source.texture, u, 'uSource');
        u2f(u, 'uDirection', 1 / temp.width, 0);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        gl.bindFramebuffer(gl.FRAMEBUFFER, bloom.framebuffer);
        gl.viewport(0, 0, bloom.width, bloom.height);
        bindTexture(0, temp.texture, u, 'uSource');
        u2f(u, 'uDirection', 0, 1 / bloom.height);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
      }
      // Unused mips must still be readable by the composite, so clear them once.
      for (let i = iterations; i < targets.bloom.length; i++) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, targets.bloom[i].framebuffer);
        gl.viewport(0, 0, targets.bloom[i].width, targets.bloom[i].height);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
      }
    }

    {
      const u = use(programs.composite);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, width, height);
      bindTexture(0, targets.scene.texture, u, 'uScene');
      bindTexture(1, targets.bloom[0].texture, u, 'uBloom0');
      bindTexture(2, targets.bloom[1].texture, u, 'uBloom1');
      bindTexture(3, targets.bloom[2].texture, u, 'uBloom2');
      u2f(u, 'uTexel', 1 / width, 1 / height);
      u1f(u, 'uBloomStrength', theme.bloom);
      u1f(u, 'uExposure', theme.exposure);
      u1f(u, 'uVignette', 0.55);
      u1f(u, 'uSaturation', theme.saturation);
      u1f(u, 'uGrain', theme.grain);
      u1f(u, 'uTime', time);
      u1f(u, 'uFxaa', caps.samples === 0 ? 1 : 0);
      gl.bindVertexArray(gridVAO);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindVertexArray(null);
    }
  }

  // ── projected labels ────────────────────────────────

  const labelPoint = vec3.create();
  const labelAnchor = vec3.create();
  // Pooled: this runs every frame, and a fresh object per node per layer at
  // 60 Hz is exactly the garbage the rest of this module goes out of its way
  // to avoid. `labels` keeps its entries; `labelCount` says how many are live.
  const labelPool = [];
  const labels = [];

  function projectLabels() {
    labels.length = 0;

    for (const node of scene.nodes) {
      vec3.set(
        labelAnchor,
        node.center.x,
        node.center.y + node.extent.h / 2 + 0.16,
        node.center.z
      );
      mat4.transformPoint(labelPoint, camera.viewProjection, labelAnchor);
      if (labelPoint[2] < -1 || labelPoint[2] > 1) continue;

      const dx = node.center.x - camera.eye[0];
      const dy = node.center.y - camera.eye[1];
      const dz = node.center.z - camera.eye[2];
      const distance = Math.hypot(dx, dy, dz);

      const slot = labels.length;
      if (!labelPool[slot]) labelPool[slot] = { id: '', index: 0, x: 0, y: 0, distance: 0, opacity: 0 };
      const entry = labelPool[slot];
      entry.id = node.id;
      entry.index = node.index;
      entry.x = (labelPoint[0] * 0.5 + 0.5) * cssWidth;
      entry.y = (1 - (labelPoint[1] * 0.5 + 0.5)) * cssHeight;
      entry.distance = distance;
      entry.opacity = clamp(1.6 - distance / (camera.radius * 1.9), 0, 1);
      labels.push(entry);
    }

    // Greedy near-to-far declutter: a wall of overlapping text is worse than
    // no text, and the nearest label is the one the user is looking at.
    labels.sort((a, b) => a.distance - b.distance);
    for (let i = 0; i < labels.length; i++) {
      if (labels[i].opacity <= 0) continue;
      for (let j = i + 1; j < labels.length; j++) {
        if (Math.abs(labels[i].x - labels[j].x) < 74 && Math.abs(labels[i].y - labels[j].y) < 20) {
          labels[j].opacity = 0;
        }
      }
    }
    return labels;
  }

  // ── input ───────────────────────────────────────────

  const pointers = new Map();
  let dragMode = null;
  let lastX = 0;
  let lastY = 0;
  let downX = 0;
  let downY = 0;
  let pinchDistance = 0;
  let pointerInside = false;
  let hasInteracted = false;
  let hoverFrame = 0;

  const toNDC = (event) => {
    const rect = canvas.getBoundingClientRect();
    return [
      ((event.clientX - rect.left) / (rect.width || 1)) * 2 - 1,
      1 - ((event.clientY - rect.top) / (rect.height || 1)) * 2,
    ];
  };

  function pickAt(ndcX, ndcY) {
    const { origin, dir } = rayFromNDC(camera, ndcX, ndcY);
    let best = null;
    let bestT = Infinity;
    for (const node of scene.nodes) {
      const min = [
        node.center.x - node.extent.w / 2,
        node.center.y - node.extent.h / 2,
        node.center.z - node.extent.d / 2,
      ];
      const max = [
        node.center.x + node.extent.w / 2,
        node.center.y + node.extent.h / 2,
        node.center.z + node.extent.d / 2,
      ];
      const t = rayBox(origin, dir, min, max);
      if (t >= 0 && t < bestT) {
        bestT = t;
        best = node;
      }
    }
    return best;
  }

  const onPointerDown = (event) => {
    canvas.setPointerCapture?.(event.pointerId);
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    hasInteracted = true;
    lastX = downX = event.clientX;
    lastY = downY = event.clientY;
    dragMode = event.button === 2 || event.shiftKey ? 'pan' : 'orbit';
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinchDistance = Math.hypot(a.x - b.x, a.y - b.y);
      dragMode = 'pinch';
    }
    camera.idleFor = 0;
  };

  const onPointerMove = (event) => {
    if (pointers.has(event.pointerId)) {
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    }

    if (dragMode === 'pinch' && pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const distance = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinchDistance > 0) dolly(camera, clamp(pinchDistance / distance, 0.5, 2));
      pinchDistance = distance;
      return;
    }

    if (dragMode === 'orbit' || dragMode === 'pan') {
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      if (dragMode === 'orbit') orbit(camera, -dx * 0.006, -dy * 0.006);
      else pan(camera, dx, dy);
      return;
    }

    // Hover picking, throttled to one test per frame. The handle is kept so
    // dispose() can cancel it: a callback that fires after teardown would touch
    // a deleted buffer and call back into an unmounted component.
    if (hoverFrame) return;
    const [nx, ny] = toNDC(event);
    hoverFrame = requestAnimationFrame(() => {
      hoverFrame = 0;
      const hit = pickAt(nx, ny);
      const id = hit ? hit.id : null;
      if (id !== hoveredId) {
        hoveredId = id;
        applySelection();
        options.onHover?.(id);
      }
      canvas.style.cursor = id ? 'pointer' : 'grab';
    });
  };

  const endDrag = (event) => {
    const wasDragging = dragMode;
    pointers.delete(event.pointerId);
    canvas.releasePointerCapture?.(event.pointerId);
    if (pointers.size < 2 && dragMode === 'pinch') dragMode = null;
    if (pointers.size === 0) dragMode = null;

    // A click is a pointerup close to where the pointer went down. Comparing
    // against `lastX/lastY` instead would always read as a click, because
    // pointermove keeps those in step with the cursor.
    if (wasDragging === 'orbit' && Math.hypot(event.clientX - downX, event.clientY - downY) < 5) {
      const [nx, ny] = toNDC(event);
      const hit = pickAt(nx, ny);
      options.onSelect?.(hit ? hit.id : null);
      if (hit) focusOn(camera, hit);
    }
  };

  const onWheel = (event) => {
    // Do not steal the page's scroll until the user has shown they want to
    // drive this panel.
    if (!pointerInside || !hasInteracted) return;
    event.preventDefault();
    dolly(camera, Math.exp(event.deltaY * 0.0012));
  };

  const onDoubleClick = () => frameAll(true);
  const onPointerEnter = () => { pointerInside = true; };
  const onPointerLeave = () => {
    pointerInside = false;
    if (hoveredId !== null) {
      hoveredId = null;
      applySelection();
      options.onHover?.(null);
    }
  };
  const onContextMenu = (event) => event.preventDefault();

  const onKeyDown = (event) => {
    const step = 0.12;
    switch (event.key) {
      case 'ArrowLeft': orbit(camera, -step, 0); break;
      case 'ArrowRight': orbit(camera, step, 0); break;
      case 'ArrowUp': orbit(camera, 0, -step); break;
      case 'ArrowDown': orbit(camera, 0, step); break;
      case '+': case '=': dolly(camera, 0.88); break;
      case '-': case '_': dolly(camera, 1.14); break;
      case 'Home': frameAll(true); break;
      default: return;
    }
    hasInteracted = true;
    event.preventDefault();
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('pointerenter', onPointerEnter);
  canvas.addEventListener('pointerleave', onPointerLeave);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('dblclick', onDoubleClick);
  canvas.addEventListener('contextmenu', onContextMenu);
  canvas.addEventListener('keydown', onKeyDown);

  // ── context loss ────────────────────────────────────

  let contextLost = false;
  const onContextLost = (event) => {
    // preventDefault is what asks the browser to try to restore the context.
    event.preventDefault();
    contextLost = true;
    stop();
    options.onContextLost?.();
  };
  const onContextRestored = () => {
    contextLost = false;
    // Every GL object created against the old context is gone: programs, VAOs
    // and the static geometry buffers, not just the render targets. Rebuilding
    // them in place would mean a second construction path that nothing else
    // exercises, so the caller disposes this engine and creates a fresh one
    // through the same code that ran at mount.
    options.onContextRestored?.();
  };
  canvas.addEventListener('webglcontextlost', onContextLost);
  canvas.addEventListener('webglcontextrestored', onContextRestored);

  // ── loop ────────────────────────────────────────────

  let raf = 0;
  let last = 0;
  let visible = true;
  let running = false;
  let slowFrames = 0;
  let fastFrames = 0;
  let qualityChanges = 0;

  const resizeObserver = typeof ResizeObserver !== 'undefined'
    ? new ResizeObserver(() => resize())
    : null;
  resizeObserver?.observe(canvas);

  const intersectionObserver = typeof IntersectionObserver !== 'undefined'
    ? new IntersectionObserver((entries) => {
      visible = entries.some(e => e.isIntersecting);
      if (visible) start(); else stop();
    }, { threshold: 0.01 })
    : null;
  intersectionObserver?.observe(canvas);

  const onVisibilityChange = () => {
    if (document.hidden) stop(); else start();
  };
  document.addEventListener('visibilitychange', onVisibilityChange);

  function adaptQuality(frameMs) {
    if (qualityChanges >= 2) return;

    if (frameMs > 24) { slowFrames++; fastFrames = 0; } else { slowFrames = 0; }
    if (frameMs < 12) fastFrames++; else fastFrames = 0;

    if (slowFrames > 30 && quality > 0) {
      quality--;
      qualityChanges++;
      slowFrames = 0;
      resize();
    } else if (fastFrames > 120 && quality < (software ? 1 : reduced ? 1 : 2)) {
      quality++;
      qualityChanges++;
      fastFrames = 0;
      resize();
    }
  }

  function tick(now) {
    raf = requestAnimationFrame(tick);
    if (contextLost) return;

    // Clamped so a tab that was hidden for a minute does not resume with a
    // 60-second dt and teleport the camera across the scene.
    const dt = last ? Math.min((now - last) / 1000, 0.05) : 0.016;
    last = now;
    if (motion && playing) clock += dt;

    updateCamera(camera, dt, aspect(), motion);

    const started = performance.now();
    renderFrame();
    adaptQuality(performance.now() - started);

    options.onFrame?.(projectLabels());
  }

  function start() {
    if (running || contextLost || !visible || (typeof document !== 'undefined' && document.hidden)) return;
    running = true;
    last = 0;
    raf = requestAnimationFrame(tick);
  }

  function stop() {
    running = false;
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
  }

  /**
   * Re-frame the model.
   *
   * `recompose` hands the orbit angle back to the framing solve. An automatic
   * re-frame after a layer is added must not do that - it would take an angle
   * the viewer chose - but an explicit reset is exactly a request for the
   * default composition again.
   */
  function frameAll(recompose = false) {
    if (recompose) camera.userPosed = false;
    frameBounds(camera, scene.bounds, aspect(), safeInsets());
    camera.idleFor = 0;
  }

  /**
   * The parts of the panel the overlay already occupies, as fractions of its
   * height: the metric chips plus a layer label's headroom at the top, the
   * family legend at the bottom. Measured in pixels because the chrome is a
   * fixed size while the panel is not - on a phone it is a fifth of the height
   * and on a desktop less than a tenth.
   */
  function safeInsets() {
    const height = Math.max(canvas.clientHeight || 0, 1);
    return { top: 64 / height, bottom: 46 / height };
  }

  // ── public surface ──────────────────────────────────

  resize();

  const engine = {
    capabilities: caps,

    setScene(next, { keepCamera = true } = {}) {
      const first = scene.nodes.length === 0;
      scene = next;
      uploadScene();
      if (first || !keepCamera) {
        frameAll();
        settle(camera);
      }
      if (!motion) engine.renderNow();
    },

    setTheme(isDarkMode) {
      const next = isDarkMode ? THEMES.dark : THEMES.light;
      const rebuild = next.additive !== theme.additive;
      theme = next;
      // The ribbons and particles bake a theme-dependent colour, so switching
      // between additive and over compositing has to rebuild them.
      if (rebuild && scene.nodes.length > 0) {
        buildRibbons();
        buildParticles();
      }
      if (!motion) engine.renderNow();
    },

    setMotion(enabled) {
      motion = enabled;
      camera.driftEnabled = enabled;
      if (!enabled) {
        settle(camera);
        engine.renderNow();
      }
    },

    setPlaying(next) {
      playing = next;
    },

    setSelection(nextSelected, nextHovered = hoveredId) {
      selectedId = nextSelected;
      hoveredId = nextHovered;
      applySelection();
      if (!motion) engine.renderNow();
    },

    focusNode(id) {
      const node = scene.nodes.find(n => n.id === id);
      if (node) focusOn(camera, node);
      if (!motion) engine.renderNow();
    },

    frameAll,

    /** Render one frame synchronously. Used before PNG export and in reduced motion. */
    renderNow() {
      if (contextLost || !targets.scene) return;
      updateCamera(camera, 0.016, aspect(), motion);
      renderFrame();
      options.onFrame?.(projectLabels());
    },

    resize,
    start,
    stop,

    dispose() {
      stop();
      if (hoverFrame) cancelAnimationFrame(hoverFrame);
      hoverFrame = 0;
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', endDrag);
      canvas.removeEventListener('pointercancel', endDrag);
      canvas.removeEventListener('pointerenter', onPointerEnter);
      canvas.removeEventListener('pointerleave', onPointerLeave);
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('dblclick', onDoubleClick);
      canvas.removeEventListener('contextmenu', onContextMenu);
      canvas.removeEventListener('keydown', onKeyDown);
      canvas.removeEventListener('webglcontextlost', onContextLost);
      canvas.removeEventListener('webglcontextrestored', onContextRestored);
      document.removeEventListener('visibilitychange', onVisibilityChange);
      resizeObserver?.disconnect();
      intersectionObserver?.disconnect();

      disposeTargets();
      Object.values(programs).forEach(p => gl.deleteProgram(p.program));
      [boxVertexBuffer, boxIndexBuffer, quadBuffer, sliceInstanceBuffer,
        nodeInstanceBuffer, particleBuffer, ribbonVertexBuffer, ribbonIndexBuffer]
        .forEach(b => gl.deleteBuffer(b));
      [slabVAO, shellVAO, shadowVAO, particleVAO, ribbonVAO, gridVAO]
        .forEach(v => gl.deleteVertexArray(v));

      // Browsers cap live contexts at about 16. Without this an unmount in
      // React Strict Mode exhausts them after eight remounts and the panel
      // silently stops rendering. Skipped when the context is already lost,
      // because asking a lost context to lose itself cancels the restoration
      // the caller is waiting on.
      if (!contextLost) gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
  };

  return engine;
}
