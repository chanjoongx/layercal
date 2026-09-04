/**
 * Model state -> renderable scene.
 *
 * This is the only place the model is turned into geometry, and it is a pure
 * function: same layers in, deep-equal scene out, no clock, no randomness, no
 * DOM. Everything that moves is a uniform the renderer varies over time, not
 * data baked into the scene, which is what makes the whole visualisation
 * unit-testable without a GPU.
 *
 * The scene is consumed by two renderers: the WebGL2 engine and the SVG
 * fallback. Neither knows anything about layers; both know about nodes and
 * links.
 */

import { paintFor } from './palette';
import {
  propagateShapes,
  extentFor,
  annotationExtent,
  ANNOTATION_TYPES,
} from './tensorShape';

/**
 * @typedef {import('./tensorShape').TensorShape} TensorShape
 * @typedef {import('./palette').LayerPaint} LayerPaint
 *
 * @typedef {{
 *   id: string,
 *   index: number,
 *   type: string,
 *   name: string,
 *   paint: LayerPaint,
 *   extent: {w:number,h:number,d:number},
 *   center: {x:number,y:number,z:number},
 *   shape: TensorShape,
 *   params: number,
 *   flops: number,
 *   paramShare: number,
 *   flopShare: number,
 *   annotation: boolean,
 *   warning: boolean,
 *   warningField: string|null,
 *   warningExpected: number|null,
 *   phase: number,
 * }} SceneNode
 *
 * @typedef {{
 *   from: number,
 *   to: number,
 *   width0: number,
 *   width1: number,
 *   broken: boolean,
 *   flow: number,
 * }} SceneLink
 *
 * @typedef {{
 *   nodes: SceneNode[],
 *   links: SceneLink[],
 *   bounds: {min:number[], max:number[], center:number[], radius:number},
 *   totals: {params:number, flops:number, depth:number},
 * }} Scene
 */

/** Height the stack floats above the grid. */
const FLOAT_HEIGHT = 0.9;

const finite = (v) => (Number.isFinite(v) ? v : 0);

/** Empty scene that is still safe to frame a camera on. */
export function emptyScene() {
  return {
    nodes: [],
    links: [],
    bounds: {
      min: [-1, 0, -1],
      max: [1, 1.8, 1],
      center: [0, FLOAT_HEIGHT, 0],
      radius: 1,
    },
    totals: { params: 0, flops: 0, depth: 0 },
  };
}

/**
 * @param {Array<{id: string, type: string, params: object}>} layers
 * @param {Record<string, {name: string, calculate: Function, calculateFLOPs?: Function}>} layerTypes
 * @param {Map<number, {field: string, expected: number}>} [issues]
 * @returns {Scene}
 */
export function buildScene(layers, layerTypes, issues) {
  const list = Array.isArray(layers) ? layers : [];
  const types = layerTypes || {};
  const problems = issues instanceof Map ? issues : new Map();

  if (list.length === 0) return emptyScene();

  const shapes = propagateShapes(list);

  // Pass 1: extents and per-layer cost. Annotations inherit the cross-section
  // of the tensor they sit on, so they need the previous extent, which is why
  // this cannot be a single map().
  const nodes = [];
  let previousExtent = null;
  let totalParams = 0;
  let totalFlops = 0;

  for (let i = 0; i < list.length; i++) {
    const layer = list[i];
    const config = types[layer.type];
    if (!config) continue;

    const shape = shapes[i];
    const annotation = ANNOTATION_TYPES.has(layer.type);
    const extent = annotation
      ? annotationExtent(previousExtent)
      : extentFor(shape, previousExtent);

    const params = Math.max(0, finite(config.calculate(layer.params)));
    const flops = Math.max(0, finite(config.calculateFLOPs ? config.calculateFLOPs(layer.params) : 0));
    totalParams += params;
    totalFlops += flops;

    const issue = problems.get(i);

    nodes.push({
      id: layer.id != null ? String(layer.id) : `node-${i}`,
      index: i,
      type: layer.type,
      name: config.name || layer.type,
      paint: paintFor(layer.type),
      extent,
      center: { x: 0, y: FLOAT_HEIGHT, z: 0 },
      shape,
      params,
      flops,
      paramShare: 0,
      flopShare: 0,
      annotation,
      warning: Boolean(issue),
      warningField: issue ? issue.field : null,
      warningExpected: issue ? issue.expected : null,
      phase: 0,
    });

    if (!annotation) previousExtent = extent;
  }

  if (nodes.length === 0) return emptyScene();

  // Pass 2: lay the stack out along +Z. The gap scales with the neighbouring
  // depths so a run of thin activation plates stays legible next to a deep
  // convolution block instead of collapsing into it.
  let z = 0;
  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    if (i === 0) {
      z = node.extent.d / 2;
    } else {
      const prev = nodes[i - 1];
      const gap = 0.42 + 0.3 * Math.max(prev.extent.d, node.extent.d);
      z += prev.extent.d / 2 + gap + node.extent.d / 2;
    }
    node.center.z = z;
    // Annotations ride slightly above the flow so they read as markers on the
    // pipe rather than as another tensor in it.
    node.center.y = FLOAT_HEIGHT + (node.annotation ? 0.08 : 0);
  }

  // Recentre the run on the origin so the camera's default target is the model.
  const span = nodes[nodes.length - 1].center.z + nodes[nodes.length - 1].extent.d / 2;
  const shift = span / 2;
  for (const node of nodes) node.center.z -= shift;

  // Pass 3: shares and animation phase. `phase` is the node's normalised
  // position along the stack, which the shader uses to time the forward-pass
  // pulse; storing it here keeps the shader free of any layout knowledge.
  const denomParams = totalParams > 0 ? totalParams : 1;
  const denomFlops = totalFlops > 0 ? totalFlops : 1;
  const lastIndex = Math.max(1, nodes.length - 1);

  for (let i = 0; i < nodes.length; i++) {
    nodes[i].paramShare = totalParams > 0 ? nodes[i].params / denomParams : 0;
    nodes[i].flopShare = totalFlops > 0 ? nodes[i].flops / denomFlops : 0;
    nodes[i].phase = i / lastIndex;
  }

  // Pass 4: links. A link is "broken" when the layer it feeds reported a
  // dimension mismatch, because that is where the tensor stops lining up.
  const links = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    const a = nodes[i];
    const b = nodes[i + 1];
    links.push({
      from: i,
      to: i + 1,
      width0: Math.min(a.extent.w, a.extent.h) * 0.34,
      width1: Math.min(b.extent.w, b.extent.h) * 0.34,
      broken: b.warning,
      flow: b.flopShare,
    });
  }

  return {
    nodes,
    links,
    bounds: computeBounds(nodes),
    totals: { params: totalParams, flops: totalFlops, depth: nodes.length },
  };
}

function computeBounds(nodes) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];

  for (const node of nodes) {
    const { center, extent } = node;
    min[0] = Math.min(min[0], center.x - extent.w / 2);
    min[1] = Math.min(min[1], center.y - extent.h / 2);
    min[2] = Math.min(min[2], center.z - extent.d / 2);
    max[0] = Math.max(max[0], center.x + extent.w / 2);
    max[1] = Math.max(max[1], center.y + extent.h / 2);
    max[2] = Math.max(max[2], center.z + extent.d / 2);
  }

  const center = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];
  const radius = Math.max(
    0.6,
    0.5 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2])
  );

  return { min, max, center, radius };
}

/**
 * Round-robin ordering for a per-link budget.
 *
 * The renderer packs one particle per entry into a single buffer, and reduced
 * quality draws a *prefix* of that buffer. A link-major order would therefore
 * delete the flow from the tail of the model rather than thinning it, so the
 * order has to be interleaved: pass 0 takes one particle from every link, pass
 * 1 takes a second from every link that has one, and so on. Any prefix is then
 * a fair sample across the whole model.
 *
 * @param {number[]} counts particles wanted per link
 * @returns {number[]} link indices, in the order they should be written
 */
export function interleaveByLink(counts) {
  const list = Array.isArray(counts) ? counts : [];
  const order = [];
  let longest = 0;
  for (const n of list) if (n > longest) longest = n;

  for (let pass = 0; pass < longest; pass++) {
    for (let i = 0; i < list.length; i++) {
      if (pass < list[i]) order.push(i);
    }
  }
  return order;
}

/** Screen-reader and `aria-label` summary of a scene. */
export function describeScene(scene, t) {
  const strings = t || {};
  if (!scene || scene.nodes.length === 0) {
    return strings.vizEmptyLabel || 'An empty neural network canvas.';
  }
  const template = strings.vizLabel
    || 'Neural network with {layers} layers and {params} parameters.';
  return template
    .replace('{layers}', String(scene.totals.depth))
    .replace('{params}', scene.totals.params.toLocaleString());
}
