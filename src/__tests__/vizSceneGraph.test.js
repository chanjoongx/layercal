import { describe, it, expect } from 'vitest'
import { buildScene, emptyScene, describeScene, interleaveByLink } from '@/viz/sceneGraph'
import { getLayerTypes, DEFAULT_LAYER_PARAMS } from '@/config/layerTypes'
import { validateModelDimensions } from '@/utils/modelValidation'
import { TRANSLATIONS } from '@/config/translations'

const LAYER_TYPES = getLayerTypes(TRANSLATIONS.en)

let counter = 0
const layer = (type, params = {}) => ({
  id: `l${++counter}`,
  type,
  params: { ...DEFAULT_LAYER_PARAMS[type], ...params },
})

const CNN = () => [
  layer('conv2d', { in_channels: 3, out_channels: 64 }),
  layer('batchnorm', { num_features: 64 }),
  layer('relu'),
  layer('maxpool2d', { kernel_size: 2 }),
  layer('conv2d', { in_channels: 64, out_channels: 128 }),
  layer('relu'),
  layer('linear', { input_dim: 128, output_dim: 64 }),
]

describe('emptyScene', () => {
  it('is safe to frame a camera on', () => {
    const scene = emptyScene()
    expect(scene.nodes).toHaveLength(0)
    expect(scene.links).toHaveLength(0)
    expect(scene.bounds.radius).toBe(1)
    expect(scene.bounds.min).toHaveLength(3)
    expect(scene.bounds.max).toHaveLength(3)
  })
})

describe('buildScene: degenerate input', () => {
  it('returns an empty scene for an empty model', () => {
    expect(buildScene([], LAYER_TYPES).nodes).toHaveLength(0)
  })

  it('returns an empty scene for null', () => {
    expect(buildScene(null, LAYER_TYPES).nodes).toHaveLength(0)
  })

  it('drops layers whose type is not in the table rather than crashing', () => {
    const scene = buildScene([layer('conv2d'), { id: 'x', type: 'wormhole', params: {} }], LAYER_TYPES)
    expect(scene.nodes).toHaveLength(1)
  })

  it('survives a model made only of unknown types', () => {
    const scene = buildScene([{ id: 'x', type: 'wormhole', params: {} }], LAYER_TYPES)
    expect(scene.nodes).toHaveLength(0)
    expect(scene.bounds.radius).toBe(1)
  })

  it('handles a single layer without dividing by zero', () => {
    const scene = buildScene([layer('linear')], LAYER_TYPES)
    expect(scene.nodes).toHaveLength(1)
    expect(scene.links).toHaveLength(0)
    expect(scene.nodes[0].paramShare).toBeCloseTo(1, 6)
    expect(scene.nodes[0].phase).toBe(0)
  })

  it('gives every node a finite share when the model has no parameters at all', () => {
    const scene = buildScene([layer('relu'), layer('softmax')], LAYER_TYPES)
    for (const node of scene.nodes) {
      expect(node.paramShare).toBe(0)
      expect(Number.isFinite(node.paramShare)).toBe(true)
    }
  })
})

describe('buildScene: layout', () => {
  const scene = buildScene(CNN(), LAYER_TYPES)

  it('marches strictly forward along +Z', () => {
    for (let i = 1; i < scene.nodes.length; i++) {
      expect(scene.nodes[i].center.z).toBeGreaterThan(scene.nodes[i - 1].center.z)
    }
  })

  it('leaves a gap between every pair of neighbours', () => {
    for (let i = 1; i < scene.nodes.length; i++) {
      const a = scene.nodes[i - 1]
      const b = scene.nodes[i]
      const gap = (b.center.z - b.extent.d / 2) - (a.center.z + a.extent.d / 2)
      expect(gap, `layers ${i - 1} and ${i} overlap`).toBeGreaterThan(0)
    }
  })

  it('centres the run on the origin', () => {
    const first = scene.nodes[0]
    const last = scene.nodes[scene.nodes.length - 1]
    const lo = first.center.z - first.extent.d / 2
    const hi = last.center.z + last.extent.d / 2
    expect(lo + hi).toBeCloseTo(0, 5)
  })

  it('lifts annotation plates above the flow so they read as markers', () => {
    const relu = scene.nodes.find(n => n.type === 'relu')
    const conv = scene.nodes.find(n => n.type === 'conv2d')
    expect(relu.annotation).toBe(true)
    expect(conv.annotation).toBe(false)
    expect(relu.center.y).toBeGreaterThan(conv.center.y)
  })

  it('gives annotations the cross-section of the tensor they sit on', () => {
    const conv = scene.nodes[0]
    const bn = scene.nodes[1]
    expect(bn.extent.w).toBeCloseTo(conv.extent.w * 0.86, 6)
    expect(bn.extent.d).toBe(0.16)
  })

  it('spreads the animation phase from 0 to 1 across the stack', () => {
    expect(scene.nodes[0].phase).toBe(0)
    expect(scene.nodes[scene.nodes.length - 1].phase).toBe(1)
    for (let i = 1; i < scene.nodes.length; i++) {
      expect(scene.nodes[i].phase).toBeGreaterThan(scene.nodes[i - 1].phase)
    }
  })
})

describe('buildScene: bounds', () => {
  const scene = buildScene(CNN(), LAYER_TYPES)

  it('encloses every node', () => {
    for (const node of scene.nodes) {
      expect(node.center.x - node.extent.w / 2).toBeGreaterThanOrEqual(scene.bounds.min[0] - 1e-9)
      expect(node.center.y - node.extent.h / 2).toBeGreaterThanOrEqual(scene.bounds.min[1] - 1e-9)
      expect(node.center.z - node.extent.d / 2).toBeGreaterThanOrEqual(scene.bounds.min[2] - 1e-9)
      expect(node.center.x + node.extent.w / 2).toBeLessThanOrEqual(scene.bounds.max[0] + 1e-9)
      expect(node.center.y + node.extent.h / 2).toBeLessThanOrEqual(scene.bounds.max[1] + 1e-9)
      expect(node.center.z + node.extent.d / 2).toBeLessThanOrEqual(scene.bounds.max[2] + 1e-9)
    }
  })

  it('reports a finite, positive radius', () => {
    expect(scene.bounds.radius).toBeGreaterThan(0)
    expect(Number.isFinite(scene.bounds.radius)).toBe(true)
  })

  it('puts the centre inside the box', () => {
    for (let i = 0; i < 3; i++) {
      expect(scene.bounds.center[i]).toBeGreaterThanOrEqual(scene.bounds.min[i])
      expect(scene.bounds.center[i]).toBeLessThanOrEqual(scene.bounds.max[i])
    }
  })
})

describe('buildScene: shares and totals', () => {
  const scene = buildScene(CNN(), LAYER_TYPES)

  it('parameter shares sum to one', () => {
    const sum = scene.nodes.reduce((t, n) => t + n.paramShare, 0)
    expect(sum).toBeCloseTo(1, 6)
  })

  it('FLOP shares sum to one', () => {
    const sum = scene.nodes.reduce((t, n) => t + n.flopShare, 0)
    expect(sum).toBeCloseTo(1, 6)
  })

  it('totals match the sum of the parts', () => {
    const params = scene.nodes.reduce((t, n) => t + n.params, 0)
    expect(scene.totals.params).toBeCloseTo(params, 6)
    expect(scene.totals.depth).toBe(scene.nodes.length)
  })

  it('agrees with the layer table it was built from', () => {
    const expected = CNN().reduce(
      (t, l) => t + LAYER_TYPES[l.type].calculate(l.params), 0
    )
    expect(scene.totals.params).toBe(expected)
  })
})

describe('buildScene: links', () => {
  it('creates one link per adjacent pair', () => {
    const scene = buildScene(CNN(), LAYER_TYPES)
    expect(scene.links).toHaveLength(scene.nodes.length - 1)
    scene.links.forEach((link, i) => {
      expect(link.from).toBe(i)
      expect(link.to).toBe(i + 1)
    })
  })

  it('marks the link that feeds a mismatched layer as broken', () => {
    // Conv2D emits 64 channels straight into a 128-wide BatchNorm.
    const layers = [
      layer('conv2d', { in_channels: 3, out_channels: 64 }),
      layer('batchnorm', { num_features: 128 }),
    ]
    const issues = validateModelDimensions(layers)
    const scene = buildScene(layers, LAYER_TYPES, issues)

    expect(scene.nodes[1].warning).toBe(true)
    expect(scene.nodes[1].warningField).toBe('num_features')
    expect(scene.nodes[1].warningExpected).toBe(64)
    expect(scene.links[0].broken).toBe(true)
  })

  it('leaves links unbroken when the stack lines up', () => {
    const layers = CNN()
    const scene = buildScene(layers, LAYER_TYPES, validateModelDimensions(layers))
    expect(scene.links.some(l => l.broken)).toBe(false)
  })

  it('tolerates a missing issues map', () => {
    const scene = buildScene(CNN(), LAYER_TYPES, undefined)
    expect(scene.nodes.every(n => n.warning === false)).toBe(true)
  })

  it('gives every link a positive width at both ends', () => {
    const scene = buildScene(CNN(), LAYER_TYPES)
    for (const link of scene.links) {
      expect(link.width0).toBeGreaterThan(0)
      expect(link.width1).toBeGreaterThan(0)
    }
  })
})

describe('buildScene: purity', () => {
  it('is deterministic', () => {
    counter = 0
    const a = buildScene(CNN(), LAYER_TYPES)
    counter = 0
    const b = buildScene(CNN(), LAYER_TYPES)
    expect(JSON.parse(JSON.stringify(a))).toEqual(JSON.parse(JSON.stringify(b)))
  })

  it('does not mutate the layers it was given', () => {
    const layers = CNN()
    const before = JSON.parse(JSON.stringify(layers))
    buildScene(layers, LAYER_TYPES)
    expect(layers).toEqual(before)
  })
})

describe('buildScene: scale', () => {
  it('lays out sixty layers without overlap or non-finite geometry', () => {
    const layers = []
    for (let i = 0; i < 20; i++) {
      layers.push(layer('transformer', { d_model: 512 }))
      layers.push(layer('layernorm', { normalized_shape: 512 }))
      layers.push(layer('relu'))
    }
    const scene = buildScene(layers, LAYER_TYPES)
    expect(scene.nodes).toHaveLength(60)
    for (const node of scene.nodes) {
      for (const v of [node.center.z, node.extent.w, node.extent.h, node.extent.d]) {
        expect(Number.isFinite(v)).toBe(true)
      }
    }
  })

  it('keeps a 50,000-word embedding inside the size clamp', () => {
    const scene = buildScene([
      layer('embedding', { vocab_size: 50000, embedding_dim: 1024 }),
      layer('linear', { input_dim: 1024, output_dim: 64 }),
    ], LAYER_TYPES)
    for (const node of scene.nodes) {
      expect(node.extent.w).toBeLessThanOrEqual(2.6)
      expect(node.extent.h).toBeLessThanOrEqual(2.6)
    }
  })
})

describe('describeScene', () => {
  it('summarises a model for assistive technology', () => {
    const scene = buildScene(CNN(), LAYER_TYPES)
    const text = describeScene(scene, TRANSLATIONS.en)
    expect(text).toContain(String(scene.totals.depth))
    expect(text).not.toContain('{layers}')
    expect(text).not.toContain('{params}')
  })

  it('has something to say about an empty canvas', () => {
    expect(describeScene(emptyScene(), TRANSLATIONS.en)).toBeTruthy()
    expect(describeScene(null, TRANSLATIONS.en)).toBeTruthy()
  })

  it('works with no translation table at all', () => {
    expect(describeScene(buildScene(CNN(), LAYER_TYPES), undefined)).toContain('7')
  })
})


describe('interleaveByLink', () => {
  it('emits exactly the requested number of entries per link', () => {
    const counts = [3, 1, 4, 2]
    const order = interleaveByLink(counts)
    expect(order).toHaveLength(10)
    for (let i = 0; i < counts.length; i++) {
      expect(order.filter(x => x === i)).toHaveLength(counts[i])
    }
  })

  it('spreads every prefix across all links', () => {
    // This is the whole point: reduced quality draws a prefix of the particle
    // buffer, and a link-major order would delete the flow from the tail of the
    // model rather than thinning it evenly.
    const counts = new Array(12).fill(20)
    const order = interleaveByLink(counts)

    for (const fraction of [0.25, 0.5, 1]) {
      const prefix = order.slice(0, Math.floor(order.length * fraction))
      const perLink = counts.map((_, i) => prefix.filter(x => x === i).length)
      expect(Math.max(...perLink) - Math.min(...perLink)).toBeLessThanOrEqual(1)
    }
  })

  it('keeps prefixes fair when the budgets are uneven', () => {
    const counts = [10, 2, 6]
    const order = interleaveByLink(counts)
    const quarter = order.slice(0, Math.floor(order.length / 4))
    // Every link that still has budget left is represented.
    expect(new Set(quarter).size).toBe(3)
  })

  it('is defensive about empty and malformed input', () => {
    expect(interleaveByLink([])).toEqual([])
    expect(interleaveByLink(null)).toEqual([])
    expect(interleaveByLink([0, 0])).toEqual([])
  })
})
