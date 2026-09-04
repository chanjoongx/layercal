import { describe, it, expect } from 'vitest'
import { outgoingWidth, wireAppendedLayer } from '@/utils/layerWiring'
import { getLayerTypes, DEFAULT_LAYER_PARAMS, LAYER_TYPE_IDS } from '@/config/layerTypes'
import { validateModelDimensions } from '@/utils/modelValidation'

// The wiring reads option lists off the same table the UI renders, so the test
// uses the real one rather than a fixture that could drift away from it.
const LAYER_TYPES = getLayerTypes(new Proxy({}, {
  get: (_, key) => new Proxy({ name: String(key), description: '' }, {
    get: (target, k) => (k in target ? target[k] : String(k)),
  }),
}))

const layer = (type, params = {}) => ({
  type,
  params: { ...DEFAULT_LAYER_PARAMS[type], ...params },
})

/** Append `type` the way the UI does, and hand back the whole stack. */
const append = (layers, type) => [
  ...layers,
  { type, params: wireAppendedLayer(type, layers, LAYER_TYPES) },
]

describe('outgoingWidth', () => {
  it('is null for an empty model', () => {
    expect(outgoingWidth([])).toBeNull()
  })

  it('reports the channel count leaving the last sized layer', () => {
    expect(outgoingWidth([layer('conv2d', { out_channels: 64 })])).toBe(64)
    expect(outgoingWidth([layer('linear', { output_dim: 256 })])).toBe(256)
  })

  it('looks past a trailing run of layers that carry no width', () => {
    // ReLU and Dropout pass the tensor through untouched. Reading only the last
    // shape would work by luck here; walking back is what makes it not luck.
    const layers = [
      layer('conv2d', { out_channels: 128 }),
      layer('relu'),
      layer('dropout'),
    ]
    expect(outgoingWidth(layers)).toBe(128)
  })

  it('tolerates a malformed layer list', () => {
    expect(outgoingWidth(undefined)).toBeNull()
    // An unrecognised type carries the tensor through untouched rather than
    // throwing, so the width is the seed's. The contract worth pinning here is
    // that it stays a usable number.
    expect(outgoingWidth([{ type: 'nonsense', params: {} }])).toBeGreaterThan(0)
  })
})

describe('wireAppendedLayer', () => {
  it('returns plain defaults for the first layer of a model', () => {
    for (const type of LAYER_TYPE_IDS) {
      expect(wireAppendedLayer(type, [], LAYER_TYPES))
        .toEqual(DEFAULT_LAYER_PARAMS[type] || {})
    }
  })

  it('never hands back a reference into the defaults table', () => {
    const params = wireAppendedLayer('conv2d', [], LAYER_TYPES)
    expect(params).not.toBe(DEFAULT_LAYER_PARAMS.conv2d)
    params.out_channels = 999
    expect(DEFAULT_LAYER_PARAMS.conv2d.out_channels).not.toBe(999)
  })

  it('wires the input of each type that has one to the incoming width', () => {
    const cases = [
      ['conv2d', 'in_channels'],
      ['linear', 'input_dim'],
      ['lstm', 'input_size'],
      ['gru', 'input_size'],
      ['transformer', 'd_model'],
      ['attention', 'd_model'],
      ['batchnorm', 'num_features'],
      ['layernorm', 'normalized_shape'],
    ]
    const stack = [layer('conv2d', { out_channels: 512 })]
    for (const [type, field] of cases) {
      expect(wireAppendedLayer(type, stack, LAYER_TYPES)[field]).toBe(512)
    }
  })

  it('leaves the output width alone - that is the modeller\'s decision', () => {
    const stack = [layer('conv2d', { out_channels: 32 })]
    const params = wireAppendedLayer('conv2d', stack, LAYER_TYPES)
    expect(params.in_channels).toBe(32)
    expect(params.out_channels).toBe(DEFAULT_LAYER_PARAMS.conv2d.out_channels)
  })

  it('leaves types with no input dimension untouched', () => {
    const stack = [layer('conv2d', { out_channels: 64 })]
    for (const type of ['relu', 'softmax', 'dropout', 'maxpool2d', 'avgpool2d']) {
      expect(wireAppendedLayer(type, stack, LAYER_TYPES))
        .toEqual(DEFAULT_LAYER_PARAMS[type] || {})
    }
  })

  it('snaps to the nearest legal option when the width is not selectable', () => {
    // Conv2D can emit 3 channels; LayerNorm's smallest option is 16. The result
    // still has to be a value the select can actually show.
    const stack = [layer('conv2d', { out_channels: 3 })]
    const params = wireAppendedLayer('layernorm', stack, LAYER_TYPES)
    const options = LAYER_TYPES.layernorm.fields
      .find(f => f.key === 'normalized_shape').options
    expect(options).toContain(params.normalized_shape)
    expect(params.normalized_shape).toBe(16)
  })

  it('keeps an exact match rather than the numerically nearest one', () => {
    const stack = [layer('conv2d', { out_channels: 256 })]
    expect(wireAppendedLayer('batchnorm', stack, LAYER_TYPES).num_features).toBe(256)
  })
})

describe('building a model by clicking through the palette', () => {
  // The regression this module exists for: every layer type appended to every
  // other one used to be able to produce a mismatch on the second click.
  it('produces no dimension mismatch for any pair of layer types', () => {
    const failures = []
    for (const first of LAYER_TYPE_IDS) {
      for (const second of LAYER_TYPE_IDS) {
        const model = append(append([], first), second)
        const issues = validateModelDimensions(model)
        if (issues.size) failures.push(`${first} -> ${second}: ${[...issues.keys()]}`)
      }
    }
    expect(failures).toEqual([])
  })

  it('stays valid down a long mixed stack', () => {
    const order = [
      'conv2d', 'batchnorm', 'relu', 'maxpool2d', 'conv2d', 'batchnorm', 'relu',
      'avgpool2d', 'linear', 'dropout', 'layernorm', 'attention', 'transformer',
      'lstm', 'gru', 'linear', 'softmax',
    ]
    const model = order.reduce(append, [])
    expect(model).toHaveLength(order.length)
    expect([...validateModelDimensions(model).keys()]).toEqual([])
  })
})
