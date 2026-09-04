import { describe, it, expect } from 'vitest'
import {
  outputShape, propagateShapes, seedShape, sizeFor, extentFor,
  annotationExtent, ANNOTATION_TYPES,
} from '@/viz/tensorShape'
import { FLOPS_ASSUMPTIONS, DEFAULT_LAYER_PARAMS, LAYER_TYPE_IDS } from '@/config/layerTypes'

const layer = (type, params = {}) => ({
  type,
  params: { ...DEFAULT_LAYER_PARAMS[type], ...params },
})

const { imageSize, seqLen, rnnSeqLen } = FLOPS_ASSUMPTIONS

describe('seedShape', () => {
  it('starts a convolution stack at the assumed image resolution', () => {
    const s = seedShape(layer('conv2d', { in_channels: 3 }))
    expect(s.kind).toBe('spatial')
    expect(s.dims).toEqual([3, imageSize, imageSize])
  })

  it('starts a transformer at the assumed sequence length', () => {
    const s = seedShape(layer('transformer', { d_model: 768 }))
    expect(s.dims).toEqual([seqLen, 768])
  })

  it('starts an RNN at the assumed timestep count', () => {
    expect(seedShape(layer('lstm', { input_size: 256 })).dims).toEqual([rnnSeqLen, 256])
  })

  it('never throws for a type it has no rule for', () => {
    expect(() => seedShape({ type: 'unknown', params: {} })).not.toThrow()
    expect(() => seedShape(null)).not.toThrow()
  })
})

describe('outputShape', () => {
  it('embedding emits the embedding dimension over the sequence', () => {
    const s = outputShape(layer('embedding', { embedding_dim: 256 }), null)
    expect(s.kind).toBe('sequence')
    expect(s.channels).toBe(256)
  })

  it('linear emits its output width as a vector', () => {
    const s = outputShape(layer('linear', { output_dim: 1024 }), null)
    expect(s).toMatchObject({ kind: 'vector', channels: 1024 })
    expect(s.dims).toEqual([1024])
  })

  it('conv2d keeps the spatial size (same padding) and swaps the channel count', () => {
    const incoming = seedShape(layer('conv2d'))
    const s = outputShape(layer('conv2d', { out_channels: 128 }), incoming)
    expect(s.dims).toEqual([128, imageSize, imageSize])
  })

  it('pooling floors the spatial size by the kernel and keeps the channels', () => {
    const incoming = { kind: 'spatial', dims: [64, 225, 225], label: '', channels: 64 }
    const s = outputShape(layer('maxpool2d', { kernel_size: 2 }), incoming)
    expect(s.dims).toEqual([64, 112, 112])
  })

  it('pooling never produces a zero-sized feature map', () => {
    const incoming = { kind: 'spatial', dims: [64, 2, 2], label: '', channels: 64 }
    const s = outputShape(layer('maxpool2d', { kernel_size: 4 }), incoming)
    expect(s.dims[1]).toBe(1)
    expect(s.dims[2]).toBe(1)
  })

  it('a bidirectional RNN doubles its output width', () => {
    const one = outputShape(layer('lstm', { hidden_size: 512, bidirectional: false }), null)
    const two = outputShape(layer('lstm', { hidden_size: 512, bidirectional: true }), null)
    expect(one.channels).toBe(512)
    expect(two.channels).toBe(1024)
  })

  it('a GRU follows the same rule as an LSTM', () => {
    const s = outputShape(layer('gru', { hidden_size: 256, bidirectional: true }), null)
    expect(s.channels).toBe(512)
  })

  it('transformer and attention pass d_model through unchanged', () => {
    for (const type of ['transformer', 'attention']) {
      const s = outputShape(layer(type, { d_model: 768 }), null)
      expect(s.channels).toBe(768)
      expect(s.kind).toBe('sequence')
    }
  })

  it('activations and dropout leave the tensor exactly as it arrived', () => {
    const incoming = { kind: 'spatial', dims: [64, 56, 56], label: '64 x 56 x 56', channels: 64 }
    for (const type of ['relu', 'softmax', 'dropout']) {
      expect(outputShape(layer(type), incoming)).toMatchObject({
        kind: 'spatial',
        dims: [64, 56, 56],
        channels: 64,
      })
    }
  })

  it('normalisation rescales without reshaping', () => {
    const incoming = { kind: 'sequence', dims: [512, 768], label: '', channels: 768 }
    for (const type of ['batchnorm', 'layernorm']) {
      expect(outputShape(layer(type), incoming).dims).toEqual([512, 768])
    }
  })

  it('degrades to a safe shape rather than emitting NaN for a half-typed value', () => {
    const s = outputShape({ type: 'linear', params: { output_dim: '' } }, null)
    expect(Number.isFinite(s.channels)).toBe(true)
    expect(s.label).not.toMatch(/NaN/)
  })

  it('an unknown layer type is a passthrough, not a crash', () => {
    const incoming = { kind: 'vector', dims: [256], label: '256', channels: 256 }
    expect(outputShape({ type: 'wormhole', params: {} }, incoming)).toMatchObject({ channels: 256 })
  })

  it('labels are human-readable and grouped', () => {
    const s = outputShape(layer('embedding', { embedding_dim: 1024 }), null)
    expect(s.label).toBe('512 x 1,024')
  })
})

describe('propagateShapes', () => {
  it('threads the shape through a convolutional stack', () => {
    const shapes = propagateShapes([
      layer('conv2d', { in_channels: 3, out_channels: 64 }),
      layer('relu'),
      layer('maxpool2d', { kernel_size: 2 }),
      layer('conv2d', { in_channels: 64, out_channels: 128 }),
    ])
    expect(shapes).toHaveLength(4)
    expect(shapes[0].dims).toEqual([64, 224, 224])
    expect(shapes[2].dims).toEqual([64, 112, 112])
    // The second convolution inherits the pooled resolution, not the seed one.
    expect(shapes[3].dims).toEqual([128, 112, 112])
  })

  it('returns an empty list for an empty model', () => {
    expect(propagateShapes([])).toEqual([])
    expect(propagateShapes(null)).toEqual([])
  })

  it('handles every layer type without throwing', () => {
    expect(() => propagateShapes(LAYER_TYPE_IDS.map(id => layer(id)))).not.toThrow()
  })
})

describe('sizeFor', () => {
  it('is clamped at both ends', () => {
    expect(sizeFor(0)).toBe(0.34)
    expect(sizeFor(-5)).toBe(0.34)
    expect(sizeFor(Number.NaN)).toBe(0.34)
    expect(sizeFor(1e12)).toBeLessThanOrEqual(2.6)
  })

  it('is monotonically increasing', () => {
    let previous = -1
    for (const n of [1, 16, 64, 256, 1024, 50000]) {
      const v = sizeFor(n)
      expect(v).toBeGreaterThanOrEqual(previous)
      previous = v
    }
  })

  it('compresses a 780x ratio into something both ends can be seen at', () => {
    // A 50,000-word vocabulary next to a 64-unit dense layer is the case that
    // makes linear scaling unusable.
    const ratio = sizeFor(50000) / sizeFor(64)
    expect(ratio).toBeLessThan(3)
    expect(ratio).toBeGreaterThan(1)
  })
})

describe('extentFor', () => {
  it('gives a spatial tensor width and height from H and W', () => {
    const e = extentFor({ kind: 'spatial', dims: [64, 224, 224], label: '', channels: 64 })
    expect(e.w).toBeCloseTo(e.h, 6)
    expect(e.d).toBeGreaterThan(0)
  })

  it('gives a vector a thin plate', () => {
    const e = extentFor({ kind: 'vector', dims: [512], label: '', channels: 512 })
    expect(e.d).toBeLessThan(e.w)
  })

  it('never returns a non-finite extent', () => {
    const e = extentFor({ kind: 'sequence', dims: [Number.NaN, Number.NaN], label: '', channels: null })
    for (const v of [e.w, e.h, e.d]) expect(Number.isFinite(v)).toBe(true)
  })

  it('is defensive about a missing shape', () => {
    expect(() => extentFor(null)).not.toThrow()
  })
})

describe('annotations', () => {
  it('treats activations, dropout and normalisation as thin plates', () => {
    expect([...ANNOTATION_TYPES].sort()).toEqual(
      ['batchnorm', 'dropout', 'layernorm', 'relu', 'softmax']
    )
  })

  it('inherits the upstream cross-section at a reduced scale', () => {
    const e = annotationExtent({ w: 2, h: 1, d: 3 })
    expect(e.w).toBeCloseTo(1.72, 6)
    expect(e.h).toBeCloseTo(0.86, 6)
    expect(e.d).toBe(0.16)
  })

  it('has a defined size with nothing upstream', () => {
    expect(annotationExtent(null).d).toBe(0.16)
  })
})
