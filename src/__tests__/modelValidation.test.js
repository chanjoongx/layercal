import { describe, it, expect } from 'vitest'
import { validateModelDimensions } from '@/utils/modelValidation'
import { DEFAULT_LAYER_PARAMS } from '@/config/layerTypes'

const L = (type, over = {}) => ({ type, params: { ...DEFAULT_LAYER_PARAMS[type], ...over } })

describe('validateModelDimensions', () => {
  it('accepts an empty or single-layer model', () => {
    expect(validateModelDimensions([]).size).toBe(0)
    expect(validateModelDimensions([L('conv2d')]).size).toBe(0)
  })

  it('never flags the first layer, which has no predecessor', () => {
    expect(validateModelDimensions([L('linear', { input_dim: 2048 })]).size).toBe(0)
  })

  it('catches the mismatch stock defaults produce', () => {
    // Conv2D defaults to 64 out_channels, BatchNorm defaults to 128 features.
    const issues = validateModelDimensions([L('conv2d'), L('batchnorm')])
    expect(issues.get(1)).toEqual({ field: 'num_features', expected: 64 })
  })

  it('stays quiet once the dimensions line up', () => {
    const issues = validateModelDimensions([L('conv2d'), L('batchnorm', { num_features: 64 })])
    expect(issues.size).toBe(0)
  })

  it('sees through passthrough layers', () => {
    const issues = validateModelDimensions([
      L('conv2d'),          // emits 64
      L('relu'),
      L('dropout'),
      L('maxpool2d'),
      L('batchnorm'),       // defaults to 128
    ])
    expect(issues.get(4)).toEqual({ field: 'num_features', expected: 64 })
  })

  it('compares Linear against the channel count after a conv stack', () => {
    // Codegen inserts a global average pool there, so (N,C,H,W) becomes (N,C).
    const issues = validateModelDimensions([L('conv2d'), L('linear')])
    expect(issues.get(1)).toEqual({ field: 'input_dim', expected: 64 })
  })

  it('tracks Embedding into an RNN', () => {
    const ok = validateModelDimensions([
      L('embedding', { embedding_dim: 256 }),
      L('lstm', { input_size: 256 }),
    ])
    expect(ok.size).toBe(0)

    const bad = validateModelDimensions([
      L('embedding', { embedding_dim: 256 }),
      L('lstm', { input_size: 128 }),
    ])
    expect(bad.get(1)).toEqual({ field: 'input_size', expected: 256 })
  })

  it('doubles the RNN output when bidirectional', () => {
    const issues = validateModelDimensions([
      L('embedding', { embedding_dim: 128 }),
      L('lstm', { input_size: 128, hidden_size: 256, bidirectional: true }),
      L('linear', { input_dim: 256 }),
    ])
    expect(issues.get(2)).toEqual({ field: 'input_dim', expected: 512 })
  })

  it('checks transformer d_model against the previous layer', () => {
    const issues = validateModelDimensions([
      L('embedding', { embedding_dim: 768 }),
      L('transformer', { d_model: 512 }),
    ])
    expect(issues.get(1)).toEqual({ field: 'd_model', expected: 768 })
  })

  it('reports every offending layer, not just the first', () => {
    const issues = validateModelDimensions([
      L('conv2d'),                                  // emits 64
      L('batchnorm'),                               // wants 128 -> flagged
      L('linear', { input_dim: 512 }),              // wants 512, gets 64 -> flagged
    ])
    expect([...issues.keys()]).toEqual([1, 2])
  })

  it('flags nothing for a correctly wired CNN', () => {
    const issues = validateModelDimensions([
      L('conv2d', { in_channels: 3, out_channels: 64 }),
      L('batchnorm', { num_features: 64 }),
      L('relu'),
      L('conv2d', { in_channels: 64, out_channels: 128 }),
      L('batchnorm', { num_features: 128 }),
      L('relu'),
      L('linear', { input_dim: 128, output_dim: 64 }),
    ])
    expect(issues.size).toBe(0)
  })

  it('ignores half-typed values rather than shouting about them', () => {
    const issues = validateModelDimensions([
      L('conv2d', { out_channels: '' }),
      L('batchnorm', { num_features: '' }),
    ])
    expect(issues.size).toBe(0)
  })
})
