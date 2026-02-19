import { describe, it, expect, beforeAll } from 'vitest'
import { getLayerTypes, formatNumber, calculateMemory, formatBytes } from '@/config/layerTypes'

/*
 * Mock translation object — getLayerTypes needs it for UI strings,
 * but calculations don't depend on any of these values.
 */
const stubT = new Proxy({}, {
  get(_, key) {
    if (typeof key === 'symbol') return undefined
    return new Proxy({ name: key, description: key }, {
      get(_, sub) { return `${key}.${sub}` }
    })
  }
})

let layers

beforeAll(() => {
  layers = getLayerTypes(stubT, false)
})


// ─────────────────────────────────────────────
//  Parameter Count Tests
// ─────────────────────────────────────────────

describe('Parameter calculations', () => {
  it('Embedding: V × E', () => {
    expect(layers.embedding.calculate({ vocab_size: 10000, embedding_dim: 128 }))
      .toBe(1_280_000)

    expect(layers.embedding.calculate({ vocab_size: 50000, embedding_dim: 512 }))
      .toBe(25_600_000)
  })

  it('Linear: I×O + bias', () => {
    expect(layers.linear.calculate({ input_dim: 512, output_dim: 256, use_bias: true }))
      .toBe(131_328) // 512*256 + 256

    expect(layers.linear.calculate({ input_dim: 512, output_dim: 256, use_bias: false }))
      .toBe(131_072)
  })

  it('Conv2D: Cin×Cout×K² + bias', () => {
    expect(layers.conv2d.calculate({ in_channels: 3, out_channels: 64, kernel_size: 3, use_bias: true }))
      .toBe(1_792) // 3*64*9 + 64

    // 1×1 convolution, no bias
    expect(layers.conv2d.calculate({ in_channels: 256, out_channels: 256, kernel_size: 1, use_bias: false }))
      .toBe(65_536)
  })

  it('LSTM: single layer, unidirectional', () => {
    // 4 gates × (I×H + H×H + 2H) × direction
    const result = layers.lstm.calculate({
      input_size: 128, hidden_size: 256, num_layers: 1, bidirectional: false
    })
    // 4 * (128*256 + 256*256 + 512) = 4 * 98816 = 395264
    expect(result).toBe(395_264)
  })

  it('LSTM: stacked bidirectional (layer 1 takes 2H input)', () => {
    const result = layers.lstm.calculate({
      input_size: 128, hidden_size: 256, num_layers: 2, bidirectional: true
    })
    // Layer 0: 4*(128*256+256²+512)*2 = 790528
    // Layer 1: input=512, 4*(512*256+256²+512)*2 = 1576960
    expect(result).toBe(2_367_488)
  })

  it('GRU: 3 gates instead of 4', () => {
    const result = layers.gru.calculate({
      input_size: 128, hidden_size: 256, num_layers: 1, bidirectional: false
    })
    // 3 * (128*256 + 256² + 512) = 3 * 98816
    expect(result).toBe(296_448)
  })

  it('GRU params should be 75% of LSTM params (same config)', () => {
    const config = { input_size: 128, hidden_size: 256, num_layers: 1, bidirectional: false }
    const lstmParams = layers.lstm.calculate(config)
    const gruParams = layers.gru.calculate(config)
    expect(gruParams / lstmParams).toBeCloseTo(0.75, 5)
  })

  it('Transformer: MHA + FFN + LayerNorm per block', () => {
    const result = layers.transformer.calculate({ d_model: 512, num_heads: 8, d_ff: 2048 })
    // MHA: 4*(512²+512) = 1050624
    // FFN: 512*2048 + 2048 + 2048*512 + 512 = 2099712
    // LN: 2*(512*2) = 2048
    expect(result).toBe(3_152_384)
  })

  it('Attention: Q,K,V,O projections only (no FFN)', () => {
    const result = layers.attention.calculate({ d_model: 512, num_heads: 8 })
    expect(result).toBe(1_050_624) // 4*(512²+512)
  })

  it('Transformer params > Attention params (same d_model)', () => {
    const tfParams = layers.transformer.calculate({ d_model: 512, num_heads: 8, d_ff: 2048 })
    const attnParams = layers.attention.calculate({ d_model: 512, num_heads: 8 })
    expect(tfParams).toBeGreaterThan(attnParams)
  })

  it('BatchNorm: gamma + beta = 2F', () => {
    expect(layers.batchnorm.calculate({ num_features: 128 })).toBe(256)
    expect(layers.batchnorm.calculate({ num_features: 1024 })).toBe(2048)
  })

  it('LayerNorm: gamma + beta = 2N', () => {
    expect(layers.layernorm.calculate({ normalized_shape: 512 })).toBe(1024)
  })

  // These layers have no learnable parameters
  it.each(['dropout', 'maxpool2d', 'avgpool2d', 'relu', 'softmax'])(
    '%s: zero trainable parameters',
    (layerName) => {
      const calc = layers[layerName].calculate
      expect(calc(layers[layerName].defaultParams)).toBe(0)
    }
  )
})


// ─────────────────────────────────────────────
//  FLOPs Tests
// ─────────────────────────────────────────────

describe('FLOPs calculations', () => {
  it('Embedding: zero FLOPs (table lookup)', () => {
    expect(layers.embedding.calculateFLOPs()).toBe(0)
  })

  it('Linear: 2×I×O multiply-accumulate', () => {
    expect(layers.linear.calculateFLOPs({ input_dim: 512, output_dim: 256 }))
      .toBe(262_144)
  })

  it('Conv2D: 2×Cin×Cout×K²×Hout×Wout', () => {
    const flops = layers.conv2d.calculateFLOPs(
      { in_channels: 3, out_channels: 64, kernel_size: 3 }, 224
    )
    // 2 * 3 * 64 * 9 * 224 * 224
    expect(flops).toBe(173_408_256)
  })

  it('LSTM FLOPs scale linearly with sequence length', () => {
    const config = { input_size: 128, hidden_size: 256, num_layers: 1, bidirectional: false }
    const flops64 = layers.lstm.calculateFLOPs(config, 64)
    const flops128 = layers.lstm.calculateFLOPs(config, 128)
    expect(flops128 / flops64).toBe(2)
  })

  it('Bidirectional LSTM: 2x FLOPs of unidirectional (single layer)', () => {
    const base = { input_size: 128, hidden_size: 256, num_layers: 1 }
    const uni = layers.lstm.calculateFLOPs({ ...base, bidirectional: false }, 128)
    const bi = layers.lstm.calculateFLOPs({ ...base, bidirectional: true }, 128)
    expect(bi / uni).toBe(2)
  })

  it('Transformer FLOPs: attention + FFN', () => {
    const flops = layers.transformer.calculateFLOPs({ d_model: 512, num_heads: 8, d_ff: 2048 }, 512)
    // qkv=805306368 + scores=268435456 + output=268435456 + ffn=2147483648
    expect(flops).toBe(3_489_660_928)
  })

  it('Attention FLOPs < Transformer FLOPs (no FFN)', () => {
    const attnFlops = layers.attention.calculateFLOPs({ d_model: 512, num_heads: 8 }, 512)
    const tfFlops = layers.transformer.calculateFLOPs({ d_model: 512, num_heads: 8, d_ff: 2048 }, 512)
    expect(attnFlops).toBeLessThan(tfFlops)
    expect(attnFlops).toBe(1_342_177_280)
  })

  it('MaxPool2D: comparisons over each window', () => {
    const flops = layers.maxpool2d.calculateFLOPs({ kernel_size: 2 }, 64, 224)
    // floor(224/2)=112, 64 * 112² * 4
    expect(flops).toBe(3_211_264)
  })

  it('AvgPool2D FLOPs equals MaxPool2D FLOPs (same config)', () => {
    const mp = layers.maxpool2d.calculateFLOPs({ kernel_size: 2 }, 64, 224)
    const ap = layers.avgpool2d.calculateFLOPs({ kernel_size: 2 }, 64, 224)
    expect(mp).toBe(ap)
  })

  it.each(['dropout', 'relu', 'softmax'])(
    '%s: zero FLOPs',
    (name) => { expect(layers[name].calculateFLOPs()).toBe(0) }
  )
})


// ─────────────────────────────────────────────
//  Memory Estimation
// ─────────────────────────────────────────────

describe('calculateMemory', () => {
  const oneM = 1_000_000

  it('inference FP32: params × 4 bytes', () => {
    expect(calculateMemory(oneM, 'inference', 'fp32')).toBe(4_000_000)
  })

  it('inference FP16: params × 2 bytes', () => {
    expect(calculateMemory(oneM, 'inference', 'fp16')).toBe(2_000_000)
  })

  it('inference INT8: params × 1 byte', () => {
    expect(calculateMemory(oneM, 'inference', 'int8')).toBe(1_000_000)
  })

  it('training mode: 4× inference (Adam overhead)', () => {
    const inf = calculateMemory(oneM, 'inference', 'fp32')
    const train = calculateMemory(oneM, 'training', 'fp32')
    expect(train).toBe(inf * 4)
  })

  it('bf16 treated same as fp16', () => {
    expect(calculateMemory(oneM, 'inference', 'bf16'))
      .toBe(calculateMemory(oneM, 'inference', 'fp16'))
  })

  it('unknown precision falls back to FP32', () => {
    expect(calculateMemory(oneM, 'inference', 'tf32')).toBe(4_000_000)
  })
})


// ─────────────────────────────────────────────
//  Formatting Utilities
// ─────────────────────────────────────────────

describe('formatNumber', () => {
  it('raw number below 1K', () => { expect(formatNumber(999)).toBe('999') })
  it('thousands',  () => { expect(formatNumber(1_500)).toBe('1.50K') })
  it('millions',   () => { expect(formatNumber(3_152_384)).toBe('3.15M') })
  it('billions',   () => { expect(formatNumber(1_500_000_000)).toBe('1.50G') })
  it('trillions',  () => { expect(formatNumber(2_000_000_000_000)).toBe('2.00T') })
})

describe('formatBytes', () => {
  it('bytes',     () => { expect(formatBytes(512)).toBe('512 B') })
  it('kilobytes', () => { expect(formatBytes(4_000)).toBe('4.00 KB') })
  it('megabytes', () => { expect(formatBytes(4_000_000)).toBe('4.00 MB') })
  it('gigabytes', () => { expect(formatBytes(4_000_000_000)).toBe('4.00 GB') })
})
