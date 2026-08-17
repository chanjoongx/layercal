import { describe, it, expect } from 'vitest'
import {
  generatePyTorchCode,
  generateTensorFlowCode,
  generateJAXCode,
} from '@/utils/codeGenerator'

const conv = (over = {}) => ({
  type: 'conv2d',
  params: { in_channels: 3, out_channels: 64, kernel_size: 3, use_bias: true, ...over },
})
const linear = (over = {}) => ({
  type: 'linear',
  params: { input_dim: 64, output_dim: 32, use_bias: true, ...over },
})
const relu = () => ({ type: 'relu', params: {} })
const pool = () => ({ type: 'maxpool2d', params: { kernel_size: 2 } })
const bn = (n = 64) => ({ type: 'batchnorm', params: { num_features: n } })
const embedding = () => ({ type: 'embedding', params: { vocab_size: 10000, embedding_dim: 128 } })
const lstm = () => ({
  type: 'lstm',
  params: { input_size: 128, hidden_size: 256, num_layers: 1, bidirectional: false },
})
const transformer = () => ({
  type: 'transformer',
  params: { d_model: 512, num_heads: 8, d_ff: 2048 },
})


// ─────────────────────────────────────────────
//  Empty model
// ─────────────────────────────────────────────

describe('empty model', () => {
  it.each([generatePyTorchCode, generateTensorFlowCode, generateJAXCode])(
    'returns a placeholder comment',
    (generate) => { expect(generate([])).toBe('# Add layers to generate code') }
  )
})


// ─────────────────────────────────────────────
//  Spatial → vector transition
// ─────────────────────────────────────────────

describe('Conv2D → Linear transition (PyTorch)', () => {
  it('inserts a global average pool and flatten', () => {
    const code = generatePyTorchCode([conv(), relu(), linear()])
    expect(code).toContain('nn.AdaptiveAvgPool2d(1)')
    expect(code).toContain('nn.Flatten()')
    // the pooling must come before the Linear call in forward()
    const forward = code.slice(code.indexOf('def forward'))
    expect(forward.indexOf('self.gap(x)')).toBeLessThan(forward.indexOf('self.fc(x)'))
  })

  it('keeps spatial context through pooling, batchnorm and activations', () => {
    const code = generatePyTorchCode([conv(), bn(), relu(), pool(), linear()])
    expect(code).toContain('nn.AdaptiveAvgPool2d(1)')
  })

  it('does NOT insert pooling for a pure MLP', () => {
    const code = generatePyTorchCode([linear({ input_dim: 256 }), relu(), linear()])
    expect(code).not.toContain('AdaptiveAvgPool2d')
    expect(code).not.toContain('nn.Flatten()')
  })

  it('does NOT insert pooling after a sequence model', () => {
    const code = generatePyTorchCode([embedding(), lstm(), linear({ input_dim: 256 })])
    expect(code).not.toContain('AdaptiveAvgPool2d')
  })

  it('inserts pooling only once per transition', () => {
    const code = generatePyTorchCode([conv(), linear(), linear({ input_dim: 32 })])
    expect(code.match(/AdaptiveAvgPool2d/g)).toHaveLength(1)
  })
})

describe('Conv2D → Dense transition (TensorFlow)', () => {
  it('inserts GlobalAveragePooling2D in the Sequential path', () => {
    const code = generateTensorFlowCode([conv(), relu(), linear()])
    expect(code).toContain('layers.GlobalAveragePooling2D()')
  })

  it('inserts GlobalAveragePooling2D in the Functional path', () => {
    // The trailing attention layer is what routes this to the Functional API;
    // the conv→linear step is still what triggers the pooling.
    const code = generateTensorFlowCode([
      conv(), relu(), linear(), { type: 'attention', params: { d_model: 256, num_heads: 4 } },
    ])
    expect(code).toContain('keras.Model(inputs, x')
    expect(code).toContain('layers.GlobalAveragePooling2D()(x)')
  })

  it('does not pool when a sequence layer already ended spatial context', () => {
    const code = generateTensorFlowCode([conv(), transformer(), linear({ input_dim: 512 })])
    expect(code).not.toContain('GlobalAveragePooling2D')
  })
})

describe('Conv → Dense transition (JAX)', () => {
  it('inserts a mean over the spatial axes', () => {
    const code = generateJAXCode([conv(), relu(), linear()])
    expect(code).toContain('jnp.mean(x, axis=(1, 2))')
  })
})


// ─────────────────────────────────────────────
//  Runnable output
// ─────────────────────────────────────────────

describe('TensorFlow Sequential', () => {
  it('declares an Input layer so model.summary() works', () => {
    const code = generateTensorFlowCode([linear({ input_dim: 256 })])
    expect(code).toContain('keras.Input(shape=input_shape)')
    expect(code).toContain('def create_generatedmodel(input_shape=(256,))')
  })

  it('routes transformer models to the Functional API', () => {
    const code = generateTensorFlowCode([transformer()])
    expect(code).toContain('Functional API')
    expect(code).not.toContain('keras.Sequential')
  })
})

describe('inferred example input', () => {
  it('PyTorch: NCHW tensor for a conv model', () => {
    expect(generatePyTorchCode([conv({ in_channels: 1 })]))
      .toContain('torch.randn(1, 1, 224, 224)')
  })

  it('PyTorch: integer token ids for an embedding model', () => {
    expect(generatePyTorchCode([embedding()]))
      .toContain('torch.randint(0, 10000, (1, 32))')
  })

  it('PyTorch: sequence tensor for an LSTM model', () => {
    expect(generatePyTorchCode([lstm()]))
      .toContain('torch.randn(1, 32, 128)')
  })

  it('PyTorch: skips activations when inferring the shape', () => {
    expect(generatePyTorchCode([relu(), conv({ in_channels: 3 })]))
      .toContain('torch.randn(1, 3, 224, 224)')
  })

  it('TensorFlow: channels-last shape for a conv model', () => {
    expect(generateTensorFlowCode([conv({ in_channels: 3 })]))
      .toContain('input_shape=(224, 224, 3)')
  })

  it('JAX: channels-last array for a conv model', () => {
    expect(generateJAXCode([conv({ in_channels: 3 })]))
      .toContain('jnp.ones((1, 224, 224, 3))')
  })

  it('JAX: integer dtype for an embedding model', () => {
    expect(generateJAXCode([embedding()]))
      .toContain('jnp.ones((1, 32), dtype=jnp.int32)')
  })
})


// ─────────────────────────────────────────────
//  Grouping and naming
// ─────────────────────────────────────────────

describe('grouping', () => {
  it('collapses repeated transformer blocks into a TransformerEncoder', () => {
    const code = generatePyTorchCode([transformer(), transformer(), transformer()])
    expect(code).toContain('nn.TransformerEncoder(')
    expect(code).toContain('num_layers=3')
  })

  it('uses semantic variable names', () => {
    const code = generatePyTorchCode([conv(), bn(), relu(), linear()])
    expect(code).toContain('self.conv =')
    expect(code).toContain('self.bn =')
    expect(code).toContain('self.fc =')
  })

  it('selects BatchNorm2d after a conv and BatchNorm1d otherwise', () => {
    expect(generatePyTorchCode([conv(), bn()])).toContain('nn.BatchNorm2d(64)')
    expect(generatePyTorchCode([linear(), bn(32)])).toContain('nn.BatchNorm1d(32)')
  })

  it('JAX RNN output is a single array, not a tuple', () => {
    const code = generateJAXCode([lstm()])
    expect(code).toContain('x = nn.RNN(nn.OptimizedLSTMCell(features=256))(x)')
    expect(code).not.toContain('x, _ = nn.RNN')
  })
})
