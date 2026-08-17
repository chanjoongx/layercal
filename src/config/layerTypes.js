/**
 * Layer definitions, parameter/FLOPs formulas, and memory estimation.
 *
 * Parameter counts follow the standard PyTorch implementations exactly.
 * FLOPs are forward-pass estimates and depend on input shape, so the
 * assumed shapes live in FLOPS_ASSUMPTIONS below and are surfaced in the UI
 * rather than left implicit.
 */

/** Input shapes assumed by the FLOPs estimates (shown in the UI). */
export const FLOPS_ASSUMPTIONS = {
  imageSize: 224,   // Conv2D / pooling: 224×224 feature map, 'same' padding
  seqLen: 512,      // Transformer / Attention: 512 tokens
  rnnSeqLen: 128,   // LSTM / GRU: 128 timesteps
  batchSize: 32,    // Normalization layers
  poolChannels: 64, // Pooling layers, when channels are unknown
};

const { imageSize, seqLen, rnnSeqLen, batchSize, poolChannels } = FLOPS_ASSUMPTIONS;

/**
 * Default parameters per layer type. This is the single source of truth, shared by the
 * canvas, the RAG validator and localStorage restore. Callers must spread these
 * rather than mutate them.
 */
export const DEFAULT_LAYER_PARAMS = {
  embedding: { vocab_size: 10000, embedding_dim: 128 },
  linear: { input_dim: 512, output_dim: 256, use_bias: true },
  conv2d: { in_channels: 3, out_channels: 64, kernel_size: 3, use_bias: true },
  lstm: { input_size: 128, hidden_size: 256, num_layers: 1, bidirectional: false },
  transformer: { d_model: 512, num_heads: 8, d_ff: 2048, dropout: 0.1 },
  batchnorm: { num_features: 128 },
  dropout: { rate: 0.1 },
  maxpool2d: { kernel_size: 2 },
  avgpool2d: { kernel_size: 2 },
  layernorm: { normalized_shape: 512 },
  gru: { input_size: 128, hidden_size: 256, num_layers: 1, bidirectional: false },
  attention: { d_model: 512, num_heads: 8 },
  relu: {},
  softmax: {},
};

/**
 * Every layer type id, for validating data that arrives from outside the app
 * (restored localStorage, LLM output). Kept in sync with getLayerTypes by a test.
 */
export const LAYER_TYPE_IDS = Object.keys(DEFAULT_LAYER_PARAMS);

export const getLayerTypes = (t, isDarkMode) => ({
  embedding: {
    name: t.embedding.name,
    icon: '📚',
    color: isDarkMode ? 'bg-purple-900/30 border-purple-700' : 'bg-purple-100 border-purple-300',
    defaultParams: DEFAULT_LAYER_PARAMS.embedding,
    fields: [
      { key: 'vocab_size', label: t.vocabSize, type: 'number', min: 1, step: 1 },
      { key: 'embedding_dim', label: t.embeddingDim, type: 'select', options: [64, 128, 256, 512, 768, 1024] }
    ],
    calculate: (params) => params.vocab_size * params.embedding_dim,
    // Embedding lookup is a table lookup, so 0 FLOPs
    calculateFLOPs: () => 0,
    description: t.embedding.description
  },
  linear: {
    name: t.linear.name,
    icon: '🔗',
    color: isDarkMode ? 'bg-blue-900/30 border-blue-700' : 'bg-blue-100 border-blue-300',
    defaultParams: DEFAULT_LAYER_PARAMS.linear,
    fields: [
      { key: 'input_dim', label: t.inputDim, type: 'select', options: [64, 128, 256, 512, 768, 1024, 2048] },
      { key: 'output_dim', label: t.outputDim, type: 'select', options: [64, 128, 256, 512, 768, 1024, 2048] },
      { key: 'use_bias', label: t.useBias, type: 'checkbox' }
    ],
    calculate: (params) => params.input_dim * params.output_dim + (params.use_bias ? params.output_dim : 0),
    // FLOPs = 2 * in * out (counting multiply-add as 2)
    calculateFLOPs: (params) => 2 * params.input_dim * params.output_dim,
    description: t.linear.description
  },
  conv2d: {
    name: t.conv2d.name,
    icon: '🖼️',
    color: isDarkMode ? 'bg-green-900/30 border-green-700' : 'bg-green-100 border-green-300',
    defaultParams: DEFAULT_LAYER_PARAMS.conv2d,
    fields: [
      { key: 'in_channels', label: t.inChannels, type: 'select', options: [1, 3, 16, 32, 64, 128, 256] },
      { key: 'out_channels', label: t.outChannels, type: 'select', options: [16, 32, 64, 128, 256, 512] },
      { key: 'kernel_size', label: t.kernelSize, type: 'select', options: [1, 3, 5, 7] },
      { key: 'use_bias', label: t.useBias, type: 'checkbox' }
    ],
    calculate: (params) => params.in_channels * params.out_channels * params.kernel_size * params.kernel_size + (params.use_bias ? params.out_channels : 0),
    // FLOPs = 2 * Cin * Cout * K^2 * Hout * Wout, 'same' padding
    calculateFLOPs: (params, inputSize = imageSize) => {
      const outputSize = inputSize; // 'same' padding
      return 2 * params.in_channels * params.out_channels * params.kernel_size * params.kernel_size * outputSize * outputSize;
    },
    description: t.conv2d.description
  },
  lstm: {
    name: t.lstm.name,
    icon: '🔄',
    color: isDarkMode ? 'bg-orange-900/30 border-orange-700' : 'bg-orange-100 border-orange-300',
    defaultParams: DEFAULT_LAYER_PARAMS.lstm,
    fields: [
      { key: 'input_size', label: t.inputSize, type: 'select', options: [64, 128, 256, 512, 768] },
      { key: 'hidden_size', label: t.hiddenSize, type: 'select', options: [128, 256, 512, 768, 1024] },
      { key: 'num_layers', label: t.numLayers, type: 'select', options: [1, 2, 3, 4] },
      { key: 'bidirectional', label: t.bidirectional, type: 'checkbox' }
    ],
    calculate: (params) => {
      const direction = params.bidirectional ? 2 : 1;
      let total = 0;
      for (let layer = 0; layer < params.num_layers; layer++) {
        const input_dim = layer === 0 ? params.input_size : params.hidden_size * direction;
        const params_per_direction = 4 * (input_dim * params.hidden_size + params.hidden_size * params.hidden_size + params.hidden_size * 2);
        total += params_per_direction * direction;
      }
      return total;
    },
    // LSTM FLOPs per timestep: 4 gates * (input*hidden + hidden*hidden) * 2
    calculateFLOPs: (params, seqLength = rnnSeqLen) => {
      const direction = params.bidirectional ? 2 : 1;
      let total = 0;
      for (let layer = 0; layer < params.num_layers; layer++) {
        const input_dim = layer === 0 ? params.input_size : params.hidden_size * direction;
        const flops_per_step = 4 * 2 * (input_dim * params.hidden_size + params.hidden_size * params.hidden_size);
        total += flops_per_step * seqLength * direction;
      }
      return total;
    },
    description: t.lstm.description
  },
  transformer: {
    name: t.transformer.name,
    icon: '⚡',
    color: isDarkMode ? 'bg-pink-900/30 border-pink-700' : 'bg-pink-100 border-pink-300',
    defaultParams: DEFAULT_LAYER_PARAMS.transformer,
    fields: [
      { key: 'd_model', label: t.modelDim, type: 'select', options: [256, 512, 768, 1024] },
      { key: 'num_heads', label: t.numHeads, type: 'select', options: [4, 8, 12, 16] },
      { key: 'd_ff', label: t.ffnDim, type: 'select', options: [1024, 2048, 3072, 4096] }
    ],
    calculate: (params) => {
      const mha = 4 * (params.d_model * params.d_model + params.d_model);
      const ffn = params.d_model * params.d_ff + params.d_ff + params.d_ff * params.d_model + params.d_model;
      const ln = 2 * (params.d_model * 2);
      return mha + ffn + ln;
    },
    // Transformer FLOPs: Attention + FFN
    calculateFLOPs: (params, seqLength = seqLen) => {
      // Attention: Q,K,V projections + attention scores + output projection
      const qkvProj = 3 * 2 * seqLength * params.d_model * params.d_model;
      const attnScores = 2 * seqLength * seqLength * params.d_model;
      const attnOutput = 2 * seqLength * params.d_model * params.d_model;
      // FFN: two linear layers (d→d_ff, d_ff→d)
      const ffn = seqLength * (2 * params.d_model * params.d_ff + 2 * params.d_ff * params.d_model);
      return qkvProj + attnScores + attnOutput + ffn;
    },
    description: t.transformer.description
  },
  batchnorm: {
    name: t.batchnorm.name,
    icon: '📊',
    color: isDarkMode ? 'bg-yellow-900/30 border-yellow-700' : 'bg-yellow-100 border-yellow-300',
    defaultParams: DEFAULT_LAYER_PARAMS.batchnorm,
    fields: [
      { key: 'num_features', label: t.numFeatures, type: 'select', options: [16, 32, 64, 128, 256, 512, 768, 1024] }
    ],
    calculate: (params) => params.num_features * 2,
    // BatchNorm: element-wise normalize + scale + shift
    calculateFLOPs: (params, batch = batchSize, spatialSize = 1) => params.num_features * batch * spatialSize * 4,
    description: t.batchnorm.description
  },
  dropout: {
    name: t.dropout.name,
    icon: '💧',
    color: isDarkMode ? 'bg-gray-800 border-gray-600' : 'bg-gray-100 border-gray-300',
    defaultParams: DEFAULT_LAYER_PARAMS.dropout,
    fields: [
      { key: 'rate', label: t.dropoutRate, type: 'number', step: 0.1, min: 0, max: 1 }
    ],
    calculate: () => 0,
    calculateFLOPs: () => 0,
    description: t.dropout.description
  },
  maxpool2d: {
    name: t.maxpool2d.name,
    icon: '⬇️',
    color: isDarkMode ? 'bg-teal-900/30 border-teal-700' : 'bg-teal-100 border-teal-300',
    defaultParams: DEFAULT_LAYER_PARAMS.maxpool2d,
    fields: [
      { key: 'kernel_size', label: t.kernelSize, type: 'select', options: [2, 3, 4] }
    ],
    calculate: () => 0,
    // MaxPool: comparison operations
    calculateFLOPs: (params, channels = poolChannels, inputSize = imageSize) => {
      const outputSize = Math.floor(inputSize / params.kernel_size);
      return channels * outputSize * outputSize * params.kernel_size * params.kernel_size;
    },
    description: t.maxpool2d.description
  },
  avgpool2d: {
    name: t.avgpool2d.name,
    icon: '📉',
    color: isDarkMode ? 'bg-cyan-900/30 border-cyan-700' : 'bg-cyan-100 border-cyan-300',
    defaultParams: DEFAULT_LAYER_PARAMS.avgpool2d,
    fields: [
      { key: 'kernel_size', label: t.kernelSize, type: 'select', options: [2, 3, 4] }
    ],
    calculate: () => 0,
    calculateFLOPs: (params, channels = poolChannels, inputSize = imageSize) => {
      const outputSize = Math.floor(inputSize / params.kernel_size);
      return channels * outputSize * outputSize * params.kernel_size * params.kernel_size;
    },
    description: t.avgpool2d.description
  },
  layernorm: {
    name: t.layernorm.name,
    icon: '🎯',
    color: isDarkMode ? 'bg-indigo-900/30 border-indigo-700' : 'bg-indigo-100 border-indigo-300',
    defaultParams: DEFAULT_LAYER_PARAMS.layernorm,
    fields: [
      { key: 'normalized_shape', label: t.numFeatures, type: 'select', options: [128, 256, 512, 768, 1024] }
    ],
    calculate: (params) => params.normalized_shape * 2,
    calculateFLOPs: (params, batch = batchSize, seqLength = seqLen) => params.normalized_shape * batch * seqLength * 5,
    description: t.layernorm.description
  },
  gru: {
    name: t.gru.name,
    icon: '🔁',
    color: isDarkMode ? 'bg-red-900/30 border-red-700' : 'bg-red-100 border-red-300',
    defaultParams: DEFAULT_LAYER_PARAMS.gru,
    fields: [
      { key: 'input_size', label: t.inputSize, type: 'select', options: [64, 128, 256, 512, 768] },
      { key: 'hidden_size', label: t.hiddenSize, type: 'select', options: [128, 256, 512, 768, 1024] },
      { key: 'num_layers', label: t.numLayers, type: 'select', options: [1, 2, 3, 4] },
      { key: 'bidirectional', label: t.bidirectional, type: 'checkbox' }
    ],
    calculate: (params) => {
      const direction = params.bidirectional ? 2 : 1;
      let total = 0;
      for (let layer = 0; layer < params.num_layers; layer++) {
        const input_dim = layer === 0 ? params.input_size : params.hidden_size * direction;
        const params_per_direction = 3 * (input_dim * params.hidden_size + params.hidden_size * params.hidden_size + params.hidden_size * 2);
        total += params_per_direction * direction;
      }
      return total;
    },
    // GRU: 3 gates instead of 4
    calculateFLOPs: (params, seqLength = rnnSeqLen) => {
      const direction = params.bidirectional ? 2 : 1;
      let total = 0;
      for (let layer = 0; layer < params.num_layers; layer++) {
        const input_dim = layer === 0 ? params.input_size : params.hidden_size * direction;
        const flops_per_step = 3 * 2 * (input_dim * params.hidden_size + params.hidden_size * params.hidden_size);
        total += flops_per_step * seqLength * direction;
      }
      return total;
    },
    description: t.gru.description
  },
  attention: {
    name: t.attention.name,
    icon: '👁️',
    color: isDarkMode ? 'bg-fuchsia-900/30 border-fuchsia-700' : 'bg-fuchsia-100 border-fuchsia-300',
    defaultParams: DEFAULT_LAYER_PARAMS.attention,
    fields: [
      { key: 'd_model', label: t.modelDim, type: 'select', options: [256, 512, 768, 1024] },
      { key: 'num_heads', label: t.numHeads, type: 'select', options: [4, 8, 12, 16] }
    ],
    calculate: (params) => {
      return 4 * (params.d_model * params.d_model + params.d_model);
    },
    calculateFLOPs: (params, seqLength = seqLen) => {
      const qkvProj = 3 * 2 * seqLength * params.d_model * params.d_model;
      const attnScores = 2 * seqLength * seqLength * params.d_model;
      const attnOutput = 2 * seqLength * params.d_model * params.d_model;
      return qkvProj + attnScores + attnOutput;
    },
    description: t.attention.description
  },
  relu: {
    name: t.relu.name,
    icon: '🔥',
    color: isDarkMode ? 'bg-lime-900/30 border-lime-700' : 'bg-lime-100 border-lime-300',
    defaultParams: DEFAULT_LAYER_PARAMS.relu,
    fields: [],
    calculate: () => 0,
    // ReLU: simple comparison, conventionally excluded from FLOPs counts
    calculateFLOPs: () => 0,
    description: t.relu.description
  },
  softmax: {
    name: t.softmax.name,
    icon: '🎲',
    color: isDarkMode ? 'bg-amber-900/30 border-amber-700' : 'bg-amber-100 border-amber-300',
    defaultParams: DEFAULT_LAYER_PARAMS.softmax,
    fields: [],
    calculate: () => 0,
    calculateFLOPs: () => 0,
    description: t.softmax.description
  }
});

/**
 * Format large numbers with units (K, M, G, T)
 */
export const formatNumber = (num) => {
  const n = Number(num);
  if (!Number.isFinite(n)) return '0';
  const sign = n < 0 ? '-' : '';
  const abs = Math.abs(n);
  if (abs >= 1e12) return sign + (abs / 1e12).toFixed(2) + 'T';
  if (abs >= 1e9) return sign + (abs / 1e9).toFixed(2) + 'G';
  if (abs >= 1e6) return sign + (abs / 1e6).toFixed(2) + 'M';
  if (abs >= 1e3) return sign + (abs / 1e3).toFixed(2) + 'K';
  return String(n);
};

/** Bytes held per parameter at a given precision. */
export const BYTES_PER_PARAM = {
  fp32: 4,
  fp16: 2,
  bf16: 2,
  int8: 1,
};

/**
 * Bytes per parameter for Adam training, independent of the weight dtype.
 *
 *   pure fp32:       weights 4 + grads 4 + m 4 + v 4              = 16
 *   mixed precision: weights 2 + grads 2 + fp32 master 4 + m 4 + v 4 = 16
 *
 * Halving the weight dtype does not halve training memory, because the fp32 master
 * copy the optimizer needs cancels the saving out. Reporting `bytes × 4` per
 * precision (as earlier versions did) understated fp16/bf16 training by 2×.
 */
export const TRAINING_BYTES_PER_PARAM = 16;

/**
 * Calculate memory footprint of the model weights.
 * Activation memory is excluded, since it depends on batch size and input shape.
 *
 * @param {number} totalParams - Total number of parameters
 * @param {string} mode - 'inference' or 'training'
 * @param {string} precision - 'fp32', 'fp16', 'bf16', 'int8'
 */
export const calculateMemory = (totalParams, mode = 'inference', precision = 'fp32') => {
  const params = Number(totalParams);
  if (!Number.isFinite(params) || params <= 0) return 0;

  if (mode === 'training') {
    return params * TRAINING_BYTES_PER_PARAM;
  }

  const bytes = BYTES_PER_PARAM[precision] ?? BYTES_PER_PARAM.fp32;
  return params * bytes;
};

/**
 * Format bytes to a human readable string (SI units, base 1000).
 */
export const formatBytes = (bytes) => {
  const b = Number(bytes);
  if (!Number.isFinite(b) || b < 0) return '0 B';
  if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
  if (b >= 1e6) return (b / 1e6).toFixed(2) + ' MB';
  if (b >= 1e3) return (b / 1e3).toFixed(2) + ' KB';
  return b + ' B';
};
