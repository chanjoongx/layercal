/**
 * Architecture Knowledge Base for AI Advisor
 *
 * Each entry represents a well-known model architecture,
 * translated into LayerCal's layer format. Used as retrieval
 * context in the RAG pipeline — when a user describes what
 * they need, we search this DB by tags and surface the most
 * relevant references for the LLM to draw from.
 *
 * IMPORTANT: Every param value must be a valid option in
 * layerTypes.js. If a real architecture uses a value we
 * don't support (e.g. embedding_dim=384), we snap to the
 * nearest available option.
 */

const ARCHITECTURE_KB = [

  // ── CNN: Image Classification ─────────────────────────

  {
    id: 'lenet-5',
    name: 'LeNet-5',
    category: 'image_classification',
    tags: ['cnn', 'image', 'classification', 'mnist', 'digit', 'small', 'simple', 'beginner'],
    description: 'Classic CNN for handwritten digit recognition (LeCun, 1998)',
    approxParams: '~62K',
    layers: [
      { type: 'conv2d', params: { in_channels: 1, out_channels: 16, kernel_size: 5, use_bias: true } },
      { type: 'relu', params: {} },
      { type: 'avgpool2d', params: { kernel_size: 2 } },
      { type: 'conv2d', params: { in_channels: 16, out_channels: 32, kernel_size: 5, use_bias: true } },
      { type: 'relu', params: {} },
      { type: 'avgpool2d', params: { kernel_size: 2 } },
      { type: 'linear', params: { input_dim: 512, output_dim: 256, use_bias: true } },
      { type: 'relu', params: {} },
      { type: 'linear', params: { input_dim: 256, output_dim: 128, use_bias: true } },
      { type: 'softmax', params: {} },
    ]
  },

  {
    id: 'simple-cnn',
    name: 'Simple CNN Classifier',
    category: 'image_classification',
    tags: ['cnn', 'image', 'classification', 'cifar', 'simple', 'beginner', 'lightweight'],
    description: 'Compact CNN for small image classification tasks (CIFAR-10 style)',
    approxParams: '~1.2M',
    layers: [
      { type: 'conv2d', params: { in_channels: 3, out_channels: 64, kernel_size: 3, use_bias: true } },
      { type: 'batchnorm', params: { num_features: 64 } },
      { type: 'relu', params: {} },
      { type: 'maxpool2d', params: { kernel_size: 2 } },
      { type: 'conv2d', params: { in_channels: 64, out_channels: 128, kernel_size: 3, use_bias: true } },
      { type: 'batchnorm', params: { num_features: 128 } },
      { type: 'relu', params: {} },
      { type: 'maxpool2d', params: { kernel_size: 2 } },
      { type: 'conv2d', params: { in_channels: 128, out_channels: 256, kernel_size: 3, use_bias: false } },
      { type: 'relu', params: {} },
      { type: 'dropout', params: { rate: 0.3 } },
      { type: 'linear', params: { input_dim: 256, output_dim: 128, use_bias: true } },
      { type: 'linear', params: { input_dim: 128, output_dim: 64, use_bias: true } },
    ]
  },

  {
    id: 'vgg-style',
    name: 'VGG-style Network',
    category: 'image_classification',
    tags: ['cnn', 'image', 'classification', 'vgg', 'deep', 'large', 'imagenet'],
    description: 'Deep CNN with repeated 3x3 convolutions, inspired by VGG (Simonyan & Zisserman, 2014)',
    approxParams: '~14M',
    layers: [
      { type: 'conv2d', params: { in_channels: 3, out_channels: 64, kernel_size: 3, use_bias: true } },
      { type: 'relu', params: {} },
      { type: 'conv2d', params: { in_channels: 64, out_channels: 64, kernel_size: 3, use_bias: true } },
      { type: 'relu', params: {} },
      { type: 'maxpool2d', params: { kernel_size: 2 } },
      { type: 'conv2d', params: { in_channels: 64, out_channels: 128, kernel_size: 3, use_bias: true } },
      { type: 'relu', params: {} },
      { type: 'conv2d', params: { in_channels: 128, out_channels: 128, kernel_size: 3, use_bias: true } },
      { type: 'relu', params: {} },
      { type: 'maxpool2d', params: { kernel_size: 2 } },
      { type: 'conv2d', params: { in_channels: 128, out_channels: 256, kernel_size: 3, use_bias: true } },
      { type: 'relu', params: {} },
      { type: 'maxpool2d', params: { kernel_size: 2 } },
      { type: 'linear', params: { input_dim: 256, output_dim: 1024, use_bias: true } },
      { type: 'dropout', params: { rate: 0.5 } },
      { type: 'linear', params: { input_dim: 1024, output_dim: 512, use_bias: true } },
      { type: 'dropout', params: { rate: 0.5 } },
      { type: 'linear', params: { input_dim: 512, output_dim: 256, use_bias: true } },
    ]
  },

  {
    id: 'mobilenet-lite',
    name: 'MobileNet-style (Lite)',
    category: 'image_classification',
    tags: ['cnn', 'image', 'classification', 'mobile', 'edge', 'lightweight', 'efficient', 'quantization', 'deploy'],
    description: 'Lightweight CNN optimized for mobile/edge deployment, inspired by MobileNet',
    approxParams: '~85K',
    layers: [
      { type: 'conv2d', params: { in_channels: 3, out_channels: 64, kernel_size: 3, use_bias: false } },
      { type: 'batchnorm', params: { num_features: 64 } },
      { type: 'relu', params: {} },
      { type: 'conv2d', params: { in_channels: 64, out_channels: 128, kernel_size: 1, use_bias: false } },
      { type: 'batchnorm', params: { num_features: 128 } },
      { type: 'relu', params: {} },
      { type: 'conv2d', params: { in_channels: 128, out_channels: 256, kernel_size: 1, use_bias: false } },
      { type: 'batchnorm', params: { num_features: 256 } },
      { type: 'relu', params: {} },
      { type: 'avgpool2d', params: { kernel_size: 4 } },
      { type: 'linear', params: { input_dim: 256, output_dim: 128, use_bias: true } },
      { type: 'linear', params: { input_dim: 128, output_dim: 64, use_bias: true } },
    ]
  },

  // ── NLP: Text Models ──────────────────────────────────

  {
    id: 'lstm-text-clf',
    name: 'LSTM Text Classifier',
    category: 'text_classification',
    tags: ['nlp', 'text', 'classification', 'lstm', 'rnn', 'sentiment', 'sequence'],
    description: 'Bidirectional LSTM for text classification (sentiment analysis, etc.)',
    approxParams: '~3.5M',
    layers: [
      { type: 'embedding', params: { vocab_size: 30000, embedding_dim: 128 } },
      { type: 'lstm', params: { input_size: 128, hidden_size: 256, num_layers: 2, bidirectional: true } },
      { type: 'dropout', params: { rate: 0.3 } },
      { type: 'linear', params: { input_dim: 512, output_dim: 128, use_bias: true } },
      { type: 'relu', params: {} },
      { type: 'dropout', params: { rate: 0.2 } },
      { type: 'linear', params: { input_dim: 128, output_dim: 64, use_bias: true } },
      { type: 'softmax', params: {} },
    ]
  },

  {
    id: 'gru-seq-model',
    name: 'GRU Sequence Model',
    category: 'sequence_modeling',
    tags: ['nlp', 'text', 'gru', 'rnn', 'sequence', 'language', 'generation', 'lightweight'],
    description: 'GRU-based sequence model — lighter alternative to LSTM',
    approxParams: '~1.8M',
    layers: [
      { type: 'embedding', params: { vocab_size: 10000, embedding_dim: 256 } },
      { type: 'gru', params: { input_size: 256, hidden_size: 512, num_layers: 2, bidirectional: false } },
      { type: 'dropout', params: { rate: 0.2 } },
      { type: 'linear', params: { input_dim: 512, output_dim: 256, use_bias: true } },
      { type: 'relu', params: {} },
      { type: 'linear', params: { input_dim: 256, output_dim: 128, use_bias: true } },
    ]
  },

  // ── Transformer / Attention ───────────────────────────

  {
    id: 'bert-encoder',
    name: 'BERT-style Encoder',
    category: 'text_encoding',
    tags: ['transformer', 'bert', 'nlp', 'text', 'encoder', 'attention', 'pretrain', 'large'],
    description: 'Transformer encoder stack similar to BERT-base (Devlin et al., 2018)',
    approxParams: '~38M',
    layers: [
      { type: 'embedding', params: { vocab_size: 30000, embedding_dim: 768 } },
      { type: 'transformer', params: { d_model: 768, num_heads: 12, d_ff: 3072 } },
      { type: 'transformer', params: { d_model: 768, num_heads: 12, d_ff: 3072 } },
      { type: 'transformer', params: { d_model: 768, num_heads: 12, d_ff: 3072 } },
      { type: 'transformer', params: { d_model: 768, num_heads: 12, d_ff: 3072 } },
      { type: 'layernorm', params: { normalized_shape: 768 } },
      { type: 'linear', params: { input_dim: 768, output_dim: 256, use_bias: true } },
    ]
  },

  {
    id: 'small-transformer',
    name: 'Small Transformer Encoder',
    category: 'text_encoding',
    tags: ['transformer', 'nlp', 'text', 'encoder', 'attention', 'small', 'lightweight', 'efficient'],
    description: 'Compact transformer for resource-constrained environments',
    approxParams: '~7M',
    layers: [
      { type: 'embedding', params: { vocab_size: 10000, embedding_dim: 256 } },
      { type: 'transformer', params: { d_model: 256, num_heads: 4, d_ff: 1024 } },
      { type: 'transformer', params: { d_model: 256, num_heads: 4, d_ff: 1024 } },
      { type: 'layernorm', params: { normalized_shape: 256 } },
      { type: 'dropout', params: { rate: 0.1 } },
      { type: 'linear', params: { input_dim: 256, output_dim: 128, use_bias: true } },
    ]
  },

  {
    id: 'gpt-decoder',
    name: 'GPT-style Decoder',
    category: 'text_generation',
    tags: ['transformer', 'gpt', 'nlp', 'text', 'generation', 'decoder', 'causal', 'language', 'large'],
    description: 'Autoregressive transformer decoder for text generation (GPT-style)',
    approxParams: '~50M',
    layers: [
      { type: 'embedding', params: { vocab_size: 50000, embedding_dim: 512 } },
      { type: 'transformer', params: { d_model: 512, num_heads: 8, d_ff: 2048 } },
      { type: 'transformer', params: { d_model: 512, num_heads: 8, d_ff: 2048 } },
      { type: 'transformer', params: { d_model: 512, num_heads: 8, d_ff: 2048 } },
      { type: 'transformer', params: { d_model: 512, num_heads: 8, d_ff: 2048 } },
      { type: 'transformer', params: { d_model: 512, num_heads: 8, d_ff: 2048 } },
      { type: 'transformer', params: { d_model: 512, num_heads: 8, d_ff: 2048 } },
      { type: 'layernorm', params: { normalized_shape: 512 } },
      { type: 'linear', params: { input_dim: 512, output_dim: 2048, use_bias: true } },
    ]
  },

  // ── General purpose ───────────────────────────────────

  {
    id: 'mlp-classifier',
    name: 'MLP Classifier',
    category: 'tabular_classification',
    tags: ['mlp', 'tabular', 'classification', 'simple', 'beginner', 'feedforward', 'dense'],
    description: 'Multi-layer perceptron for tabular/structured data',
    approxParams: '~400K',
    layers: [
      { type: 'linear', params: { input_dim: 256, output_dim: 512, use_bias: true } },
      { type: 'batchnorm', params: { num_features: 512 } },
      { type: 'relu', params: {} },
      { type: 'dropout', params: { rate: 0.3 } },
      { type: 'linear', params: { input_dim: 512, output_dim: 256, use_bias: true } },
      { type: 'relu', params: {} },
      { type: 'dropout', params: { rate: 0.2 } },
      { type: 'linear', params: { input_dim: 256, output_dim: 128, use_bias: true } },
      { type: 'relu', params: {} },
      { type: 'linear', params: { input_dim: 128, output_dim: 64, use_bias: true } },
      { type: 'softmax', params: {} },
    ]
  },

  {
    id: 'autoencoder',
    name: 'Convolutional Autoencoder',
    category: 'unsupervised',
    tags: ['autoencoder', 'unsupervised', 'cnn', 'image', 'reconstruction', 'representation', 'compression'],
    description: 'Conv autoencoder for image reconstruction and feature extraction',
    approxParams: '~200K',
    layers: [
      // Encoder
      { type: 'conv2d', params: { in_channels: 1, out_channels: 32, kernel_size: 3, use_bias: true } },
      { type: 'relu', params: {} },
      { type: 'maxpool2d', params: { kernel_size: 2 } },
      { type: 'conv2d', params: { in_channels: 32, out_channels: 64, kernel_size: 3, use_bias: true } },
      { type: 'relu', params: {} },
      { type: 'maxpool2d', params: { kernel_size: 2 } },
      // Bottleneck
      { type: 'linear', params: { input_dim: 64, output_dim: 128, use_bias: true } },
      // Decoder (simplified with linear layers in LayerCal)
      { type: 'linear', params: { input_dim: 128, output_dim: 256, use_bias: true } },
      { type: 'relu', params: {} },
      { type: 'linear', params: { input_dim: 256, output_dim: 512, use_bias: true } },
    ]
  },

  {
    id: 'self-attention-clf',
    name: 'Self-Attention Classifier',
    category: 'text_classification',
    tags: ['attention', 'nlp', 'text', 'classification', 'self-attention', 'lightweight'],
    description: 'Lightweight attention-based text classifier without full transformer overhead',
    approxParams: '~4M',
    layers: [
      { type: 'embedding', params: { vocab_size: 10000, embedding_dim: 256 } },
      { type: 'attention', params: { d_model: 256, num_heads: 4 } },
      { type: 'layernorm', params: { normalized_shape: 256 } },
      { type: 'attention', params: { d_model: 256, num_heads: 4 } },
      { type: 'layernorm', params: { normalized_shape: 256 } },
      { type: 'dropout', params: { rate: 0.1 } },
      { type: 'linear', params: { input_dim: 256, output_dim: 128, use_bias: true } },
      { type: 'relu', params: {} },
      { type: 'linear', params: { input_dim: 128, output_dim: 64, use_bias: true } },
    ]
  },
];

export default ARCHITECTURE_KB;
