<p align="center">
  <img src="public/calculator-icon.svg" alt="LayerCal" width="80" />
</p>

<h1 align="center">LayerCal</h1>

<p align="center">
  <a href="https://github.com/chanjoongx/layercal/actions/workflows/test.yml">
    <img src="https://github.com/chanjoongx/layercal/actions/workflows/test.yml/badge.svg" alt="CI" />
  </a>
</p>

<p align="center">
  Browser-based deep learning model parameter calculator.<br/>
  Drag layers, get instant parameter counts, FLOPs, memory estimates, and framework code.
</p>

<p align="center">
  <a href="https://layercal.com"><img src="https://img.shields.io/badge/Live_Demo-layercal.com-7c3aed?style=for-the-badge" alt="Live Demo" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/chanjoongx/layercal?style=flat-square" alt="License" />
  <img src="https://img.shields.io/badge/React-18-61dafb?style=flat-square&logo=react&logoColor=white" alt="React" />
  <img src="https://img.shields.io/badge/Vite-5-646cff?style=flat-square&logo=vite&logoColor=white" alt="Vite" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-3.4-38bdf8?style=flat-square&logo=tailwindcss&logoColor=white" alt="Tailwind" />
  <img src="https://img.shields.io/badge/Deploy-Cloudflare_Pages-f38020?style=flat-square&logo=cloudflarepages&logoColor=white" alt="Cloudflare Pages" />
</p>

<br/>

<p align="center">
  <img src="https://github.com/user-attachments/assets/092c8620-b337-4d01-813c-3a24d47945c8" alt="LayerCal Screenshot" width="900" />
</p>

---

## Features

Build a neural network by dragging layers onto a canvas. Everything runs client-side — no backend, no signup.

| | |
|---|---|
| **14 Layer Types** | Embedding · Linear · Conv2D · LSTM · GRU · Transformer · Attention · BatchNorm · LayerNorm · Dropout · MaxPool2D · AvgPool2D · ReLU · Softmax |
| **Real-time Computation** | Parameter counts, FLOPs (forward pass), memory estimation with FP32 / FP16 / INT8 precision across inference and training modes |
| **Code Generation** | PyTorch `nn.Module` · TensorFlow Sequential / Functional API · JAX/Flax `nn.compact` — with smart layer grouping and context-aware naming |
| **Multi-language** | EN · KO · JA · ZH · ES · FR · DE · PT |
| **Dark Mode** | System preference detection + manual toggle |

## Getting Started

```bash
git clone https://github.com/chanjoongx/layercal.git
cd layercal
npm install
npm run dev
```

## Testing

```bash
npm test            # run all tests
npm run test:watch  # watch mode during development
```

Tests cover parameter counting, FLOPs estimation, and memory calculation for all 14 layer types, including edge cases like bidirectional LSTM stacking and bias toggle behavior.

## Project Structure

```
src/
├── components/
│   ├── LayerCal.jsx            # Main app — UI, state, drag-drop
│   └── ui/                     # shadcn/ui primitives
├── config/
│   ├── layerTypes.js           # Layer definitions + calculation formulas
│   └── translations.js         # i18n for 8 languages
└── utils/
    ├── codeGenerator.js        # PyTorch / TensorFlow / JAX codegen
    ├── imageExport.js          # PNG export via html2canvas
    └── localStorage.js         # Safe storage + system detection
```

## Calculation Reference

### Parameter Formulas

| Layer | Formula | Notes |
|-------|---------|-------|
| Embedding | `V × E` | V = vocab, E = dim |
| Linear | `I × O + O` | with bias |
| Conv2D | `Cin × Cout × K² + Cout` | with bias |
| LSTM | `4(IH + H² + 2H) × L × dir` | per-layer input adjustment |
| GRU | `3(IH + H² + 2H) × L × dir` | 3 gates |
| Transformer | `12d² + 13d` | per block, d_ff = 4d |
| Attention | `4(d² + d)` | Q, K, V, O projections |
| BatchNorm | `2F` | γ + β |

### Memory Estimation

| Mode | Formula |
|------|---------|
| Inference | `params × bytes_per_param` |
| Training (Adam) | `params × bytes_per_param × 4` |

FP32 = 4 bytes, FP16 = 2 bytes, INT8 = 1 byte

### Code Generation

Not simple template substitution — the generator handles:

- **Grouping**: 3× consecutive Transformer → `TransformerEncoder(num_layers=3)`
- **Context**: Auto-selects `BatchNorm1d` / `BatchNorm2d` from preceding layer
- **Naming**: `self.embed`, `self.fc1`, `self.conv` instead of `self.layer_0`

## Tech Stack

React 18 · Vite 5 · Tailwind CSS 3.4 · shadcn/ui · html2canvas · Cloudflare Pages

## Documentation

- [Technical Guide](docs/LayerCal-Guide.pdf) — Detailed formulas, code generation examples, and architecture overview

## License

[MIT](LICENSE)

---

<p align="center">
  Built by <a href="https://github.com/chanjoongx">@chanjoongx</a>
</p>
