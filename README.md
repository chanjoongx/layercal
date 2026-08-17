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
  <img src="https://img.shields.io/badge/Vitest-3-6e9f18?style=flat-square&logo=vitest&logoColor=white" alt="Vitest" />
  <img src="https://img.shields.io/badge/Gemini_API-1a73e8?style=flat-square&logo=googlegemini&logoColor=white" alt="Gemini" />
  <img src="https://img.shields.io/badge/OpenAI_API-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI" />
  <img src="https://img.shields.io/badge/Claude_API-d4a27f?style=flat-square&logo=anthropic&logoColor=white" alt="Claude" />
  <img src="https://img.shields.io/badge/Deploy-Cloudflare_Pages-f38020?style=flat-square&logo=cloudflarepages&logoColor=white" alt="Cloudflare Pages" />
</p>

<br/>

<p align="center">
  <img src="https://github.com/user-attachments/assets/092c8620-b337-4d01-813c-3a24d47945c8" alt="LayerCal Screenshot" width="900" />
</p>

---

## What it does

Build a neural network by dragging layers onto a canvas and watch the parameter count, FLOPs and
memory footprint update as you go. When the model looks right, export it as PyTorch, TensorFlow or
JAX code that actually runs.

Everything happens in the browser. There is no backend, no account, and nothing is uploaded.

| | |
|---|---|
| **14 layer types** | Embedding, Linear, Conv2D, LSTM, GRU, Transformer, Attention, BatchNorm, LayerNorm, Dropout, MaxPool2D, AvgPool2D, ReLU, Softmax |
| **Live computation** | Parameter counts, forward-pass FLOPs, and memory across FP32, FP16, BF16 and INT8 for both inference and Adam training |
| **Code generation** | PyTorch `nn.Module`, TensorFlow Sequential and Functional API, JAX/Flax `nn.compact` |
| **AI architecture advisor** | Describe what you need in plain English and get a validated layer stack back |
| **Dimension checking** | Layers whose input does not match the previous layer's output are flagged on the canvas |
| **Persistent canvas** | Your model is saved locally and restored on the next visit |
| **Eight languages** | EN, KO, JA, ZH, ES, FR, DE, PT |
| **Dark mode** | Follows the system setting, with a manual override |

## AI architecture advisor

Describe your requirements in natural language. The advisor returns a layer stack, shows you its
estimated parameter count, and applies it to the canvas once you approve.

1. The query is matched against a knowledge base of 12 reference architectures such as LeNet-5, VGG, MobileNet, BERT and GPT
2. Matching references go into the prompt. If nothing matches, none are injected, because an unrelated example steers the model in the wrong direction
3. The model returns a JSON layer stack, using native JSON mode where the provider supports it
4. A deterministic pass validates every layer, snaps parameters to legal values, and repairs cross-layer dimension mismatches
5. You see the proposal and its parameter count before anything touches the canvas

### Providers

Bring your own API key. Nothing is proxied through a server.

| Provider | Default model | Cost |
|----------|---------------|------|
| Google Gemini | `gemini-3.5-flash-lite` | Free tier available |
| OpenAI | `gpt-5.6-luna` | Requires billing |
| Anthropic Claude | `claude-haiku-4-5` | Requires billing |

Your key is stored in your browser and sent only to the provider you pick. You can switch that
storage off entirely under **Advanced**, and clear a saved key at any time.

### Surviving model retirements

Models get retired, and a tool with no backend cannot patch itself when that happens. Two things
guard against it:

- **Model override.** Every provider accepts a custom model ID under **Advanced**, saved per
  provider. A retired default is reported as `MODEL_NOT_FOUND` with instructions, so you can point
  the app at a current model yourself.
- **Per-family request shaping.** The OpenAI client detects the model family and sends
  `max_completion_tokens` for GPT-5.x and o-series models, or `max_tokens` plus `temperature` for
  the GPT-4 generation. A regression test pins the defaults so a retired ID cannot creep back in.

## Getting started

```bash
git clone https://github.com/chanjoongx/layercal.git
cd layercal
npm install
npm run dev
```

Node 20 or newer.

To enable Google Analytics, copy `env.example` to `.env` and set `VITE_GA_ID`. Without it, no
analytics code loads at all.

## Testing

```bash
npm test            # run once
npm run test:watch  # watch mode
```

200 tests across six suites:

| Suite | What it covers |
|-------|----------------|
| `layerTypes` | Parameter, FLOPs and memory formulas for all 14 types, including stacked bidirectional RNNs, bias toggles, and guards against non-finite input |
| `modelValidation` | Cross-layer dimension checking, including passthrough layers, bidirectional RNN output width, and half-typed values |
| `ragPipeline` | JSON extraction from every LLM output shape seen in practice, parameter snapping, cross-layer repair, retrieval ranking |
| `codeGenerator` | Spatial to vector transitions, framework routing, inferred input shapes, layer grouping and naming |
| `llmClient` | Model ID regressions and per-family OpenAI request shaping |
| `render` | Server-renders every component in all eight locales and checks translation key parity |

The render suite uses `react-dom/server` rather than jsdom, so it needs no extra dependencies. It
catches undefined identifiers and bad hook usage, but effects do not run, so it is a smoke test
rather than a substitute for clicking through the app.

## Project structure

```
public/
├── _headers                    Cloudflare Pages security headers and caching
├── llms.txt                    Site summary for AI crawlers
├── robots.txt                  Crawler policy, including AI answer engines
└── site.webmanifest            PWA manifest
src/
├── components/
│   ├── LayerCal.jsx            Main app: UI, state, drag and drop, persistence
│   ├── AIAdvisor.jsx           Advisor dialog: provider, key, model, query
│   └── ui/                     Card, alert and accessible modal primitives
├── config/
│   ├── layerTypes.js           Layer definitions and calculation formulas
│   ├── translations.js         Strings for eight languages
│   └── architectureKB.js       12 reference architectures for retrieval
└── utils/
    ├── llmClient.js            Unified client for OpenAI, Gemini and Claude
    ├── ragPipeline.js          Retrieval, prompt building, parsing, validation
    ├── codeGenerator.js        PyTorch, TensorFlow and JAX code generation
    ├── imageExport.js          PNG export, lazy-loads html2canvas
    └── localStorage.js         Storage that degrades safely when blocked
```

## Calculation reference

### Parameters

| Layer | Formula | Notes |
|-------|---------|-------|
| Embedding | `V x E` | Vocabulary size by embedding dimension |
| Linear | `I x O + O` | With bias |
| Conv2D | `Cin x Cout x K² + Cout` | With bias, independent of resolution |
| LSTM | `4(IH + H² + 2H) x L x dir` | Layer 2 onward takes the previous hidden size as input |
| GRU | `3(IH + H² + 2H) x L x dir` | Three gates, so 75% of an equivalent LSTM |
| Transformer | `12d² + 13d` | Per block, when `d_ff = 4d` |
| Attention | `4(d² + d)` | Q, K, V and output projections |
| BatchNorm, LayerNorm | `2F` | One scale and one shift per feature |

### Memory

| Mode | Bytes per parameter |
|------|---------------------|
| Inference | 4 (FP32), 2 (FP16 or BF16), 1 (INT8) |
| Training with Adam | **16, whatever the weight precision** |

Training memory does not scale with precision, which is the estimate people usually get wrong:

```
pure FP32:        weights 4 + grads 4 + m 4 + v 4                  = 16 B/param
mixed precision:  weights 2 + grads 2 + FP32 master 4 + m 4 + v 4  = 16 B/param
```

The FP32 master copy the optimiser keeps cancels out the saving from narrower weights. Activation
memory is excluded, since it depends on batch size and input shape.

### FLOPs

Forward pass only, counting a multiply-accumulate as two operations. The estimates assume 224x224
images with `same` padding, 512-token sequences for attention, 128 timesteps for RNNs, and batch
size 32 for normalisation. Those assumptions are printed under the FLOPs figure in the app, because
a FLOPs number without its input shape is meaningless.

## Code generation

The generator is not template substitution. It handles:

- **Grouping.** Three consecutive Transformer blocks become `TransformerEncoder(num_layers=3)`
- **Context.** BatchNorm resolves to `BatchNorm1d` or `BatchNorm2d` based on the preceding layer
- **Shape transitions.** A convolution stack feeding a dense layer gets a global average pool and
  flatten inserted automatically, so a 4D feature map never reaches `nn.Linear`
- **Naming.** `self.conv`, `self.fc1`, `self.embed` rather than `self.layer_0`
- **A runnable entry point.** `__main__` builds a correctly shaped example tensor from the first
  layer, counts parameters, and runs one forward pass

The generator reproduces the model you built, so a stack whose dimensions do not line up still
exports code that fails at runtime. That is what the canvas warnings are for: fix the flagged
layers first, and the generated script runs end to end.

## Deployment

Built for Cloudflare Pages. `npm run build` emits `dist/`, and `public/_headers` ships with it.

`_headers` sets HSTS, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`,
`Permissions-Policy`, immutable caching for hashed assets, and a Content Security Policy. The
load-bearing part of that policy is `connect-src`: users paste their own API keys into this page,
so an injected script must not be able to send them anywhere except the three provider endpoints.

A CSP can only be validated in a real browser. Check a preview deployment before promoting it. If
something is blocked, rename the header to `Content-Security-Policy-Report-Only`, read the console
violations, then rename it back.

## Privacy

- No backend, no account, no server-side storage
- Calculations, code generation and PNG export run entirely in the browser
- API keys stay in `localStorage` and go only to the provider you select. Storage can be turned off
- Google Analytics loads only when `VITE_GA_ID` is set at build time

## Tech stack

React 18, Vite 5, Tailwind CSS 3.4, shadcn/ui primitives, Vitest, html2canvas, Cloudflare Pages.

## Documentation

- [Technical guide](docs/LayerCal-Guide.pdf)

## License

[MIT](LICENSE)

---

<p align="center">
  Built by <a href="https://github.com/chanjoongx">@chanjoongx</a>
</p>
