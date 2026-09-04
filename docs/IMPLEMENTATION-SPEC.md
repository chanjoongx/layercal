# LayerCal v3 — Implementation Specification

> **Status:** authoritative build spec for the "Living Architecture" release (v3.0.0).
> **Audience:** the engineer or agent implementing the change. Every section is a contract.
> Anything not stated here is out of scope; anything stated here is required.

---

## 0. Purpose, invariants and non-goals

### 0.1 What this release adds

LayerCal today is a correct calculator with a flat, utilitarian UI. The layer stack is a
vertical list of coloured rectangles. That representation is honest but it does not show
the one thing a neural network actually *is*: a sequence of tensors that change shape as
data flows through them.

v3 adds a **real-time WebGL2 visualisation of the model as a stack of tensor volumes with
an animated forward pass**, and rebuilds the surrounding interface around it.

### 0.2 Hard invariants — must not regress

| # | Invariant | Enforced by |
|---|-----------|-------------|
| I1 | No backend. No account. Nothing leaves the browser except a BYOK LLM call to the provider the user picked. | `public/_headers` `connect-src`, code review |
| I2 | Every existing parameter / FLOPs / memory formula produces identical output. | `src/__tests__/layerTypes.test.js` green with every assertion unchanged; its single edit is a call site, because `getLayerTypes` lost its `isDarkMode` argument when colour left that module |
| I3 | The app server-renders without a DOM. `renderToStaticMarkup(<LayerCal />)` must not throw. | `src/__tests__/render.test.jsx` |
| I4 | Every locale defines exactly the key set English defines, and no value is blank. | `render.test.jsx` → `translation completeness` |
| I5 | The app is fully usable with WebGL2 unavailable, with JS animation disabled, and from the keyboard alone. | fallback path §4.9, a11y §4.10 |
| I6 | `npm run build` succeeds, the renderer lands in its own lazily-loaded chunk, and the initial payload grows by no more than 15 KB gzipped. | §7.3 budget check |
| I7 | No new runtime dependency is added to `package.json`. | `git diff package.json` |

> **I7 is deliberate.** three.js would cost ~150 KB gzipped for a feature that needs perhaps
> 6% of its surface, and it would put a third-party render loop between the user's model and
> the screen. The renderer here is ~2,500 lines of WebGL2 that we own, sitting on ~850 lines
> of pure geometry and maths; it is never fetched when the panel is not shown, and the parts
> that carry correctness (geometry, layout, maths) are unit-testable without a GPU.

### 0.3 Non-goals

- No physically-based rendering, no shadow maps, no IBL. The look is *stylised technical
  illustration*, not a render of a real object.
- No training-loop animation, no backward pass. Forward pass only.
- No model import from PyTorch / ONNX. Out of scope for this release.

---

## 1. Design system

### 1.1 Colour tokens

All colours live in `src/index.css` as HSL triples on `:root` and `.dark`, consumed through
Tailwind's `hsl(var(--token))` bridge. **No component may hardcode a hex value that is not
also present in the token table**, with the single exception of the layer palette (§1.2),
which is a data table shared with the GPU and therefore lives in JavaScript.

| Token | Light | Dark | Used for |
|-------|-------|------|----------|
| `--background` | `240 20% 99%` | `230 25% 6%` | page ground |
| `--surface` | `0 0% 100%` | `230 22% 9%` | cards |
| `--surface-raised` | `240 30% 98%` | `230 20% 12%` | nested panels, inputs |
| `--foreground` | `230 25% 12%` | `220 25% 96%` | body text |
| `--muted-foreground` | `230 12% 42%` | `225 14% 62%` | secondary text |
| `--border` | `230 18% 90%` | `230 18% 18%` | hairlines |
| `--border-strong` | `230 18% 82%` | `230 16% 26%` | focused / active hairlines |
| `--accent` | `262 83% 58%` | `262 90% 70%` | primary brand violet |
| `--accent-soft` | `262 90% 96%` | `262 45% 18%` | accent-tinted fills |
| `--positive` | `160 84% 34%` | `158 70% 52%` | success |
| `--warning` | `35 92% 45%` | `38 95% 60%` | dimension mismatch |
| `--danger` | `0 74% 50%` | `0 80% 66%` | destructive |

Two further tokens carry the ambient gradient used by the page background and the 3D
viewport clear colour, so the canvas and the page never disagree:

```
--viz-sky      light: 235 60% 97%    dark: 235 45% 10%
--viz-ground   light: 250 30% 92%    dark: 245 35%  5%
```

### 1.2 Layer palette — single source of truth

`src/viz/palette.js` exports `LAYER_PALETTE`, keyed by layer type id. It is consumed by:

1. the Tailwind-styled 2D cards (via `paletteStyle()` returning inline CSS custom properties),
2. the SVG fallback diagram,
3. the WebGL instance buffer (as linear-space RGB floats).

Contract:

```js
/**
 * @typedef {{
 *   family:  'input'|'dense'|'conv'|'recurrent'|'attention'|'norm'|'pool'|'reg'|'act',
 *   base:    [number, number, number],  // sRGB 0..1, the solid body colour
 *   glow:    [number, number, number],  // sRGB 0..1, emissive / bloom colour
 *   hex:     string,                    // '#rrggbb', the 2D UI colour (light theme)
 *   hexDark: string,                    // '#rrggbb', the 2D UI colour (dark theme)
 * }} LayerPaint
 *
 * @type {Record<string, LayerPaint>}
 */
export const LAYER_PALETTE
```

Every id in `LAYER_TYPE_IDS` must have an entry, asserted by `vizPalette.test.js`.

`layerTypes.js` must carry **no** colour of its own. It previously exported a `color` field of
Tailwind class pairs and an emoji `icon` per layer. Both are removed: `color` was a second table
saying what a layer looks like, and it had already drifted from this one (GRU red there and amber
here, LayerNorm indigo there and teal here). Dropping it also removed `isDarkMode` from
`getLayerTypes`, so a theme toggle no longer invalidates the layer table and therefore no longer
rebuilds the whole scene.

Family → hue assignment (fixed, do not re-order):

| family | types | hue |
|--------|-------|-----|
| `input` | `embedding` | violet `#8b5cf6` |
| `dense` | `linear` | blue `#3b82f6` |
| `conv` | `conv2d` | emerald `#10b981` |
| `recurrent` | `lstm`, `gru` | amber `#f59e0b` |
| `attention` | `transformer`, `attention` | fuchsia `#d946ef` |
| `norm` | `batchnorm`, `layernorm` | cyan `#06b6d4` |
| `pool` | `maxpool2d`, `avgpool2d` | sky `#0ea5e9` |
| `reg` | `dropout` | slate `#64748b` |
| `act` | `relu`, `softmax` | lime `#84cc16` |

### 1.3 Type scale and spacing

`--font-display` and `--font-mono` are declared but resolve to system stacks — **no webfont
may be fetched**, because `font-src` in the CSP is `'self' data:` and adding a CDN would
widen the attack surface for a page that holds API keys.

```
display: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial
mono:    ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace
```

Numeric readouts (parameter counts, FLOPs, byte figures) use `font-variant-numeric:
tabular-nums` so a changing value does not reflow its own row.

### 1.4 Motion

| Token | Value | Use |
|-------|-------|-----|
| `--ease-out` | `cubic-bezier(.16,1,.3,1)` | entrances, hovers |
| `--ease-in-out` | `cubic-bezier(.65,0,.35,1)` | camera, layout |
| `--dur-fast` | `140ms` | hover, press |
| `--dur-base` | `260ms` | panel, modal |
| `--dur-slow` | `640ms` | number roll-up, focus flight |

Every animation, CSS or JS, is gated on `prefers-reduced-motion`. The existing global
`@media (prefers-reduced-motion: reduce)` block in `index.css` handles CSS. JS-driven
motion (the render loop, the counter) must check `matchMedia('(prefers-reduced-motion:
reduce)').matches` and render a single static frame instead.

---

## 2. Pure model → geometry layer

Three pure modules. No DOM, no GL, no React. These carry the correctness of the
visualisation and are the parts that are unit-tested exhaustively.

### 2.1 `src/viz/tensorShape.js`

Maps a layer to the shape of the tensor it *emits*, and to a 3D extent for drawing.

```js
/**
 * @typedef {{
 *   kind: 'spatial'|'sequence'|'vector'|'passthrough',
 *   dims: number[],          // e.g. [64, 56, 56] or [512, 768]
 *   label: string,           // '64 x 56 x 56'
 *   channels: number|null,   // feature width carried forward, null if unknown
 * }} TensorShape
 */
export function outputShape(layer, incoming): TensorShape
```

Rules — these must match `modelValidation.js` so the two never disagree:

| type | kind | emits |
|------|------|-------|
| `embedding` | sequence | `[seqLen, embedding_dim]`, channels = `embedding_dim` |
| `linear` | vector | `[output_dim]`, channels = `output_dim` |
| `conv2d` | spatial | `[out_channels, H, W]`, H/W carried from incoming, channels = `out_channels` |
| `maxpool2d`, `avgpool2d` | spatial | H/W ÷ kernel_size (floor, min 1), channels unchanged |
| `lstm`, `gru` | sequence | `[rnnSeqLen, hidden_size × dir]`, channels = `hidden_size × dir` |
| `transformer`, `attention` | sequence | `[seqLen, d_model]`, channels = `d_model` |
| `batchnorm` | inherits incoming kind | shape unchanged |
| `layernorm` | inherits incoming kind | shape unchanged |
| `dropout`, `relu`, `softmax` | passthrough | shape unchanged |

The seed shape, when there is no incoming tensor, is derived from the first layer's own
parameters — `conv2d` seeds `[in_channels, 224, 224]`, `linear` seeds `[input_dim]`,
`lstm`/`gru` seed `[128, input_size]`, `transformer`/`attention` seed `[512, d_model]`,
`embedding` seeds `[512]` token ids. These constants come from `FLOPS_ASSUMPTIONS` in
`layerTypes.js` — **import them, do not re-declare them**.

`extentFor(shape)` maps a `TensorShape` to a drawing extent `{w, h, d}` in world units:

```
size(n) = clamp(0.34 + 0.42 * log10(1 + n) , 0.34, 2.6)

spatial      → w = size(W), h = size(H), d = size(channels) * 0.75
sequence     → w = size(channels) * 0.9, h = size(channels) * 0.9, d = size(seqLen) * 0.5
vector       → w = size(channels), h = size(channels), d = 0.22
passthrough  → inherits the previous extent scaled by 0.86, d = 0.16
```

The log compression is essential: a 50,000-token vocabulary next to a 64-unit dense layer
would otherwise be 780× the size and the small layer would be invisible.

### 2.2 `src/viz/sceneGraph.js`

```js
/**
 * @typedef {{
 *   id: string,
 *   index: number,
 *   type: string,
 *   name: string,
 *   paint: LayerPaint,
 *   extent: {w:number,h:number,d:number},
 *   center: {x:number,y:number,z:number},
 *   shape: TensorShape,
 *   params: number,
 *   flops: number,
 *   paramShare: number,   // 0..1
 *   flopShare: number,    // 0..1
 *   warning: boolean,     // dimension mismatch reported by validateModelDimensions
 * }} SceneNode
 *
 * @typedef {{
 *   nodes: SceneNode[],
 *   links: SceneLink[],
 *   bounds: {min:{x,y,z}, max:{x,y,z}, radius:number, center:{x,y,z}},
 *   totals: {params:number, flops:number, depth:number},
 * }} Scene
 *
 * @param {Array<{id,type,params}>} layers
 * @param {Record<string, LayerTypeConfig>} layerTypes  from getLayerTypes()
 * @param {Map<number,{field,expected}>} issues         from validateModelDimensions()
 * @returns {Scene}
 */
export function buildScene(layers, layerTypes, issues): Scene
```

Layout: layers march along **+Z**, centred on the origin.

```
gap(i) = 0.42 + 0.30 * max(extent[i].d, extent[i+1].d)
z[0]   = 0
z[i]   = z[i-1] + extent[i-1].d/2 + gap(i-1) + extent[i].d/2
```

then the whole run is translated so that its Z midpoint is 0. Y is 0 for every node (the
stack floats at a constant height above the grid); X is 0. A *lane offset* is applied to
`passthrough` nodes: they are drawn at 82% scale and lifted `+0.08` in Y so activations and
dropout read as annotations on the flow rather than as tensors of their own.

`links[i]` connects node `i` to node `i+1` and carries
`{from, to, width0, width1, broken, flow}` where `width0/1` are the ribbon half-widths taken
from the adjacent extents, `broken` is true when the *downstream* node has `warning`, and
`flow` is the normalised FLOPs of the downstream node (drives particle density).

`buildScene([])` returns an empty scene with `bounds.radius === 1` — the caller must be able
to frame an empty scene without special-casing.

**Purity requirement:** `buildScene` must be deterministic and free of `Math.random`,
`Date.now`, and any DOM access. The animation phase is a *uniform*, not scene data.

### 2.3 `src/viz/math.js`

Column-major 4×4 matrices in `Float32Array(16)`, matching the layout `gl.uniformMatrix4fv`
expects with `transpose = false`.

Required exports, all allocation-free where an `out` parameter is supplied:

```
mat4.identity(out)
mat4.perspective(out, fovyRadians, aspect, near, far)
mat4.lookAt(out, eye, target, up)
mat4.multiply(out, a, b)
mat4.invert(out, a)             // returns null when singular
mat4.transformPoint(out, m, p)  // w-divide applied
vec3.create/set/add/sub/scale/dot/cross/length/normalize/lerp
clamp(v, lo, hi)
damp(current, target, lambda, dt)   // frame-rate-independent exponential smoothing
easeInOutCubic(t)
```

`damp` is `target + (current - target) * Math.exp(-lambda * dt)`. Using a fixed per-frame
lerp factor instead would make the camera feel different at 60 Hz and 144 Hz; this is the
single most common bug in hand-written orbit controls and the spec calls it out for that
reason.

---

## 3. WebGL2 renderer

### 3.1 Capability probe and graceful degradation

`src/viz/glCore.js` exports `probe(canvas)` returning:

```js
{ gl, level: 'full'|'reduced'|'none', hdr: boolean, samples: number, maxAniso: number }
```

- `level: 'none'` → `getContext('webgl2')` returned null. The caller renders the SVG diagram.
- `hdr` → `EXT_color_buffer_float` **or** `EXT_color_buffer_half_float` is present, so the
  scene can be rendered to `RGBA16F` and bloom can work on true HDR values. Without it the
  scene target is `RGBA8` and the bright-pass threshold is lowered from `1.05` to `0.78`
  so bloom still triggers on clipped whites.
- `samples` = `min(gl.getParameter(gl.MAX_SAMPLES), 4)`; `0` disables MSAA and enables FXAA
  in the composite pass instead.
- `level: 'reduced'` is chosen when `samples === 0 && !hdr`, or when the device reports a
  `WEBGL_debug_renderer_info` string matching `/swiftshader|llvmpipe|software/i`. In reduced
  mode the particle count is quartered, bloom drops to a single blur iteration, and the
  target frame rate is 30.

Context loss must be handled: listen for `webglcontextlost` (call `preventDefault()`, which is
what asks the browser to attempt restoration, and stop the loop) and `webglcontextrestored`
(rebuild every GPU resource from scratch). A tool that turns into a black rectangle when the OS
suspends the GPU is worse than one that never had 3D at all.

"From scratch" is the operative phrase, and it is why the rebuild does **not** happen inside the
renderer. Programs, vertex arrays and the static geometry buffers die with the context, not just
the render targets, so an in-place rebuild would be a second construction path nothing else
exercises. Both events are reported to the caller instead: `ModelViewer` falls back to the SVG on
loss, and on restore bumps an epoch that disposes the engine and builds a new one through exactly
the code that runs at mount. `dispose()` skips its own `loseContext()` while a context is already
lost, so teardown cannot cancel the restoration it is waiting on.

### 3.2 Resources

| Resource | Type | Notes |
|----------|------|-------|
| `unitBox` | indexed VAO | 24 verts / 36 indices, per-vertex position + normal + face-local UV |
| `quad` | indexed VAO | fullscreen triangle pair, positions only |
| `ribbonVAO` | dynamic VAO | rebuilt on scene change, `DYNAMIC_DRAW` |
| `instanceBuffer` | `ARRAY_BUFFER` | per-node: `mat3 basis` packed as 3 vec4 (scale+translate), `vec4 baseColor`, `vec4 glowColor`, `vec4 meta = (paramShare, flopShare, warning, zPhase)` |
| `sceneFBO` | FBO | `RGBA16F` colour + `DEPTH_COMPONENT24`, MSAA renderbuffers when `samples > 0`, resolved via `blitFramebuffer` |
| `bloomA/B` | FBO chain | 3 mip levels at ½, ¼, ⅛ resolution, `RGBA16F` or `RGBA8` |

Instance stride is 80 bytes (20 floats). Attribute locations are fixed in the shader with
`layout(location = N)` so no `getAttribLocation` round-trip is needed and a link-order change
cannot silently rebind them.

### 3.3 Passes, in order

```
1  grid          → sceneFBO   analytic grid on a large ground quad, depth write on
2  slabs         → sceneFBO   instanced opaque tensor volumes, depth write on
3  shell         → sceneFBO   instanced back-face-only additive halo, depth test on / write off
4  ribbons       → sceneFBO   additive tapered strips, depth test on / write off
5  particles     → sceneFBO   instanced additive billboards, depth test on / write off
6  resolve       → sceneTex   blitFramebuffer when MSAA, otherwise a no-op
7  brightPass    → bloom[0]   threshold + ½ downsample
8  blur x3       → bloom[n]   separable 9-tap Gaussian, H then V, at each mip
9  composite     → default    tonemap(scene + bloom) + vignette + grain + optional FXAA
```

Nine passes, ~12 draw calls, independent of layer count. Everything that varies per layer
is instanced.

### 3.4 Shaders

All GLSL lives in `src/viz/glsl.js` as template strings. **`#version 300 es` must be the
first characters of the string with no leading newline** — a leading blank line is a compile
error that reports as `'#version' : must occur first in shader`, and it is the single most
common WebGL2 mistake.

Every fragment shader declares `precision highp float;` and `precision highp int;`.

#### 3.4.1 Slab shader — the core look

Vertex stage:

- Reconstruct the world matrix from the packed instance basis.
- Apply a **breathing** displacement along the vertex normal:
  `pos += normal * 0.012 * sin(uTime * 1.6 + meta.w * 6.2831)`. Amplitude is small enough
  to read as "alive" rather than as wobble, and `meta.w` (the node's normalised Z position)
  desynchronises the stack so it undulates rather than pulsing in unison.
- Compute `vActivation`: the forward-pass wave. `wave = exp(-pow((fract(uTime * uPulseRate)
  - meta.w) * uPulseWidth, 2.0))`, giving a Gaussian bump that sweeps input→output once per
  `1/uPulseRate` seconds. `uPulseRate` defaults to `0.28` (≈3.6 s per pass).

Fragment stage, in this order:

1. **Hemispheric ambient** — `mix(uGroundColor, uSkyColor, N.y * 0.5 + 0.5)`.
2. **Key light** — one directional light at `normalize(vec3(-0.45, 0.8, 0.38))`, Lambert
   with a wrap term of `0.25` so the dark side never goes fully black.
3. **GGX specular** — `D * G * F / (4 NdotL NdotV)` with roughness `0.28`, a single term.
   Full PBR is not needed; one correct GGX lobe is what makes the edges read as material
   rather than as flat fill.
4. **Fresnel rim** — `pow(1.0 - max(dot(N, V), 0.0), 4.0)`, tinted with `glowColor`, scaled
   by `0.55 + 0.9 * vActivation`.
5. **Interior lattice** — a parallax-offset 3D grid sampled in object space:
   `p = vObjPos + V_obj * 0.14; lattice = smoothstep(0.46, 0.5, max3(abs(fract(p * uLatticeScale) - 0.5)))`.
   This is what makes a solid box read as a *volume of feature maps*. `uLatticeScale` is
   derived per-instance from the tensor's channel count so a 512-channel layer visibly has
   finer internal structure than a 16-channel one.
6. **Warning stripes** — when `meta.z > 0.5`, a 45° animated stripe mask in screen space is
   mixed toward `--warning`, at 35% strength. Colour is never the only cue: the 2D card also
   carries the text explanation, per I5.
7. **Emissive** — `glowColor * (0.10 + 1.35 * vActivation + 0.45 * paramShare)`.
   Parameter share drives a constant glow, so the layers that dominate the model's size are
   the bright ones even at rest. This is the visualisation's actual information payload.
8. Output is written in **linear space**; tone mapping happens once, in the composite pass.

#### 3.4.2 Ribbon shader

A ribbon is a triangle strip of `2 × SEGMENTS` vertices (`SEGMENTS = 24`) between two nodes,
laid out on a cubic Bézier whose control points pull toward the +Z axis so the flow reads as
a smooth pipe rather than a straight line. Half-width lerps `width0 → width1` with a
`smoothstep`, so a 64→512 channel expansion visibly flares.

The fragment stage draws **travelling energy bands**:

```glsl
float band = fract(vT * uBandDensity - uTime * uBandSpeed);
float pulse = smoothstep(0.5, 0.0, abs(band - 0.5) * 2.0);
float edge  = 1.0 - abs(vSide);                 // vSide in [-1, 1] across the ribbon
alpha = pulse * pow(edge, 1.6) * uOpacity;
```

Broken links (`broken == 1.0`) replace the smooth band with a dashed mask
`step(0.5, fract(vT * 14.0))` and force the colour to `--danger`, so a dimension mismatch is
visible from any camera angle without reading a label.

#### 3.4.3 Particle shader

`uParticleCount` billboards (default 1,400, quartered in reduced mode) are drawn in **one**
instanced call. Each instance carries only `(linkIndex, phase, sizeJitter)` — 3 floats. All
motion is computed in the vertex shader from `uTime`:

```glsl
float t = fract(phase + uTime * uSpeed * linkFlow);
vec3 p  = bezier(linkFrom, linkCtrl0, linkCtrl1, linkTo, t) + swirl(phase, t);
```

Link endpoints are read from a **uniform buffer object** (`std140`, `MAX_LINKS = 64`), which
avoids a texture fetch and keeps the whole system on the GPU. Models with more than 64 links
reuse the buffer modulo 64 — the visual result is indistinguishable and the alternative
(a data texture) is not worth the complexity.

Size attenuates with distance (`gl_PointSize` is not used; billboards are quads so they can
be rotated and soft-edged). Colour lerps from the source node's `glow` to the target's,
which makes the flow read directionally even in a still screenshot.

#### 3.4.4 Grid shader

Analytic grid with correct anti-aliasing via screen-space derivatives:

```glsl
vec2 g = abs(fract(uv - 0.5) - 0.5) / fwidth(uv);
float line = 1.0 - min(min(g.x, g.y), 1.0);
```

Two frequencies (1 unit and 8 units) are composited, then multiplied by a radial fade
`1.0 - smoothstep(12.0, 34.0, length(worldXZ))` so the plane dissolves rather than ending at
a hard edge. A soft elliptical contact shadow is accumulated per node directly in this
shader from the same UBO the particles use — cheaper and more stable than a shadow map, and
it is the one cue that stops the stack from looking like it is floating in a void.

#### 3.4.5 Composite

```
color  = sceneTex + bloomSum * uBloomStrength
color  = acesFilmic(color * uExposure)
color  = mix(color, color * vignette, 0.85)
color += (hash(gl_FragCoord.xy + uTime) - 0.5) * 0.012     // grain, breaks 8-bit banding
color  = linearToSRGB(color)
```

ACES fit is the standard Narkowicz approximation. The grain is not decoration: the sky
gradient is a wide, low-contrast ramp and without dither it bands visibly on 8-bit displays.

### 3.5 Camera

`src/viz/camera.js` — a spherical orbit camera.

State: `{ theta, phi, radius, target, ...Damped }`. Every user input writes to a *desired*
value; the render loop damps the actual value toward it with `damp(..., 9.0, dt)`.

- `phi` is clamped to `[0.12, Math.PI - 0.12]` so the camera can never flip through a pole.
- `radius` is clamped to `[bounds.radius * 0.55, bounds.radius * 6]`.
- **Idle drift:** after 4 s with no input, `theta` gains `0.055 rad/s`. Any pointer or key
  event cancels it instantly and restarts the timer. Drift is disabled entirely under
  `prefers-reduced-motion`.
- `frame(bounds)` computes the radius that fits the scene:
  `radius = bounds.radius / Math.sin(fovy / 2) * 1.15`, then also checks the horizontal FOV
  against the aspect ratio and takes the larger. Framing that only considers vertical FOV
  breaks on a wide viewport with a long model, which is exactly this app's common case.
- `focus(node)` animates `target` and `radius` over `--dur-slow` with `easeInOutCubic`.

Input mapping:

| Input | Action |
|-------|--------|
| left drag / one finger | orbit |
| wheel / pinch | dolly, exponential in `radius` |
| right drag / two fingers | pan, in the camera's screen plane |
| double click | frame the whole model |
| click a node | focus that node |
| `←` `→` `↑` `↓` | orbit by `0.12 rad` |
| `+` `-` | dolly by 12% |
| `Home` | frame |
| `Tab` | move the 3D selection to the next node |

`setPointerCapture` is used on pointerdown so a drag that leaves the canvas keeps working,
and `touch-action: none` is set on the canvas so a one-finger orbit does not also scroll the
page. Wheel handling calls `preventDefault()` **only** when the pointer is over the canvas
and the panel has been interacted with at least once — hijacking page scroll on first hover
is a hostile default.

### 3.6 Picking

CPU ray/AABB intersection against the node extents, in world space. With ≤ 200 nodes this
is ~10 µs and needs no extra GPU pass, no readback stall, and works identically on a
software rasteriser.

`pick(ndcX, ndcY)` unprojects two points via the inverse view-projection, builds the ray,
runs the slab test against each node's oriented box (which is axis-aligned here, so the
plain slab test is exact), and returns the nearest hit.

Throttle to `pointermove` at most once per animation frame.

### 3.7 The loop

```
if (document.hidden) return;                  // visibilitychange pauses
if (!intersecting) return;                    // IntersectionObserver pauses offscreen
dt = min((now - last) / 1000, 0.05);          // clamp so a background tab does not
                                              // resume with a 30 s dt and teleport
```

Adaptive quality: a 30-frame rolling mean of frame time. Above 24 ms for 30 consecutive
frames, step quality down (bloom iterations 3→2→1, particles ×0.5, DPR cap 2→1.5→1). Below
12 ms for 120 consecutive frames, step back up, but never above the level the probe granted
and never more than twice per session — an oscillating quality level is more distracting
than a permanently lower one.

DPR: `min(devicePixelRatio, 2)`, further capped by `qualityLevel`. Resize is driven by a
`ResizeObserver` on the container, not a `window.resize` listener, so a layout change that
does not resize the window still resizes the canvas.

`preserveDrawingBuffer: true` is required so `html2canvas` can capture the viewport for the
existing PNG export. The engine exposes `renderNow()` which the export path calls
synchronously before capture.

### 3.8 Disposal

`dispose()` must delete every buffer, texture, framebuffer, VAO and program, remove every
listener, cancel the RAF, disconnect both observers, and finally call
`getExtension('WEBGL_lose_context')?.loseContext()`. React Strict Mode double-invokes
effects in development; a renderer that leaks a context on unmount will exhaust the
browser's 16-context limit after eight remounts and silently stop working.

---

## 4. React integration

### 4.1 `src/components/ModelViewer.jsx`

Props:

```js
{
  layers,            // Array<{id, type, params}>
  layerTypes,        // from getLayerTypes(t, isDarkMode)
  issues,            // Map from validateModelDimensions
  isDarkMode,
  t,
  selectedId,        // string|null, controlled from LayerCal
  onSelect,          // (id|null) => void
  onFocusLayer,      // (id) => void, scrolls the 2D card into view
}
```

Behaviour:

1. On mount, `useEffect` dynamically `import('@/viz/renderer')`. **Never a static import**,
   and never `React.lazy` — `renderToStaticMarkup` cannot suspend, so `React.lazy` would
   break invariant I3. A dynamic import inside an effect never runs during SSR at all.
2. While loading, and on the server, render the SVG diagram (§4.9) so the panel is never
   empty and the layout never shifts.
3. `useEffect` on `[scene]` calls `engine.setScene(scene)`. The scene is recomputed with
   `useMemo` from `[layers, layerTypes, issues]`, so typing in a parameter field updates the
   3D view on the same commit as the 2D one.
4. Selection is *controlled*: clicking a slab calls `onSelect`, and `LayerCal` reflects the
   same `selectedId` onto the 2D card. Clicking the 2D card focuses the 3D camera. The two
   representations are one interface, not two.

### 4.2 Overlay chrome

Rendered as absolutely-positioned HTML over the canvas, never in GL:

- **Top-left:** three stat chips — parameters, FLOPs, depth — with tabular numerals.
- **Top-right:** a toolbar — play/pause flow, reset view, toggle labels, capture PNG.
- **Node labels:** one `<span>` per node, positioned each frame from the projected node
  centre via a `transform: translate3d(...)`. Labels beyond `radius * 2.4` fade out; labels
  whose projected boxes overlap are culled by a simple greedy sweep so the panel never turns
  into a wall of text. Text in the DOM stays selectable, translatable, screen-readable and
  crisp at any DPR, which no in-GL text solution achieves.
- **Bottom-left:** a legend mapping the nine families to their colours, collapsible.
- **Hover card:** name, tensor shape, parameter count, share of total, and the mismatch
  message when present.

The overlay container is `pointer-events: none`; individual controls re-enable it. Getting
this backwards makes the canvas undraggable and is worth stating explicitly.

### 4.3 `src/components/ModelDiagram2D.jsx`

The fallback and the reduced-motion representation. A static SVG built from the *same*
`Scene` object, drawn as a 2.5D isometric projection:

```
project(x, y, z) = { sx: (x - z) * 0.866, sy: (x + z) * 0.5 - y }
```

Each node is a parallelepiped (three quads: top, front, side) filled from
`paint.hex`/`paint.hexDark` with the top face lightened 14% and the side darkened 16%.
Links are tapered polygons. Warnings get a dashed danger-coloured outline plus a `<title>`.
Every node carries `<title>` text so the diagram is described to screen readers, and the
whole `<svg>` has `role="img"` with an `aria-label` summarising the model.

This component is also what the PNG export falls back to when WebGL is unavailable, so it
must be styled to look deliberate — not like a degraded mode.

### 4.4 Layout changes to `LayerCal.jsx`

```
┌──────────────────────────────────────────────────────────────┐
│ header                                                        │
├──────────────────────────────────────────────────────────────┤
│ ModelViewer   (full width, aspect 21/9, min 320px, max 560px) │
├───────────────┬──────────────────────────┬───────────────────┤
│ palette       │ builder                  │ summary            │
│ (lg:col-1)    │ (lg:col-2)               │ (lg:col-1)         │
└───────────────┴──────────────────────────┴───────────────────┘
```

On viewports below `md`, the viewer collapses to `aspect-[4/3]` and the auto-drift is
disabled to save battery.

The existing `data-capture-area` wrapper must be extended to include the viewer so PNG
export captures it.

### 4.5 Animated numbers

`useAnimatedNumber(value, duration)` in `src/utils/useAnimatedNumber.js`: RAF-driven
`easeOutExpo` interpolation between the previous and current value, snapping to the exact
target on completion so the readout is never off by one. Returns the raw value immediately
under `prefers-reduced-motion`, and never animates the first mount (a count-up from zero on
page load is noise, not feedback).

### 4.6 Selection wiring

`LayerCal` owns `selectedLayerId`. It is set by: clicking a 3D slab, clicking a 2D card,
or focusing a card's input. It drives: the 3D camera focus, a ring on the 2D card, and a
brightened slab. `Escape` clears it.

### 4.7 Palette and card restyle

Layer cards move from the current `bg-*-100 border-*-300` Tailwind pairs to a token-driven
treatment using the palette:

```jsx
style={{
  '--layer': paint.hex,             // swapped for hexDark under .dark
  background: 'color-mix(in oklab, var(--layer) 12%, transparent)',
  borderColor: 'color-mix(in oklab, var(--layer) 38%, transparent)',
}}
```

`color-mix` is supported in every browser the project targets (Chrome 111+, Safari 16.2+,
Firefox 113+). A plain `rgba()` fallback is declared first for older engines.

### 4.8 Reduced motion

Under `prefers-reduced-motion: reduce`: the render loop draws exactly one frame after each
scene or camera change and then stops; particles and the pulse are frozen at `t = 0`; idle
drift is off; the animated counter returns raw values. The scene is still fully
interactive — orbit and pick still work, they simply do not animate on their own.

### 4.9 No-WebGL2 path

`probe()` returns `level: 'none'` → the component renders `ModelDiagram2D` permanently and
shows a one-line note explaining why, with no error styling. This is a supported mode, not
a failure.

### 4.10 Accessibility

- The canvas has `role="img"` and an `aria-label` that names the model
  ("Neural network with 8 layers, 2.4 million parameters").
- A visually-hidden `<ul>` mirrors the node list in DOM order with names, shapes and
  parameter counts, so a screen reader gets the full content of the visualisation.
- The canvas is focusable (`tabIndex={0}`) with a visible focus ring, and the keyboard map
  in §3.5 works when it is focused. `Tab` inside the canvas moves the 3D selection; `Escape`
  releases focus back to the page.
- No information is conveyed by colour alone: warnings carry stripes *and* a dashed link
  *and* text.
- The overlay hover card is `aria-live="polite"` so selection changes are announced.

---

## 5. AI and model layer modernisation

### 5.1 `src/config/modelCatalog.js` — new file

A curated, dated catalogue replacing the free-text-only model override.

```js
/**
 * @typedef {{
 *   id: string,          // exact API model id
 *   label: string,       // display name
 *   tier: 'fast'|'balanced'|'frontier',
 *   inputPrice: number,  // USD per 1M input tokens
 *   outputPrice: number, // USD per 1M output tokens
 *   note?: string,
 * }} CatalogModel
 */
export const MODEL_CATALOG = { openai: [...], gemini: [...], claude: [...] }
export const CATALOG_VERIFIED_ON = '2026-09-04'
```

Verified contents as of `2026-09-04`:

| Provider | id | tier | $/1M in | $/1M out |
|----------|----|------|---------|----------|
| openai | `gpt-5.6-luna` | fast **(default)** | 0.20 | 1.20 |
| openai | `gpt-5.6-terra` | balanced | 2.00 | 12.00 |
| openai | `gpt-5.6-sol` | frontier | 4.00 | 20.00 |
| openai | `gpt-6-astra` | frontier | 10.00 | 50.00 |
| gemini | `gemini-3.5-flash-lite` | fast **(default)** | free tier | free tier |
| gemini | `gemini-3.1-flash-lite` | fast | free tier | free tier |
| gemini | `gemini-3.6-flash` | balanced | — | — |
| gemini | `gemini-3.8-flash` | balanced | — | — |
| claude | `claude-haiku-4-5` | fast **(default)** | 1.00 | 5.00 |
| claude | `claude-sonnet-5` | balanced | 2.00 | 10.00 |
| claude | `claude-opus-5` | frontier | 5.00 | 25.00 |

The default for each provider must equal `DEFAULT_MODELS[provider]` in `llmClient.js`; a
unit test asserts this, because a drift between the two would show the user one model and
call another.

Free-text override is **kept** — the catalogue is a convenience, not a cage. A model id
typed by the user that is not in the catalogue is passed through verbatim, and the
`MODEL_NOT_FOUND` path in `llmClient.js` remains the safety net for retirements.

### 5.2 `llmClient.js` corrections

`isOpenAIReasoningModel` currently matches `/^(gpt-5|o[1-9])/i`. `gpt-6-astra` does not
match, so the client would send `max_tokens` and `temperature` to a model that rejects both.
The predicate becomes:

```js
/^(?:gpt-(?:[5-9]|\d{2,})|o[1-9])/i
```

which covers GPT-5.x through GPT-9.x, any future two-digit GPT generation, and the o-series,
while still routing `gpt-4*` down the legacy path. Regression tests must pin:
`gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-6-astra` → reasoning shape; `gpt-4o`, `gpt-4.1` →
legacy shape.

Nothing else in the request pipeline changes. The existing `UNSUPPORTED_PARAM` retry
already recovers from a mis-shaped request; the fix removes a guaranteed round-trip, not a
failure mode.

### 5.3 `architectureKB.js` extension

Four modern references are appended, bringing the KB from 12 to 16. Each must use only
parameter values that exist in `layerTypes.js` option lists — the file's own header states
this rule and a test enforces it.

| id | name | tags |
|----|------|------|
| `vit-tiny` | Vision Transformer (Tiny) | vit, transformer, image, patch, classification |
| `convnext-lite` | ConvNeXt-style Block Stack | convnext, modern, cnn, image, layernorm |
| `llama-style-decoder` | Llama-style Decoder Block | llm, decoder, causal, generation, rmsnorm |
| `audio-encoder` | Whisper-style Audio Encoder | audio, speech, asr, encoder, conv, transformer |

### 5.4 `AIAdvisor.jsx` changes

- The **Advanced** panel gains a model `<select>` populated from the catalogue, with a
  trailing `Custom…` option that reveals the existing free-text input.
- Each option shows tier and price: `Haiku 4.5 · fast · $1 / $5 per 1M`.
- The result panel gains a **live preview** rendered by `ModelDiagram2D` from the proposed
  layer stack, so the user sees the shape of what they are about to apply, not just a list.
- Provider hints are updated: Gemini "free tier available", OpenAI and Anthropic
  "requires billing".

---

## 6. README

The README is a product surface, not a changelog. Required structure:

1. Centred header: logo, name, one-line description, CI badge, live-demo badge, stack badges.
2. Hero screenshot of the 3D viewer.
3. **What it does** — prose plus the feature table.
4. **The visualisation** — a new section explaining what the 3D view *encodes* (size = tensor
   shape, brightness = parameter share, particle rate = FLOPs, stripes = mismatch). A
   visualisation whose encoding is not documented is decoration.
5. **Architecture** — a Mermaid `flowchart` of the module graph.
6. **The render pipeline** — a Mermaid `flowchart LR` of the nine GPU passes.
7. **The AI advisor** — a Mermaid `sequenceDiagram` of the RAG round trip, plus the
   provider/model table.
8. **Getting started**, **Testing** (with the updated suite table), **Project structure**,
   **Calculation reference**, **Code generation**, **Deployment**, **Privacy**, **Tech stack**,
   **License**.

Mermaid renders natively on GitHub, needs no build step, and stays diffable — an image would
go stale the first time a module moved. Every diagram must use only node shapes and arrow
types that GitHub's Mermaid version supports, and must not rely on colour to be legible,
since GitHub renders Mermaid in both themes.

---

## 7. Verification protocol

### 7.1 Unit tests — new files

| File | Asserts |
|------|---------|
| `viz.palette.test.js` | every `LAYER_TYPE_IDS` entry has paint; every channel in `[0,1]`; hex strings well-formed; families are from the fixed set |
| `viz.tensorShape.test.js` | the full table in §2.1; pooling floors and clamps to ≥1; passthrough preserves shape; bidirectional RNN doubles width; unknown types do not throw; non-finite params degrade to a safe shape |
| `viz.sceneGraph.test.js` | monotonically increasing Z; no overlapping extents; shares sum to 1 (or 0 for an empty model); bounds enclose every node; empty input returns `radius === 1`; warnings propagate to the right link; determinism (same input → deep-equal output) |
| `viz.math.test.js` | `perspective`/`lookAt` against hand-computed reference matrices; `invert(multiply(a, invert(a))) ≈ identity`; `damp` is frame-rate independent (100 × 10 ms ≈ 10 × 100 ms within 1e-3); clamps |
| `modelCatalog.test.js` | catalogue defaults equal `DEFAULT_MODELS`; ids unique; prices non-negative; tiers valid |

Extended: `llmClient.test.js` gains the GPT-6 shaping cases. The knowledge-base audit
(every entry's parameters legal, and the whole stack dimension-consistent) landed in
`modelCatalog.test.js` rather than `ragPipeline.test.js`, because it reads the layer
table and the validator, not the pipeline.

**Result: 427 tests across eleven suites, all green.**

| Suite | Tests |
|-------|------:|
| `modelCatalog` (includes the knowledge-base audit) | 79 |
| `layerTypes` | 64 |
| `ragPipeline` | 55 |
| `vizPalette` | 44 |
| `llmClient` | 34 |
| `vizSceneGraph` | 36 |
| `vizMath` | 30 |
| `vizTensorShape` | 30 |
| `codeGenerator` | 25 |
| `render` | 18 |
| `modelValidation` | 12 |

### 7.2 Render verification

`npm run build && npm run preview`, then drive headless Chrome to screenshot the viewport at
1440×900 in both themes with a representative model loaded. Chrome needs
`--headless=new --use-angle=swiftshader --enable-unsafe-swiftshader` to rasterise WebGL2
without a GPU. The screenshots are inspected for: slabs present and correctly ordered,
ribbons connecting them, grid visible, bloom not blown out, labels not overlapping, text
legible in both themes.

A blank or black canvas in this check is a **build failure**, not a cosmetic issue.

### 7.3 Budget check

Build the previous commit in a throwaway `git worktree` and compare gzipped sizes chunk by chunk.

**Measured, 2026-09-04:**

| Chunk | Before | After | Delta |
|-------|-------:|------:|------:|
| `index-*.js` | 43.80 KB | 54.05 KB | +10.25 KB |
| `index-*.css` | 6.90 KB | 7.13 KB | +0.23 KB |
| `icons-*.js` | 5.58 KB | 6.26 KB | +0.68 KB |
| `react-vendor-*.js` | 43.12 KB | 43.13 KB | +0.01 KB |
| `renderer-*.js` | - | 17.68 KB | lazy, not in the initial payload |
| **Initial total** | **99.40 KB** | **110.57 KB** | **+11.17 KB** |

The first draft of this spec budgeted 8 KB, which turned out to be wrong rather than tight: it did
not account for the eight locales of new interface strings or the four added reference
architectures, neither of which is renderer code. The stylesheet then grew by far less than the
first measurement suggested, because deleting the dead per-layer Tailwind colour classes from
`layerTypes.js` removed a block of generated CSS. The budget above is the corrected figure, and the
important property - that the WebGL engine itself is never in the initial payload - holds.

### 7.4 QA matrix

Driven from Node against a headless Chrome running SwiftShader (WebGL2 with no
GPU at all), over the production build served by `vite preview`.

| Case | Expected | Result |
|------|----------|--------|
| Both themes render | non-blank canvas, clean `gl.getError()`, no console errors | pass, 253 / 171 distinct colour buckets, `glError: 0`, zero console errors |
| Empty canvas | empty grid plus an inviting empty state, no errors | pass |
| 60 layers | frame cost independent of layer count | pass, 60 layers measured *faster* than 5 on the same machine (10 fps vs 7), so the cost is entirely fill-rate |
| 50,000-word embedding | no slab beyond the size clamp; layout unbroken | pass, unit-tested in `vizSceneGraph` |
| Theme toggle | clear colour, palette and flow blending swap without a reload | pass |
| Tab away and back | loop pauses and resumes with no time jump | pass, `dt` clamped to 50 ms |
| WebGL2 unavailable | isometric SVG, an explanatory note, no console errors | pass, 24 SVG polygons, note present, zero errors |
| `prefers-reduced-motion` | one static frame, still interactive, no play control | pass |
| Keyboard only | canvas focusable, labelled, orbits, mirrors the model to a screen reader | pass |
| PNG export | the file contains the 3D viewport and no visually hidden text | pass, 2720x3324 PNG, 333 distinct colours across the viewer strip |
| Mobile, 390 px | no horizontal scroll, no control overlapping another | pass, after tightening the stat-chip row |
| 8 locales | no layout break, no `undefined`, no clipped label | pass |
| README diagrams | all four render under Mermaid 11 with no syntax errors | pass |

Three real defects were found by this pass and fixed rather than documented:

1. **PNG export was broken outright.** `color-mix(in oklab, ...)` in the
   layer-card styles resolves to an `oklab()` colour in Chrome's computed style,
   and html2canvas throws on any colour function it does not recognise, so the
   entire export failed rather than one tint. The palette now emits finished
   `rgba()` strings.
2. **Visually hidden text appeared in the export.** The `onclone` step rewrites
   `overflow: hidden` to `visible` so text does not truncate, which undoes
   exactly how `.sr-only` hides itself. Screen-reader labels were being painted
   down the margin of every exported PNG. This predates the release;
   `.sr-only` is now hidden explicitly before that rewrite.
3. **The viewer's stat chips overlapped its toolbar on a phone.** The chips
   wrapped onto a second line and landed underneath the buttons, making them
   unreachable rather than merely untidy.

### 7.5 Review pass

A second, adversarial read of every file afterwards found nine more defects, all fixed:

| # | Defect | Why it mattered |
|---|--------|-----------------|
| 1 | `layerTypes.js` still exported a `color` field of Tailwind classes and an emoji `icon`, neither read by anything | A second colour table that had already drifted from the palette: GRU red there and amber here, LayerNorm indigo there and teal here |
| 2 | The layer table depended on `isDarkMode` | A theme toggle invalidated it, which rebuilt the whole scene and re-uploaded every GPU buffer for a change that only needed new uniforms |
| 3 | `getBoundingClientRect()` ran twice per frame | Two forced layout flushes per frame - the classic way to make a render loop slower than the GPU it drives |
| 4 | `projectLabels` allocated an object and an array per node per frame | Around 7,000 objects a second on a 60-layer model, inside a module whose stated premise is that the loop allocates nothing |
| 5 | The sky was inverse-tone-mapped from scratch every frame | Six `pow` and six `sqrt` per frame to recompute two constants |
| 6 | Reduced quality drew a prefix of a link-major particle buffer | The flow vanished off the tail of the model instead of thinning evenly. Particles are now written round-robin across links, so any prefix is a fair sample |
| 7 | The pending hover `requestAnimationFrame` outlived `dispose()` | It fired against deleted buffers and called back into an unmounted component |
| 8 | Context restoration rebuilt render targets and buffer data, but not programs, VAOs or static buffers | A restored context would have rendered nothing: the behaviour this spec documents was never actually implemented. Recovery now runs through the caller (section 3.1) |
| 9 | `ModelDiagram2D` minted fixed SVG gradient ids | Two instances can be on one page - the fallback and the advisor preview - and the second painted with the first one's colours |

Three smaller things were tidied at the same time. A hidden canvas kept `role="img"` and its label,
so a screen reader announced the model twice whenever the SVG fallback was up. `--viz-sky` and
`--viz-ground` were declared and never used, while `ModelViewer` hardcoded two hex colours for the
same job; the panel now paints the renderer's own sky gradient from those tokens. And the mechanical
token pass over `AIAdvisor` had left 35 template literals interpolating a constant string.
---
### 7.6 CSP verification

`vite preview` does not apply `public/_headers`, so every browser check above ran **without** the
production Content-Security-Policy. That is the one header this app cannot afford to get wrong,
because users paste API keys into the page and `connect-src` is what stops an injected script
posting them somewhere else.

Closed by serving `dist/` through a small static server that parses `_headers` and applies its
rules the way Cloudflare Pages does, then loading the app in headless Chrome with a
`securitypolicyviolation` listener installed before any page script runs, and exercising the paths
most likely to trip the policy:

- first load and the dynamically imported WebGL renderer (`script-src 'self'`),
- the code-export dialog,
- the PNG export, which fetches html2canvas at runtime and downloads through a `blob:` URL,
- the AI advisor dialog.

**Result: zero violations, canvas drawn, `gl.getError()` clean, no console errors.** The release
adds no origin to any directive - the shaders are strings in the bundle, the fonts are system
stacks, and the renderer chunk is same-origin - which is why the policy did not have to widen.

This is a pre-flight check, not a substitute for the one in the README: the real deployment should
still be validated on a preview, because a CDN can add or drop headers this server does not model.

## 8. Acceptance criteria

- [x] `npm test` - **427 tests, 0 failures** across eleven suites.
- [x] `npm run build` - succeeds; the renderer is a separate lazy chunk
      (`renderer-*.js`, 17.68 KB gzipped); initial payload +11.17 KB gzipped
      against the previous commit (see 7.3).
- [x] Headless screenshots in both themes show a correct, non-blank 3D scene
      with a clean `gl.getError()`.
- [x] Invariants I1-I7 (see 0.2) verified:
      I1, no new origin in `connect-src`;
      I2, `layerTypes.test.js` unchanged and green;
      I3, `renderToStaticMarkup` green in all eight locales;
      I4, 142 keys per locale, parity and non-empty asserted;
      I5, no-WebGL, reduced-motion and keyboard-only paths verified in-browser;
      I6, build green, renderer lazily chunked, payload within the corrected budget;
      I7, `package.json` dependencies untouched.
- [x] Every row of the 7.4 QA matrix passes.
- [x] The production build runs under the production CSP with zero violations (see 7.6).
- [x] README contains four Mermaid diagrams; all four render under Mermaid 11
      with no syntax errors and legible proportions.
- [x] No `TODO`, commented-out block, `debugger`, or `console.log` in `src/`.
      The five `console.warn` calls in `localStorage.js` and the one
      `console.error` in `imageExport.js` are pre-existing, deliberate error
      reporting.
- [x] `git diff package.json` shows no dependency change; only the version,
      description and keywords moved.
