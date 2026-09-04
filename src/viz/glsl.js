/**
 * GLSL ES 3.00 sources for the model viewer.
 *
 * Two rules hold throughout this file:
 *
 *   1. `#version 300 es` is the first token of every source, with no leading
 *      newline. A blank line before it is a compile error that reports as
 *      "'#version' : must occur first in shader", and it is the most common
 *      way a working WebGL2 shader gets broken by a reformat.
 *   2. Attribute locations are pinned with `layout(location = N)`. Nothing
 *      calls getAttribLocation, so a change in declaration order cannot
 *      silently rebind a buffer to the wrong attribute.
 *
 * All lighting is computed in linear space. Tone mapping and the sRGB encode
 * happen exactly once, in the composite pass.
 */

/** Shared helpers, textually included where needed. */
const COMMON = /* glsl */ `
const float PI = 3.14159265359;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
`;

/** Cubic Bezier position and tangent, used by the particle system. */
const BEZIER = /* glsl */ `
vec3 bezier(vec3 a, vec3 b, vec3 c, vec3 d, float t) {
  float u = 1.0 - t;
  return u*u*u*a + 3.0*u*u*t*b + 3.0*u*t*t*c + t*t*t*d;
}
vec3 bezierTangent(vec3 a, vec3 b, vec3 c, vec3 d, float t) {
  float u = 1.0 - t;
  return 3.0*u*u*(b-a) + 6.0*u*t*(c-b) + 3.0*t*t*(d-c);
}
`;

// ─── Tensor slabs ──────────────────────────────────────────────

export const SLAB_VERT = `#version 300 es
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec4 iCenter;   // xyz = world centre, w = normalised depth
layout(location = 4) in vec4 iExtent;   // xyz = size, w = feature-map grid cells
layout(location = 5) in vec4 iBase;     // rgb = albedo (linear), a = parameter share
layout(location = 6) in vec4 iGlow;     // rgb = emissive (linear), a = warning flag
layout(location = 7) in vec4 iState;    // x = selected, y = hovered, z = annotation, w = flop share

uniform mat4 uViewProj;
uniform vec3 uCameraPos;
uniform float uTime;
uniform float uMotion;      // 0 freezes every animation (reduced motion)
uniform float uPulseRate;
uniform float uPulseWidth;
uniform float uShellScale;  // 1.0 for the solid pass, >1 for the halo shell

out vec3 vNormal;
out vec3 vWorld;
out vec2 vUV;
out vec4 vBase;
out vec4 vGlow;
out vec4 vState;
out float vActivation;
out float vCells;

void main() {
  float t = uTime * uMotion;

  // Breathing: a sub-millimetre displacement along the normal, desynchronised
  // by the node's depth so the stack undulates instead of pulsing in unison.
  float breathe = 0.012 * sin(t * 1.6 + iCenter.w * 6.28318);

  vec3 scale = iExtent.xyz * uShellScale;
  vec3 world = iCenter.xyz + aPosition * scale + aNormal * breathe;

  // Forward-pass wave: a Gaussian bump sweeping input to output. The offset is
  // wrapped into [-0.5, 0.5] so the pulse loops seamlessly instead of jumping
  // when the sweep restarts.
  float head = fract(t * uPulseRate);
  float d = head - iCenter.w;
  d = d - floor(d + 0.5);
  vActivation = exp(-(d * uPulseWidth) * (d * uPulseWidth));

  vNormal = aNormal;
  vWorld = world;
  vUV = aUV;
  vBase = iBase;
  vGlow = iGlow;
  vState = iState;
  vCells = iExtent.w;

  gl_Position = uViewProj * vec4(world, 1.0);
}
`;

export const SLAB_FRAG = `#version 300 es
precision highp float;
precision highp int;
${COMMON}

in vec3 vNormal;
in vec3 vWorld;
in vec2 vUV;
in vec4 vBase;
in vec4 vGlow;
in vec4 vState;
in float vActivation;
in float vCells;

uniform vec3 uCameraPos;
uniform vec3 uSkyColor;
uniform vec3 uGroundColor;
uniform vec3 uKeyColor;
uniform vec3 uWarnColor;
uniform vec3 uFogColor;
uniform float uFogNear;
uniform float uFogDensity;
uniform float uTime;
uniform float uMotion;

out vec4 fragColor;

const vec3 KEY_DIR = vec3(-0.4767, 0.8474, 0.4025);   // normalised (-0.45, 0.8, 0.38)
const float ROUGHNESS = 0.28;

float distributionGGX(float NoH, float a) {
  float a2 = a * a;
  float d = NoH * NoH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-6);
}

float visibilitySmith(float NoV, float NoL, float a) {
  float a2 = a * a;
  float gv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
  float gl = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
  return 0.5 / max(gv + gl, 1e-4);
}

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(uCameraPos - vWorld);
  vec3 H = normalize(KEY_DIR + V);

  float NoV = clamp(dot(N, V), 1e-4, 1.0);
  float NoL = clamp(dot(N, KEY_DIR), 0.0, 1.0);
  float NoH = clamp(dot(N, H), 0.0, 1.0);
  float VoH = clamp(dot(V, H), 0.0, 1.0);

  // Hemispheric ambient. Two colours beat one because it gives the top and
  // bottom faces of every slab a different tint, which is most of what makes
  // a box read as a solid rather than a silhouette.
  vec3 ambient = mix(uGroundColor, uSkyColor, N.y * 0.5 + 0.5);

  // A derivative-anti-aliased grid on the face of the plate, showing the
  // spatial resolution of the feature map. The layer is already drawn as a
  // stack of plates, so the structure is real; this only says how finely each
  // plate is sampled. vCells is zero for layers that have no spatial extent.
  float grid = 0.0;
  if (vCells > 0.5) {
    vec2 cell = vUV * vCells;
    vec2 d = abs(fract(cell - 0.5) - 0.5) / max(fwidth(cell), vec2(1e-5));
    // Only the faces of the plate, not its edges.
    grid = (1.0 - min(min(d.x, d.y), 1.0)) * step(0.5, abs(N.z));
  }

  // Wrapped diffuse: the shadow side keeps its hue instead of going to black.
  float wrap = clamp((dot(N, KEY_DIR) + 0.25) / 1.25, 0.0, 1.0);
  vec3 albedo = vBase.rgb * (1.0 - 0.16 * grid);
  // Keeping the key below 1.0 matters: a diffuse term that clips drags every
  // saturated hue toward white once the tone map runs.
  vec3 diffuse = albedo * (ambient * 0.40 + uKeyColor * wrap * 0.88);

  // One correct GGX lobe. Full PBR is not needed here, but a specular term
  // that obeys the Fresnel curve is what stops the edges looking like flat fill.
  float a = ROUGHNESS * ROUGHNESS;
  vec3 F0 = mix(vec3(0.04), albedo, 0.20);
  vec3 F = F0 + (1.0 - F0) * pow(1.0 - VoH, 5.0);
  vec3 specular = uKeyColor * distributionGGX(NoH, a) * visibilitySmith(NoV, NoL, a) * NoL * F * 1.15;

  // Fresnel rim in the emissive colour, brightened as the pulse arrives.
  float fresnel = pow(1.0 - NoV, 4.0);
  vec3 rim = vGlow.rgb * fresnel * (0.30 + 0.55 * vActivation);

  vec3 color = diffuse + specular + rim;

  // Emissive. Parameter share is a constant lift and the pulse is transient, so
  // at rest the brightest layers are the ones that dominate the model's size -
  // that is the information this view carries. The coefficients are a fraction
  // of what they were: at the old strength every layer glowed like a sign.
  vec3 emissive = mix(albedo, vGlow.rgb, 0.55);
  color += emissive * (0.02 + 0.55 * vActivation + 0.22 * vBase.a);

  // Selection and hover.
  color += vGlow.rgb * (vState.x * 0.34 + vState.y * 0.16);

  // Dimension mismatch. Concentrated on the silhouette rather than painted over
  // the whole layer: the stripes used to obliterate the colour that says what
  // kind of layer it is. Still a pattern and not only a hue, and the link into
  // the layer is dashed, and the card spells it out.
  if (vGlow.a > 0.5) {
    float edge = pow(1.0 - NoV, 2.5);
    float s = fract((vWorld.x + vWorld.y + vWorld.z) * 3.0 - uTime * uMotion * 0.5);
    float stripe = smoothstep(0.34, 0.5, abs(s - 0.5) * 2.0);
    color = mix(color, uWarnColor, clamp((0.14 + 0.5 * edge) * (0.4 + 0.6 * stripe), 0.0, 0.75));
  }

  // Distance fog. Without it every layer sits at the same apparent depth and a
  // long stack reads as flat; this is most of what makes the far end recede.
  float distance = length(uCameraPos - vWorld);
  float fog = 1.0 - exp(-max(distance - uFogNear, 0.0) * uFogDensity);
  color = mix(color, uFogColor, clamp(fog, 0.0, 0.9));

  fragColor = vec4(color, 1.0);
}
`;

/**
 * Back-face-only additive halo, reusing the slab vertex shader. It declares
 * only the varyings it reads; GLSL ES 3.00 permits a fragment stage to consume
 * a subset of what the vertex stage produces.
 */
export const SHELL_FRAG = `#version 300 es
precision highp float;

in vec3 vNormal;
in vec3 vWorld;
in vec4 vBase;
in vec4 vGlow;
in vec4 vState;
in float vActivation;

uniform vec3 uCameraPos;
uniform float uShellOpacity;

out vec4 fragColor;

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(uCameraPos - vWorld);
  // Back faces, so the silhouette is where the normal turns away from the eye.
  float f = pow(1.0 - abs(dot(N, V)), 2.5);
  float amount = f * (0.05 + 0.30 * vActivation + 0.14 * vBase.a + 0.24 * vState.x) * uShellOpacity;
  fragColor = vec4(vGlow.rgb * amount, amount);
}
`;

// ─── Connection ribbons ────────────────────────────────────────

export const RIBBON_VERT = `#version 300 es
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aParams;   // x = t along the link, y = side (-1..1), z = broken
layout(location = 2) in vec3 aColor;

uniform mat4 uViewProj;

out float vT;
out float vSide;
out float vBroken;
out vec3 vColor;

void main() {
  vT = aParams.x;
  vSide = aParams.y;
  vBroken = aParams.z;
  vColor = aColor;
  gl_Position = uViewProj * vec4(aPosition, 1.0);
}
`;

export const RIBBON_FRAG = `#version 300 es
precision highp float;

in float vT;
in float vSide;
in float vBroken;
in vec3 vColor;

uniform float uTime;
uniform float uMotion;
uniform float uBandDensity;
uniform float uBandSpeed;
uniform float uOpacity;
uniform vec3 uWarnColor;

out vec4 fragColor;

void main() {
  float edge = 1.0 - abs(vSide);
  float profile = pow(max(edge, 0.0), 1.6);

  // A constant tube keeps the connection readable between pulses. Kept low:
  // the ribbon is a cross of two strips, so the centre gets both, and on top of
  // the particles the sum used to clip to white and lose the colour that says
  // where the data is coming from.
  float base = 0.11 * profile;

  float band = fract(vT * uBandDensity - uTime * uMotion * uBandSpeed);
  float pulse = smoothstep(0.5, 0.0, abs(band - 0.5) * 2.0);

  vec3 color = vColor;
  float alpha = (base + 0.60 * pulse * profile) * uOpacity;

  if (vBroken > 0.5) {
    // A dashed, danger-coloured link: a dimension mismatch has to be visible
    // from any camera angle, without reading a label.
    float dash = step(0.5, fract(vT * 14.0 - uTime * uMotion * 0.5));
    alpha = (0.08 + dash * 0.85) * profile * uOpacity * 1.15;
    color = uWarnColor;
  }

  fragColor = vec4(color * alpha, alpha);
}
`;

// ─── Activation particles ──────────────────────────────────────

export const PARTICLE_VERT = `#version 300 es
layout(location = 0) in vec2 aCorner;    // unit quad corner in [-1, 1]
layout(location = 1) in vec3 iP0;
layout(location = 2) in vec3 iP1;
layout(location = 3) in vec3 iP2;
layout(location = 4) in vec3 iP3;
layout(location = 5) in vec4 iParticle;  // x = phase, y = speed, z = size, w = swirl radius
layout(location = 6) in vec3 iColorA;
layout(location = 7) in vec3 iColorB;

uniform mat4 uViewProj;
uniform mat4 uView;
uniform float uTime;
uniform float uMotion;
uniform float uSpeed;

out vec2 vUV;
out vec3 vColor;
out float vFade;
${BEZIER}

void main() {
  float t = fract(iParticle.x + uTime * uMotion * uSpeed * iParticle.y);

  vec3 pos = bezier(iP0, iP1, iP2, iP3, t);

  // Swirl around the path so the flow reads as volume rather than as a wire.
  vec3 tangent = normalize(bezierTangent(iP0, iP1, iP2, iP3, t) + vec3(0.0, 0.0, 1e-4));
  vec3 helper = abs(tangent.y) > 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
  vec3 right = normalize(cross(helper, tangent));
  vec3 binormal = cross(tangent, right);

  float angle = iParticle.x * 6.28318 + t * 9.0;
  float radius = iParticle.w * (0.35 + 0.65 * sin(t * 3.14159265));
  pos += (right * cos(angle) + binormal * sin(angle)) * radius;

  // Camera-facing billboard: the view matrix rows are the camera basis.
  vec3 camRight = vec3(uView[0][0], uView[1][0], uView[2][0]);
  vec3 camUp    = vec3(uView[0][1], uView[1][1], uView[2][1]);
  pos += (camRight * aCorner.x + camUp * aCorner.y) * iParticle.z;

  vUV = aCorner;
  vColor = mix(iColorA, iColorB, t);
  // Fade in and out at the endpoints so particles never pop inside a slab.
  vFade = sin(t * 3.14159265);

  gl_Position = uViewProj * vec4(pos, 1.0);
}
`;

export const PARTICLE_FRAG = `#version 300 es
precision highp float;

in vec2 vUV;
in vec3 vColor;
in float vFade;

uniform float uOpacity;

out vec4 fragColor;

void main() {
  float d = length(vUV);
  if (d > 1.0) discard;
  float a = pow(smoothstep(1.0, 0.0, d), 3.0) * vFade * uOpacity;
  fragColor = vec4(vColor * a, a);
}
`;

// ─── Sky ───────────────────────────────────────────────────────

/**
 * The background is a gradient, not a flat clear. A single colour behind a
 * grid that recedes to the horizon reads as a wall; a vertical ramp with a
 * soft bloom where the model sits reads as space. Colours arrive already
 * inverse-tone-mapped, so what lands on screen is exactly what was asked for.
 */
export const SKY_FRAG = `#version 300 es
precision highp float;

in vec2 vUV;

uniform vec3 uSkyTop;
uniform vec3 uSkyBottom;
uniform vec3 uSkyGlow;

out vec4 fragColor;

void main() {
  vec3 color = mix(uSkyBottom, uSkyTop, smoothstep(0.0, 1.0, vUV.y));
  float d = distance(vUV, vec2(0.5, 0.44));
  color += uSkyGlow * exp(-d * d * 7.0);
  fragColor = vec4(color, 1.0);
}
`;

// ─── Ground grid and contact shadows ───────────────────────────

export const GRID_VERT = `#version 300 es
layout(location = 0) in vec2 aCorner;

uniform mat4 uViewProj;
uniform float uGridSize;

out vec2 vPlane;

void main() {
  vPlane = aCorner * uGridSize;
  gl_Position = uViewProj * vec4(vPlane.x, 0.0, vPlane.y, 1.0);
}
`;

export const GRID_FRAG = `#version 300 es
precision highp float;

in vec2 vPlane;

uniform vec3 uGridColor;
uniform vec3 uAccentColor;
uniform float uFadeStart;
uniform float uFadeEnd;
uniform float uOpacity;
uniform float uCellScale;

out vec4 fragColor;

// Analytic, derivative-anti-aliased grid. Sampling a texture here would alias
// badly at grazing angles, which is most of the frame for an orbiting camera.
float gridLine(vec2 p, float scale) {
  vec2 coord = p * scale;
  vec2 g = abs(fract(coord - 0.5) - 0.5) / max(fwidth(coord), vec2(1e-5));
  return 1.0 - min(min(g.x, g.y), 1.0);
}

void main() {
  float minor = gridLine(vPlane, uCellScale);
  float major = gridLine(vPlane, uCellScale * 0.2);

  float r = length(vPlane);
  float fade = 1.0 - smoothstep(uFadeStart, uFadeEnd, r);
  // Squaring the fade pulls the grid back from the frame edges, so it reads as
  // ground the model sits on rather than as wallpaper behind it.
  fade *= fade;

  // The centre line along the flow axis is tinted, so the model's direction is
  // legible even when the stack is viewed end-on.
  float axis = 1.0 - smoothstep(0.0, 0.05 * uCellScale, abs(vPlane.x));

  vec3 color = mix(uGridColor, uAccentColor, max(major * 0.5, axis * 0.85));
  float alpha = (minor * 0.26 + major * 0.72 + axis * 0.32) * fade * uOpacity;

  fragColor = vec4(color * alpha, alpha);
}
`;

export const SHADOW_VERT = `#version 300 es
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec4 iCenter;
layout(location = 2) in vec4 iExtent;

uniform mat4 uViewProj;

out vec2 vUV;
out float vStrength;

void main() {
  // Wider than the slab and softer the higher it floats, which is how a real
  // contact shadow behaves and the cheapest way to anchor the stack to the grid.
  float lift = max(iCenter.y, 0.001);
  float spread = 1.35 + lift * 0.5;
  vec3 world = vec3(
    iCenter.x + aCorner.x * iExtent.x * spread,
    0.004,
    iCenter.z + aCorner.y * iExtent.z * spread
  );
  vUV = aCorner;
  vStrength = clamp(0.55 / (0.6 + lift), 0.0, 1.0);
  gl_Position = uViewProj * vec4(world, 1.0);
}
`;

export const SHADOW_FRAG = `#version 300 es
precision highp float;

in vec2 vUV;
in float vStrength;

uniform float uStrength;

out vec4 fragColor;

void main() {
  float d = length(vUV);
  float a = pow(smoothstep(1.0, 0.0, d), 1.6) * vStrength * uStrength;
  fragColor = vec4(0.0, 0.0, 0.0, a);
}
`;

// ─── Post-processing ───────────────────────────────────────────

export const FULLSCREEN_VERT = `#version 300 es
layout(location = 0) in vec2 aCorner;

out vec2 vUV;

void main() {
  vUV = aCorner * 0.5 + 0.5;
  gl_Position = vec4(aCorner, 0.0, 1.0);
}
`;

export const BRIGHT_FRAG = `#version 300 es
precision highp float;

in vec2 vUV;

uniform sampler2D uScene;
uniform float uThreshold;
uniform float uKnee;

out vec4 fragColor;

void main() {
  vec3 c = texture(uScene, vUV).rgb;
  float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
  // Soft knee, so a surface hovering at the threshold does not flicker in and
  // out of the bloom as the camera moves.
  float soft = clamp(lum - uThreshold + uKnee, 0.0, 2.0 * uKnee);
  soft = soft * soft / (4.0 * uKnee + 1e-4);
  float contribution = max(soft, lum - uThreshold) / max(lum, 1e-4);
  fragColor = vec4(c * contribution, 1.0);
}
`;

export const BLUR_FRAG = `#version 300 es
precision highp float;

in vec2 vUV;

uniform sampler2D uSource;
uniform vec2 uDirection;   // texel-sized step, horizontal or vertical

out vec4 fragColor;

// Nine-tap Gaussian folded into five bilinear fetches.
const float OFFSETS[3] = float[3](0.0, 1.3846153846, 3.2307692308);
const float WEIGHTS[3] = float[3](0.2270270270, 0.3162162162, 0.0702702703);

void main() {
  vec3 sum = texture(uSource, vUV).rgb * WEIGHTS[0];
  for (int i = 1; i < 3; i++) {
    vec2 offset = uDirection * OFFSETS[i];
    sum += texture(uSource, vUV + offset).rgb * WEIGHTS[i];
    sum += texture(uSource, vUV - offset).rgb * WEIGHTS[i];
  }
  fragColor = vec4(sum, 1.0);
}
`;

export const COMPOSITE_FRAG = `#version 300 es
precision highp float;
${COMMON}

in vec2 vUV;

uniform sampler2D uScene;
uniform sampler2D uBloom0;
uniform sampler2D uBloom1;
uniform sampler2D uBloom2;
uniform vec2 uTexel;
uniform float uBloomStrength;
uniform float uExposure;
uniform float uVignette;
uniform float uGrain;
uniform float uSaturation;
uniform float uTime;
uniform float uFxaa;

out vec4 fragColor;

// Narkowicz's ACES fit: one polynomial, no LUT, and it keeps saturated
// emissives from clipping to white, which is exactly what bloom would do here.
vec3 aces(vec3 x) {
  const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 linearToSRGB(vec3 c) {
  return mix(c * 12.92, 1.055 * pow(max(c, 1e-5), vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

/**
 * Display-referred colour at uv, using the centre pixel's bloom for every
 * sample. Bloom is a heavily blurred signal, so reusing it costs one texture
 * fetch per FXAA tap instead of four, and the error is below the dither floor.
 */
vec3 shadeAt(vec2 uv, vec3 bloom) {
  return linearToSRGB(aces((texture(uScene, uv).rgb + bloom) * uExposure));
}

void main() {
  vec3 bloom = (texture(uBloom0, vUV).rgb * 0.55
              + texture(uBloom1, vUV).rgb * 0.30
              + texture(uBloom2, vUV).rgb * 0.15) * uBloomStrength;

  vec3 color = shadeAt(vUV, bloom);

  // FXAA runs only when the device could not give us MSAA. It works on the
  // display-referred signal, which is where the edges the eye sees actually are.
  if (uFxaa > 0.5) {
    float lNW = luma(shadeAt(vUV + vec2(-uTexel.x,  uTexel.y), bloom));
    float lNE = luma(shadeAt(vUV + vec2( uTexel.x,  uTexel.y), bloom));
    float lSW = luma(shadeAt(vUV + vec2(-uTexel.x, -uTexel.y), bloom));
    float lSE = luma(shadeAt(vUV + vec2( uTexel.x, -uTexel.y), bloom));
    float lM  = luma(color);

    vec2 dir = vec2(
      -((lNW + lNE) - (lSW + lSE)),
       ((lNW + lSW) - (lNE + lSE))
    );
    float reduce = max((lNW + lNE + lSW + lSE) * 0.03125, 0.0078125);
    float rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcp, -8.0, 8.0) * uTexel;

    vec3 inner = 0.5 * (
      shadeAt(vUV + dir * (1.0 / 3.0 - 0.5), bloom) +
      shadeAt(vUV + dir * (2.0 / 3.0 - 0.5), bloom)
    );
    vec3 outer = inner * 0.5 + 0.25 * (
      shadeAt(vUV - dir * 0.5, bloom) +
      shadeAt(vUV + dir * 0.5, bloom)
    );

    float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
    float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
    float lOuter = luma(outer);
    color = (lOuter < lMin || lOuter > lMax) ? inner : outer;
  }

  // ACES desaturates as it rolls off, which is right for photography and
  // wrong for a diagram whose colours carry meaning. Push it back afterwards.
  color = clamp(mix(vec3(luma(color)), color, uSaturation), 0.0, 1.0);

  vec2 centred = vUV - 0.5;
  color *= mix(1.0, 1.0 - dot(centred, centred) * uVignette, 0.85);

  // Dither. The sky is a wide, low-contrast ramp and without this it bands
  // visibly on an 8-bit display.
  color += (hash12(gl_FragCoord.xy + fract(uTime) * 137.0) - 0.5) * uGrain;

  fragColor = vec4(color, 1.0);
}
`;
