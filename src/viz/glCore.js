/**
 * WebGL2 plumbing: capability probing, program compilation, buffers, vertex
 * array objects and render targets.
 *
 * Nothing here knows what a neural network is. It is the layer that makes
 * renderer.js readable.
 */

/**
 * @typedef {{
 *   gl: WebGL2RenderingContext|null,
 *   level: 'full'|'reduced'|'none',
 *   hdr: boolean,
 *   samples: number,
 *   renderer: string,
 *   reason: string,
 * }} Capabilities
 */

const SOFTWARE_RENDERER = /swiftshader|llvmpipe|software|basic render/i;

/**
 * Create a context and work out what it can actually do.
 *
 * `preserveDrawingBuffer` is required, not optional: the existing PNG export
 * runs html2canvas over the page, and without it the canvas reads back blank.
 *
 * @param {HTMLCanvasElement} canvas
 * @returns {Capabilities}
 */
export function probe(canvas) {
  /** @type {WebGL2RenderingContext|null} */
  let gl = null;
  try {
    gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,          // we resolve MSAA ourselves, off-screen
      depth: false,              // the default framebuffer only ever gets a
      stencil: false,            // fullscreen composite
      premultipliedAlpha: true,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
      failIfMajorPerformanceCaveat: false,
    });
  } catch {
    gl = null;
  }

  if (!gl) {
    return { gl: null, level: 'none', hdr: false, samples: 0, renderer: '', reason: 'no-webgl2' };
  }

  const hdr = Boolean(
    gl.getExtension('EXT_color_buffer_float') ||
    gl.getExtension('EXT_color_buffer_half_float')
  );
  // Half-float targets still need a linear filter to be blurred smoothly.
  gl.getExtension('OES_texture_float_linear');

  const maxSamples = gl.getParameter(gl.MAX_SAMPLES) || 0;
  const samples = Math.min(maxSamples, 4);

  let renderer = '';
  const info = gl.getExtension('WEBGL_debug_renderer_info');
  if (info) {
    try {
      renderer = String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL) || '');
    } catch {
      renderer = '';
    }
  }

  const software = SOFTWARE_RENDERER.test(renderer);
  const level = software || (samples === 0 && !hdr) ? 'reduced' : 'full';

  return {
    gl,
    level,
    hdr,
    samples,
    renderer,
    reason: software ? 'software-renderer' : '',
  };
}

/**
 * Compile and link. On failure the info log is thrown with the offending
 * source line quoted, because "ERROR: 0:42" with no line 42 in view is the
 * least useful error message in graphics programming.
 *
 * @param {WebGL2RenderingContext} gl
 */
export function createProgram(gl, vertexSource, fragmentSource, label = 'program') {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertexSource, `${label}.vert`);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource, `${label}.frag`);

  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);

  // Shaders can be detached and deleted the moment the link succeeds; keeping
  // them alive leaks one object per program for the life of the context.
  gl.detachShader(program, vs);
  gl.detachShader(program, fs);
  gl.deleteShader(vs);
  gl.deleteShader(fs);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`[viz] link failed for ${label}: ${log}`);
  }

  return {
    program,
    uniforms: cacheUniforms(gl, program),
  };
}

function compileShader(gl, type, source, label) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) || '';
    gl.deleteShader(shader);
    throw new Error(`[viz] compile failed for ${label}:\n${log}\n${annotate(source, log)}`);
  }
  return shader;
}

/** Quote the lines the driver complained about. */
function annotate(source, log) {
  const lines = source.split('\n');
  const seen = new Set();
  const out = [];
  const pattern = /:(\d+):(\d+)|ERROR:\s*\d+:(\d+)/g;
  let match;
  while ((match = pattern.exec(log)) !== null) {
    const n = Number(match[2] || match[3]);
    if (!Number.isFinite(n) || seen.has(n)) continue;
    seen.add(n);
    for (let i = Math.max(0, n - 2); i < Math.min(lines.length, n + 1); i++) {
      out.push(`${String(i + 1).padStart(4)} | ${lines[i]}`);
    }
  }
  return out.join('\n');
}

function cacheUniforms(gl, program) {
  const uniforms = Object.create(null);
  const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i++) {
    const info = gl.getActiveUniform(program, i);
    if (!info) continue;
    // Array uniforms come back as "name[0]"; store both spellings.
    const name = info.name.replace(/\[0\]$/, '');
    uniforms[name] = gl.getUniformLocation(program, info.name);
  }
  return uniforms;
}

/**
 * @param {WebGL2RenderingContext} gl
 * @param {ArrayBufferView|number} data or a byte length for a sized-but-empty buffer
 */
export function createBuffer(gl, target, data, usage) {
  const buffer = gl.createBuffer();
  gl.bindBuffer(target, buffer);
  gl.bufferData(target, data, usage);
  gl.bindBuffer(target, null);
  return buffer;
}

/**
 * Describe one attribute for `createVAO`.
 * @typedef {{ location: number, size: number, offset: number, divisor?: number }} AttribSpec
 */

/**
 * @param {WebGL2RenderingContext} gl
 * @param {Array<{buffer: WebGLBuffer, stride: number, attribs: AttribSpec[]}>} streams
 * @param {WebGLBuffer|null} indexBuffer
 */
export function createVAO(gl, streams, indexBuffer = null) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  for (const stream of streams) {
    gl.bindBuffer(gl.ARRAY_BUFFER, stream.buffer);
    for (const attrib of stream.attribs) {
      gl.enableVertexAttribArray(attrib.location);
      gl.vertexAttribPointer(
        attrib.location,
        attrib.size,
        gl.FLOAT,
        false,
        stream.stride,
        attrib.offset
      );
      if (attrib.divisor) gl.vertexAttribDivisor(attrib.location, attrib.divisor);
    }
  }

  // The element buffer binding is part of VAO state, so it must be bound
  // while the VAO is bound and must NOT be unbound before the VAO is.
  if (indexBuffer) gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);

  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  return vao;
}

/**
 * A colour target, optionally multisampled with a matching resolve texture.
 *
 * @typedef {{
 *   width: number, height: number,
 *   texture: WebGLTexture,
 *   framebuffer: WebGLFramebuffer,
 *   msaaFramebuffer: WebGLFramebuffer|null,
 *   colorRenderbuffer: WebGLRenderbuffer|null,
 *   depthRenderbuffer: WebGLRenderbuffer|null,
 *   samples: number,
 *   dispose: () => void,
 * }} RenderTarget
 */

/**
 * @param {WebGL2RenderingContext} gl
 * @param {{width:number,height:number,hdr:boolean,depth?:boolean,samples?:number}} options
 * @returns {RenderTarget}
 */
export function createTarget(gl, { width, height, hdr, depth = false, samples = 0 }) {
  const w = Math.max(1, Math.floor(width));
  const h = Math.max(1, Math.floor(height));
  const internalFormat = hdr ? gl.RGBA16F : gl.RGBA8;
  const type = hdr ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;

  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, gl.RGBA, type, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  // CLAMP_TO_EDGE matters for the blur: REPEAT would wrap the bright right
  // edge of the frame onto the left one.
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);

  const framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

  let msaaFramebuffer = null;
  let colorRenderbuffer = null;
  let depthRenderbuffer = null;

  if (samples > 0) {
    msaaFramebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, msaaFramebuffer);

    colorRenderbuffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, colorRenderbuffer);
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, internalFormat, w, h);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, colorRenderbuffer);

    if (depth) {
      depthRenderbuffer = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, depthRenderbuffer);
      gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.DEPTH_COMPONENT24, w, h);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthRenderbuffer);
    }
  } else if (depth) {
    depthRenderbuffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, depthRenderbuffer);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthRenderbuffer);
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindRenderbuffer(gl.RENDERBUFFER, null);

  const target = {
    width: w,
    height: h,
    texture,
    framebuffer,
    msaaFramebuffer,
    colorRenderbuffer,
    depthRenderbuffer,
    samples,
    dispose() {
      gl.deleteTexture(texture);
      gl.deleteFramebuffer(framebuffer);
      if (msaaFramebuffer) gl.deleteFramebuffer(msaaFramebuffer);
      if (colorRenderbuffer) gl.deleteRenderbuffer(colorRenderbuffer);
      if (depthRenderbuffer) gl.deleteRenderbuffer(depthRenderbuffer);
    },
  };

  return target;
}

/** The framebuffer geometry should be drawn into (multisampled when available). */
export function drawFramebuffer(target) {
  return target.msaaFramebuffer || target.framebuffer;
}

/** Resolve multisampled colour into the sampleable texture. */
export function resolveTarget(gl, target) {
  if (!target.msaaFramebuffer) return;
  gl.bindFramebuffer(gl.READ_FRAMEBUFFER, target.msaaFramebuffer);
  gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, target.framebuffer);
  gl.blitFramebuffer(
    0, 0, target.width, target.height,
    0, 0, target.width, target.height,
    gl.COLOR_BUFFER_BIT, gl.NEAREST
  );
  gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
  gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
}

/**
 * A unit cube with outward normals and per-face UVs, centred on the origin
 * with each side one unit long, so `position * extent` yields the extent.
 */
export function unitBoxGeometry() {
  // +X, -X, +Y, -Y, +Z, -Z
  const faces = [
    { n: [1, 0, 0], u: [0, 0, -1], v: [0, 1, 0] },
    { n: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0] },
    { n: [0, 1, 0], u: [1, 0, 0], v: [0, 0, -1] },
    { n: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] },
    { n: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] },
    { n: [0, 0, -1], u: [-1, 0, 0], v: [0, 1, 0] },
  ];

  const vertices = new Float32Array(6 * 4 * 8);
  const indices = new Uint16Array(6 * 6);
  let v = 0;
  let i = 0;

  faces.forEach((face, f) => {
    const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
    for (const [cu, cv] of corners) {
      vertices[v++] = (face.n[0] + face.u[0] * cu + face.v[0] * cv) * 0.5;
      vertices[v++] = (face.n[1] + face.u[1] * cu + face.v[1] * cv) * 0.5;
      vertices[v++] = (face.n[2] + face.u[2] * cu + face.v[2] * cv) * 0.5;
      vertices[v++] = face.n[0];
      vertices[v++] = face.n[1];
      vertices[v++] = face.n[2];
      vertices[v++] = cu * 0.5 + 0.5;
      vertices[v++] = cv * 0.5 + 0.5;
    }
    const base = f * 4;
    indices[i++] = base; indices[i++] = base + 1; indices[i++] = base + 2;
    indices[i++] = base; indices[i++] = base + 2; indices[i++] = base + 3;
  });

  return { vertices, indices, stride: 32 };
}

/** Two triangles covering clip space, for fullscreen passes and billboards. */
export const QUAD_CORNERS = new Float32Array([
  -1, -1, 1, -1, 1, 1,
  -1, -1, 1, 1, -1, 1,
]);
