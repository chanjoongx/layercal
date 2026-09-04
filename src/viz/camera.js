/**
 * Orbit camera with damped input.
 *
 * Every gesture writes a *desired* value; the render loop damps the actual
 * value toward it with a frame-rate-independent filter. That is the difference
 * between a camera that feels like a physical object and one that feels like a
 * spreadsheet cell.
 */

import { mat4, vec3, clamp, damp, easeInOutCubic } from './math';

const TAU = Math.PI * 2;
const POLE_MARGIN = 0.12;

/** Seconds of stillness before the camera starts drifting on its own. */
const IDLE_DELAY = 4;
const DRIFT_RATE = 0.055; // rad/s

export function createCamera() {
  const state = {
    fovy: (42 * Math.PI) / 180,
    near: 0.05,
    far: 200,

    // Three-quarter view from above the flow axis: the stack reads as a
    // sequence rather than as a single silhouette.
    theta: -1.02,
    phi: 1.12,
    radius: 8,
    target: vec3.set(vec3.create(), 0, 0.9, 0),

    desiredTheta: -1.02,
    desiredPhi: 1.12,
    desiredRadius: 8,
    desiredTarget: vec3.set(vec3.create(), 0, 0.9, 0),

    minRadius: 1,
    maxRadius: 60,

    idleFor: 0,
    driftEnabled: true,

    // In-flight focus animation, or null.
    flight: null,

    eye: vec3.create(),
    view: mat4.create(),
    projection: mat4.create(),
    viewProjection: mat4.create(),
    inverseViewProjection: mat4.create(),
    aspect: 1,
  };

  return state;
}

/**
 * Frame the scene so every node fits.
 *
 * The naive version fits the bounding *sphere* into the vertical field of view.
 * That is wrong for this app's normal case by a wide margin: a 16-layer stack
 * is a long, thin box, its bounding sphere is dominated by its length, and
 * fitting that sphere vertically backs the camera off far enough to render the
 * model as a smudge in the middle of an empty grid.
 *
 * So project the actual box half-extents onto the camera's own right/up/forward
 * axes and fit those. The result is tight on both axes at any orbit angle.
 */
export function frameBounds(camera, bounds, aspect) {
  const right = vec3.create();
  const up = vec3.create();
  const forward = vec3.create();
  orbitBasis(camera.desiredTheta, camera.desiredPhi, right, up, forward);

  const half = [
    Math.max((bounds.max[0] - bounds.min[0]) / 2, 0.2),
    Math.max((bounds.max[1] - bounds.min[1]) / 2, 0.2),
    Math.max((bounds.max[2] - bounds.min[2]) / 2, 0.2),
  ];

  // |h . axis| summed over the three axes is the half-extent of an AABB along
  // an arbitrary direction.
  const extentAlong = (axis) =>
    Math.abs(half[0] * axis[0]) + Math.abs(half[1] * axis[1]) + Math.abs(half[2] * axis[2]);

  const halfWidth = extentAlong(right);
  const halfHeight = extentAlong(up);
  const halfDepth = extentAlong(forward);

  const tanV = Math.tan(camera.fovy / 2);
  const tanH = tanV * Math.max(aspect, 0.25);

  // The extra margin is not padding for its own sake: the floating HTML labels
  // sit above each slab and would be clipped by a frame that fits exactly.
  const distance = Math.max(halfHeight / tanV, halfWidth / tanH) * 1.22 + halfDepth + 0.4;

  camera.minRadius = Math.max(0.6, distance * 0.14);
  camera.maxRadius = distance * 5;
  camera.desiredRadius = clamp(distance, camera.minRadius, camera.maxRadius);

  vec3.set(camera.desiredTarget, bounds.center[0], bounds.center[1], bounds.center[2]);
  camera.flight = null;
}

/** Orbit angles to the camera's right / up / backward (target-to-eye) axes. */
function orbitBasis(theta, phi, right, up, forward) {
  const sinPhi = Math.sin(phi);
  vec3.set(forward, sinPhi * Math.sin(theta), Math.cos(phi), sinPhi * Math.cos(theta));
  vec3.normalize(forward, forward);
  vec3.normalize(right, vec3.cross(right, [0, 1, 0], forward));
  vec3.normalize(up, vec3.cross(up, forward, right));
}

/** Snap the animated values onto their targets, for the first frame. */
export function settle(camera) {
  camera.theta = camera.desiredTheta;
  camera.phi = camera.desiredPhi;
  camera.radius = camera.desiredRadius;
  vec3.copy(camera.target, camera.desiredTarget);
}

export function orbit(camera, dTheta, dPhi) {
  camera.desiredTheta += dTheta;
  camera.desiredPhi = clamp(camera.desiredPhi + dPhi, POLE_MARGIN, Math.PI - POLE_MARGIN);
  camera.idleFor = 0;
  camera.flight = null;
}

export function dolly(camera, factor) {
  camera.desiredRadius = clamp(camera.desiredRadius * factor, camera.minRadius, camera.maxRadius);
  camera.idleFor = 0;
  camera.flight = null;
}

/** Pan in the camera's screen plane, scaled so the drag tracks the cursor. */
export function pan(camera, dx, dy) {
  const right = vec3.create();
  const up = vec3.create();
  cameraBasis(camera, right, up);

  const scale = camera.radius * 0.0018;
  camera.desiredTarget[0] += (-right[0] * dx + up[0] * dy) * scale;
  camera.desiredTarget[1] += (-right[1] * dx + up[1] * dy) * scale;
  camera.desiredTarget[2] += (-right[2] * dx + up[2] * dy) * scale;
  camera.idleFor = 0;
  camera.flight = null;
}

/** Animate to look at one node. */
export function focusOn(camera, node, duration = 0.64) {
  const extent = Math.max(node.extent.w, node.extent.h, node.extent.d);
  camera.flight = {
    elapsed: 0,
    duration,
    fromTarget: vec3.set(vec3.create(), camera.target[0], camera.target[1], camera.target[2]),
    toTarget: vec3.set(vec3.create(), node.center.x, node.center.y, node.center.z),
    fromRadius: camera.radius,
    toRadius: clamp(extent * 4.2 + 1.6, camera.minRadius, camera.maxRadius),
  };
  camera.idleFor = 0;
}

function cameraBasis(camera, right, up) {
  orbitBasis(camera.theta, camera.phi, right, up, vec3.create());
}

/**
 * Advance the camera and rebuild its matrices.
 *
 * @param {number} dt seconds
 * @param {number} aspect viewport width / height
 * @param {boolean} motion false under prefers-reduced-motion: no drift, no easing
 */
export function updateCamera(camera, dt, aspect, motion = true) {
  camera.aspect = aspect;

  if (camera.flight) {
    const flight = camera.flight;
    flight.elapsed += dt;
    const t = motion ? clamp(flight.elapsed / flight.duration, 0, 1) : 1;
    const e = easeInOutCubic(t);
    vec3.lerp(camera.desiredTarget, flight.fromTarget, flight.toTarget, e);
    camera.desiredRadius = flight.fromRadius + (flight.toRadius - flight.fromRadius) * e;
    if (t >= 1) camera.flight = null;
  }

  if (motion) {
    camera.idleFor += dt;
    if (camera.driftEnabled && camera.idleFor > IDLE_DELAY) {
      camera.desiredTheta += DRIFT_RATE * dt;
    }
    camera.theta = damp(camera.theta, camera.desiredTheta, 9, dt);
    camera.phi = damp(camera.phi, camera.desiredPhi, 9, dt);
    camera.radius = damp(camera.radius, camera.desiredRadius, 9, dt);
    camera.target[0] = damp(camera.target[0], camera.desiredTarget[0], 9, dt);
    camera.target[1] = damp(camera.target[1], camera.desiredTarget[1], 9, dt);
    camera.target[2] = damp(camera.target[2], camera.desiredTarget[2], 9, dt);
  } else {
    settle(camera);
  }

  // Keep theta bounded so it never loses float precision in a long session.
  if (camera.theta > TAU || camera.theta < -TAU) {
    const wrap = Math.floor(camera.theta / TAU) * TAU;
    camera.theta -= wrap;
    camera.desiredTheta -= wrap;
  }

  const sinPhi = Math.sin(camera.phi);
  camera.eye[0] = camera.target[0] + camera.radius * sinPhi * Math.sin(camera.theta);
  camera.eye[1] = camera.target[1] + camera.radius * Math.cos(camera.phi);
  camera.eye[2] = camera.target[2] + camera.radius * sinPhi * Math.cos(camera.theta);

  // Far plane tracks the orbit distance: a fixed 200 with a 0.05 near plane
  // spends most of the depth buffer's precision on empty space and z-fights the
  // thin annotation plates.
  camera.near = Math.max(0.02, camera.radius * 0.02);
  camera.far = camera.radius * 6 + 40;

  mat4.perspective(camera.projection, camera.fovy, aspect, camera.near, camera.far);
  mat4.lookAt(camera.view, camera.eye, camera.target, [0, 1, 0]);
  mat4.multiply(camera.viewProjection, camera.projection, camera.view);
  mat4.invert(camera.inverseViewProjection, camera.viewProjection);

  return camera;
}

/**
 * Build a world-space ray through a normalised device coordinate.
 * @returns {{origin: Float32Array, dir: Float32Array}}
 */
export function rayFromNDC(camera, ndcX, ndcY) {
  const near = vec3.create();
  const far = vec3.create();
  mat4.transformPoint(near, camera.inverseViewProjection, [ndcX, ndcY, -1]);
  mat4.transformPoint(far, camera.inverseViewProjection, [ndcX, ndcY, 1]);
  const dir = vec3.create();
  vec3.normalize(dir, vec3.sub(dir, far, near));
  return { origin: near, dir };
}
