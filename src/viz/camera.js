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

    // Nearly perpendicular to the flow axis. The obliqueness matters more than
    // it looks: at a three-quarter angle one end of a long stack is much nearer
    // than the other, so fitting the near end leaves the far end at two thirds
    // of the frame and the panel looks half empty. Square-on, both ends sit at
    // almost the same depth and the model fills the width. The remaining tilt
    // is what keeps it reading as three-dimensional.
    theta: -1.42,
    phi: 1.16,
    radius: 8,
    target: vec3.set(vec3.create(), 0, 0.9, 0),

    desiredTheta: -1.42,
    desiredPhi: 1.16,
    desiredRadius: 8,
    desiredTarget: vec3.set(vec3.create(), 0, 0.9, 0),

    minRadius: 1,
    maxRadius: 60,

    idleFor: 0,
    userPosed: false,
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

/** Margin enough for the floating HTML labels, and no more. */
const FIT_MARGIN = 1.035;
const FIT_PAD = 0.14;

/** Square-on to the flow axis, and the most oblique the composer may go. */
const COMPOSE_SQUARE = -Math.PI / 2;
const COMPOSE_OBLIQUE = -0.86;

// Scratch vectors: the framing solve scans a couple of dozen candidate angles
// and would otherwise allocate three vectors for each one.
const fitRight = vec3.create();
const fitUp = vec3.create();
const fitForward = vec3.create();
const fitCorners = new Float64Array(24);

/**
 * Fit the bounding box at one orbit angle.
 *
 * The naive version fits the bounding *sphere* into the vertical field of view.
 * That is wrong for this app's normal case by a wide margin: a 16-layer stack is
 * a long, thin box, its bounding sphere is dominated by its length, and fitting
 * that sphere vertically backs the camera off far enough to render the model as
 * a smudge in the middle of an empty grid.
 *
 * So solve per corner instead. A corner sits at depth `f` in front of the
 * target, so it fits when `distance >= |offset| / tan + f`; the maximum over the
 * eight corners is exact and tight at any angle. Summing the half-extents is the
 * other near-miss - it adds the box's whole depth, which for a long stack seen
 * nearly side-on is a quarter of its length, and the camera ends up that much
 * too far back.
 *
 * @returns {{distance: number, coverage: number}} `coverage` is the fraction of
 *   the panel's area the model covers once fitted, which is what the composer
 *   maximises.
 */
function fitAt(half, theta, phi, tanV, tanH, halfY, midY) {
  orbitBasis(theta, phi, fitRight, fitUp, fitForward);

  for (let i = 0; i < 8; i++) {
    const cx = (i & 1 ? half[0] : -half[0]);
    const cy = (i & 2 ? half[1] : -half[1]);
    const cz = (i & 4 ? half[2] : -half[2]);
    fitCorners[i * 3] = cx * fitRight[0] + cy * fitRight[1] + cz * fitRight[2];
    fitCorners[i * 3 + 1] = cx * fitUp[0] + cy * fitUp[1] + cz * fitUp[2];
    fitCorners[i * 3 + 2] = cx * fitForward[0] + cy * fitForward[1] + cz * fitForward[2];
  }

  // Fitting alone is not enough to use the panel. Under perspective the near end
  // of a long stack is magnified, so a frame that fits every corner has the
  // model touching one edge with a quarter of the panel empty at the other - it
  // measured 26% dead on the left against 4% on the right. So solve for the
  // aim point as well: fit, see where the projection actually sits, slide the
  // target to centre it, and fit again. Three passes is comfortably convergent.
  // The panel is not all usable: the metric chips sit across the top and the
  // family legend across the bottom, and the layer labels ride above the layers
  // they name. Fitting to the raw canvas tucked the first label behind the
  // chips. `halfY` is the share of the frame's height that is actually free and
  // `midY` where the middle of it sits.
  const tanVFit = tanV * halfY;

  let offsetR = 0;
  let offsetU = 0;
  let distance = 0;
  let coverage = 0;

  for (let pass = 0; pass < 3; pass++) {
    distance = 0;
    for (let i = 0; i < 8; i++) {
      const r = fitCorners[i * 3] - offsetR;
      const u = fitCorners[i * 3 + 1] - offsetU;
      const f = fitCorners[i * 3 + 2];
      distance = Math.max(distance, Math.abs(r) / tanH + f, Math.abs(u) / tanVFit + f);
    }
    distance = distance * FIT_MARGIN + FIT_PAD;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < 8; i++) {
      const z = Math.max(distance - fitCorners[i * 3 + 2], 1e-3);
      const x = (fitCorners[i * 3] - offsetR) / (z * tanH);
      const y = (fitCorners[i * 3 + 1] - offsetU) / (z * tanVFit);
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    // The frame spans -1..1 on both axes, so a span of 2 covers it completely.
    coverage = ((maxX - minX) / 2) * ((maxY - minY) / 2);

    const errorX = (minX + maxX) / 2;
    const errorY = (minY + maxY) / 2;
    if (Math.abs(errorX) < 1e-4 && Math.abs(errorY) < 1e-4) break;
    offsetR += errorX * distance * tanH;
    offsetU += errorY * distance * tanVFit;
  }

  // Finally bias the model into the free part of the frame. Moving the target
  // up moves the model down the screen, which is why this is negated.
  return { distance, coverage, offsetR, offsetU: offsetU - midY * distance * tanV };
}

/**
 * The orbit azimuth that fills the panel best.
 *
 * Seen square-on, a sixteen-layer chain is about six times wider than it is
 * tall. In a three-to-one panel that wastes two thirds of the height, which is
 * the single biggest reason the view used to look empty. Turning the camera
 * toward the end of the chain foreshortens it into something closer to the
 * panel's own proportions.
 *
 * It has to be solved rather than tuned, because the right angle depends on the
 * model: the same rotation that gains a long chain twenty-five points of height
 * costs a four-layer model - already about as tall as it is wide - a third of
 * its width. The scan lands the short one back at square-on on its own.
 */
function composeTheta(half, phi, tanV, tanH, halfY, midY, current) {
  let bestTheta = COMPOSE_SQUARE;
  let best = -1;
  const steps = 24;
  for (let i = 0; i <= steps; i++) {
    const theta = COMPOSE_SQUARE + (i / steps) * (COMPOSE_OBLIQUE - COMPOSE_SQUARE);
    const { coverage } = fitAt(half, theta, phi, tanV, tanH, halfY, midY);
    if (coverage > best) {
      best = coverage;
      bestTheta = theta;
    }
  }

  // A box looks the same from any of four quadrants, so take the equivalent
  // angle nearest where the camera already is. Otherwise adding one layer can
  // swing the model half a turn, which reads as the view being taken away.
  let nearest = bestTheta;
  let nearestDistance = Infinity;
  for (let sign = -1; sign <= 1; sign += 2) {
    for (let k = -2; k <= 2; k++) {
      const candidate = sign * bestTheta + k * Math.PI;
      const d = Math.abs(candidate - current);
      if (d < nearestDistance) {
        nearestDistance = d;
        nearest = candidate;
      }
    }
  }
  return nearest;
}

/**
 * Frame the scene so every node fits, and - until the viewer takes hold of the
 * camera - from the angle that fills the panel best.
 */
export function frameBounds(camera, bounds, aspect, insets) {
  const half = [
    Math.max((bounds.max[0] - bounds.min[0]) / 2, 0.2),
    Math.max((bounds.max[1] - bounds.min[1]) / 2, 0.2),
    Math.max((bounds.max[2] - bounds.min[2]) / 2, 0.2),
  ];

  const tanV = Math.tan(camera.fovy / 2);
  const tanH = tanV * Math.max(aspect, 0.25);

  // Overlay chrome, as fractions of the panel's height. Clamped so a very short
  // panel cannot reserve so much that nothing is left to frame into.
  const top = clamp(insets && insets.top ? insets.top : 0, 0, 0.3);
  const bottom = clamp(insets && insets.bottom ? insets.bottom : 0, 0, 0.3);
  const yMax = 1 - 2 * top;
  const yMin = -1 + 2 * bottom;
  const halfY = (yMax - yMin) / 2;
  const midY = (yMax + yMin) / 2;

  // Once someone has orbited, the angle is theirs and the fit only moves the
  // distance and the centre.
  if (!camera.userPosed) {
    camera.desiredTheta = composeTheta(half, camera.desiredPhi, tanV, tanH, halfY, midY, camera.desiredTheta);
  }

  const fit = fitAt(half, camera.desiredTheta, camera.desiredPhi, tanV, tanH, halfY, midY);

  camera.minRadius = Math.max(0.6, fit.distance * 0.14);
  camera.maxRadius = fit.distance * 5;
  camera.desiredRadius = clamp(fit.distance, camera.minRadius, camera.maxRadius);

  // fitRight / fitUp still hold the basis for the angle just fitted.
  vec3.set(
    camera.desiredTarget,
    bounds.center[0] + fit.offsetR * fitRight[0] + fit.offsetU * fitUp[0],
    bounds.center[1] + fit.offsetR * fitRight[1] + fit.offsetU * fitUp[1],
    bounds.center[2] + fit.offsetR * fitRight[2] + fit.offsetU * fitUp[2]
  );
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
  // From here on the framing solve keeps the angle instead of choosing one.
  camera.userPosed = true;
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
