import { describe, it, expect } from 'vitest'
import { createCamera, frameBounds, settle, updateCamera, orbit } from '@/viz/camera'
import { mat4 } from '@/viz/math'

/**
 * The framing solve, tested through the matrix it actually produces rather than
 * through its internals. Two things have to hold for every model and every panel
 * shape: nothing is ever cut off, and the model is never left as a speck in the
 * middle of an empty grid. Both have regressed before.
 */

/** A box `long` deep along the flow axis and `wide` x `tall` in cross-section. */
const boundsOf = (wide, tall, long) => ({
  min: [-wide / 2, -tall / 2, -long / 2],
  max: [wide / 2, tall / 2, long / 2],
  center: [0, 0, 0],
  radius: Math.hypot(wide, tall, long) / 2,
})

const cornersOf = (b) => {
  const out = []
  for (let i = 0; i < 8; i++) {
    out.push([
      i & 1 ? b.max[0] : b.min[0],
      i & 2 ? b.max[1] : b.min[1],
      i & 4 ? b.max[2] : b.min[2],
    ])
  }
  return out
}

/** Frame the box, then report where its corners land in clip space. */
function frameAndProject(bounds, aspect) {
  const camera = createCamera()
  frameBounds(camera, bounds, aspect)
  settle(camera)
  updateCamera(camera, 1 / 60, aspect, false)

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  const p = [0, 0, 0]
  for (const corner of cornersOf(bounds)) {
    mat4.transformPoint(p, camera.viewProjection, corner)
    minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0])
    minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1])
  }
  return {
    camera,
    minX, maxX, minY, maxY,
    // The frame spans -1..1, so these are fractions of the panel.
    fillX: (maxX - minX) / 2,
    fillY: (maxY - minY) / 2,
    // 0 is square-on to the flow axis; larger is more turned. Invariant under
    // the quadrant symmetries the composer is free to use.
    obliqueness: Math.abs(Math.cos(camera.desiredTheta)),
  }
}

// A short stack, a typical one, a very long one, and a wide flat one - across a
// phone, a laptop and an ultrawide panel.
const SHAPES = [
  ['4 layers', boundsOf(2.6, 2.6, 5)],
  ['16 layers', boundsOf(2.6, 2.6, 22)],
  ['60 layers', boundsOf(2.6, 2.6, 84)],
  ['one layer', boundsOf(2.6, 2.6, 1.2)],
]
const ASPECTS = [['phone', 1.3], ['laptop', 2.69], ['ultrawide', 4.2]]

describe('framing', () => {
  it('never cuts off any corner of the model', () => {
    const clipped = []
    for (const [name, bounds] of SHAPES) {
      for (const [panel, aspect] of ASPECTS) {
        const r = frameAndProject(bounds, aspect)
        if (r.minX < -1 || r.maxX > 1 || r.minY < -1 || r.maxY > 1) {
          clipped.push(`${name} on ${panel}: x ${r.minX.toFixed(2)}..${r.maxX.toFixed(2)} y ${r.minY.toFixed(2)}..${r.maxY.toFixed(2)}`)
        }
      }
    }
    expect(clipped).toEqual([])
  })

  it('leaves only a small margin on the constraining axis', () => {
    // The fit is exact plus a margin for the floating labels. If this drops, the
    // model is adrift in the panel; if it exceeds 1 the test above fails.
    const loose = []
    for (const [name, bounds] of SHAPES) {
      for (const [panel, aspect] of ASPECTS) {
        const r = frameAndProject(bounds, aspect)
        const fill = Math.max(r.fillX, r.fillY)
        if (fill < 0.82) loose.push(`${name} on ${panel}: ${(fill * 100).toFixed(0)}%`)
      }
    }
    expect(loose).toEqual([])
  })

  it('covers a real share of the panel, not a band across the middle', () => {
    // Square-on, a 16-layer chain covered 35% of the height of a laptop panel.
    // The composer turns the camera until the model's projected shape is closer
    // to the panel's, which is what this number is measuring.
    const r = frameAndProject(boundsOf(2.6, 2.6, 22), 2.69)
    expect(r.fillY).toBeGreaterThan(0.5)
    expect(r.fillX).toBeGreaterThan(0.5)
  })

  it('turns further for a long chain than for a short one', () => {
    const short = frameAndProject(boundsOf(2.6, 2.6, 5), 2.69)
    const long = frameAndProject(boundsOf(2.6, 2.6, 22), 2.69)
    expect(long.obliqueness).toBeGreaterThan(short.obliqueness)
  })

  it('fills the panel for a compact model too', () => {
    // A single layer is about as tall as it is wide, so it can never span an
    // ultrawide panel - height is the binding axis and that is the one to check.
    // The solve turns this one to three-quarters, which is both wider on screen
    // and the angle that reads as a solid rather than a rectangle; what matters
    // is that it ends up large, so that is what is asserted.
    for (const aspect of [1.3, 2.69, 4.2]) {
      const r = frameAndProject(boundsOf(2.6, 2.6, 1.2), aspect)
      expect(Math.max(r.fillX, r.fillY)).toBeGreaterThan(0.85)
    }
  })

  it('adapts to the panel: a phone gets a more turned view than an ultrawide', () => {
    const phone = frameAndProject(boundsOf(2.6, 2.6, 22), 1.3)
    const wide = frameAndProject(boundsOf(2.6, 2.6, 22), 4.2)
    expect(phone.obliqueness).toBeGreaterThan(wide.obliqueness)
  })
})

describe('who owns the orbit angle', () => {
  it('composes the angle until the viewer touches it', () => {
    const camera = createCamera()
    expect(camera.userPosed).toBe(false)
    frameBounds(camera, boundsOf(2.6, 2.6, 22), 2.69)
    const composed = camera.desiredTheta

    // Re-framing an untouched camera may recompose freely.
    frameBounds(camera, boundsOf(2.6, 2.6, 22), 2.69)
    expect(camera.desiredTheta).toBeCloseTo(composed, 6)
  })

  it('keeps the angle once the viewer has orbited', () => {
    const camera = createCamera()
    frameBounds(camera, boundsOf(2.6, 2.6, 22), 2.69)

    orbit(camera, 0.4, 0.1)
    expect(camera.userPosed).toBe(true)
    const chosen = camera.desiredTheta

    // Adding a layer re-frames the distance but must not take the angle back.
    frameBounds(camera, boundsOf(2.6, 2.6, 26), 2.69)
    expect(camera.desiredTheta).toBeCloseTo(chosen, 6)
  })

  it('still fits the model from an angle the viewer chose', () => {
    const bounds = boundsOf(2.6, 2.6, 22)
    const camera = createCamera()
    orbit(camera, 0.9, 0.2)
    frameBounds(camera, bounds, 2.69)
    settle(camera)
    updateCamera(camera, 1 / 60, 2.69, false)

    const p = [0, 0, 0]
    for (const corner of cornersOf(bounds)) {
      mat4.transformPoint(p, camera.viewProjection, corner)
      expect(Math.abs(p[0])).toBeLessThanOrEqual(1)
      expect(Math.abs(p[1])).toBeLessThanOrEqual(1)
    }
  })
})

describe('the overlay safe area', () => {
  // The metric chips run across the top of the panel and the family legend
  // across the bottom. Framing to the raw canvas put the first layer's label
  // behind the chips, so the solve is given the free region instead.
  const INSETS = { top: 0.12, bottom: 0.09 }

  const projectWithInsets = (bounds, aspect) => {
    const camera = createCamera()
    frameBounds(camera, bounds, aspect, INSETS)
    settle(camera)
    updateCamera(camera, 1 / 60, aspect, false)
    let minY = Infinity, maxY = -Infinity
    const p = [0, 0, 0]
    for (const corner of cornersOf(bounds)) {
      mat4.transformPoint(p, camera.viewProjection, corner)
      minY = Math.min(minY, p[1])
      maxY = Math.max(maxY, p[1])
    }
    return { minY, maxY }
  }

  it('keeps every model inside the free part of the panel', () => {
    const outside = []
    for (const [name, bounds] of SHAPES) {
      for (const [panel, aspect] of ASPECTS) {
        const { minY, maxY } = projectWithInsets(bounds, aspect)
        if (maxY > 1 - 2 * INSETS.top + 1e-6 || minY < -1 + 2 * INSETS.bottom - 1e-6) {
          outside.push(`${name} on ${panel}: ${minY.toFixed(3)}..${maxY.toFixed(3)}`)
        }
      }
    }
    expect(outside).toEqual([])
  })

  it('still uses the room it does have', () => {
    // Reserving space must not turn into backing the camera off: the model
    // should nearly span the free region.
    const { minY, maxY } = projectWithInsets(boundsOf(2.6, 2.6, 22), 2.69)
    const free = (1 - 2 * INSETS.top) - (-1 + 2 * INSETS.bottom)
    expect((maxY - minY) / free).toBeGreaterThan(0.7)
  })

  it('ignores insets that would leave nothing to frame into', () => {
    const bounds = boundsOf(2.6, 2.6, 22)
    const camera = createCamera()
    frameBounds(camera, bounds, 2.69, { top: 5, bottom: 5 })
    expect(Number.isFinite(camera.desiredRadius)).toBe(true)
    expect(camera.desiredRadius).toBeGreaterThan(0)
  })
})
