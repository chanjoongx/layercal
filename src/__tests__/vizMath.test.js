import { describe, it, expect } from 'vitest'
import { mat4, vec3, clamp, damp, easeInOutCubic, easeOutExpo, rayBox } from '@/viz/math'

const near = (a, b, digits = 5) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(b[i], digits)
}

describe('clamp', () => {
  it('bounds on both sides and passes the middle through', () => {
    expect(clamp(-1, 0, 1)).toBe(0)
    expect(clamp(2, 0, 1)).toBe(1)
    expect(clamp(0.5, 0, 1)).toBe(0.5)
  })
})

describe('easing', () => {
  it('pins the endpoints', () => {
    expect(easeInOutCubic(0)).toBe(0)
    expect(easeInOutCubic(1)).toBe(1)
    expect(easeOutExpo(0)).toBe(0)
    expect(easeOutExpo(1)).toBe(1)
  })

  it('is symmetric about the midpoint', () => {
    expect(easeInOutCubic(0.5)).toBeCloseTo(0.5, 6)
    for (const t of [0.1, 0.25, 0.4]) {
      expect(easeInOutCubic(t) + easeInOutCubic(1 - t)).toBeCloseTo(1, 6)
    }
  })

  it('is monotonic', () => {
    let previous = -1
    for (let i = 0; i <= 20; i++) {
      const v = easeInOutCubic(i / 20)
      expect(v).toBeGreaterThanOrEqual(previous)
      previous = v
    }
  })
})

describe('damp', () => {
  it('converges toward the target', () => {
    let v = 0
    for (let i = 0; i < 60; i++) v = damp(v, 10, 9, 1 / 60)
    expect(v).toBeGreaterThan(9.9)
    expect(v).toBeLessThan(10)
  })

  it('is frame-rate independent', () => {
    // The whole reason this is not a fixed per-frame lerp: the same camera has
    // to feel the same at 60 Hz and at 144 Hz.
    let fast = 0
    for (let i = 0; i < 100; i++) fast = damp(fast, 1, 6, 0.01)

    let slow = 0
    for (let i = 0; i < 10; i++) slow = damp(slow, 1, 6, 0.1)

    expect(fast).toBeCloseTo(slow, 3)
  })

  it('does nothing over zero time', () => {
    expect(damp(3, 10, 9, 0)).toBe(3)
  })
})

describe('vec3', () => {
  it('does the usual algebra', () => {
    const out = vec3.create()
    near([...vec3.add(out, [1, 2, 3], [4, 5, 6])], [5, 7, 9])
    near([...vec3.sub(out, [4, 5, 6], [1, 2, 3])], [3, 3, 3])
    near([...vec3.scale(out, [1, 2, 3], 2)], [2, 4, 6])
    expect(vec3.dot([1, 2, 3], [4, 5, 6])).toBe(32)
    expect(vec3.length([3, 4, 0])).toBe(5)
  })

  it('crosses right-handed', () => {
    near([...vec3.cross(vec3.create(), [1, 0, 0], [0, 1, 0])], [0, 0, 1])
  })

  it('normalises to unit length', () => {
    const out = vec3.normalize(vec3.create(), [3, 4, 0])
    expect(vec3.length(out)).toBeCloseTo(1, 6)
  })

  it('returns zero rather than NaN for a zero vector', () => {
    near([...vec3.normalize(vec3.create(), [0, 0, 0])], [0, 0, 0])
  })

  it('lerps the endpoints exactly', () => {
    near([...vec3.lerp(vec3.create(), [0, 0, 0], [2, 4, 6], 0.5)], [1, 2, 3])
    near([...vec3.lerp(vec3.create(), [0, 0, 0], [2, 4, 6], 1)], [2, 4, 6])
  })
})

describe('mat4.perspective', () => {
  it('matches the reference matrix', () => {
    const m = mat4.perspective(mat4.create(), Math.PI / 2, 1, 1, 101)
    // f = 1 / tan(45deg) = 1
    near([...m], [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, -1.02, -1,
      0, 0, -2.02, 0,
    ], 4)
  })

  it('maps the near plane to -1 and the far plane to +1', () => {
    const m = mat4.perspective(mat4.create(), Math.PI / 3, 1.5, 0.5, 50)
    const onNear = mat4.transformPoint(vec3.create(), m, [0, 0, -0.5])
    const onFar = mat4.transformPoint(vec3.create(), m, [0, 0, -50])
    expect(onNear[2]).toBeCloseTo(-1, 5)
    expect(onFar[2]).toBeCloseTo(1, 5)
  })

  it('narrows horizontally as the aspect ratio widens', () => {
    const wide = mat4.perspective(mat4.create(), 1, 2, 0.1, 100)
    const square = mat4.perspective(mat4.create(), 1, 1, 0.1, 100)
    expect(wide[0]).toBeLessThan(square[0])
    expect(wide[5]).toBeCloseTo(square[5], 6)
  })
})

describe('mat4.lookAt', () => {
  it('places the target at the origin of view space', () => {
    const view = mat4.lookAt(mat4.create(), [0, 0, 5], [0, 0, 0], [0, 1, 0])
    near([...mat4.transformPoint(vec3.create(), view, [0, 0, 0])], [0, 0, -5])
  })

  it('keeps up pointing up', () => {
    const view = mat4.lookAt(mat4.create(), [0, 0, 5], [0, 0, 0], [0, 1, 0])
    const above = mat4.transformPoint(vec3.create(), view, [0, 1, 0])
    expect(above[1]).toBeGreaterThan(0)
  })

  it('returns identity rather than NaN when eye and target coincide', () => {
    const m = mat4.lookAt(mat4.create(), [1, 1, 1], [1, 1, 1], [0, 1, 0])
    near([...m], [...mat4.identity(mat4.create())])
  })

  it('does not produce NaN when up is parallel to the view direction', () => {
    const m = mat4.lookAt(mat4.create(), [0, 5, 0], [0, 0, 0], [0, 1, 0])
    for (const v of m) expect(Number.isNaN(v)).toBe(false)
  })
})

describe('mat4.multiply and invert', () => {
  const view = mat4.lookAt(mat4.create(), [3, 4, 5], [0, 1, 0], [0, 1, 0])
  const proj = mat4.perspective(mat4.create(), 0.9, 1.6, 0.1, 120)
  const vp = mat4.multiply(mat4.create(), proj, view)

  it('multiplying by the identity changes nothing', () => {
    const id = mat4.identity(mat4.create())
    near([...mat4.multiply(mat4.create(), vp, id)], [...vp])
    near([...mat4.multiply(mat4.create(), id, vp)], [...vp])
  })

  it('applies the right operand first', () => {
    // proj * view means: transform by view, then project.
    const direct = mat4.transformPoint(vec3.create(), vp, [1, 2, 3])
    const staged = mat4.transformPoint(
      vec3.create(), proj,
      mat4.transformPoint(vec3.create(), view, [1, 2, 3])
    )
    near([...direct], [...staged], 4)
  })

  it('inverts to the identity', () => {
    const inv = mat4.invert(mat4.create(), vp)
    expect(inv).not.toBeNull()
    near([...mat4.multiply(mat4.create(), vp, inv)], [...mat4.identity(mat4.create())], 4)
  })

  it('round-trips a point through the inverse', () => {
    const inv = mat4.invert(mat4.create(), vp)
    const p = [0.4, -1.2, 2.5]
    const ndc = mat4.transformPoint(vec3.create(), vp, p)
    near([...mat4.transformPoint(vec3.create(), inv, ndc)], p, 4)
  })

  it('returns null for a singular matrix', () => {
    const zero = new Float32Array(16)
    expect(mat4.invert(mat4.create(), zero)).toBeNull()
  })
})

describe('rayBox', () => {
  const min = [-1, -1, -1]
  const max = [1, 1, 1]

  it('hits a box straight ahead and reports the near distance', () => {
    expect(rayBox([0, 0, -5], [0, 0, 1], min, max)).toBeCloseTo(4, 5)
  })

  it('misses a box off to the side', () => {
    expect(rayBox([5, 5, -5], [0, 0, 1], min, max)).toBe(-1)
  })

  it('ignores a box behind the ray', () => {
    expect(rayBox([0, 0, 5], [0, 0, 1], min, max)).toBe(-1)
  })

  it('reports the exit distance when the origin is inside', () => {
    expect(rayBox([0, 0, 0], [0, 0, 1], min, max)).toBeCloseTo(1, 5)
  })

  it('handles an axis-parallel ray without dividing by zero', () => {
    expect(Number.isNaN(rayBox([0, 5, 0], [1, 0, 0], min, max))).toBe(false)
  })

  it('hits on a diagonal', () => {
    const d = vec3.normalize(vec3.create(), [1, 1, 1])
    expect(rayBox([-5, -5, -5], [...d], min, max)).toBeGreaterThan(0)
  })
})
