import { describe, it, expect } from 'vitest'
import {
  LAYER_PALETTE, LAYER_FAMILIES, FAMILY_LABELS, FALLBACK_PAINT,
  paintFor, srgbToLinear, toLinear, paletteStyle,
} from '@/viz/palette'
import { LAYER_TYPE_IDS } from '@/config/layerTypes'

const HEX = /^#[0-9a-f]{6}$/

describe('layer palette coverage', () => {
  it('paints every layer type the app can build', () => {
    for (const id of LAYER_TYPE_IDS) {
      expect(LAYER_PALETTE[id], `missing paint for ${id}`).toBeDefined()
    }
  })

  it('does not carry paint for a type that no longer exists', () => {
    const known = new Set(LAYER_TYPE_IDS)
    for (const id of Object.keys(LAYER_PALETTE)) {
      expect(known.has(id), `stale paint for ${id}`).toBe(true)
    }
  })

  it('uses only declared families', () => {
    const families = new Set(LAYER_FAMILIES)
    for (const [id, paint] of Object.entries(LAYER_PALETTE)) {
      expect(families.has(paint.family), `${id} has family ${paint.family}`).toBe(true)
    }
  })

  it('labels every family', () => {
    for (const family of LAYER_FAMILIES) {
      expect(FAMILY_LABELS[family]).toBeTruthy()
    }
  })
})

describe('paint values', () => {
  const entries = Object.entries(LAYER_PALETTE)

  it.each(entries)('%s has channels inside the unit range', (_id, paint) => {
    for (const channel of [...paint.base, ...paint.glow]) {
      expect(channel).toBeGreaterThanOrEqual(0)
      expect(channel).toBeLessThanOrEqual(1)
    }
  })

  it.each(entries)('%s has well-formed hex strings', (_id, paint) => {
    expect(paint.hex).toMatch(HEX)
    expect(paint.hexDark).toMatch(HEX)
  })

  it('makes the glow lighter than the body, so it reads as light from inside', () => {
    for (const [id, paint] of Object.entries(LAYER_PALETTE)) {
      const sum = (c) => c[0] + c[1] + c[2]
      expect(sum(paint.glow), `${id} glow is not lighter than its base`)
        .toBeGreaterThan(sum(paint.base))
    }
  })
})

describe('paintFor', () => {
  it('returns the declared paint for a known type', () => {
    expect(paintFor('conv2d')).toBe(LAYER_PALETTE.conv2d)
  })

  it('falls back rather than throwing on an unknown type', () => {
    expect(paintFor('quantum_flux')).toBe(FALLBACK_PAINT)
    expect(paintFor(undefined)).toBe(FALLBACK_PAINT)
  })
})

describe('srgbToLinear', () => {
  it('pins the endpoints', () => {
    expect(srgbToLinear(0)).toBe(0)
    expect(srgbToLinear(1)).toBeCloseTo(1, 6)
  })

  it('uses the linear segment below the knee', () => {
    // The piecewise form matters at the dark end, which is where the fresnel
    // rim colours live; pow(c, 2.2) is ~15% too dark there.
    expect(srgbToLinear(0.04)).toBeCloseTo(0.04 / 12.92, 6)
    expect(srgbToLinear(0.04)).toBeGreaterThan(Math.pow(0.04, 2.2))
  })

  it('is monotonic', () => {
    let previous = -1
    for (let i = 0; i <= 20; i++) {
      const v = srgbToLinear(i / 20)
      expect(v).toBeGreaterThan(previous)
      previous = v
    }
  })

  it('darkens the midtones, as a decode should', () => {
    expect(srgbToLinear(0.5)).toBeLessThan(0.5)
    expect(srgbToLinear(0.5)).toBeCloseTo(0.2140, 3)
  })
})

describe('toLinear', () => {
  it('converts all three channels', () => {
    expect(toLinear([0, 0.5, 1])).toEqual([
      srgbToLinear(0), srgbToLinear(0.5), srgbToLinear(1),
    ])
  })
})

describe('paletteStyle', () => {
  it('emits the hex, its rgb triple and every tinted role', () => {
    const light = paletteStyle('conv2d', false)
    expect(light['--layer']).toBe(LAYER_PALETTE.conv2d.hex)
    expect(light['--layer-rgb']).toMatch(/^\d{1,3} \d{1,3} \d{1,3}$/)
    for (const key of ['--layer-fill', '--layer-fill-hover', '--layer-line']) {
      expect(light[key], key).toMatch(/^rgba\(\d{1,3}, \d{1,3}, \d{1,3}, [\d.]+\)$/)
    }
  })

  it('uses only colour syntax html2canvas can parse', () => {
    // color-mix() resolves to an oklab() colour in Chrome's computed style, and
    // html2canvas throws on any function it does not know — which took the whole
    // PNG export down, not just one tint.
    for (const dark of [false, true]) {
      for (const value of Object.values(paletteStyle('transformer', dark))) {
        expect(String(value)).not.toMatch(/color-mix|oklab|oklch/)
      }
    }
  })

  it('swaps to the dark variant', () => {
    expect(paletteStyle('conv2d', true)['--layer']).toBe(LAYER_PALETTE.conv2d.hexDark)
  })

  it('does not throw for an unknown type', () => {
    expect(() => paletteStyle('nope', false)).not.toThrow()
  })
})
