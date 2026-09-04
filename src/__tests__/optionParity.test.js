import { describe, it, expect } from 'vitest'
import { buildPrompt, parseAndValidateLayers } from '@/utils/ragPipeline'
import { getLayerTypes } from '@/config/layerTypes'

/**
 * The legal values for a parameter are written down in three places: the field
 * definitions the `<select>` renders, the schema text in the LLM prompt, and the
 * snapping table that validates what the LLM sends back. Nothing linked them,
 * so widening one list and forgetting another produced exactly the failure this
 * project keeps hitting - a value the UI offers, the advisor is never told
 * about, and the validator silently rewrites.
 *
 * These are behavioural checks rather than an import of the private tables: the
 * schema is read back out of `buildPrompt`, and the snapping table is probed
 * through `parseAndValidateLayers`.
 */

const LAYER_TYPES = getLayerTypes(new Proxy({}, {
  get: (_, key) => new Proxy({ name: String(key), description: '' }, {
    get: (target, k) => (k in target ? target[k] : String(k)),
  }),
}))

/** Every numeric `<select>` in the builder, as [type, key, options]. */
const NUMERIC_FIELDS = Object.entries(LAYER_TYPES).flatMap(([type, cfg]) =>
  (cfg.fields || [])
    .filter(f => f.type === 'select' && (f.options || []).every(o => typeof o === 'number'))
    .map(f => [type, f.key, f.options]))

/** `  - key: one of [1, 2, 3]` lines from the schema block of the prompt. */
function schemaOptions() {
  // buildPrompt returns the request parts, not one string.
  const lines = Object.values(buildPrompt('a small convolutional network', []))
    .filter(v => typeof v === 'string')
    .flatMap(v => v.split(/\r?\n/))

  const found = new Map()
  let type = null
  for (const line of lines) {
    const header = line.match(/^([a-z0-9]+):$/)
    if (header) { type = header[1]; continue }
    const field = line.match(/^\s+- (\w+): one of \[([^\]]*)\]/)
    if (field && type) {
      found.set(type + '.' + field[1], field[2].split(',').map(v => Number(v.trim())))
    }
  }
  return found
}

describe('parameter option lists agree across all three sources', () => {
  it('covers every numeric field, so the sweep below is not vacuous', () => {
    expect(NUMERIC_FIELDS.length).toBeGreaterThanOrEqual(20)
  })

  it('the advisor prompt offers exactly what the builder offers', () => {
    const schema = schemaOptions()
    expect(schema.size).toBeGreaterThan(10)

    const mismatches = []
    for (const [type, key, options] of NUMERIC_FIELDS) {
      const listed = schema.get(type + '.' + key)
      // Not every field is described in the prompt; the ones that are must match.
      if (!listed) continue
      if (JSON.stringify(listed) !== JSON.stringify(options)) {
        mismatches.push(`${type}.${key}: prompt ${listed} vs builder ${options}`)
      }
    }
    expect(mismatches).toEqual([])
  })

  it('the validator accepts every value the builder offers, unchanged', () => {
    const rewritten = []
    for (const [type, key, options] of NUMERIC_FIELDS) {
      for (const value of options) {
        const { layers } = parseAndValidateLayers(
          JSON.stringify([{ type, params: { [key]: value } }]))
        const got = layers[0]?.params?.[key]
        // A legal value must survive. If the snapping table holds a narrower
        // list it moves the value to its own nearest option - that is the drift.
        if (got !== value) rewritten.push(`${type}.${key}: ${value} -> ${got}`)
      }
    }
    expect(rewritten).toEqual([])
  })
})
