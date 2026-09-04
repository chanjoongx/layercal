import { describe, it, expect } from 'vitest'
import {
  MODEL_CATALOG, CATALOG_PROVIDERS, CATALOG_VERIFIED_ON,
  modelsFor, findModel, priceLabel,
} from '@/config/modelCatalog'
import { DEFAULT_MODELS } from '@/utils/llmClient'
import ARCHITECTURE_KB from '@/config/architectureKB'
import { DEFAULT_LAYER_PARAMS, LAYER_TYPE_IDS, getLayerTypes } from '@/config/layerTypes'
import { validateModelDimensions } from '@/utils/modelValidation'
import { TRANSLATIONS } from '@/config/translations'

const TIERS = new Set(['fast', 'balanced', 'frontier'])

describe('model catalogue', () => {
  it('covers exactly the providers the client can call', () => {
    expect([...CATALOG_PROVIDERS].sort()).toEqual(Object.keys(DEFAULT_MODELS).sort())
  })

  it('contains the default model for every provider', () => {
    // Showing one model in the picker and calling another is worse than
    // showing none, so the two tables are pinned together here.
    for (const [provider, id] of Object.entries(DEFAULT_MODELS)) {
      expect(findModel(provider, id), `${provider} default ${id} is not in the catalogue`)
        .not.toBeNull()
    }
  })

  it('lists the default first, so the picker opens on it', () => {
    for (const [provider, id] of Object.entries(DEFAULT_MODELS)) {
      expect(modelsFor(provider)[0].id).toBe(id)
    }
  })

  it('records when it was last checked', () => {
    expect(CATALOG_VERIFIED_ON).toMatch(/^\d{4}-\d{2}-\d{2}$/)
  })

  it('has unique ids within each provider', () => {
    for (const provider of CATALOG_PROVIDERS) {
      const ids = modelsFor(provider).map(m => m.id)
      expect(new Set(ids).size).toBe(ids.length)
    }
  })

  it('uses only declared tiers, and offers a fast one everywhere', () => {
    for (const provider of CATALOG_PROVIDERS) {
      const tiers = modelsFor(provider).map(m => m.tier)
      for (const tier of tiers) expect(TIERS.has(tier)).toBe(true)
      expect(tiers).toContain('fast')
    }
  })

  it('gives every entry a label and a non-negative price or none at all', () => {
    for (const provider of CATALOG_PROVIDERS) {
      for (const model of modelsFor(provider)) {
        expect(model.label.trim()).not.toBe('')
        for (const price of [model.inputPrice, model.outputPrice]) {
          if (price !== null) {
            expect(Number.isFinite(price)).toBe(true)
            expect(price).toBeGreaterThanOrEqual(0)
          }
        }
      }
    }
  })

  it('only claims a free tier where there actually is one', () => {
    // A UI that falls back to "Free tier" whenever a price is missing would
    // label the paid Gemini Flash models free, so the note is explicit.
    for (const provider of CATALOG_PROVIDERS) {
      for (const model of modelsFor(provider)) {
        if (model.noteKey === 'freeTier') {
          expect(model.inputPrice).toBeNull()
          expect(model.id).toMatch(/flash-lite/)
        }
        if (model.note) expect(model.noteKey).toBeTruthy()
      }
    }
  })

  it('never prices output below input', () => {
    for (const provider of CATALOG_PROVIDERS) {
      for (const model of modelsFor(provider)) {
        if (model.inputPrice === null) continue
        expect(model.outputPrice).toBeGreaterThanOrEqual(model.inputPrice)
      }
    }
  })

  it('orders each provider from cheapest to most expensive', () => {
    for (const provider of CATALOG_PROVIDERS) {
      const priced = modelsFor(provider).filter(m => m.inputPrice !== null)
      for (let i = 1; i < priced.length; i++) {
        expect(priced[i].inputPrice).toBeGreaterThanOrEqual(priced[i - 1].inputPrice)
      }
    }
  })
})

describe('catalogue lookups', () => {
  it('returns an empty list for an unknown provider rather than throwing', () => {
    expect(modelsFor('nope')).toEqual([])
    expect(modelsFor(undefined)).toEqual([])
  })

  it('finds nothing for an id the user typed by hand', () => {
    expect(findModel('claude', 'claude-3-opus-20240229')).toBeNull()
  })

  it('formats a price line', () => {
    expect(priceLabel({ inputPrice: 1, outputPrice: 5 })).toBe('$1 / $5 per 1M')
    expect(priceLabel({ inputPrice: 0.2, outputPrice: 1.2 })).toBe('$0.20 / $1.20 per 1M')
  })

  it('returns null when there is no published per-token price', () => {
    expect(priceLabel({ inputPrice: null, outputPrice: null })).toBeNull()
    expect(priceLabel(null)).toBeNull()
  })
})

describe('architecture knowledge base', () => {
  const LAYER_TYPES = getLayerTypes(TRANSLATIONS.en)
  const known = new Set(LAYER_TYPE_IDS)

  // Every param the LLM prompt advertises as legal is a select option, and the
  // KB is injected into that prompt verbatim. A KB entry using a value the
  // canvas cannot represent teaches the model to emit one too.
  const OPTIONS = {}
  for (const [type, config] of Object.entries(LAYER_TYPES)) {
    OPTIONS[type] = {}
    for (const field of config.fields) {
      if (field.type === 'select') OPTIONS[type][field.key] = field.options
    }
  }

  it('has a unique id and a name for every entry', () => {
    const ids = ARCHITECTURE_KB.map(a => a.id)
    expect(new Set(ids).size).toBe(ids.length)
    for (const arch of ARCHITECTURE_KB) {
      expect(arch.name.trim()).not.toBe('')
      expect(arch.description.trim()).not.toBe('')
      expect(arch.tags.length).toBeGreaterThan(2)
    }
  })

  it.each(ARCHITECTURE_KB.map(a => [a.id, a]))('%s uses only real layer types', (_id, arch) => {
    for (const layer of arch.layers) {
      expect(known.has(layer.type), `${arch.id} uses ${layer.type}`).toBe(true)
    }
  })

  it.each(ARCHITECTURE_KB.map(a => [a.id, a]))('%s uses only legal parameter values', (_id, arch) => {
    for (const layer of arch.layers) {
      const options = OPTIONS[layer.type] || {}
      for (const [key, value] of Object.entries(layer.params)) {
        if (!options[key]) continue
        expect(options[key], `${arch.id}: ${layer.type}.${key} = ${value}`).toContain(value)
      }
    }
  })

  it.each(ARCHITECTURE_KB.map(a => [a.id, a]))('%s names only parameters the layer has', (_id, arch) => {
    for (const layer of arch.layers) {
      const defaults = DEFAULT_LAYER_PARAMS[layer.type]
      for (const key of Object.keys(layer.params)) {
        // `dropout` on a transformer is documented in the schema but not a
        // parameter LayerCal stores, so it is the one accepted extra.
        if (layer.type === 'transformer' && key === 'dropout') continue
        expect(key in defaults, `${arch.id}: ${layer.type} has no ${key}`).toBe(true)
      }
    }
  })

  it.each(ARCHITECTURE_KB.map(a => [a.id, a]))('%s connects end to end', (_id, arch) => {
    const issues = validateModelDimensions(arch.layers)
    const described = [...issues.entries()].map(
      ([index, issue]) => `layer ${index + 1} (${arch.layers[index].type}) wants ${issue.field} = ${issue.expected}`
    )
    expect(described, `${arch.id} has dimension mismatches`).toEqual([])
  })
})
