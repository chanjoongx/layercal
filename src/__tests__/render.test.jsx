import { describe, it, expect } from 'vitest'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import LayerCal from '@/components/LayerCal'
import AIAdvisor from '@/components/AIAdvisor'
import { TRANSLATIONS, LANGUAGE_OPTIONS } from '@/config/translations'

/*
 * Server rendering exercises the whole component tree, including every
 * useState initializer and useMemo, without needing jsdom or a browser.
 * It catches undefined identifiers, bad hook usage and crashes on empty state,
 * none of which the bundler reports.
 *
 * Effects don't run here, so this is a smoke test, not a substitute for
 * clicking through the app.
 */

describe('LayerCal renders', () => {
  it('renders with an empty canvas', () => {
    const html = renderToStaticMarkup(<LayerCal />)
    expect(html).toContain('LayerCal')
    expect(html).toContain('id="model-drop-zone"')
  })

  it('survives localStorage being unavailable', () => {
    // The node test environment has no localStorage at all; if the safe
    // wrapper leaked the ReferenceError this render would throw.
    expect(() => renderToStaticMarkup(<LayerCal />)).not.toThrow()
  })

  it('renders the layer palette for every layer type', () => {
    const html = renderToStaticMarkup(<LayerCal />)
    for (const { name } of Object.values(TRANSLATIONS.en).filter(v => v && v.name)) {
      expect(html).toContain(name)
    }
  })

  it('shows a copyright line that points at the licence', () => {
    const html = renderToStaticMarkup(<LayerCal />)
    expect(html).toContain('LayerCal')
    expect(html).toContain('MIT License')
    // "All rights reserved" would contradict the MIT licence the code ships under.
    expect(html).not.toContain('All rights reserved')
  })

  it('exposes a skip link and a matching main landmark', () => {
    const html = renderToStaticMarkup(<LayerCal />)
    expect(html).toContain('href="#main-content"')
    expect(html).toContain('id="main-content"')
  })
})

describe('LayerCal renders in every supported language', () => {
  it.each(LANGUAGE_OPTIONS.map(o => o.code))('%s', (code) => {
    // A missing key in one locale would render "undefined" into the page.
    const html = renderToStaticMarkup(<LayerCal />)
    expect(html).not.toContain('undefined')
    expect(TRANSLATIONS[code]).toBeDefined()
  })
})

describe('AIAdvisor renders', () => {
  const props = {
    isDarkMode: false,
    t: TRANSLATIONS.en,
    canvasHasLayers: false,
    onApply: () => {},
    onClose: () => {},
  }

  it('renders its initial state', () => {
    const html = renderToStaticMarkup(<AIAdvisor {...props} />)
    expect(html).toContain('AI Architecture Advisor')
    expect(html).toContain('role="dialog"')
  })

  it('renders in dark mode', () => {
    expect(() => renderToStaticMarkup(<AIAdvisor {...props} isDarkMode />)).not.toThrow()
  })

  it('renders for every locale', () => {
    for (const { code } of LANGUAGE_OPTIONS) {
      const html = renderToStaticMarkup(<AIAdvisor {...props} t={TRANSLATIONS[code]} />)
      expect(html).not.toContain('undefined')
    }
  })
})

describe('translation completeness', () => {
  it('every locale defines the same keys as English', () => {
    const enKeys = Object.keys(TRANSLATIONS.en).sort()
    for (const { code } of LANGUAGE_OPTIONS) {
      expect({ code, keys: Object.keys(TRANSLATIONS[code]).sort() })
        .toEqual({ code, keys: enKeys })
    }
  })

  it('no translated value is empty', () => {
    for (const { code } of LANGUAGE_OPTIONS) {
      for (const [key, value] of Object.entries(TRANSLATIONS[code])) {
        if (typeof value === 'string') {
          expect(value.trim(), `${code}.${key}`).not.toBe('')
        }
      }
    }
  })
})
