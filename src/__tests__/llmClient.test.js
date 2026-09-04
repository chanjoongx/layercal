import { describe, it, expect } from 'vitest'
import {
  DEFAULT_MODELS,
  isOpenAIReasoningModel,
  buildOpenAIBody,
  LLMError,
} from '@/utils/llmClient'

/*
 * These guard the two things that silently rot in a BYOK client with no
 * backend: a model ID that gets retired, and a request body shaped for the
 * wrong OpenAI model family.
 */

describe('DEFAULT_MODELS', () => {
  it('covers every provider the UI offers', () => {
    expect(Object.keys(DEFAULT_MODELS).sort()).toEqual(['claude', 'gemini', 'openai'])
  })

  it.each(Object.entries(DEFAULT_MODELS))(
    '%s default is a non-empty string',
    (_provider, model) => {
      expect(typeof model).toBe('string')
      expect(model.length).toBeGreaterThan(0)
    }
  )

  it('never reverts to models that have been retired', () => {
    // claude-3-5-haiku-20241022 was retired 2026-02-19 and returns 404.
    const retired = ['claude-3-5-haiku-20241022', 'claude-3-opus-20240229', 'gemini-2.0-flash-001']
    expect(retired).not.toContain(DEFAULT_MODELS.claude)
    expect(retired).not.toContain(DEFAULT_MODELS.gemini)
    expect(retired).not.toContain(DEFAULT_MODELS.openai)
  })
})


describe('isOpenAIReasoningModel', () => {
  it.each([
    'gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-5.6-sol', 'gpt-5', 'GPT-5-Mini',
    'o1', 'o3-mini', 'o4-mini',
  ])(
    'treats %s as a reasoning model',
    (model) => { expect(isOpenAIReasoningModel(model)).toBe(true) }
  )

  it.each(['gpt-6-astra', 'gpt-6', 'GPT-7-whatever', 'gpt-12-future'])(
    'treats %s as a reasoning model, so a new generation is not sent legacy parameters',
    (model) => {
      // An earlier pattern matched only `gpt-5`, so gpt-6 fell through to the
      // legacy branch and every request paid for a rejected round trip before
      // the UNSUPPORTED_PARAM retry recovered it.
      expect(isOpenAIReasoningModel(model)).toBe(true)
    }
  )

  it.each(['gpt-4o-mini', 'gpt-4o', 'gpt-4.1-nano', 'gpt-3.5-turbo', '', undefined, null])(
    'treats %s as a classic chat model',
    (model) => { expect(isOpenAIReasoningModel(model)).toBe(false) }
  )

  it('does not match a model from another provider that happens to start with o', () => {
    expect(isOpenAIReasoningModel('gemini-3.5-flash-lite')).toBe(false)
    expect(isOpenAIReasoningModel('claude-opus-5')).toBe(false)
  })
})


describe('buildOpenAIBody', () => {
  const sys = 'system prompt mentioning JSON'
  const user = 'user prompt'

  it('uses max_completion_tokens and omits temperature for reasoning models', () => {
    const body = buildOpenAIBody('gpt-5.6-luna', sys, user)
    expect(body.max_completion_tokens).toBeGreaterThan(0)
    expect(body).not.toHaveProperty('max_tokens')
    expect(body).not.toHaveProperty('temperature')
  })

  it('uses max_tokens and temperature for classic models', () => {
    const body = buildOpenAIBody('gpt-4o-mini', sys, user)
    expect(body.max_tokens).toBeGreaterThan(0)
    expect(body).not.toHaveProperty('max_completion_tokens')
    expect(body.temperature).toBe(0.3)
  })

  it('requests JSON mode by default', () => {
    expect(buildOpenAIBody('gpt-4o-mini', sys, user).response_format)
      .toEqual({ type: 'json_object' })
  })

  it('can drop JSON mode for models that reject it', () => {
    expect(buildOpenAIBody('gpt-4o-mini', sys, user, { jsonMode: false }))
      .not.toHaveProperty('response_format')
  })

  it('gives reasoning models enough headroom to think and still answer', () => {
    // A tight ceiling gets consumed entirely by hidden reasoning, leaving
    // finish_reason "length" and no content.
    expect(buildOpenAIBody('gpt-5.6-luna', sys, user).max_completion_tokens)
      .toBeGreaterThanOrEqual(4096)
  })

  it.each(['gpt-6-astra', 'gpt-5.6-sol'])(
    'shapes %s for the reasoning family',
    (model) => {
      const body = buildOpenAIBody(model, sys, user)
      expect(body.max_completion_tokens).toBeGreaterThan(0)
      expect(body).not.toHaveProperty('max_tokens')
      expect(body).not.toHaveProperty('temperature')
    }
  )

  it('passes the system and user prompts through in order', () => {
    const body = buildOpenAIBody('gpt-4o-mini', sys, user)
    expect(body.messages).toEqual([
      { role: 'system', content: sys },
      { role: 'user', content: user },
    ])
  })
})


describe('LLMError', () => {
  it('carries a machine-readable code alongside the message', () => {
    const err = new LLMError('RATE_LIMIT', 'slow down')
    expect(err).toBeInstanceOf(Error)
    expect(err.code).toBe('RATE_LIMIT')
    expect(err.message).toBe('slow down')
    expect(err.name).toBe('LLMError')
  })
})
