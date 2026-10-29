import { describe, expect, it } from 'vitest'

import { backoffDelay } from './health'

describe('backoffDelay', () => {
  it('doubles delay for each attempt step', () => {
    expect(backoffDelay(500, 1)).toBe(500)
    expect(backoffDelay(500, 2)).toBe(1000)
    expect(backoffDelay(500, 3)).toBe(2000)
  })
})
