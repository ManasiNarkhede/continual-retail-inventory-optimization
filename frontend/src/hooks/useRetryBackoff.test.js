import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'

import { useRetryBackoff } from './useRetryBackoff'

describe('useRetryBackoff', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('returns initial state', () => {
    const { result } = renderHook(() => useRetryBackoff())
    expect(result.current.attempt).toBe(0)
    expect(result.current.delay).toBe(1000)
  })

  it('increments attempt and sets delay on retry', () => {
    const { result } = renderHook(() => useRetryBackoff())
    const callback = vi.fn()

    act(() => {
      result.current.retry(callback)
    })

    expect(result.current.attempt).toBe(1)
    expect(result.current.delay).toBeGreaterThanOrEqual(1000)

    act(() => {
      vi.advanceTimersByTime(result.current.delay)
    })

    expect(callback).toHaveBeenCalled()
  })

  it('does not retry beyond maxAttempts', () => {
    const { result } = renderHook(() => useRetryBackoff({ maxAttempts: 2 }))
    const callback = vi.fn()

    act(() => {
      result.current.retry(callback)
    })
    act(() => {
      vi.advanceTimersByTime(result.current.delay)
    })
    expect(result.current.attempt).toBe(1)
    expect(callback).toHaveBeenCalledTimes(1)

    act(() => {
      result.current.retry(callback)
    })
    act(() => {
      vi.advanceTimersByTime(result.current.delay)
    })
    expect(result.current.attempt).toBe(2)
    expect(callback).toHaveBeenCalledTimes(2)

    act(() => {
      result.current.retry(callback)
    })
    expect(result.current.attempt).toBe(2)
    expect(callback).toHaveBeenCalledTimes(2)
  })

  it('resets attempt and delay and clears timer', () => {
    const { result } = renderHook(() => useRetryBackoff())
    const callback = vi.fn()

    act(() => {
      result.current.retry(callback)
    })
    expect(result.current.attempt).toBe(1)

    act(() => {
      result.current.reset()
    })

    expect(result.current.attempt).toBe(0)
    expect(result.current.delay).toBe(1000)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('caps base delay at maxDelay (jitter may exceed slightly)', () => {
    const { result } = renderHook(() => useRetryBackoff({ maxDelay: 2000, maxAttempts: 10 }))

    for (let i = 0; i < 5; i++) {
      act(() => {
        result.current.retry(() => {})
      })
      act(() => {
        vi.advanceTimersByTime(result.current.delay)
      })
    }

    expect(result.current.delay).toBeGreaterThanOrEqual(1000)
    expect(result.current.delay).toBeLessThanOrEqual(2400)
  })

  it('applies jitter to delay', () => {
    const { result } = renderHook(() => useRetryBackoff())

    const delays = []
    for (let i = 0; i < 5; i++) {
      act(() => {
        result.current.retry(() => {})
      })
      delays.push(result.current.delay)
      act(() => {
        result.current.reset()
      })
    }

    const uniqueDelays = new Set(delays)
    expect(uniqueDelays.size).toBeGreaterThan(1)
  })
})
