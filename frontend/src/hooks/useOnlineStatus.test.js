import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'

import { useOnlineStatus } from './useOnlineStatus'

describe('useOnlineStatus', () => {
  let onlineHandler = null
  let offlineHandler = null

  beforeEach(() => {
    onlineHandler = null
    offlineHandler = null

    Object.defineProperty(globalThis, 'navigator', {
      value: { onLine: true },
      writable: true,
      configurable: true,
    })

    vi.spyOn(window, 'addEventListener').mockImplementation((event, handler) => {
      if (event === 'online') onlineHandler = handler
      if (event === 'offline') offlineHandler = handler
    })
    vi.spyOn(window, 'removeEventListener').mockImplementation((event, handler) => {
      if (event === 'online' && onlineHandler === handler) onlineHandler = null
      if (event === 'offline' && offlineHandler === handler) offlineHandler = null
    })

    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it('returns initial online state from navigator.onLine', () => {
    globalThis.navigator = { onLine: true }
    const { result } = renderHook(() => useOnlineStatus())
    expect(result.current.isOnline).toBe(true)
    expect(result.current.wasOffline).toBe(false)
    expect(result.current.offlineSince).toBeNull()
  })

  it('returns initial offline state from navigator.onLine', () => {
    globalThis.navigator = { onLine: false }
    const { result } = renderHook(() => useOnlineStatus())
    expect(result.current.isOnline).toBe(false)
  })

  it('sets wasOffline and offlineSince after offline event with debounce', () => {
    globalThis.navigator = { onLine: true }
    const { result } = renderHook(() => useOnlineStatus())

    act(() => {
      offlineHandler?.()
    })

    expect(result.current.isOnline).toBe(true)

    act(() => {
      vi.advanceTimersByTime(1000)
    })

    expect(result.current.isOnline).toBe(false)
    expect(result.current.wasOffline).toBe(true)
    expect(result.current.offlineSince).not.toBeNull()
  })

  it('clears offline state after online event with debounce', () => {
    globalThis.navigator = { onLine: false }
    const { result } = renderHook(() => useOnlineStatus())

    act(() => {
      offlineHandler?.()
      vi.advanceTimersByTime(1000)
    })

    expect(result.current.isOnline).toBe(false)

    act(() => {
      onlineHandler?.()
      vi.advanceTimersByTime(1000)
    })

    expect(result.current.isOnline).toBe(true)
    expect(result.current.offlineSince).toBeNull()
  })

  it('resets wasOffline when reset is called', () => {
    globalThis.navigator = { onLine: true }
    const { result } = renderHook(() => useOnlineStatus())

    act(() => {
      offlineHandler?.()
      vi.advanceTimersByTime(1000)
    })

    expect(result.current.wasOffline).toBe(true)

    act(() => {
      result.current.reset()
    })

    expect(result.current.wasOffline).toBe(false)
  })

  it('clears pending timeout on unmount', () => {
    globalThis.navigator = { onLine: true }
    const { unmount } = renderHook(() => useOnlineStatus())

    act(() => {
      offlineHandler?.()
    })

    expect(vi.getTimerCount()).toBe(1)

    unmount()

    expect(vi.getTimerCount()).toBe(0)
  })
})
