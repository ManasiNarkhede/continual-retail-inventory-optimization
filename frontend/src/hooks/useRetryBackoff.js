import { useState, useCallback, useRef } from 'react'

export function useRetryBackoff({ maxDelay = 30000, maxAttempts = 10 } = {}) {
  const [attempt, setAttempt] = useState(0)
  const [delay, setDelay] = useState(1000)
  const timerRef = useRef(null)

  const calculateDelay = useCallback((attemptNum) => {
    const base = Math.min(1000 * Math.pow(2, attemptNum), maxDelay)
    const jitter = base * 0.2 * (Math.random() * 2 - 1)
    return Math.max(1000, Math.floor(base + jitter))
  }, [maxDelay])

  const retry = useCallback((callback) => {
    if (attempt >= maxAttempts) return
    const nextAttempt = attempt + 1
    const nextDelay = calculateDelay(nextAttempt)
    setAttempt(nextAttempt)
    setDelay(nextDelay)
    timerRef.current = setTimeout(() => {
      callback()
    }, nextDelay)
  }, [attempt, maxAttempts, calculateDelay])

  const reset = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    setAttempt(0)
    setDelay(1000)
  }, [])

  return { retry, reset, attempt, delay }
}
