import { useState, useEffect, useRef } from 'react'

export default function AnimatedCounter({ value, duration = 1000, prefix = '', suffix = '' }) {
  const [display, setDisplay] = useState(0)
  const startTime = useRef(null)
  const startValue = useRef(0)

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (prefersReducedMotion) {
      setDisplay(value)
      return
    }

    startValue.current = display
    startTime.current = null

    const animate = (timestamp) => {
      if (!startTime.current) startTime.current = timestamp
      const progress = Math.min((timestamp - startTime.current) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3) // ease-out cubic
      const current = startValue.current + (value - startValue.current) * eased
      setDisplay(current)
      if (progress < 1) requestAnimationFrame(animate)
    }

    requestAnimationFrame(animate)
  }, [value, duration])

  return (
    <span>
      {prefix}{typeof value === 'number' ? display.toFixed(2) : display}{suffix}
    </span>
  )
}
