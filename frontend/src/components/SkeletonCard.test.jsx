import React from 'react'
import { describe, expect, it } from 'vitest'
import { render } from '@testing-library/react'

import SkeletonCard from './SkeletonCard'

describe('SkeletonCard', () => {
  it('renders with shimmer animation class', () => {
    const { container } = render(<SkeletonCard />)
    expect(container.querySelector('.animate-shimmer')).toBeTruthy()
  })

  it('applies custom className when provided', () => {
    const { container } = render(<SkeletonCard className="custom-class" />)
    const root = container.firstChild
    expect(root.classList.contains('custom-class')).toBe(true)
    expect(root.classList.contains('animate-shimmer')).toBe(true)
  })

  it('renders placeholder bars', () => {
    const { container } = render(<SkeletonCard />)
    const bars = container.querySelectorAll('.bg-slate-200\\/50, .dark\\:bg-slate-700\\/50')
    expect(bars.length).toBeGreaterThanOrEqual(2)
  })
})
