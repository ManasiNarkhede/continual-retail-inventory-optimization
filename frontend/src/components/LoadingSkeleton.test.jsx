import React from 'react'
import { describe, expect, it } from 'vitest'
import { render } from '@testing-library/react'

import LoadingSkeleton from './LoadingSkeleton'

describe('LoadingSkeleton', () => {
  it('renders shimmer placeholders for cards and chart region', () => {
    const { container } = render(<LoadingSkeleton />)
    const shimmerCount = container.querySelectorAll('.animate-shimmer').length
    expect(shimmerCount).toBeGreaterThanOrEqual(2)
    expect(container.querySelector('.h-56')).toBeTruthy()
  })
})
