import React from 'react'
import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'

import AnimatedCounter from './AnimatedCounter'

describe('AnimatedCounter', () => {
  beforeEach(() => {
    // Mock prefers-reduced-motion to skip animation and set value directly
    window.matchMedia = vi.fn().mockImplementation((query) => ({
      matches: query === '(prefers-reduced-motion: reduce)',
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }))
  })

  it('renders with prefix and suffix', () => {
    render(<AnimatedCounter value={42} prefix="$" suffix=" USD" />)
    expect(screen.getByText('$42.00 USD')).toBeTruthy()
  })

  it('renders non-numeric value', () => {
    render(<AnimatedCounter value="N/A" />)
    expect(screen.getByText('N/A')).toBeTruthy()
  })
})
