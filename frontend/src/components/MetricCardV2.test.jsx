import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'

import MetricCardV2 from './MetricCardV2'

describe('MetricCardV2', () => {
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

  it('renders label, value, and unit', () => {
    render(<MetricCardV2 label="Revenue" value={1234.56} unit="USD" />)
    expect(screen.getByText('Revenue')).toBeTruthy()
    expect(screen.getByText('USD')).toBeTruthy()
  })

  it('renders with prefix and suffix', () => {
    render(<MetricCardV2 label="Growth" value={5.4} prefix="+" suffix="%" />)
    expect(screen.getByText('+5.40%')).toBeTruthy()
  })

  it('renders explanation tooltip when provided', () => {
    render(<MetricCardV2 label="Test" value={42} explanation="This is a test explanation" />)
    const button = screen.getByRole('button', { name: 'Show explanation' })
    expect(button).toBeTruthy()
    fireEvent.click(button)
    expect(screen.getByText('This is a test explanation')).toBeTruthy()
  })

  it('applies Good status color and badge', () => {
    const { container } = render(<MetricCardV2 label="Health" value={95} status="Good" />)
    expect(container.querySelector('.metric-good')).toBeTruthy()
    expect(container.querySelector('.border-emerald-400')).toBeTruthy()
    expect(screen.getByText('Good')).toBeTruthy()
  })

  it('applies Watch status color and badge', () => {
    const { container } = render(<MetricCardV2 label="Health" value={75} status="Watch" />)
    expect(container.querySelector('.metric-watch')).toBeTruthy()
    expect(container.querySelector('.border-amber-400')).toBeTruthy()
    expect(screen.getByText('Watch')).toBeTruthy()
  })

  it('applies Risk status color and badge', () => {
    const { container } = render(<MetricCardV2 label="Health" value={45} status="Risk" />)
    expect(container.querySelector('.metric-risk')).toBeTruthy()
    expect(container.querySelector('.border-rose-400')).toBeTruthy()
    expect(screen.getByText('Risk')).toBeTruthy()
  })

  it('does not apply status classes when no status', () => {
    const { container } = render(<MetricCardV2 label="Health" value={50} />)
    expect(container.querySelector('.metric-good')).toBeNull()
    expect(container.querySelector('.metric-watch')).toBeNull()
    expect(container.querySelector('.metric-risk')).toBeNull()
    expect(screen.queryByText('Good')).toBeNull()
    expect(screen.queryByText('Watch')).toBeNull()
    expect(screen.queryByText('Risk')).toBeNull()
  })

  it('renders with empty string status', () => {
    const { container } = render(<MetricCardV2 label="Health" value={50} status="" />)
    expect(container.querySelector('.border-transparent')).toBeTruthy()
  })

  it('converts non-numeric value to 0', () => {
    render(<MetricCardV2 label="Status" value="N/A" />)
    // parseFloat('N/A') is NaN, so || 0 gives 0, rendered as 0.00
    expect(screen.getByText('0.00')).toBeTruthy()
  })
})
