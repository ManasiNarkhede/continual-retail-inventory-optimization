import React from 'react'
import { fireEvent, render, screen } from '@testing-library/react'

import MetricCard from './MetricCard'

describe('MetricCard', () => {
  it('renders label and value', () => {
    render(<MetricCard label="Mean SMAPE" value="12.34" unit="%" />)
    expect(screen.getByText('Mean SMAPE')).toBeTruthy()
    expect(screen.getByText('12.34')).toBeTruthy()
  })

  it('toggles explanation text', () => {
    render(<MetricCard label="BWT" value="-0.02" explanation="Backward transfer guidance" />)
    const toggle = screen.getByRole('button', { name: 'What does this mean?' })
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    fireEvent.click(toggle)
    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    expect(screen.getByText('Backward transfer guidance')).toBeTruthy()
  })
})
