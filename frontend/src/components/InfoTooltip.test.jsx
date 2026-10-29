import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'

import InfoTooltip from './InfoTooltip'

describe('InfoTooltip', () => {
  it('renders the info button', () => {
    render(<InfoTooltip explanation="Test explanation" />)
    expect(screen.getByRole('button', { name: 'Show explanation' })).toBeTruthy()
  })

  it('toggles explanation on click', () => {
    render(<InfoTooltip explanation="Test explanation" />)
    const button = screen.getByRole('button', { name: 'Show explanation' })
    expect(button.getAttribute('aria-expanded')).toBe('false')
    fireEvent.click(button)
    expect(button.getAttribute('aria-expanded')).toBe('true')
    expect(screen.getByText('Test explanation')).toBeTruthy()
  })

  it('closes on Escape key', () => {
    render(<InfoTooltip explanation="Test explanation" />)
    const button = screen.getByRole('button', { name: 'Show explanation' })
    fireEvent.click(button)
    expect(screen.getByText('Test explanation')).toBeTruthy()
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(button.getAttribute('aria-expanded')).toBe('false')
  })
})
