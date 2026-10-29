import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import HomePage from './HomePage'

describe('HomePage', () => {
  it('renders title and tagline', () => {
    render(
      <MemoryRouter>
        <HomePage />
      </MemoryRouter>
    )
    expect(screen.getByText('RetailPulse AI')).toBeTruthy()
    expect(screen.getByText('Turn Sales Data Into Better Inventory Decisions')).toBeTruthy()
  })

  it('renders go to dashboard button', () => {
    render(
      <MemoryRouter>
        <HomePage />
      </MemoryRouter>
    )
    expect(screen.getByRole('button', { name: 'Go to Dashboard' })).toBeTruthy()
  })

  it('renders feature cards', () => {
    render(
      <MemoryRouter>
        <HomePage />
      </MemoryRouter>
    )
    expect(screen.getByText('AI Forecasting')).toBeTruthy()
    expect(screen.getByText('Inventory Optimization')).toBeTruthy()
    expect(screen.getByText('Continual Learning')).toBeTruthy()
    expect(screen.getByText('HPO Analysis')).toBeTruthy()
  })
})
