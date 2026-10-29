import React from 'react'
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Routes, Route } from 'react-router-dom'

import PageTransition from './PageTransition'

describe('PageTransition', () => {
  it('renders children', () => {
    render(
      <MemoryRouter initialEntries={['/']}>
        <PageTransition>
          <div data-testid="child">Hello</div>
        </PageTransition>
      </MemoryRouter>
    )
    expect(screen.getByTestId('child')).toBeTruthy()
    expect(screen.getByText('Hello')).toBeTruthy()
  })
})
