import React from 'react'
import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'

import ErrorBoundary from './ErrorBoundary'

class Boom extends React.Component {
  render() {
    throw new Error('boom')
  }
}

function MaybeBoom({ shouldThrow }) {
  if (shouldThrow) {
    throw new Error('boom')
  }
  return <div>Recovered</div>
}

describe('ErrorBoundary', () => {
  it('renders fallback UI on child render error', () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    render(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>,
    )

    expect(screen.getByText('Something went wrong')).toBeTruthy()
    expect(screen.getByText('boom')).toBeTruthy()
    consoleSpy.mockRestore()
  })

  it('resets fallback when Try Again is clicked', () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const { rerender } = render(
      <ErrorBoundary>
        <MaybeBoom shouldThrow />
      </ErrorBoundary>,
    )

    rerender(
      <ErrorBoundary>
        <MaybeBoom shouldThrow={false} />
      </ErrorBoundary>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Try Again' }))

    expect(screen.getByText('Recovered')).toBeTruthy()
    consoleSpy.mockRestore()
  })
})
