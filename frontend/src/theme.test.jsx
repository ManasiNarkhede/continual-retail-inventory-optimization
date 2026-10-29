import React from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import useAppStore from './store/appStore'

vi.mock('./api/client', async () => {
  const actual = await vi.importActual('./api/client')
  return {
    ...actual,
    useAlerts: () => ({ data: [] }),
  }
})

describe('Theme toggle', () => {
  beforeEach(() => {
    useAppStore.setState({ darkModeEnabled: false })
  })

  function renderApp() {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    })
    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={['/hpo']}>
          <App />
        </MemoryRouter>
      </QueryClientProvider>,
    )
  }

  it('exposes a dark mode toggle with aria-pressed state', () => {
    renderApp()

    const toggle = screen.getByRole('button', { name: 'Toggle dark mode' })
    expect(toggle.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(toggle)
    expect(toggle.getAttribute('aria-pressed')).toBe('true')
  })

  it('renders landmark regions for navigation and main content', () => {
    renderApp()
    expect(screen.getByRole('navigation', { name: 'Primary navigation' })).toBeTruthy()
    expect(screen.getByRole('main')).toBeTruthy()
  })
})
