import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'

import OfflineBanner from './OfflineBanner'
import { useOnlineStatus } from '../hooks/useOnlineStatus'
import { useRetryBackoff } from '../hooks/useRetryBackoff'

vi.mock('../hooks/useOnlineStatus')
vi.mock('../hooks/useRetryBackoff')

describe('OfflineBanner', () => {
  const mockReset = vi.fn()
  const originalLocation = window.location

  beforeEach(() => {
    vi.clearAllMocks()
    delete window.location
    window.location = { reload: vi.fn() }
  })

  afterEach(() => {
    window.location = originalLocation
  })

  it('returns null when online and never was offline', () => {
    useOnlineStatus.mockReturnValue({ isOnline: true, wasOffline: false })
    useRetryBackoff.mockReturnValue({ retry: vi.fn(), reset: mockReset, attempt: 0, delay: 1000 })

    const { container } = render(<OfflineBanner />)
    expect(container.firstChild).toBeNull()
  })

  it('shows connection restored message when online but was offline', () => {
    useOnlineStatus.mockReturnValue({ isOnline: true, wasOffline: true })
    useRetryBackoff.mockReturnValue({ retry: vi.fn(), reset: mockReset, attempt: 0, delay: 1000 })

    render(<OfflineBanner />)
    expect(screen.getByText('Connection restored')).toBeTruthy()
  })

  it('shows connection lost message when offline', () => {
    useOnlineStatus.mockReturnValue({ isOnline: false, wasOffline: true })
    useRetryBackoff.mockReturnValue({ retry: vi.fn(), reset: mockReset, attempt: 2, delay: 5000 })

    render(<OfflineBanner />)
    expect(screen.getByText(/Connection lost\. Retrying in 5s\.\.\./)).toBeTruthy()
    expect(screen.getByText(/\(attempt 2\)/)).toBeTruthy()
  })

  it('calls reset and reloads when retry button is clicked', () => {
    useOnlineStatus.mockReturnValue({ isOnline: false, wasOffline: true })
    useRetryBackoff.mockReturnValue({ retry: vi.fn(), reset: mockReset, attempt: 1, delay: 2000 })

    render(<OfflineBanner />)
    fireEvent.click(screen.getByRole('button', { name: /Retry Now/ }))
    expect(mockReset).toHaveBeenCalledTimes(1)
    expect(window.location.reload).toHaveBeenCalledTimes(1)
  })
})
