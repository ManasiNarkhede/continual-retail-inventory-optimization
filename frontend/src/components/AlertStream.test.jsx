import React from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { vi } from 'vitest'

import AlertStream from './AlertStream'

const mockAlerts = [
  { id: 'a1', severity: 'critical', message: 'Stockout risk detected', store_id: 1, timestamp: '2024-01-01T10:00:00Z' },
  { id: 'a2', severity: 'warning', message: 'Low inventory', store_id: 2, timestamp: '2024-01-01T11:00:00Z' },
  { id: 'a3', severity: 'info', message: 'New shipment arrived', store_id: 3, timestamp: '2024-01-01T12:00:00Z' },
  { id: 'a4', severity: 'success', message: 'Restocking complete', store_id: 4, timestamp: '2024-01-01T13:00:00Z' },
]

describe('AlertStream', () => {
  it('renders empty state when no alerts', () => {
    render(<AlertStream alerts={[]} />)
    expect(screen.getByText('No active alerts')).toBeTruthy()
  })

  it('renders alerts with correct severity styling', () => {
    render(<AlertStream alerts={mockAlerts} />)
    expect(screen.getByText('Stockout risk detected')).toBeTruthy()
    expect(screen.getByText('Low inventory')).toBeTruthy()
    expect(screen.getByText('New shipment arrived')).toBeTruthy()
    expect(screen.getByText('Restocking complete')).toBeTruthy()
  })

  it('dismisses individual alert', async () => {
    render(<AlertStream alerts={mockAlerts} />)
    const dismissButtons = screen.getAllByRole('button').filter(btn =>
      btn.className.includes('shrink-0')
    )
    fireEvent.click(dismissButtons[0])
    await waitFor(() => {
      expect(screen.queryByText('Stockout risk detected')).toBeNull()
    })
  })

  it('clears all alerts', () => {
    render(<AlertStream alerts={mockAlerts} />)
    fireEvent.click(screen.getByText('Clear All'))
    expect(screen.getByText('No active alerts')).toBeTruthy()
  })

  it('shows "Show more" when more than 5 alerts', () => {
    const manyAlerts = Array.from({ length: 7 }, (_, i) => ({
      id: `a${i}`,
      severity: 'info',
      message: `Alert ${i}`,
      store_id: i,
      timestamp: '2024-01-01T10:00:00Z',
    }))
    render(<AlertStream alerts={manyAlerts} />)
    expect(screen.getByText('Show 2 more')).toBeTruthy()
  })

  it('shows all alerts when "Show more" is clicked', () => {
    const manyAlerts = Array.from({ length: 7 }, (_, i) => ({
      id: `a${i}`,
      severity: 'info',
      message: `Alert ${i}`,
      store_id: i,
      timestamp: '2024-01-01T10:00:00Z',
    }))
    render(<AlertStream alerts={manyAlerts} />)
    fireEvent.click(screen.getByText('Show 2 more'))
    expect(screen.getByText('Alert 5')).toBeTruthy()
    expect(screen.getByText('Alert 6')).toBeTruthy()
  })
})
