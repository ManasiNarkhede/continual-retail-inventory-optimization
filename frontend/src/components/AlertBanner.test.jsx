import React from 'react'
import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'

import AlertBanner from './AlertBanner'
import useAppStore from '../store/appStore'

vi.mock('../api/client', () => ({
  useAlerts: () => ({
    data: [{ alert_id: 'a1', severity: 'HIGH', message: 'Stockout risk detected' }],
  }),
}))

describe('AlertBanner', () => {
  it('renders alert and notification rows', () => {
    useAppStore.setState({
      selectedStore: 1,
      forecastHorizon: 14,
      isTraining: false,
      apiStatuses: {
        forecast: { unavailable: false, message: '' },
        simulation: { unavailable: false, message: '' },
        kpis: { unavailable: false, message: '' },
      },
      notifications: [{ type: 'warning', message: 'Training in progress', ts: 1 }],
    })
    render(<AlertBanner />)
    expect(screen.getByText('Stockout risk detected')).toBeTruthy()
    expect(screen.getByText('Training in progress')).toBeTruthy()
  })
})
