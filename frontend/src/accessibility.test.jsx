import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { axe, toHaveNoViolations } from 'jest-axe'

import App from './App'
import useAppStore from './store/appStore'

expect.extend(toHaveNoViolations)

vi.mock('./api/client', async () => {
  const actual = await vi.importActual('./api/client')
  return {
    ...actual,
    useAlerts: () => ({ data: [] }),
    useStores: () => ({
      data: {
        stores: [1, 2, 3],
        forecastable_stores: [1, 2],
        total: 3,
        forecastable_total: 2,
      },
      isLoading: false,
      error: null,
    }),
    useKpis: () => ({
      data: {
        smape: 12.0,
        baseline_smape: 15.0,
        smape_uplift_vs_baseline: 3.0,
        rmse: 23.0,
        coverage_80pct: 0.78,
        service_level: 96.0,
        fill_rate: 95.5,
        total_cost: 1200.0,
        cost_saved: 350.0,
        bwt: -0.01,
        model_version: 'test',
        timestamp_utc: '2026-01-01T00:00:00Z',
        is_demo: false,
      },
      isLoading: false,
    }),
    useForecast: () => ({
      data: {
        store_id: 1,
        horizon: 14,
        dates: ['2026-01-01', '2026-01-02'],
        p10: [10, 9],
        p50: [12, 11],
        p90: [16, 14],
      },
      isLoading: false,
      error: null,
    }),
    useCLMetrics: () => ({
      data: {
        bwt: -0.01,
        fwt: 0.03,
        plasticity: 0.9,
        stability: 0.92,
        experience_accs: [0.85, 0.87, 0.88, 0.9],
        strategy: 'Replay',
      },
      isLoading: false,
      refetch: vi.fn(),
    }),
    useSimulation: () => ({
      mutate: vi.fn(),
      isPending: false,
      data: {
        store_id: 1,
        total_cost: 1250.0,
        service_level: 0.96,
        risk_aversion: 0.5,
        steps: [
          { day: 1, stock: 80, demand: 40, order_qty: 30, reward: -12.3, service_level: 1.0 },
          { day: 2, stock: 70, demand: 45, order_qty: 35, reward: -14.1, service_level: 1.0 },
        ],
      },
    }),
    useInventoryAction: () => ({
      mutate: vi.fn(),
      isPending: false,
      data: { store_id: 1, order_qty: 42.0, confidence: 0.9 },
    }),
  }
})

function renderAppAt(pathname) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[pathname]}>
        <App />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('WCAG smoke checks', () => {
  beforeEach(() => {
    useAppStore.setState({
      selectedStore: 1,
      darkModeEnabled: false,
    })
  })

  it('dashboard view has no obvious axe violations', async () => {
    const { container } = renderAppAt('/dashboard')
    const results = await axe(container)
    expect(results).toHaveNoViolations()
  })

  it('hpo table view has no obvious axe violations', async () => {
    const { container } = renderAppAt('/hpo')
    const results = await axe(container)
    expect(results).toHaveNoViolations()
  })

  it('forecast workspace has no obvious axe violations', async () => {
    const { container } = renderAppAt('/forecast')
    const results = await axe(container)
    expect(results).toHaveNoViolations()
  })

  it('simulation workspace has no obvious axe violations', async () => {
    const { container } = renderAppAt('/simulation')
    const results = await axe(container)
    expect(results).toHaveNoViolations()
  })

  it('cl metrics workspace has no obvious axe violations', async () => {
    const { container } = renderAppAt('/cl-metrics')
    const results = await axe(container)
    expect(results).toHaveNoViolations()
  })
})
