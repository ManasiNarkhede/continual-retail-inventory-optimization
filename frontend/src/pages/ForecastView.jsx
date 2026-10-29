import { useEffect, useState } from 'react'

import ForecastChart from '../components/ForecastChart'
import ErrorBoundary from '../components/ErrorBoundary'
import InventoryPanel from '../components/InventoryPanel'
import PageTransition from '../components/PageTransition'
import StoreSelector from '../components/StoreSelector'
import useAppStore from '../store/appStore'
import useForecastData from '../hooks/useForecast'

function GlassSkeleton({ className = '' }) {
  return (
    <div className={`glass glass-dark animate-shimmer p-5 ${className}`}>
      <div className="h-4 w-40 rounded bg-slate-200/60 dark:bg-slate-600/60" />
      <div className="mt-4 h-56 rounded-lg bg-slate-100/60 dark:bg-slate-700/60" />
      <div className="mt-4 grid grid-cols-3 gap-3">
        <div className="h-3 rounded bg-slate-200/60 dark:bg-slate-600/60" />
        <div className="h-3 rounded bg-slate-200/60 dark:bg-slate-600/60" />
        <div className="h-3 rounded bg-slate-200/60 dark:bg-slate-600/60" />
      </div>
    </div>
  )
}

export default function ForecastView() {
  const selectedStore = useAppStore((state) => state.selectedStore)
  const setSelectedStore = useAppStore((state) => state.setSelectedStore)
  const forecast = useForecastData()
  const [lastUpdated, setLastUpdated] = useState(null)

  useEffect(() => {
    if (forecast.data && !forecast.isLoading) {
      setLastUpdated(new Date().toLocaleString())
    }
  }, [forecast.data, forecast.isLoading])

  return (
    <PageTransition>
      <div className="space-y-6">
        {/* Breadcrumb + Header */}
        <div>
          <nav aria-label="Breadcrumb" className="mb-2">
            <ol className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
              <li>
                <span className="font-medium text-slate-600 dark:text-slate-300">RetailPulse AI</span>
              </li>
              <li aria-hidden="true">/</li>
              <li aria-current="page" className="text-slate-900 dark:text-slate-100">
                Demand Forecast Workspace
              </li>
            </ol>
          </nav>
          <h1 className="font-display text-3xl text-slate-900 dark:text-slate-100">Demand Forecast Workspace</h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Explore demand forecasts by store, review quantile trajectories, and plan inventory actions.
          </p>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Store Selector - Glass Card */}
          <div className="glass glass-dark p-5">
            <StoreSelector selected={selectedStore} onSelect={setSelectedStore} />
          </div>

          {/* Forecast Chart - Glass Card */}
          <div className="lg:col-span-2">
            <div className="glass glass-dark p-5">
              <ErrorBoundary>
                {forecast.isLoading ? (
                  <GlassSkeleton />
                ) : (
                  <ForecastChart storeId={selectedStore} horizon={14} />
                )}
              </ErrorBoundary>
              {lastUpdated && (
                <p className="mt-4 text-right text-xs text-slate-400 dark:text-slate-500">
                  Last Updated: {lastUpdated}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Inventory Panel */}
        <ErrorBoundary>
          {forecast.isLoading ? (
            <div className="glass glass-dark animate-shimmer p-5">
              <div className="h-4 w-48 rounded bg-slate-200/60 dark:bg-slate-600/60" />
              <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                {Array.from({ length: 4 }).map((_, i) => (
                  <div key={i} className="h-20 rounded-lg bg-slate-100/60 dark:bg-slate-700/60" />
                ))}
              </div>
            </div>
          ) : (
            <div className="glass glass-dark p-5">
              <InventoryPanel forecast={forecast.data} />
            </div>
          )}
        </ErrorBoundary>
      </div>
    </PageTransition>
  )
}
