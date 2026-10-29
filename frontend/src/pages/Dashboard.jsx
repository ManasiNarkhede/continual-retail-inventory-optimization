import { lazy, Suspense, useState } from 'react'
import ForecastChart from '../components/ForecastChart'
import ErrorBoundary from '../components/ErrorBoundary'
import SkeletonCard from '../components/SkeletonCard'
import MetricCardV2 from '../components/MetricCardV2'
import PageTransition from '../components/PageTransition'
import StoreSelector from '../components/StoreSelector'
import { useKpis } from '../api/client'
import useAppStore from '../store/appStore'

export default function Dashboard() {
  const selectedStore = useAppStore((state) => state.selectedStore)
  const setSelectedStore = useAppStore((state) => state.setSelectedStore)
  const { data: kpis, isLoading } = useKpis()

  const hasRlAgent =
    typeof kpis?.service_level === 'number' &&
    kpis.service_level > 0 &&
    typeof kpis?.fill_rate === 'number' &&
    kpis.fill_rate > 0

  const metricStatus = {
    smape:
      typeof kpis?.smape === 'number'
        ? kpis.smape < 12
          ? 'Good'
          : kpis.smape < 20
            ? 'Watch'
            : 'Risk'
        : '',
    uplift:
      typeof kpis?.smape_uplift_vs_baseline === 'number'
        ? kpis.smape_uplift_vs_baseline >= 3
          ? 'Good'
          : kpis.smape_uplift_vs_baseline >= 1
            ? 'Watch'
            : 'Risk'
        : '',
    coverage:
      typeof kpis?.coverage_80pct === 'number'
        ? kpis.coverage_80pct >= 0.75
          ? 'Good'
          : kpis.coverage_80pct >= 0.65
            ? 'Watch'
            : 'Risk'
        : '',
    bwt:
      typeof kpis?.bwt === 'number'
        ? kpis.bwt >= -0.02
          ? 'Good'
          : kpis.bwt >= -0.05
            ? 'Watch'
            : 'Risk'
        : '',
    rmse:
      typeof kpis?.rmse === 'number'
        ? kpis.rmse < 1500
          ? 'Good'
          : kpis.rmse < 2500
            ? 'Watch'
            : 'Risk'
        : '',
    service:
      typeof kpis?.service_level === 'number'
        ? kpis.service_level >= 95
          ? 'Good'
          : kpis.service_level >= 90
            ? 'Watch'
            : 'Risk'
        : '',
  }

  return (
    <PageTransition>
      <div className="page-enter space-y-6">
        <h1 className="font-display text-3xl text-slate-900 dark:text-slate-100">Inventory Intelligence Dashboard</h1>

        {isLoading ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <SkeletonCard key={`dashboard-skeleton-${i}`} />
            ))}
          </div>
        ) : null}

        {!isLoading ? (
          <div className="space-y-6">
            {/* Forecast Metrics */}
            <h2 className="font-display text-lg text-slate-600 dark:text-slate-300">Forecast Metrics</h2>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCardV2
                label="Mean SMAPE"
                value={typeof kpis?.smape === 'number' ? kpis.smape.toFixed(2) : '--'}
                unit="%"
                explanation="Symmetric Mean Absolute Percentage Error — measures forecast accuracy. Lower is better. Below 12% is production-grade, 12-20% is acceptable, above 20% needs improvement."
                status={metricStatus.smape}
              />
              <MetricCardV2
                label="SMAPE vs Baseline"
                value={typeof kpis?.smape_uplift_vs_baseline === 'number' ? kpis.smape_uplift_vs_baseline.toFixed(2) : '--'}
                prefix="+"
                unit="pp"
                explanation={`How much better the TFT model is compared to the seasonal naive baseline (${typeof kpis?.baseline_smape === 'number' ? `${kpis.baseline_smape.toFixed(2)}%` : '--'}). Higher is better.`}
                status={metricStatus.uplift}
              />
              <MetricCardV2
                label="Coverage (p10–p90)"
                value={typeof kpis?.coverage_80pct === 'number' ? (kpis.coverage_80pct * 100).toFixed(1) : '--'}
                unit="%"
                explanation="Share of actual sales falling inside the p10–p90 prediction interval. Target is ~80%. Low values mean the model's uncertainty bands are too narrow."
                status={metricStatus.coverage}
              />
              <MetricCardV2
                label="BWT"
                value={typeof kpis?.bwt === 'number' ? kpis.bwt.toFixed(4) : '--'}
                explanation="Backward Transfer — measures knowledge retention after learning new data. Values near 0 or positive mean minimal forgetting. Below -0.05 indicates catastrophic forgetting."
                status={metricStatus.bwt}
              />
            </div>

            {/* RMSE */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCardV2
                label="RMSE"
                value={typeof kpis?.rmse === 'number' ? kpis.rmse.toFixed(0) : '--'}
                unit="units"
                explanation="Root Mean Square Error in sales units. Gives the absolute scale of forecast errors for operational planning. Below 1500 is good, 1500-2500 is acceptable."
                status={metricStatus.rmse}
              />
            </div>

            {/* RL-dependent metrics */}
            {hasRlAgent ? (
              <>
                <h2 className="font-display text-lg text-slate-600 dark:text-slate-300">RL Policy Performance</h2>
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
                  <MetricCardV2
                    label="Service Level"
                    value={typeof kpis?.service_level === 'number' ? kpis.service_level.toFixed(1) : '--'}
                    unit="%"
                    explanation="Percentage of days where demand was fully met from available stock. Target ≥ 95%."
                    status={metricStatus.service}
                  />
                  <MetricCardV2
                    label="Fill Rate"
                    value={typeof kpis?.fill_rate === 'number' ? kpis.fill_rate.toFixed(1) : '--'}
                    unit="%"
                    explanation="Percentage of total demand volume fulfilled from inventory."
                  />
                  <MetricCardV2
                    label="Total Cost"
                    value={typeof kpis?.total_cost === 'number' ? kpis.total_cost.toFixed(0) : '--'}
                    unit="EUR"
                    explanation="Combined holding + stockout + ordering cost from RL simulation."
                  />
                </div>
              </>
            ) : null}

            <div className="grid gap-4 lg:grid-cols-3">
              <StoreSelector selected={selectedStore} onSelect={setSelectedStore} />
              <div className="lg:col-span-2">
                <ErrorBoundary>
                  <div className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-panel dark:border-slate-700 dark:bg-slate-800/90 dark:shadow-panel-dark">
                    <ForecastChart storeId={selectedStore} horizon={14} />
                  </div>
                </ErrorBoundary>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </PageTransition>
  )
}
