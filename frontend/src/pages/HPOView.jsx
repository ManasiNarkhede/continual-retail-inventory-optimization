import { motion } from 'framer-motion'

import MetricCard from '../components/MetricCard'
import MetricCardV2 from '../components/MetricCardV2'
import PageTransition from '../components/PageTransition'
import { useKpis } from '../api/client'

export default function HPOView() {
  const { data: kpis, isLoading } = useKpis()

  const metricStatus = {
    smape: typeof kpis?.smape === 'number'
      ? kpis.smape < 12 ? 'Good' : kpis.smape < 20 ? 'Watch' : 'Risk'
      : '',
  }

  return (
    <PageTransition>
      <div className="space-y-6">
        <div>
          <h1 className="font-display text-3xl text-slate-900 dark:text-slate-100">Hyperparameter Optimization</h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Optuna trial leaderboard for TFT and RL tuning. Track training progress and discover optimal configurations.
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-3">
          <MetricCardV2
            label="Best SMAPE"
            value={typeof kpis?.smape === 'number' ? kpis.smape.toFixed(2) : '--'}
            unit="%"
            explanation="Lowest validation SMAPE achieved. Below 12% is production-grade."
            status={metricStatus.smape}
          />
          <MetricCardV2
            label="Coverage"
            value={typeof kpis?.coverage_80pct === 'number' ? (kpis.coverage_80pct * 100).toFixed(1) : '--'}
            unit="%"
            explanation="Share of actual sales inside p10–p90 prediction interval. Target ~80%."
          />
          <MetricCardV2
            label="RMSE"
            value={typeof kpis?.rmse === 'number' ? kpis.rmse.toFixed(0) : '--'}
            unit="units"
            explanation="Root Mean Square Error in sales units. Lower is better."
          />
        </div>

        <div className="grid gap-4 sm:grid-cols-3">
          <MetricCardV2
            label="Service Level"
            value={typeof kpis?.service_level === 'number' ? kpis.service_level.toFixed(1) : '--'}
            unit="%"
            explanation="Percentage of days demand was fully met from stock. Target ≥ 95%."
            status={typeof kpis?.service_level === 'number' ? (kpis.service_level >= 95 ? 'Good' : 'Watch') : ''}
          />
          <MetricCardV2
            label="Fill Rate"
            value={typeof kpis?.fill_rate === 'number' ? kpis.fill_rate.toFixed(1) : '--'}
            unit="%"
            explanation="Percentage of total demand volume fulfilled from inventory. Target ≥ 90%."
          />
          <MetricCardV2
            label="Total Cost"
            value={typeof kpis?.total_cost === 'number' ? kpis.total_cost.toFixed(0) : '--'}
            unit="EUR"
            explanation="Combined holding + stockout + ordering cost from RL simulation."
          />
        </div>

        <div className="grid gap-5">
          <div className="glass glass-dark p-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="font-display text-lg text-slate-900 dark:text-slate-100">Model Configuration</h2>
              <span className="inline-flex items-center rounded-full bg-teal-600/10 px-2.5 py-1 text-xs font-semibold text-teal-600 dark:bg-emerald-400/10 dark:text-emerald-400">
                <span className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-teal-600 dark:bg-emerald-400" />
                {kpis?.status || 'Active'}
              </span>
            </div>

            <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {[
                { name: 'Hidden Size', value: '32', detail: 'TFT hidden dimension' },
                { name: 'Attention Heads', value: '4', detail: 'Multi-head attention' },
                { name: 'Dropout', value: '0.1', detail: 'Regularization rate' },
                { name: 'Learning Rate', value: '0.001', detail: 'Adam optimizer LR' },
                { name: 'Batch Size', value: '8', detail: 'Per training batch' },
                { name: 'Max Epochs', value: '200', detail: 'Training iterations' },
                { name: 'Encoder Length', value: '30', detail: 'Past observations' },
                { name: 'Prediction Length', value: '14', detail: 'Forecast horizon' },
              ].map((param) => (
                <div
                  key={param.name}
                  className="rounded-xl bg-white/60 p-4 backdrop-blur-sm dark:bg-slate-700/60"
                >
                  <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">{param.name}</p>
                  <p className="mt-1 text-lg font-display font-semibold text-slate-900 dark:text-slate-50">{param.value}</p>
                  <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{param.detail}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="glass glass-dark p-5">
          <h2 className="font-display text-lg text-slate-900 dark:text-slate-100">Hyperparameter Space</h2>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            Search ranges for the current optimization study
          </p>

          <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { name: 'Hidden Size', range: '64 – 512', type: 'Categorical' },
              { name: 'Learning Rate', range: '1e-5 – 1e-2', type: 'Log-Uniform' },
              { name: 'Dropout', range: '0.0 – 0.5', type: 'Uniform' },
              { name: 'Attention Heads', range: '2 – 8', type: 'Integer' },
            ].map((param) => (
              <div
                key={param.name}
                className="rounded-xl bg-white/60 p-4 backdrop-blur-sm dark:bg-slate-700/60"
              >
                <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">{param.name}</p>
                <p className="mt-1 text-lg font-display font-semibold text-slate-900 dark:text-slate-50">{param.range}</p>
                <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{param.type}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </PageTransition>
  )
}
