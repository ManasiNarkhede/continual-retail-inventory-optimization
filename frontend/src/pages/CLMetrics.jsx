import CLCurveChart from '../components/CLCurveChart'
import SkeletonCard from '../components/SkeletonCard'
import MetricCardV2 from '../components/MetricCardV2'
import PageTransition from '../components/PageTransition'
import { useCLMetrics } from '../api/client'
import useAppStore from '../store/appStore'

export default function CLMetrics() {
  const { data, isLoading, refetch } = useCLMetrics()
  const status = useAppStore((state) => state.apiStatuses.kpis)

  const statusOf = (metric, value) => {
    if (metric === 'bwt') {
      if (value >= -0.02) return 'Good'
      if (value >= -0.05) return 'Watch'
      return 'Risk'
    }
    if (metric === 'fwt') {
      if (value > 0) return 'Good'
      if (value >= -0.01) return 'Watch'
      return 'Risk'
    }
    if (metric === 'plasticity') {
      if (value >= 0.85) return 'Good'
      if (value >= 0.75) return 'Watch'
      return 'Risk'
    }
    if (metric === 'stability') {
      if (value >= 0.9) return 'Good'
      if (value >= 0.8) return 'Watch'
      return 'Risk'
    }
    return ''
  }

  if (isLoading) {
    return (
      <div className="page-enter grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <SkeletonCard key={`clmetrics-skeleton-${i}`} />
        ))}
      </div>
    )
  }

  if (status?.unavailable || data?.error === 'SERVICE_UNAVAILABLE') {
    return (
      <div className="page-enter space-y-4">
        <div className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-600 dark:bg-amber-900/30 dark:text-amber-200">
          <p className="font-semibold">Service Unavailable</p>
          <p className="mt-1">{data?.message || status?.message || 'Model training in progress. Retry shortly.'}</p>
          <button
            type="button"
            className="mt-3 rounded-lg bg-amber-600 px-3 py-2 text-xs font-semibold text-white hover:bg-amber-700 dark:bg-amber-700 dark:hover:bg-amber-600"
            onClick={() => refetch()}
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!data) {
    return null
  }

  const curveData = data.experience_accs.map((acc, idx) => ({
    experience: `Exp ${idx + 1}`,
    accuracy: Number((acc * 100).toFixed(2)),
  }))

  return (
    <PageTransition>
      <div className="space-y-6">
        <div>
          <h1 className="font-display text-3xl text-slate-900 dark:text-slate-100">Continual Learning Metrics</h1>
          <div className="mt-3 flex items-start gap-3 rounded-lg bg-white/60 p-3 backdrop-blur-sm dark:bg-slate-700/40">
            <div className="mt-0.5 h-2 w-2 shrink-0 rounded-full bg-teal-600 dark:bg-emerald-400" />
            <div>
              <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
                Strategy: <span className="text-teal-600 dark:text-emerald-400">{data.strategy}</span>
              </p>
              <p className="mt-1 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
                {data.strategy === 'EWC'
                  ? 'Elastic Weight Consolidation protects important parameters during new learning.'
                  : data.strategy === 'Replay'
                    ? 'Replay rehearses past data alongside new data to prevent forgetting.'
                    : `The ${data.strategy} strategy balances learning new patterns while retaining old knowledge.`}
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCardV2
            label="BWT"
            value={data.bwt.toFixed(4)}
            explanation="Backward Transfer — measures knowledge retention after learning new data. Values near 0 or positive mean minimal forgetting."
            status={statusOf('bwt', data.bwt)}
          />
          <MetricCardV2
            label="FWT"
            value={data.fwt.toFixed(4)}
            explanation="Forward Transfer — how much old learning helps new tasks. Positive is good."
            status={statusOf('fwt', data.fwt)}
          />
          <MetricCardV2
            label="Plasticity"
            value={data.plasticity.toFixed(4)}
            explanation="Adaptability — higher means the model can learn new demand patterns quickly."
            status={statusOf('plasticity', data.plasticity)}
          />
          <MetricCardV2
            label="Stability"
            value={data.stability.toFixed(4)}
            explanation="Consistency — higher means the model remembers old behavior while learning new data."
            status={statusOf('stability', data.stability)}
          />
        </div>

        <CLCurveChart data={curveData} />
      </div>
    </PageTransition>
  )
}
