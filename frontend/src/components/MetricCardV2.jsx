import { useState } from 'react'
import { Info } from 'lucide-react'
import AnimatedCounter from './AnimatedCounter'

export default function MetricCardV2({ label, value, unit, explanation, status, prefix = '', suffix = '' }) {
  const [showExplanation, setShowExplanation] = useState(false)

  const statusColors = {
    Good: 'metric-good border-emerald-400',
    Watch: 'metric-watch border-amber-400',
    Risk: 'metric-risk border-rose-400',
    '': 'border-transparent',
  }

  return (
    <div className={`glass glass-dark relative overflow-hidden rounded-xl border-l-4 p-5 transition hover:shadow-xl ${statusColors[status] || ''}`}>
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-1.5">
            <h3 className="text-sm font-medium text-slate-500 dark:text-slate-400">{label}</h3>
            {explanation && (
              <button
                type="button"
                onClick={() => setShowExplanation(!showExplanation)}
                className={`ml-1 inline-flex items-center justify-center rounded-full p-0.5 transition-colors ${
                  showExplanation
                    ? 'bg-teal-100 text-teal-700 dark:bg-teal-900/50 dark:text-teal-300'
                    : 'text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:text-slate-500 dark:hover:bg-slate-800 dark:hover:text-slate-300'
                }`}
                aria-expanded={showExplanation}
                aria-label={showExplanation ? 'Hide explanation' : 'Show explanation'}
              >
                <Info size={14} />
              </button>
            )}
          </div>
          <div className="mt-2 flex items-baseline gap-1">
            <span className="text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
              <AnimatedCounter value={parseFloat(value) || 0} prefix={prefix} suffix={suffix} />
            </span>
            {unit && <span className="text-sm text-slate-500 dark:text-slate-400">{unit}</span>}
          </div>

          {/* Explanation shown inline below the metric value */}
          {showExplanation && explanation ? (
            <div className="mt-3 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
              <span className="inline-flex items-center gap-1 font-medium text-teal-600 dark:text-teal-400">
                <Info size={12} /> What does this mean?
              </span>
              <span className="mt-1 block text-slate-500 dark:text-slate-400">{explanation}</span>
            </div>
          ) : null}
        </div>
        {status && (
          <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${
            status === 'Good' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-300' :
            status === 'Watch' ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/50 dark:text-amber-300' :
            'bg-rose-100 text-rose-700 dark:bg-rose-900/50 dark:text-rose-300'
          }`}>
            {status}
          </span>
        )}
      </div>
    </div>
  )
}
