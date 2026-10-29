import { useState } from 'react'

const statusStyles = {
  Good: 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-900/40 dark:text-emerald-300 dark:border-emerald-700',
  Watch: 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-900/40 dark:text-amber-300 dark:border-amber-700',
  Risk: 'bg-red-50 text-red-700 border-red-200 dark:bg-red-900/40 dark:text-red-300 dark:border-red-700',
}

export default function MetricCard({ label, value, unit = '', explanation = '', status = '' }) {
  const [expanded, setExpanded] = useState(false)
  const panelId = `metric-explain-${label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`

  return (
    <div className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-panel dark:border-slate-700 dark:bg-slate-800/90 dark:shadow-panel-dark">
      <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">{label}</p>
      <p className="mt-2 text-3xl font-display font-semibold text-slate-900 dark:text-slate-50">
        {value}
        <span className="ml-1 text-base font-body text-slate-500 dark:text-slate-400">{unit}</span>
      </p>

      {status ? (
        <span
          className={`mt-2 inline-flex rounded-full border px-2 py-0.5 text-[11px] font-semibold ${statusStyles[status] || 'bg-slate-50 text-slate-700 border-slate-200 dark:bg-slate-700 dark:text-slate-300 dark:border-slate-600'}`}
        >
          {status}
        </span>
      ) : null}

      {explanation ? (
        <div className="mt-3">
          <button
            type="button"
            onClick={() => setExpanded((current) => !current)}
            aria-expanded={expanded}
            aria-controls={panelId}
            className="text-xs font-semibold text-teal-600 hover:underline dark:text-emerald-400"
          >
            {expanded ? 'Hide meaning' : 'What does this mean?'}
          </button>
          {expanded ? <p id={panelId} className="mt-2 text-xs text-slate-600 dark:text-slate-300">{explanation}</p> : null}
        </div>
      ) : null}
    </div>
  )
}
