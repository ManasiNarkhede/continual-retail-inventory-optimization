import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { useForecast } from '../api/client'
import useAppStore from '../store/appStore'
import LoadingSkeleton from './LoadingSkeleton'

function ForecastTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) {
    return null
  }

  const darkModeEnabled = document.documentElement.classList.contains('dark')
  const bg = darkModeEnabled ? 'rgba(30, 41, 59, 0.95)' : 'rgba(255, 255, 255, 0.95)'
  const border = darkModeEnabled ? '#475569' : '#e2e8f0'
  const text = darkModeEnabled ? '#e2e8f0' : '#1e293b'
  const muted = darkModeEnabled ? '#94a3b8' : '#64748b'

  const items = payload.map((entry) => ({
    name: entry.name,
    value: entry.value,
    color: entry.color,
  }))

  return (
    <div
      style={{
        background: bg,
        border: `1px solid ${border}`,
        borderRadius: 12,
        padding: '12px 16px',
        boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.15), 0 4px 10px -6px rgba(0, 0, 0, 0.1)',
        backdropFilter: 'blur(8px)',
        minWidth: 180,
      }}
    >
      <p style={{ color: text, fontSize: 13, fontWeight: 600, marginBottom: 8, marginTop: 0 }}>
        {label}
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {items.map((item) => (
          <div key={item.name} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span
                style={{
                  display: 'inline-block',
                  width: 10,
                  height: 10,
                  borderRadius: '50%',
                  backgroundColor: item.color,
                  flexShrink: 0,
                }}
              />
              <span style={{ color: muted, fontSize: 12 }}>{item.name}</span>
            </div>
            <span style={{ color: text, fontSize: 12, fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
              {item.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function ForecastChart({ storeId, horizon = 14 }) {
  const { data, isLoading, error } = useForecast(storeId, horizon)
  const apiStatuses = useAppStore((state) => state.apiStatuses)
  const darkModeEnabled = useAppStore((state) => state.darkModeEnabled)

  if (isLoading) {
    return <LoadingSkeleton className="h-72" />
  }
  if (apiStatuses.forecast.unavailable) {
    return (
      <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-600 dark:bg-amber-900/30 dark:text-amber-200">
        <p className="font-semibold">Model training in progress or artifacts missing.</p>
        <p className="mt-1 text-xs text-amber-700 dark:text-amber-300">{apiStatuses.forecast.message}</p>
      </div>
    )
  }
  if (error) {
    const detail = error?.response?.data?.detail || error?.message || 'Unknown error'
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-700 dark:bg-red-900/30 dark:text-red-300">
        <p className="font-semibold">Forecast unavailable for this store right now.</p>
        <p className="mt-1 text-xs text-red-600 dark:text-red-400">Reason: {detail}</p>
        <p className="mt-2 text-xs text-red-600 dark:text-red-400">
          Tip: train TFT first, then restart API so model artifacts are loaded at startup.
        </p>
      </div>
    )
  }
  if (!data) {
    return null
  }

  const chartData = data.dates.map((date, i) => ({
    date,
    p10: Math.round(data.p10[i]),
    p50: Math.round(data.p50[i]),
    p90: Math.round(data.p90[i]),
    sampleMean: Math.round((data.p10[i] + data.p50[i] + data.p90[i]) / 3),
  }))

  const first = chartData[0]
  const avgMid = Math.round(chartData.reduce((acc, row) => acc + row.p50, 0) / chartData.length)
  const descriptionId = `forecast-chart-description-${storeId}-${horizon}`

  // Adaptive chart colors for dark mode
  const gridColor = darkModeEnabled ? 'rgba(51, 65, 85, 0.4)' : 'rgba(226, 232, 240, 0.6)'
  const axisTickColor = darkModeEnabled ? '#94a3b8' : undefined
  const p10Stroke = darkModeEnabled ? '#2dd4bf' : '#0f766e'
  const p50Stroke = darkModeEnabled ? '#e2e8f0' : '#16213e'
  const p90Stroke = darkModeEnabled ? '#fbbf24' : '#d97706'
  const meanStroke = darkModeEnabled ? '#94a3b8' : '#64748b'

  return (
    <div
      role="img"
      aria-label={`Forecast chart for store ${storeId}`}
      aria-describedby={descriptionId}
    >
      <h2 className="font-display text-lg text-slate-900 dark:text-slate-100">Store {storeId} Forecast</h2>
      <p className="mb-2 text-sm text-slate-500 dark:text-slate-400">Quantile trajectory for the next {horizon} days.</p>
      <div className="mb-3 rounded-lg bg-slate-50/70 px-3 py-2 text-xs text-slate-600 dark:bg-slate-700/40 dark:text-slate-300">
        <p>
          Reading guide: <strong>p50</strong> is the most likely demand, <strong>p10</strong> is safer lower demand,
          <strong> p90</strong> is higher-demand risk. Hover any day to compare range and median.
        </p>
        <p className="mt-1">
          Example: first day p10={first?.p10 ?? '--'}, p50={first?.p50 ?? '--'}, p90={first?.p90 ?? '--'}, sample
          mean={first?.sampleMean ?? '--'}; average p50 across horizon is {avgMid}.
        </p>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={chartData} role="application" aria-label={`Demand quantile chart for store ${storeId}`}>
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
          <XAxis dataKey="date" tick={{ fontSize: 11, fill: axisTickColor }} aria-label="Forecast date" />
          <YAxis tick={{ fontSize: 11, fill: axisTickColor }} aria-label="Predicted demand units" />
          <Tooltip content={<ForecastTooltip />} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line type="monotone" dataKey="p10" stroke={p10Stroke} dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey="p50" stroke={p50Stroke} dot={false} strokeWidth={2.4} />
          <Line type="monotone" dataKey="p90" stroke={p90Stroke} dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey="sampleMean" stroke={meanStroke} dot={false} strokeDasharray="4 2" />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="mt-3 overflow-x-auto">
        <table className="min-w-full text-xs" aria-label={`Forecast table for store ${storeId}`}>
          <caption className="sr-only">Forecast quantiles for each forecast day.</caption>
          <thead>
            <tr className="border-b border-slate-200 text-left text-slate-500 dark:border-slate-600 dark:text-slate-400">
              <th scope="col" className="py-1 pr-3">Date</th>
              <th scope="col" className="py-1 pr-3">p10</th>
              <th scope="col" className="py-1 pr-3">p50</th>
              <th scope="col" className="py-1 pr-3">p90</th>
            </tr>
          </thead>
          <tbody>
            {chartData.slice(0, 7).map((row) => (
              <tr key={row.date} className="border-b border-slate-100 text-slate-700 dark:border-slate-700 dark:text-slate-300">
                <th scope="row" className="py-1 pr-3 font-medium">{row.date}</th>
                <td className="py-1 pr-3">{row.p10}</td>
                <td className="py-1 pr-3">{row.p50}</td>
                <td className="py-1 pr-3">{row.p90}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p id={descriptionId} className="sr-only" role="status" aria-live="polite">
        Forecast chart rendered for store {storeId} with {horizon} day horizon.
      </p>
    </div>
  )
}
