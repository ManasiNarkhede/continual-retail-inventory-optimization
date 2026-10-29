import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Label,
} from 'recharts'

import useAppStore from '../store/appStore'

export default function CLCurveChart({ data }) {
  const chartDescriptionId = 'cl-curve-chart-description'
  const darkModeEnabled = useAppStore((state) => state.darkModeEnabled)

  const gridColor = darkModeEnabled ? '#334155' : '#e2e8f0'
  const axisTickColor = darkModeEnabled ? '#94a3b8' : '#64748b'
  const lineStroke = darkModeEnabled ? '#e2e8f0' : '#16213e'
  const refLineStroke = darkModeEnabled ? '#2dd4bf' : '#0f766e'
  const tooltipBg = darkModeEnabled ? '#1e293b' : '#ffffff'
  const tooltipBorder = darkModeEnabled ? '#475569' : '#e2e8f0'
  const tooltipColor = darkModeEnabled ? '#e2e8f0' : '#1e293b'
  const labelColor = darkModeEnabled ? '#cbd5e1' : '#475569'

  return (
    <div
      className="rounded-2xl border border-slate-200 bg-white/80 p-4 shadow-panel backdrop-blur-sm dark:border-slate-700 dark:bg-slate-800/70 dark:shadow-panel-dark"
      role="img"
      aria-label="Continual learning experience accuracy chart"
      aria-describedby={chartDescriptionId}
    >
      <h3 className="font-display text-lg text-slate-900 dark:text-slate-100">Experience Accuracy Curve</h3>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data} role="application" aria-label="Continual learning accuracy trend chart" margin={{ top: 16, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
          <XAxis dataKey="experience" tick={{ fontSize: 11, fill: axisTickColor }} aria-label="Experience index">
            <Label value="Experience" offset={-4} position="insideBottom" style={{ fill: labelColor, fontSize: 12, fontWeight: 500 }} />
          </XAxis>
          <YAxis tick={{ fontSize: 11, fill: axisTickColor }} unit="%" aria-label="Accuracy percent" domain={[0, 100]}>
            <Label value="Accuracy (%)" angle={-90} position="insideLeft" offset={10} style={{ fill: labelColor, fontSize: 12, fontWeight: 500 }} />
          </YAxis>
          <Tooltip
            contentStyle={{ fontSize: 12, backgroundColor: tooltipBg, border: `1px solid ${tooltipBorder}`, color: tooltipColor, borderRadius: 8 }}
            labelStyle={{ color: tooltipColor }}
            itemStyle={{ color: tooltipColor }}
          />
          <ReferenceLine y={80} stroke={refLineStroke} strokeDasharray="4 4" label={{ value: 'Threshold (80%)', fill: labelColor, fontSize: 11, position: 'insideTopRight' }} />
          <Line type="monotone" dataKey="accuracy" stroke={lineStroke} strokeWidth={2.5} dot={{ r: 4, fill: lineStroke, strokeWidth: 0 }} activeDot={{ r: 6, stroke: refLineStroke, strokeWidth: 2 }} />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-3 overflow-x-auto">
        <table className="min-w-full text-xs" aria-label="Continual learning accuracy table">
          <caption className="sr-only">Accuracy by experience.</caption>
          <thead>
            <tr className="border-b border-slate-200 text-left text-slate-500 dark:border-slate-600 dark:text-slate-400">
              <th scope="col" className="py-1 pr-3">Experience</th>
              <th scope="col" className="py-1 pr-3">Accuracy (%)</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row) => (
              <tr key={row.experience} className="border-b border-slate-100 text-slate-700 dark:border-slate-700 dark:text-slate-300">
                <th scope="row" className="py-1 pr-3 font-medium">{row.experience}</th>
                <td className="py-1 pr-3">{row.accuracy}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p id={chartDescriptionId} className="sr-only">
        Line chart showing experience accuracy by experience index with a reference threshold at 80 percent.
      </p>
    </div>
  )
}
