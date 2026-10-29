import { useAlerts } from '../api/client'
import useAppStore from '../store/appStore'

const severityStyles = {
  LOW: 'border-blue-200 bg-blue-50 text-blue-900 dark:border-blue-700 dark:bg-blue-900/30 dark:text-blue-200',
  MEDIUM: 'border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-600 dark:bg-amber-900/30 dark:text-amber-200',
  HIGH: 'border-orange-200 bg-orange-50 text-orange-900 dark:border-orange-700 dark:bg-orange-900/30 dark:text-orange-200',
  CRITICAL: 'border-red-200 bg-red-50 text-red-900 dark:border-red-700 dark:bg-red-900/30 dark:text-red-200',
}

export default function AlertBanner() {
  const { data: alerts = [] } = useAlerts()
  const notifications = useAppStore((state) => state.notifications)
  if (!alerts.length && !notifications.length) {
    return null
  }

  return (
    <div className="mx-auto mt-4 w-full max-w-6xl space-y-2 px-4" role="region" aria-label="System alerts">
      {alerts.map((alert) => (
        <div
          key={alert.alert_id}
          role="status"
          aria-live="polite"
          className={`rounded-lg border px-4 py-3 text-sm ${severityStyles[alert.severity] || 'border-slate-200 bg-slate-50 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200'}`}
        >
          <span className="mr-2 font-semibold">[{alert.severity}]</span>
          <span>{alert.message}</span>
        </div>
      ))}
      {notifications.map((note, idx) => (
        <div
          key={`note-${note.ts}-${idx}`}
          role="status"
          aria-live="polite"
          className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-600 dark:bg-amber-900/30 dark:text-amber-200"
        >
          <span className="mr-2 font-semibold">[INFO]</span>
          <span>{note.message}</span>
        </div>
      ))}
    </div>
  )
}
