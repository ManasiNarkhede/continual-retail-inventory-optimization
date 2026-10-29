import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { AlertTriangle, Info, X, CheckCircle, Bell } from 'lucide-react'

const severityConfig = {
  critical: { icon: AlertTriangle, color: 'bg-rose-100 text-rose-700 dark:bg-rose-900/50 dark:text-rose-300', border: 'border-rose-200 dark:border-rose-800' },
  warning: { icon: AlertTriangle, color: 'bg-amber-100 text-amber-700 dark:bg-amber-900/50 dark:text-amber-300', border: 'border-amber-200 dark:border-amber-800' },
  info: { icon: Info, color: 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300', border: 'border-blue-200 dark:border-blue-800' },
  success: { icon: CheckCircle, color: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-300', border: 'border-emerald-200 dark:border-emerald-800' },
}

export default function AlertStream({ alerts = [] }) {
  const [dismissed, setDismissed] = useState(new Set())
  const [showAll, setShowAll] = useState(false)

  const visibleAlerts = alerts.filter(a => !dismissed.has(a.id)).slice(0, showAll ? undefined : 5)

  const dismiss = (id) => {
    setDismissed(prev => new Set([...prev, id]))
  }

  const clearAll = () => {
    setDismissed(new Set(alerts.map(a => a.id)))
  }

  if (visibleAlerts.length === 0) {
    return (
      <div className="glass glass-dark rounded-xl p-5 text-center">
        <Bell size={20} className="mx-auto mb-2 text-slate-400" />
        <p className="text-sm text-slate-500 dark:text-slate-400">No active alerts</p>
      </div>
    )
  }

  return (
    <div className="glass glass-dark rounded-xl p-5">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
          Active Alerts ({visibleAlerts.length})
        </h3>
        <button
          onClick={clearAll}
          className="text-xs text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
        >
          Clear All
        </button>
      </div>

      <div className="space-y-2">
        <AnimatePresence>
          {visibleAlerts.map((alert) => {
            const config = severityConfig[alert.severity] || severityConfig.info
            const Icon = config.icon
            return (
              <motion.div
                key={alert.id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className={`flex items-start gap-3 rounded-lg border p-3 ${config.color} ${config.border}`}
              >
                <Icon size={16} className="mt-0.5 shrink-0" />
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium">{alert.message}</p>
                  <p className="text-xs opacity-75">
                    Store {alert.store_id} • {new Date(alert.timestamp).toLocaleTimeString()}
                  </p>
                </div>
                <button
                  onClick={() => dismiss(alert.id)}
                  className="shrink-0 opacity-50 hover:opacity-100"
                  aria-label={`Dismiss alert for store ${alert.store_id}`}
                >
                  <X size={14} />
                </button>
              </motion.div>
            )
          })}
        </AnimatePresence>
      </div>

      {alerts.length > 5 && !showAll && (
        <button
          onClick={() => setShowAll(true)}
          className="mt-3 w-full text-center text-xs text-slate-500 hover:text-slate-700 dark:text-slate-400"
        >
          Show {alerts.length - 5} more
        </button>
      )}
    </div>
  )
}
