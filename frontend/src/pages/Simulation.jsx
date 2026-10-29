import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

import ErrorBoundary from '../components/ErrorBoundary'
import LoadingSkeleton from '../components/LoadingSkeleton'
import AnimatedCounter from '../components/AnimatedCounter'
import PageTransition from '../components/PageTransition'
import StoreSelector from '../components/StoreSelector'
import useSimulationRun from '../hooks/useSimulation'
import useAppStore from '../store/appStore'

export default function Simulation() {
  const selectedStore = useAppStore((state) => state.selectedStore)
  const setSelectedStore = useAppStore((state) => state.setSelectedStore)
  const apiStatuses = useAppStore((state) => state.apiStatuses)
  const [days, setDays] = useState(30)
  const [riskAversion, setRiskAversion] = useState(0.5)
  const simulation = useSimulationRun()

  const runSimulation = () => {
    simulation.mutate({
      store_id: selectedStore,
      n_days: Number(days),
      initial_stock: 100,
      algorithm: 'PPO',
      risk_aversion: Number(riskAversion),
    })
  }

  const handleRetry = () => {
    simulation.reset()
    runSimulation()
  }

  return (
    <PageTransition>
      <div className="space-y-6">
        <div>
          <h1 className="font-display text-3xl text-slate-900 dark:text-slate-100">Inventory Simulation</h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Run reinforcement learning simulations to evaluate inventory policies under varying risk preferences.
          </p>
        </div>

        <div className="grid gap-5 lg:grid-cols-3">
          <div className="lg:col-span-1">
            <StoreSelector selected={selectedStore} onSelect={setSelectedStore} />
          </div>

          <div className="glass glass-dark p-5 lg:col-span-2">
            <h2 className="font-display text-lg text-slate-900 dark:text-slate-100">Simulation Controls</h2>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Configure parameters and run a virtual month-long stress test before real execution.
            </p>

            <div className="mt-4 grid gap-5 sm:grid-cols-2">
              <div>
                <label className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                  Simulation Days
                </label>
                <input
                  type="number"
                  aria-label="Simulation days"
                  className="mt-2 w-full rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-slate-800 backdrop-blur-sm dark:border-slate-600 dark:bg-slate-700/80 dark:text-slate-100"
                  value={days}
                  onChange={(event) => setDays(event.target.value)}
                  min={1}
                  max={365}
                />
                <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                  Duration: 1–365 days
                </p>
              </div>

              <div>
                <label className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                  Risk Aversion
                </label>
                <input
                  type="range"
                  aria-label="Risk aversion"
                  min={0}
                  max={1}
                  step={0.05}
                  value={riskAversion}
                  onChange={(event) => setRiskAversion(Number(event.target.value))}
                  className="mt-3 w-full accent-teal"
                />
                <div className="mt-1 flex justify-between text-xs text-slate-500 dark:text-slate-400">
                  <span>Cost focus (0)</span>
                  <span className="font-semibold text-teal-600 dark:text-emerald-400">
                    {riskAversion.toFixed(2)}
                  </span>
                  <span>Service focus (1)</span>
                </div>
              </div>
            </div>

            <div className="mt-5 flex flex-wrap items-center gap-3">
            <button
              type="button"
              className="rounded-lg bg-teal-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md transition-transform hover:scale-[1.02] hover:bg-emerald-700 active:scale-[0.98] dark:hover:bg-emerald-600"
              onClick={runSimulation}
              disabled={simulation.isPending}
            >
              {simulation.isPending ? 'Running Simulation…' : 'Run Simulation'}
            </button>
            {simulation.isPending && (
              <span className="text-xs text-slate-500 dark:text-slate-400">
                Processing {days} days of inventory data…
              </span>
            )}
          </div>

          {/* Compact guide */}
          <div className="mt-4 grid grid-cols-1 gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3 text-[11px] dark:border-slate-700 dark:bg-slate-900">
            <div className="grid grid-cols-3 gap-2">
              <div className="rounded-md bg-slate-50 px-2 py-1.5 dark:bg-slate-800">
                <span className="font-semibold text-slate-700 dark:text-slate-200">Risk=0.0 (Cost)</span>
                <p className="mt-0.5 leading-tight text-slate-500 dark:text-slate-400">Orders to average demand. Frequent stockouts on spikes. ~50-70% SL.</p>
              </div>
              <div className="rounded-md bg-slate-50 px-2 py-1.5 dark:bg-slate-800">
                <span className="font-semibold text-slate-700 dark:text-slate-200">Risk=0.5 (Balanced)</span>
                <p className="mt-0.5 leading-tight text-slate-500 dark:text-slate-400">Moderate safety buffer. Still stockouts on spikes. ~85-95% SL.</p>
              </div>
              <div className="rounded-md bg-slate-50 px-2 py-1.5 dark:bg-slate-800">
                <span className="font-semibold text-slate-700 dark:text-slate-200">Risk=1.0 (Service)</span>
                <p className="mt-0.5 leading-tight text-slate-500 dark:text-slate-400">Large safety buffer. Rarely stockouts. ~97-99% SL.</p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-slate-100 pt-2 dark:border-slate-800">
              <span className="text-slate-400">Status:</span>
              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300">● Good</span>
              <span className="inline-flex items-center gap-1 rounded-full bg-rose-100 px-2 py-0.5 text-[10px] font-semibold text-rose-700 dark:bg-rose-900/40 dark:text-rose-300">● Stockout</span>
              <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-semibold text-amber-700 dark:bg-amber-900/40 dark:text-amber-300">● Overstock</span>
            </div>
          </div>
        </div>
        </div>

        {apiStatuses.simulation.unavailable ? (
          <div className="rounded-xl border border-amber-300 bg-amber-50/80 px-4 py-3 text-sm text-amber-900 backdrop-blur-sm dark:border-amber-600 dark:bg-amber-900/30 dark:text-amber-200">
            Model training in progress or artifacts missing. {apiStatuses.simulation.message}
          </div>
        ) : null}

        <AnimatePresence mode="wait">
          {simulation.isPending ? (
            <motion.div
              key="loading"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.25 }}
            >
              <div className="glass glass-dark p-5">
                <div className="flex items-center gap-3">
                  <div className="h-5 w-5 animate-spin rounded-full border-2 border-slate-300 border-t-teal dark:border-slate-600 dark:border-t-emerald-400" />
                  <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                    Running simulation…
                  </span>
                </div>
                <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
                  <motion.div
                    className="h-full rounded-full bg-teal-600 dark:bg-emerald-400"
                    initial={{ width: '0%' }}
                    animate={{ width: '100%' }}
                    transition={{ duration: 2, ease: 'easeInOut', repeat: Infinity }}
                  />
                </div>
              </div>
              <LoadingSkeleton className="mt-4 h-48" />
            </motion.div>
          ) : null}

          {simulation.error ? (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.25 }}
              className="glass glass-dark border-rose-200 p-5 dark:border-rose-800"
            >
              <div className="flex items-start gap-3">
                <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-rose-100 text-rose-600 dark:bg-rose-900/50 dark:text-rose-300">
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </div>
                <div className="flex-1">
                  <h3 className="text-sm font-semibold text-rose-700 dark:text-rose-300">
                    Simulation failed
                  </h3>
                  <p className="mt-1 text-xs text-rose-600 dark:text-rose-400">
                    {simulation.error?.message || 'An unexpected error occurred while running the simulation.'}
                  </p>
                  <button
                    type="button"
                    onClick={handleRetry}
                    className="mt-3 rounded-md bg-rose-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-rose-700 dark:bg-rose-700 dark:hover:bg-rose-600"
                  >
                    Retry
                  </button>
                </div>
              </div>
            </motion.div>
          ) : null}

          {simulation.data ? (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
            >
              <ErrorBoundary>
                <div className="glass glass-dark p-5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <h2 className="font-display text-lg text-slate-900 dark:text-slate-100">Simulation Results</h2>
                      <p className="text-xs text-slate-500 dark:text-slate-400">Store {simulation.data.store_id}</p>
                    </div>
                    <span className="inline-flex items-center rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300">
                      Completed
                    </span>
                  </div>

                  <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                    <div className="rounded-xl bg-white/60 p-4 backdrop-blur-sm dark:bg-slate-700/60">
                      <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Total Cost</p>
                      <p className="mt-1 text-2xl font-display text-slate-900 dark:text-slate-50">
                        <AnimatedCounter value={simulation.data.total_cost} prefix="$" />
                      </p>
                    </div>
                    <div className="rounded-xl bg-white/60 p-4 backdrop-blur-sm dark:bg-slate-700/60">
                      <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Service Level</p>
                      <p className="mt-1 text-2xl font-display text-slate-900 dark:text-slate-50">
                        <AnimatedCounter value={simulation.data.service_level * 100} suffix="%" />
                      </p>
                    </div>
                    <div className="rounded-xl bg-white/60 p-4 backdrop-blur-sm dark:bg-slate-700/60">
                      <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Risk Aversion</p>
                      <p className="mt-1 text-2xl font-display text-slate-900 dark:text-slate-50">
                        <AnimatedCounter value={Number(simulation.data.risk_aversion || 0.5)} />
                      </p>
                    </div>
                  </div>

                  {Array.isArray(simulation.data.steps) && simulation.data.steps.length > 0 ? (
                    <div className="mt-5 overflow-x-auto">
                      <h3 className="mb-2 text-sm font-semibold text-slate-700 dark:text-slate-300">Daily Trajectory</h3>
                      <table className="min-w-full text-xs" aria-label="Simulation trajectory table">
                        <caption className="sr-only">Daily simulation trajectory showing stock, demand, actions, and status.</caption>
                        <thead>
                          <tr className="border-b border-slate-200 text-left text-slate-500 dark:border-slate-600 dark:text-slate-400">
                            <th scope="col" className="py-2 pr-3">Day</th>
                            <th scope="col" className="py-2 pr-3">Stock</th>
                            <th scope="col" className="py-2 pr-3">Demand</th>
                            <th scope="col" className="py-2 pr-3">Order</th>
                            <th scope="col" className="py-2 pr-3">Reward</th>
                            <th scope="col" className="py-2 pr-3">Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {simulation.data.steps.slice(0, 15).map((step) => (
                            <tr key={step.day} className="border-b border-slate-100 text-slate-700 dark:border-slate-700 dark:text-slate-300">
                              <th scope="row" className="py-2 pr-3 font-medium">{step.day}</th>
                              <td className="py-2 pr-3">{Number(step.stock).toFixed(1)}</td>
                              <td className="py-2 pr-3">{Number(step.demand).toFixed(1)}</td>
                              <td className="py-2 pr-3">{Number(step.order_qty).toFixed(1)}</td>
                              <td className="py-2 pr-3">{Number(step.reward).toFixed(2)}</td>
                              <td className="py-2 pr-3">
                                <span className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                                  step.status === 'Stockout'
                                    ? 'bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300'
                                    : step.status === 'Overstock'
                                      ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300'
                                      : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300'
                                }`}>
                                  {step.status}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : null}
                </div>
              </ErrorBoundary>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </PageTransition>
  )
}
