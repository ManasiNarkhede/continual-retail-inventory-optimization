import { lazy, Suspense } from 'react'
import { NavLink, Route, Routes } from 'react-router-dom'
import ErrorBoundary from './components/ErrorBoundary'
import useAppStore from './store/appStore'
import { Home } from 'lucide-react'

const HomePage = lazy(() => import('./pages/HomePage'))
const Dashboard = lazy(() => import('./pages/Dashboard'))
const ForecastView = lazy(() => import('./pages/ForecastView'))
const Simulation = lazy(() => import('./pages/Simulation'))
const CLMetrics = lazy(() => import('./pages/CLMetrics'))
const HPOView = lazy(() => import('./pages/HPOView'))

const links = [
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/forecast', label: 'Forecast' },
  { to: '/simulation', label: 'Simulation' },
  { to: '/cl-metrics', label: 'CL Metrics' },
  { to: '/hpo', label: 'HPO' },
]

export default function App() {
  const darkModeEnabled = useAppStore((state) => state.darkModeEnabled)
  const toggleDarkMode = useAppStore((state) => state.toggleDarkMode)

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-transparent dark:text-slate-100">
        <header className="border-b border-slate-200 bg-white/80 backdrop-blur dark:border-slate-700 dark:bg-slate-900/75">
          <div className="mx-auto flex w-full max-w-6xl flex-nowrap items-center justify-between gap-3 px-4 py-4">
            <div className="flex items-center gap-3">
              <NavLink
                to="/"
                className="flex items-center gap-2 rounded-full px-3 py-2 text-sm font-semibold transition hover:bg-slate-100 dark:hover:bg-slate-800"
                aria-label="Home"
              >
                <Home size={18} />
              </NavLink>
              <div>
                <p className="font-display text-xl text-slate-900 dark:text-slate-100">RetailPulse AI</p>
                <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-300">
                  Turn Sales Data Into Better Inventory Decisions
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button
                type="button"
                aria-label="Toggle dark mode"
                aria-pressed={darkModeEnabled}
                onClick={toggleDarkMode}
                className="rounded-full border-2 border-slate-400 bg-white px-3 py-2 text-xs font-semibold text-slate-800 shadow-sm transition hover:bg-slate-50 hover:border-slate-500 dark:border-slate-500 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-700"
              >
                {darkModeEnabled ? 'Dark On' : 'Light On'}
              </button>
              <nav className="flex flex-nowrap gap-2" aria-label="Primary navigation">
                {links.map((link) => (
                  <NavLink
                    key={link.to}
                    to={link.to}
                    className={({ isActive }) =>
                      `min-w-[90px] text-center rounded-full px-4 py-2 text-sm font-semibold transition shadow-sm ${
                        isActive
                          ? 'bg-slate-900 text-white border border-slate-900 dark:bg-teal-600 dark:text-white dark:border-teal-600'
                          : 'bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 hover:border-slate-400 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700'
                      }`
                    }
                  >
                    {link.label}
                  </NavLink>
                ))}
              </nav>
            </div>
          </div>
        </header>

        <main className="mx-auto w-full max-w-6xl px-4 py-6" role="main">
          <Suspense fallback={null}>
            <ErrorBoundary>
              <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/forecast" element={<ForecastView />} />
                <Route path="/simulation" element={<Simulation />} />
                <Route path="/cl-metrics" element={<CLMetrics />} />
                <Route path="/hpo" element={<HPOView />} />
              </Routes>
            </ErrorBoundary>
          </Suspense>
        </main>
      </div>
    </ErrorBoundary>
  )
}
