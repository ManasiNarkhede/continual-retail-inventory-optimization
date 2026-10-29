import { useNavigate } from 'react-router-dom'
import { BarChart3, Brain, Boxes, LineChart } from 'lucide-react'
import PageTransition from '../components/PageTransition'

export default function HomePage() {
  const navigate = useNavigate()

  return (
    <PageTransition>
      <div className="page-enter flex flex-col items-center space-y-10 py-8">
        {/* Title + tagline */}
        <div className="text-center space-y-4">
          <h1 className="font-display text-5xl font-bold text-slate-900 dark:text-white sm:text-6xl">
            RetailPulse AI
          </h1>
          <p className="mx-auto max-w-2xl text-xl text-slate-600 dark:text-slate-300">
            Turn Sales Data Into Better Inventory Decisions
          </p>
          <p className="mx-auto max-w-xl text-base text-slate-500 dark:text-slate-400">
            Pretrained TFT model with production-grade forecasts for 1,115 retail stores — instantly explore demand predictions, simulate inventory policies, and optimize your supply chain.
          </p>
        </div>

        {/* Center CTA */}
        <button
          onClick={() => navigate('/dashboard')}
          className="rounded-full bg-teal-600 px-8 py-3 text-lg font-semibold text-white shadow-lg transition hover:scale-105 hover:bg-teal-700 active:scale-95 dark:bg-teal-500 dark:text-white dark:hover:bg-teal-400"
        >
          Go to Dashboard
        </button>

        {/* Features grid */}
        <div className="grid w-full max-w-4xl gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {[
            {
              icon: LineChart,
              title: 'AI Forecasting',
              desc: '14-day demand forecasts per store using Temporal Fusion Transformer with 7.78% SMAPE',
            },
            {
              icon: Boxes,
              title: 'Inventory Optimization',
              desc: 'Service-level simulation and cost-minimization recommendations per store',
            },
            {
              icon: Brain,
              title: 'Continual Learning',
              desc: 'Track model drift and knowledge retention across retraining cycles',
            },
            {
              icon: BarChart3,
              title: 'HPO Analysis',
              desc: 'Hyperparameter search results and tuning recommendations for TFT & RL',
            },
          ].map((f) => (
            <div
              key={f.title}
              className="rounded-2xl border border-slate-200 bg-white/90 p-6 text-center shadow-sm transition hover:shadow-md dark:border-slate-700 dark:bg-slate-800/90"
            >
              <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-teal-100 dark:bg-teal-900/50">
                <f.icon className="h-6 w-6 text-teal-600 dark:text-teal-400" />
              </div>
              <h3 className="font-display text-base font-semibold text-slate-900 dark:text-slate-100">
                {f.title}
              </h3>
              <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </PageTransition>
  )
}
