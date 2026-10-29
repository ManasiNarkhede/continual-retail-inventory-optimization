import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, ChevronUp, Activity } from 'lucide-react'

export default function TrainingProgress({ 
  epoch = 0, 
  totalEpochs = 50, 
  currentLoss = 0, 
  bestLoss = 0,
  isTraining = false 
}) {
  const [isExpanded, setIsExpanded] = useState(true)
  const progress = totalEpochs > 0 ? (epoch / totalEpochs) * 100 : 0
  const eta = totalEpochs > 0 ? Math.ceil((totalEpochs - epoch) * 2.5) : 0 // ~2.5 min per epoch

  if (!isTraining && epoch === 0) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        className="glass glass-dark mb-6 overflow-hidden rounded-xl"
      >
        <div 
          className="flex items-center justify-between px-5 py-4"
        >
          <div className="flex items-center gap-3">
            <Activity size={18} className="text-teal-500 animate-pulse" />
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                {isTraining ? 'Training in Progress' : 'Training Complete'}
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Epoch {epoch} of {totalEpochs} {eta > 0 && `• ~${eta} min remaining`}
              </p>
            </div>
          </div>
          <button 
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
            aria-label={isExpanded ? 'Collapse training details' : 'Expand training details'}
            aria-expanded={isExpanded}
          >
            {isExpanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </button>
        </div>

        {isExpanded && (
          <div className="px-5 pb-4">
            <div className="h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
              <motion.div
                className="h-full rounded-full bg-gradient-to-r from-teal-400 to-emerald-400"
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
              />
            </div>
            <div className="mt-3 grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-xs text-slate-500 dark:text-slate-400">Current Loss</p>
                <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">{currentLoss.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-xs text-slate-500 dark:text-slate-400">Best Loss</p>
                <p className="text-sm font-semibold text-emerald-600 dark:text-emerald-400">{bestLoss.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-xs text-slate-500 dark:text-slate-400">Progress</p>
                <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">{progress.toFixed(1)}%</p>
              </div>
            </div>
          </div>
        )}
      </motion.div>
    </AnimatePresence>
  )
}
