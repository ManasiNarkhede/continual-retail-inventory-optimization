import { useState, useEffect } from 'react'
import { Info } from 'lucide-react'

export default function InfoTooltip({ explanation }) {
  const [isOpen, setIsOpen] = useState(false)

  useEffect(() => {
    const handleEsc = (e) => { if (e.key === 'Escape') setIsOpen(false) }
    document.addEventListener('keydown', handleEsc)
    return () => document.removeEventListener('keydown', handleEsc)
  }, [])

  if (!explanation) return null

  return (
    <>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={`ml-1 inline-flex items-center justify-center rounded-full p-0.5 transition-colors ${
          isOpen 
            ? 'bg-teal-100 text-teal-700 dark:bg-teal-900/50 dark:text-teal-300' 
            : 'text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:text-slate-500 dark:hover:bg-slate-800 dark:hover:text-slate-300'
        }`}
        aria-expanded={isOpen}
        aria-label={isOpen ? 'Hide explanation' : 'Show explanation'}
      >
        <Info size={14} />
      </button>
      
      {isOpen ? (
        <div className="mt-2 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
          <span className="inline-flex items-center gap-1 text-teal-600 dark:text-teal-400">
            <Info size={12} /> What does this mean?
          </span>
          <span className="block text-slate-500 dark:text-slate-400">{explanation}</span>
        </div>
      ) : null}
    </>
  )
}
