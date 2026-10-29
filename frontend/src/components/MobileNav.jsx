import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { Menu, X } from 'lucide-react'

const links = [
  { to: '/', label: 'Dashboard' },
  { to: '/forecast', label: 'Forecast' },
  { to: '/simulation', label: 'Simulation' },
  { to: '/cl-metrics', label: 'CL Metrics' },
  { to: '/hpo', label: 'HPO' },
]

export default function MobileNav() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div className="md:hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="rounded-lg p-2 text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
        aria-label="Toggle navigation"
        aria-expanded={isOpen}
      >
        {isOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      {isOpen && (
        <div className="fixed inset-0 z-40 bg-white/95 backdrop-blur-sm dark:bg-slate-900/95">
          <div className="flex h-full flex-col items-center justify-center gap-6">
            {links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                onClick={() => setIsOpen(false)}
                className={({ isActive }) =>
                  `text-2xl font-semibold transition ${
                    isActive
                      ? 'text-teal-600 dark:text-teal-400'
                      : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100'
                  }`
                }
              >
                {link.label}
              </NavLink>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
