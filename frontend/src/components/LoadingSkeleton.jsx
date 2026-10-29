export default function LoadingSkeleton({ className = '' }) {
  return (
    <div className={`space-y-5 ${className}`}>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 8 }).map((_, idx) => (
          <div key={`metric-skeleton-${idx}`} className="glass glass-dark animate-shimmer overflow-hidden rounded-xl p-4">
            <div className="h-3 w-24 rounded bg-slate-200/50 dark:bg-slate-700/50" />
            <div className="mt-4 h-8 w-20 rounded bg-slate-200/50 dark:bg-slate-700/50" />
            <div className="mt-3 h-2.5 w-32 rounded bg-slate-200/50 dark:bg-slate-700/50" />
          </div>
        ))}
      </div>

      <div className="glass glass-dark animate-shimmer overflow-hidden rounded-xl p-5">
        <div className="h-4 w-40 rounded bg-slate-200/50 dark:bg-slate-700/50" />
        <div className="mt-4 h-56 rounded-lg bg-slate-200/30 dark:bg-slate-700/30" />
        <div className="mt-4 grid grid-cols-3 gap-3">
          <div className="h-3 rounded bg-slate-200/50 dark:bg-slate-700/50" />
          <div className="h-3 rounded bg-slate-200/50 dark:bg-slate-700/50" />
          <div className="h-3 rounded bg-slate-200/50 dark:bg-slate-700/50" />
        </div>
      </div>
    </div>
  )
}
