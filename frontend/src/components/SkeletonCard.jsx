export default function SkeletonCard({ className = '' }) {
  return (
    <div className={`glass glass-dark animate-shimmer overflow-hidden rounded-xl ${className}`}>
      <div className="p-5">
        <div className="h-4 w-24 rounded bg-slate-200/50 dark:bg-slate-700/50" />
        <div className="mt-3 h-8 w-32 rounded bg-slate-200/50 dark:bg-slate-700/50" />
      </div>
    </div>
  )
}
