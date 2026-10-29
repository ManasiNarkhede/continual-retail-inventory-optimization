export default function DemoModeBanner({ isDemo, onDismiss }) {
  if (!isDemo) {
    return null
  }

  return (
    <div className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-600 dark:bg-amber-900/30 dark:text-amber-200">
      <div className="flex items-start justify-between gap-3">
        <p>Demo Mode: displaying fallback data while training artifacts are unavailable.</p>
        {typeof onDismiss === 'function' ? (
          <button
            type="button"
            onClick={onDismiss}
            className="rounded bg-amber-200 px-2 py-1 text-xs font-semibold text-amber-900 hover:bg-amber-300 dark:bg-amber-700 dark:text-amber-100 dark:hover:bg-amber-600"
          >
            Dismiss
          </button>
        ) : null}
      </div>
    </div>
  )
}
