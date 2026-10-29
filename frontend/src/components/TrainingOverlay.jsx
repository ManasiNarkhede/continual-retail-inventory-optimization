import useAppStore from '../store/appStore'

export default function TrainingOverlay() {
  const isTraining = useAppStore((state) => state.isTraining)
  if (!isTraining) {
    return null
  }

  return (
    <div className="pointer-events-none fixed inset-0 z-50 flex items-center justify-center bg-slate-900/45 p-4">
      <div className="rounded-xl bg-white px-5 py-4 text-center shadow-panel dark:bg-slate-800 dark:shadow-panel-dark" role="status" aria-live="assertive">
        <p className="text-lg font-semibold text-slate-900 dark:text-slate-100">Model Training in Progress</p>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">Live predictions will resume once artifacts are ready.</p>
      </div>
    </div>
  )
}
