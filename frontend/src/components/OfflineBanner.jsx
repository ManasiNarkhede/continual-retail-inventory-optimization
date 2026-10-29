import { useOnlineStatus } from '../hooks/useOnlineStatus'
import { useRetryBackoff } from '../hooks/useRetryBackoff'
import { WifiOff, RefreshCw } from 'lucide-react'

export default function OfflineBanner() {
  const { isOnline, wasOffline } = useOnlineStatus()
  const { retry, reset, attempt, delay } = useRetryBackoff()

  if (isOnline && !wasOffline) return null

  const handleRetry = () => {
    reset()
    window.location.reload()
  }

  return (
    <div className="fixed top-0 left-0 right-0 z-50 border-b border-amber-200 bg-amber-50/90 px-4 py-3 backdrop-blur-sm dark:border-amber-700 dark:bg-amber-900/90">
      <div className="mx-auto flex max-w-6xl items-center justify-between">
        <div className="flex items-center gap-2 text-sm text-amber-900 dark:text-amber-100">
          <WifiOff size={16} />
          <span>
            {isOnline ? 'Connection restored' : `Connection lost. Retrying in ${Math.ceil(delay / 1000)}s...`}
            {attempt > 0 && ` (attempt ${attempt})`}
          </span>
        </div>
        <button
          onClick={handleRetry}
          className="flex items-center gap-1 rounded-lg bg-amber-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-amber-700 dark:bg-amber-700 dark:hover:bg-amber-600"
        >
          <RefreshCw size={12} />
          Retry Now
        </button>
      </div>
    </div>
  )
}
