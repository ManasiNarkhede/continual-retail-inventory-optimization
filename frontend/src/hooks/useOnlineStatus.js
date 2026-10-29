import { useState, useEffect, useCallback } from 'react'

export function useOnlineStatus() {
  const [isOnline, setIsOnline] = useState(navigator.onLine)
  const [wasOffline, setWasOffline] = useState(false)
  const [offlineSince, setOfflineSince] = useState(null)

  useEffect(() => {
    let timeout
    const handleOnline = () => {
      clearTimeout(timeout)
      timeout = setTimeout(() => {
        setIsOnline(true)
        setOfflineSince(null)
      }, 1000)
    }
    const handleOffline = () => {
      clearTimeout(timeout)
      timeout = setTimeout(() => {
        setIsOnline(false)
        setWasOffline(true)
        setOfflineSince(Date.now())
      }, 1000)
    }

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    return () => {
      clearTimeout(timeout)
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  const reset = useCallback(() => {
    setWasOffline(false)
  }, [])

  return { isOnline, wasOffline, offlineSince, reset }
}
