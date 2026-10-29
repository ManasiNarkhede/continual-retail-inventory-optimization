import axios from 'axios'
import { useMutation, useQuery } from '@tanstack/react-query'

const api = axios.create({ baseURL: '/api/v1', timeout: 30000 })

const CACHE_KEY = 'retailpulse_cache'
const CACHE_TTL = 5 * 60 * 1000 // 5 minutes
const MAX_CACHE_SIZE = 1024 * 1024 // 1MB

function getCached(key) {
  try {
    const data = localStorage.getItem(`${CACHE_KEY}_${key}`)
    if (!data) return null
    const { value, timestamp } = JSON.parse(data)
    if (Date.now() - timestamp > CACHE_TTL) {
      localStorage.removeItem(`${CACHE_KEY}_${key}`)
      return null
    }
    const ageMinutes = Math.round((Date.now() - timestamp) / 60000)
    return { ...value, is_stale: true, cached_at: timestamp, cached_age_minutes: ageMinutes }
  } catch {
    return null
  }
}

function setCached(key, value) {
  try {
    const payload = JSON.stringify({ value, timestamp: Date.now() })
    if (payload.length > MAX_CACHE_SIZE) return
    localStorage.setItem(`${CACHE_KEY}_${key}`, payload)
  } catch {
    // Ignore quota errors
  }
}

export async function fetchWithFallback(requestFn) {
  try {
    const response = await requestFn()
    return response.data
  } catch (error) {
    throw error
  }
}

export const useForecast = (storeId, horizon = 7) =>
  useQuery({
    queryKey: ['forecast', storeId, horizon],
    queryFn: async () => {
      const response = await api.post('/forecast/', { store_id: storeId, horizon })
      return response.data
    },
    enabled: Boolean(storeId),
    staleTime: 5 * 60 * 1000,
    retry: 2,
  })

export const useInventoryAction = () =>
  useMutation({
    mutationFn: (req) => api.post('/inventory/action', req).then((r) => r.data),
  })

export const useSimulation = () =>
  useMutation({
    mutationFn: async (req) => {
      const response = await api.post('/simulate/', req)
      return response.data
    },
  })

export const useCLMetrics = () =>
  useQuery({
    queryKey: ['cl-metrics'],
    queryFn: async () => {
      const response = await api.get('/metrics/cl')
      return response.data
    },
    staleTime: 60 * 1000,
    retry: 2,
  })

export const useKpis = () =>
  useQuery({
    queryKey: ['kpis'],
    queryFn: async () => {
      try {
        const response = await api.get('/metrics/kpis')
        setCached('kpis', response.data)
        return response.data
      } catch (error) {
        const cached = getCached('kpis')
        if (cached) {
          return cached
        }
        throw error
      }
    },
    refetchInterval: 30 * 1000,
    retry: 2,
  })

export const useStores = () =>
  useQuery({
    queryKey: ['stores'],
    queryFn: () => api.get('/metrics/stores').then((r) => r.data),
    staleTime: 10 * 60 * 1000,
  })
