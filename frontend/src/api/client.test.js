import { describe, expect, it, vi } from 'vitest'

import { fetchWithFallback } from './client'

vi.mock('@tanstack/react-query', () => ({
  useMutation: vi.fn(),
  useQuery: vi.fn(),
}))

vi.mock('../store/appStore', () => ({
  default: vi.fn(),
}))

describe('fetchWithFallback', () => {
  it('returns data on successful request', async () => {
    const request = vi.fn().mockResolvedValue({ data: { smape: 7.78 } })
    const payload = await fetchWithFallback(request)
    expect(payload.smape).toBe(7.78)
  })

  it('throws on error response', async () => {
    const request = vi.fn().mockRejectedValue(new Error('Network error'))
    await expect(fetchWithFallback(request)).rejects.toThrow('Network error')
  })
})
