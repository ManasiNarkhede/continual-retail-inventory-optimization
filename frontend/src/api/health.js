import axios from 'axios'

const healthClient = axios.create({ baseURL: '/api', timeout: 5000 })

function backoffDelay(baseDelayMs, attempt) {
  const cappedAttempt = Math.max(0, attempt - 1)
  return baseDelayMs * (2 ** cappedAttempt)
}

export async function waitForBackend(maxAttempts = 30, delayMs = 500, maxDelayMs = 4000) {
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      const response = await healthClient.get('/metrics/kpis')
      if (response.status === 200) {
        return true
      }
    } catch {
      // keep retrying while backend starts
    }
    const waitMs = Math.min(backoffDelay(delayMs, attempt), maxDelayMs)
    // eslint-disable-next-line no-await-in-loop
    await new Promise((resolve) => setTimeout(resolve, waitMs))
  }
  return false
}

export { backoffDelay }
