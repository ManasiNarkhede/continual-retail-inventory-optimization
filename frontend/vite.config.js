import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import http from 'node:http'
import { execSync, spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

const apiTarget = process.env.VITE_API_TARGET || 'http://127.0.0.1:8000'

const backendRoot = path.resolve(process.cwd(), '..')
const backendHealth = path.join(backendRoot, 'scripts', 'check_backend.py')
let backendStarted = false

function ensureBackendRunning() {
  try {
    execSync(`python "${backendHealth}" --base-url ${apiTarget}`, {
      stdio: 'ignore',
      cwd: backendRoot,
    })
    return
  } catch {
    // backend is not reachable; attempt auto-start
  }

  const pythonExe = path.join(backendRoot, '.venv39', 'Scripts', 'python.exe')
  if (!fs.existsSync(pythonExe)) {
    console.warn('[vite-proxy] backend auto-start skipped (.venv39 missing)')
    return
  }
  if (backendStarted) {
    return
  }
  backendStarted = true

  try {
    const child = spawn(
      pythonExe,
      ['-m', 'uvicorn', 'src.api.main:app', '--host', '127.0.0.1', '--port', '8000'],
      {
        cwd: backendRoot,
        detached: true,
        stdio: 'ignore',
        windowsHide: true,
      },
    )
    child.unref()
    console.info('[vite-proxy] backend launch requested at http://127.0.0.1:8000')
  } catch (error) {
    backendStarted = false
    console.warn('[vite-proxy] backend auto-start failed:', error.message)
  }

  try {
    execSync(`python "${backendHealth}" --base-url ${apiTarget}`, {
      cwd: backendRoot,
      stdio: 'ignore',
    })
  } catch {
    console.warn('[vite-proxy] backend is still starting; first API call may retry')
  }
}

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          'vendor-charts': ['recharts'],
          'vendor-query': ['@tanstack/react-query', 'axios', 'zustand'],
        },
      },
    },
  },
  server: {
    host: 'localhost',
    port: 5173,
    open: true,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
        secure: false,
        agent: new http.Agent({ keepAlive: true, family: 4 }),
        configure: (proxy) => {
          ensureBackendRunning()
          proxy.on('error', (error, req) => {
            console.error(
              '[vite-proxy] API proxy error:',
              req?.method,
              req?.url,
              '->',
              apiTarget,
              '|',
              error.code || error.name,
              error.message,
            )
          })
          proxy.on('proxyReq', (proxyReq, req) => {
            const host = proxyReq.getHeader('host')
            console.info('[vite-proxy] ->', req.method, req.url, 'to', `${proxyReq.protocol}//${host}`)
          })
          proxy.on('proxyRes', (proxyRes, req) => {
            console.info('[vite-proxy] <-', proxyRes.statusCode, req.method, req.url)
          })
        },
      },
    },
  },
})
