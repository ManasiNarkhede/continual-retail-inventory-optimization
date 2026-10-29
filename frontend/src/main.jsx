import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

import App from './App'
import { waitForBackend } from './api/health'
import useAppStore from './store/appStore'
import './index.css'

const queryClient = new QueryClient()

function applyDarkMode(enabled) {
  const root = document.documentElement
  if (enabled) {
    root.classList.add('dark')
  } else {
    root.classList.remove('dark')
  }
}

applyDarkMode(useAppStore.getState().darkModeEnabled)
useAppStore.subscribe((state) => {
  applyDarkMode(state.darkModeEnabled)
})

const root = ReactDOM.createRoot(document.getElementById('root'))

root.render(
  <React.StrictMode>
    <div className="mx-auto mt-12 max-w-xl rounded-2xl border border-slate-200 bg-white/90 p-6 text-center text-slate-700 shadow-panel">
      Starting backend and dashboard...
    </div>
  </React.StrictMode>,
)

waitForBackend().finally(() => {
  root.render(
    <React.StrictMode>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </QueryClientProvider>
    </React.StrictMode>,
  )
})
