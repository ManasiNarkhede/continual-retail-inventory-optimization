import { create } from 'zustand'

const useAppStore = create((set) => ({
  selectedStore: 1,
  forecastHorizon: 14,
  darkModeEnabled: false,
  apiStatuses: {
    forecast: { unavailable: false, message: '' },
    simulation: { unavailable: false, message: '' },
    kpis: { unavailable: false, message: '' },
  },
  setSelectedStore: (storeId) => set({ selectedStore: storeId }),
  setForecastHorizon: (horizon) => set({ forecastHorizon: horizon }),
  setDarkModeEnabled: (enabled) =>
    set({
      darkModeEnabled: Boolean(enabled),
    }),
  toggleDarkMode: () =>
    set((state) => ({
      darkModeEnabled: !state.darkModeEnabled,
    })),
}))

export default useAppStore
