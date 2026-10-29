import useAppStore from '../store/appStore'
import { useForecast } from '../api/client'

export default function useForecastData() {
  const selectedStore = useAppStore((state) => state.selectedStore)
  const forecastHorizon = useAppStore((state) => state.forecastHorizon)
  return useForecast(selectedStore, forecastHorizon)
}
