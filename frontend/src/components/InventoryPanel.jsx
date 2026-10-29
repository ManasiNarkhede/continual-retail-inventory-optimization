import { useMemo, useState } from 'react'

import useInventory from '../hooks/useInventory'
import useAppStore from '../store/appStore'

export default function InventoryPanel({ forecast }) {
  const selectedStore = useAppStore((state) => state.selectedStore)
  const [stock, setStock] = useState(100)
  const inventoryMutation = useInventory()

  const forecastInput = useMemo(() => {
    if (!forecast?.p50) {
      return []
    }
    return forecast.p50.map((value) => Number(value))
  }, [forecast])

  const submit = () => {
    inventoryMutation.mutate({
      store_id: selectedStore,
      current_stock: Number(stock),
      forecast: forecastInput,
    })
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-panel dark:border-slate-700 dark:bg-slate-800/90 dark:shadow-panel-dark">
      <h3 className="font-display text-lg text-slate-900 dark:text-slate-100">Inventory Action</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400">Get a recommended order quantity based on store demand patterns and current inventory.</p>

      <label className="mt-4 block text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Current Stock</label>
      <input
        type="number"
        min="0"
        max="1000000"
        aria-label="Current stock"
        className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-800 dark:border-slate-600 dark:bg-slate-700 dark:text-slate-100"
        value={stock}
        onChange={(event) => setStock(Number(event.target.value))}
      />

      <button
        type="button"
        className="mt-4 rounded-lg bg-teal-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-700 dark:hover:bg-emerald-600"
        onClick={submit}
        disabled={inventoryMutation.isPending}
      >
        {inventoryMutation.isPending ? 'Evaluating...' : 'Recommend Order'}
      </button>

      {inventoryMutation.data ? (
        <div className="mt-4 rounded-lg bg-mist p-3 text-sm dark:bg-slate-700/60">
          <p className="text-slate-700 dark:text-slate-200">
            Recommended order: <strong>{inventoryMutation.data.order_qty.toFixed(2)}</strong>
          </p>
          <p className="text-slate-700 dark:text-slate-200">
            Confidence: <strong>{(inventoryMutation.data.confidence * 100).toFixed(1)}%</strong>
          </p>
          <table className="mt-3 min-w-full text-xs" aria-label="Inventory recommendation details">
            <caption className="sr-only">Recommended order quantity and confidence values.</caption>
            <tbody>
              <tr className="border-b border-slate-200 text-slate-700 dark:border-slate-600 dark:text-slate-300">
                <th scope="row" className="py-1 pr-3 text-left font-medium">Store</th>
                <td className="py-1">{selectedStore}</td>
              </tr>
              <tr className="border-b border-slate-200 text-slate-700 dark:border-slate-600 dark:text-slate-300">
                <th scope="row" className="py-1 pr-3 text-left font-medium">Order Qty</th>
                <td className="py-1">{inventoryMutation.data.order_qty.toFixed(2)}</td>
              </tr>
              <tr className="text-slate-700 dark:text-slate-300">
                <th scope="row" className="py-1 pr-3 text-left font-medium">Confidence</th>
                <td className="py-1">{(inventoryMutation.data.confidence * 100).toFixed(1)}%</td>
              </tr>
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  )
}
