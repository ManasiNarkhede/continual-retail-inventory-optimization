import { useEffect, useMemo, useState } from 'react'
import { FixedSizeList as List } from 'react-window'

import { useStores } from '../api/client'

function Row({ data, index, style }) {
  const { stores, selected, onSelect } = data
  const storeId = stores[index]
  const isSelected = selected === storeId

  return (
    <button
      type="button"
      role="option"
      style={style}
      onClick={() => onSelect(storeId)}
      aria-selected={isSelected}
      aria-label={`Select store ${storeId}`}
      className={`w-full px-3 text-left text-sm ${
        isSelected
          ? 'bg-slate-900 text-white dark:bg-teal-600 dark:text-white'
          : 'bg-white text-slate-700 hover:bg-slate-100 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700'
      }`}
    >
      <span>Store {storeId}</span>
    </button>
  )
}

export default function StoreSelector({ selected, onSelect }) {
  const { data, isLoading, error } = useStores()
  const [query, setQuery] = useState('')

  const stores = data?.stores ?? []

  const filteredStores = useMemo(() => {
    if (!query.trim()) {
      return stores
    }
    const normalized = query.trim().toLowerCase()
    return stores.filter((storeId) => String(storeId).includes(normalized))
  }, [stores, query])

  useEffect(() => {
    if (!data) {
      return
    }
    if (filteredStores.length === 0) {
      return
    }
    if (!filteredStores.includes(selected)) {
      onSelect(filteredStores[0])
    }
  }, [data, filteredStores, selected, onSelect])

  return (
    <div className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-panel dark:border-slate-700 dark:bg-slate-800/90 dark:shadow-panel-dark">
      <h2 id="store-selector-title" className="font-display text-lg text-slate-900 dark:text-slate-100">Store Selector</h2>
      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
        Search by store number. This list shows all 1,115 stores.
      </p>

      <input
        className="mt-3 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-700 dark:text-slate-100 dark:placeholder:text-slate-400"
        placeholder="Type store id (e.g. 512)"
        aria-label="Search store by id"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />

      {isLoading ? <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">Loading stores...</p> : null}

      {error ? (
        <p className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700 dark:bg-red-900/30 dark:text-red-300">
          Could not load store list. Please verify backend connectivity.
        </p>
      ) : null}

      {data?.total ? (
        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
          Total stores: {data.total}
        </p>
      ) : null}

      <div className="mt-3 overflow-hidden rounded-lg border border-slate-200 dark:border-slate-600" role="region" aria-labelledby="store-selector-title">
        <List
          height={260}
          width="100%"
          role="listbox"
          aria-label="Store results"
          aria-describedby="store-selector-help"
          itemCount={filteredStores.length}
          itemSize={36}
          itemData={{ stores: filteredStores, selected, onSelect }}
        >
          {Row}
        </List>
      </div>
      <p id="store-selector-help" className="sr-only">
        Use arrow keys or mouse to choose a store.
      </p>
    </div>
  )
}
