import React from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { vi } from 'vitest'

import StoreSelector from './StoreSelector'

vi.mock('../api/client', () => ({
  useStores: () => ({
    data: {
      stores: [1, 2, 3],
      total: 3,
    },
    isLoading: false,
    error: null,
  }),
}))

describe('StoreSelector', () => {
  it('renders selector title and stats', () => {
    render(<StoreSelector selected={1} onSelect={() => {}} />)
    expect(screen.getByText('Store Selector')).toBeTruthy()
    expect(screen.getByText(/Total stores: 3/)).toBeTruthy()
  })

  it('exposes accessibility labels for search control', () => {
    render(<StoreSelector selected={1} onSelect={() => {}} />)
    expect(screen.getByLabelText('Search store by id')).toBeTruthy()
  })

  it('renders store options', () => {
    render(<StoreSelector selected={1} onSelect={() => {}} />)
    expect(screen.getByRole('option', { name: 'Select store 1' })).toBeTruthy()
    expect(screen.getByRole('option', { name: 'Select store 2' })).toBeTruthy()
    expect(screen.getByRole('option', { name: 'Select store 3' })).toBeTruthy()
  })
})
