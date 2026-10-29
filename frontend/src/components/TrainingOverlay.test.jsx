import React from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { act, render, screen } from '@testing-library/react'

import TrainingOverlay from './TrainingOverlay'
import useAppStore from '../store/appStore'

describe('TrainingOverlay', () => {
  const initialState = useAppStore.getState()

  beforeEach(() => {
    act(() => {
      useAppStore.setState({ ...initialState, isTraining: false })
    })
  })

  afterEach(() => {
    act(() => {
      useAppStore.setState({ ...initialState })
    })
  })

  it('renders message when training is active', () => {
    const { rerender } = render(<TrainingOverlay />)
    act(() => {
      useAppStore.setState({ isTraining: true })
    })
    rerender(<TrainingOverlay />)
    expect(screen.getByText('Model Training in Progress')).toBeTruthy()
  })

  it('renders nothing when training is inactive', () => {
    act(() => {
      useAppStore.setState({ isTraining: false })
    })
    const { container } = render(<TrainingOverlay />)
    expect(container.innerHTML).toBe('')
  })
})
