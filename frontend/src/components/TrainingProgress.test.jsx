import React from 'react'
import { describe, expect, it } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'

import TrainingProgress from './TrainingProgress'

describe('TrainingProgress', () => {
  it('renders training in progress state', () => {
    render(
      <TrainingProgress
        epoch={5}
        totalEpochs={50}
        currentLoss={2.34}
        bestLoss={1.89}
        isTraining={true}
      />
    )
    expect(screen.getByText('Training in Progress')).toBeTruthy()
    expect(screen.getByText(/Epoch 5 of 50/)).toBeTruthy()
  })

  it('renders training complete state', () => {
    render(
      <TrainingProgress
        epoch={50}
        totalEpochs={50}
        currentLoss={1.23}
        bestLoss={1.23}
        isTraining={false}
      />
    )
    expect(screen.getByText('Training Complete')).toBeTruthy()
  })

  it('renders nothing when not training and epoch is 0', () => {
    const { container } = render(
      <TrainingProgress
        epoch={0}
        totalEpochs={50}
        currentLoss={0}
        bestLoss={0}
        isTraining={false}
      />
    )
    expect(container.innerHTML).toBe('')
  })

  it('displays current loss, best loss, and progress', () => {
    render(
      <TrainingProgress
        epoch={25}
        totalEpochs={50}
        currentLoss={2.3456}
        bestLoss={1.8912}
        isTraining={true}
      />
    )
    expect(screen.getByText('2.3456')).toBeTruthy()
    expect(screen.getByText('1.8912')).toBeTruthy()
    expect(screen.getByText('50.0%')).toBeTruthy()
  })

  it('toggles expand/collapse on click', () => {
    render(
      <TrainingProgress
        epoch={10}
        totalEpochs={50}
        currentLoss={2.0}
        bestLoss={1.5}
        isTraining={true}
      />
    )
    // Initially expanded — progress bar and metrics visible
    expect(screen.getByText('Current Loss')).toBeTruthy()

    // Click to collapse
    fireEvent.click(screen.getByLabelText('Collapse training details'))
    expect(screen.queryByText('Current Loss')).toBeNull()

    // Click to expand again
    fireEvent.click(screen.getByLabelText('Expand training details'))
    expect(screen.getByText('Current Loss')).toBeTruthy()
  })

  it('shows ETA when training and epochs remain', () => {
    render(
      <TrainingProgress
        epoch={10}
        totalEpochs={50}
        currentLoss={2.0}
        bestLoss={1.5}
        isTraining={true}
      />
    )
    // 40 epochs remaining * 2.5 min = 100 min
    expect(screen.getByText(/~100 min remaining/)).toBeTruthy()
  })
})
