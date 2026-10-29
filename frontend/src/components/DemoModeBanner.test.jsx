import React from 'react'
import { fireEvent, render, screen } from '@testing-library/react'

import DemoModeBanner from './DemoModeBanner'

describe('DemoModeBanner', () => {
  it('renders banner when demo mode is enabled', () => {
    render(<DemoModeBanner isDemo />)
    expect(screen.getByText(/Demo Mode/)).toBeTruthy()
  })

  it('calls dismiss callback', () => {
    const onDismiss = vi.fn()
    render(<DemoModeBanner isDemo onDismiss={onDismiss} />)
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }))
    expect(onDismiss).toHaveBeenCalledTimes(1)
  })
})
