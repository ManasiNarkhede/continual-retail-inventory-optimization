import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'

import MobileNav from './MobileNav'

const mockNavLink = vi.fn()

vi.mock('react-router-dom', () => ({
  NavLink: (props) => {
    mockNavLink(props)
    return <a href={props.to} onClick={props.onClick} className={typeof props.className === 'function' ? props.className({ isActive: false }) : props.className}>{props.children}</a>
  },
}))

describe('MobileNav', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders toggle button', () => {
    render(<MobileNav />)
    expect(screen.getByLabelText('Toggle navigation')).toBeTruthy()
  })

  it('opens menu when toggle button is clicked', () => {
    render(<MobileNav />)
    const toggleButton = screen.getByLabelText('Toggle navigation')
    fireEvent.click(toggleButton)
    expect(screen.getByText('Dashboard')).toBeTruthy()
    expect(screen.getByText('Forecast')).toBeTruthy()
    expect(screen.getByText('Simulation')).toBeTruthy()
    expect(screen.getByText('CL Metrics')).toBeTruthy()
    expect(screen.getByText('HPO')).toBeTruthy()
  })

  it('closes menu when a link is clicked', () => {
    render(<MobileNav />)
    const toggleButton = screen.getByLabelText('Toggle navigation')
    fireEvent.click(toggleButton)
    expect(screen.getByText('Dashboard')).toBeTruthy()

    const dashboardLink = screen.getByText('Dashboard')
    fireEvent.click(dashboardLink)
    expect(screen.queryByText('Forecast')).toBeNull()
  })

  it('toggles menu open and closed', () => {
    render(<MobileNav />)
    const toggleButton = screen.getByLabelText('Toggle navigation')

    fireEvent.click(toggleButton)
    expect(screen.getByText('Dashboard')).toBeTruthy()

    fireEvent.click(toggleButton)
    expect(screen.queryByText('Dashboard')).toBeNull()
  })
})
