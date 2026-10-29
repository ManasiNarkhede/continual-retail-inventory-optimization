import React from 'react'

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, message: '' }
    this.handleReset = this.handleReset.bind(this)
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, message: error?.message || 'Unexpected UI error.' }
  }

  componentDidCatch(error) {
    // eslint-disable-next-line no-console
    console.error('UI boundary error:', error)
  }

  handleReset() {
    this.setState({ hasError: false, message: '' })
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="mx-auto max-w-xl rounded-xl border border-red-200 bg-red-50 p-5 text-red-800 shadow-sm">
          <p className="text-lg font-semibold">Something went wrong</p>
          <p className="mt-1 text-sm text-red-700">A rendering error interrupted this view.</p>
          {this.state.message ? <p className="mt-2 text-xs text-red-600">{this.state.message}</p> : null}
          <button
            type="button"
            onClick={this.handleReset}
            className="mt-4 rounded-lg bg-red-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-red-800"
          >
            Try Again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
