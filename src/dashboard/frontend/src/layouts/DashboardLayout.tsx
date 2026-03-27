/**
 * Main Dashboard Layout
 */

import { Card } from '../components/common/Card'
import FederationPanel from '../components/federation/FederationPanel'
import PerformancePanel from '../components/performance/PerformancePanel'
import PrivacyPanel from '../components/privacy/PrivacyPanel'
import SecurityPanel from '../components/security/SecurityPanel'

interface DashboardLayoutProps {
  ws: ReturnType<typeof import('../hooks/useWebSocket').useWebSocket>
  scenario: string
}

export default function DashboardLayout({ ws, scenario }: DashboardLayoutProps) {
  if (!ws.status) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading...</p>
        </div>
      </div>
    )
  }

  const { status, start, pause, resume, reset } = ws

  return (
    <div className="space-y-6">
      {/* Control Bar */}
      <Card>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Scenario</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-white">
                {status.scenario}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Round</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-white">
                {status.round} / {status.total_rounds}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Accuracy</p>
              <p className="text-lg font-semibold text-primary-600">
                {(status.global_accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Privacy (ε)</p>
              <p className="text-lg font-semibold text-warning-600">
                {status.privacy_epsilon_spent.toFixed(1)}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {!status.is_running ? (
              <button
                onClick={start}
                className="flex items-center gap-2 px-4 py-2 bg-success-600 text-white rounded-lg hover:bg-success-700 transition-colors"
              >
                <Play className="w-4 h-4" />
                Start
              </button>
            ) : status.is_paused ? (
              <button
                onClick={resume}
                className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
              >
                <Play className="w-4 h-4" />
                Resume
              </button>
            ) : (
              <button
                onClick={pause}
                className="flex items-center gap-2 px-4 py-2 bg-warning-600 text-white rounded-lg hover:bg-warning-700 transition-colors"
              >
                <Pause className="w-4 h-4" />
                Pause
              </button>
            )}

            <button
              onClick={reset}
              className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              Reset
            </button>
          </div>
        </div>
      </Card>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Federation Overview */}
        <div className="lg:col-span-2">
          <FederationPanel ws={ws} />
        </div>

        {/* Performance Metrics */}
        <PerformancePanel ws={ws} />

        {/* Privacy Panel */}
        <PrivacyPanel ws={ws} />

        {/* Security Panel */}
        <SecurityPanel ws={ws} />
      </div>
    </div>
  )
}
