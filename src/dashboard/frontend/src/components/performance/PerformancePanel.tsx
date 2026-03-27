/**
 * Performance Metrics Panel
 */

import Card from '../common/Card'
import { useWebSocket } from '../../hooks/useWebSocket'

interface PerformancePanelProps {
  ws: ReturnType<typeof useWebSocket>
}

export default function PerformancePanel({ ws }: PerformancePanelProps) {
  const { status } = ws

  if (!status) return null

  // Mock accuracy history (in real app, this would come from WebSocket)
  const accuracyHistory = [
    0.75, 0.78, 0.81, 0.83, 0.85, 0.87, 0.88, 0.89, 0.90, 0.91,
    0.915, 0.92, 0.923, 0.925, 0.927, 0.93, 0.932, 0.934, 0.935,
  ].slice(0, status.round)

  return (
    <Card title="Model Performance">
      {/* Accuracy Chart */}
      <div className="mb-6">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Global Accuracy Over Rounds
        </h3>
        <div className="h-48 flex items-end gap-1">
          {accuracyHistory.map((acc, idx) => (
            <div
              key={idx}
              className="flex-1 bg-primary-500 dark:bg-primary-600 rounded-t transition-all hover:bg-primary-600"
              style={{ height: `${(acc - 0.7) * 250}px` }}
              title={`Round ${idx + 1}: ${(acc * 100).toFixed(1)}%`}
            />
          ))}
        </div>
        <div className="flex justify-between mt-2 text-xs text-gray-600 dark:text-gray-400">
          <span>0</span>
          <span>Round {status.round}</span>
        </div>
      </div>

      {/* Bank Comparison */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Bank Accuracy Comparison
        </h3>
        <div className="space-y-2">
          {status.banks.map((bank) => (
            <div key={bank.bank_id} className="flex items-center gap-2">
              <span className="text-xs text-gray-600 dark:text-gray-400 w-20">
                Bank {bank.bank_id}
              </span>
              <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-4">
                <div
                  className={`h-4 rounded-full transition-all ${
                    bank.is_malicious
                      ? 'bg-danger-500'
                      : 'bg-success-500'
                  }`}
                  style={{ width: `${bank.accuracy * 100}%` }}
                />
              </div>
              <span className="text-xs font-semibold text-gray-900 dark:text-white w-12 text-right">
                {(bank.accuracy * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  )
}
