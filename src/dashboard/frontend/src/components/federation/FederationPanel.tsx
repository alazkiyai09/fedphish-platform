/**
 * Federation Overview Panel
 */

import Card from '../common/Card'
import { useWebSocket } from '../../hooks/useWebSocket'

interface FederationPanelProps {
  ws: ReturnType<typeof useWebSocket>
}

export default function FederationPanel({ ws }: FederationPanelProps) {
  const { status } = ws

  if (!status) return null

  const bankStatuses = status.banks.map((bank) => {
    if (bank.is_malicious) return 'malicious'
    if (status.is_running) return 'training'
    if (status.round >= status.total_rounds) return 'complete'
    return 'idle'
  })

  return (
    <Card title="Federation Overview">
      {/* Bank Map Visualization */}
      <div className="mb-6">
        <div className="flex flex-wrap gap-4">
          {status.banks.map((bank, idx) => (
            <div
              key={bank.bank_id}
              className={`flex-1 min-w-[200px] p-4 rounded-lg border-2 transition-all ${
                bank.is_malicious
                  ? 'bg-danger-50 border-danger-500 dark:bg-danger-900/20'
                  : bankStatuses[idx] === 'training'
                  ? 'bg-primary-50 border-primary-500 dark:bg-primary-900/20'
                  : bankStatuses[idx] === 'complete'
                  ? 'bg-success-50 border-success-500 dark:bg-success-900/20'
                  : 'bg-gray-50 border-gray-300 dark:bg-gray-700 dark:border-gray-600'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  {bank.name}
                </h3>
                {bank.is_malicious && (
                  <span className="px-2 py-1 text-xs font-medium bg-danger-600 text-white rounded">
                    MALICIOUS
                  </span>
                )}
              </div>

              <div className="space-y-1 text-sm">
                <p className="text-gray-600 dark:text-gray-400">
                  📍 {bank.location}
                </p>
                <p className="text-gray-600 dark:text-gray-400">
                  Accuracy: <span className="font-semibold text-gray-900 dark:text-white">
                    {(bank.accuracy * 100).toFixed(1)}%
                  </span>
                </p>
                <p className="text-gray-600 dark:text-gray-400">
                  Status: <span className="font-semibold text-gray-900 dark:text-white capitalize">
                    {bankStatuses[idx]}
                  </span>
                </p>
                <p className="text-gray-600 dark:text-gray-400">
                  Reputation: <span className="font-semibold text-gray-900 dark:text-white">
                    {(bank.reputation * 100).toFixed(0)}%
                  </span>
                </p>
              </div>

              {/* Training Progress Bar */}
              {bankStatuses[idx] === 'training' && (
                <div className="mt-3">
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${bank.accuracy * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Communication Flow Visualization */}
      <div className="mt-6">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Communication Flow
        </h3>
        <div className="flex items-center justify-between">
          {status.banks.map((bank) => (
            <div
              key={bank.bank_id}
              className="flex flex-col items-center"
            >
              <div className="w-12 h-12 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center">
                <span className="text-primary-600 dark:text-primary-400 font-bold">
                  {bank.bank_id}
                </span>
              </div>
              <span className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                Bank {bank.bank_id}
              </span>
            </div>
          ))}

          {/* Animated data flow */}
          {status.is_running && !status.is_paused && (
            <>
              <div className="flex-1 h-1 bg-gradient-to-r from-primary-500 to-success-500 mx-2 animate-pulse" />
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary-500 to-success-500 flex items-center justify-center">
                <span className="text-white font-bold text-xl">S</span>
              </div>
              <div className="flex-1 h-1 bg-gradient-to-r from-success-500 to-primary-500 mx-2 animate-pulse" />
            </>
          )}
        </div>

        {status.is_running && !status.is_paused && (
          <div className="mt-4 flex items-center justify-center gap-2 text-sm text-gray-600 dark:text-gray-400">
            <span className="animate-pulse">🔒 Encrypted Updates</span>
            <span>→</span>
            <span className="animate-pulse">✓ ZK Proofs Verified</span>
            <span>→</span>
            <span className="animate-pulse">⚡ Aggregation</span>
          </div>
        )}
      </div>
    </Card>
  )
}
