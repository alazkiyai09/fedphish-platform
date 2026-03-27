/**
 * Security Status Panel
 */

import Card from '../common/Card'
import { useWebSocket } from '../../hooks/useWebSocket'
import { AlertTriangle, ShieldCheck, UserX } from 'lucide-react'

interface SecurityPanelProps {
  ws: ReturnType<typeof useWebSocket>
}

export default function SecurityPanel({ ws }: SecurityPanelProps) {
  const { status, injectAttack } = ws

  if (!status) return null

  const hasMalicious = status.banks.some((b) => b.is_malicious)

  return (
    <Card title="Security Status">
      {/* Attack Detection Alert */}
      {hasMalicious && (
        <div className="mb-4 p-4 bg-danger-50 dark:bg-danger-900/20 border border-danger-500 rounded-lg">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-danger-600 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="font-semibold text-danger-900 dark:text-danger-100">
                Malicious Activity Detected!
              </h4>
              <p className="text-sm text-danger-700 dark:text-danger-300 mt-1">
                Bank {status.banks.find((b) => b.is_malicious)?.bank_id} is sending malicious updates.
                Defense systems activated.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ZK Proof Status */}
      <div className="mb-6">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Zero-Knowledge Proofs
        </h3>
        <div className="space-y-2">
          {status.banks.map((bank) => (
            <div
              key={bank.bank_id}
              className="flex items-center justify-between text-sm"
            >
              <span className="text-gray-600 dark:text-gray-400">
                Bank {bank.bank_id}
              </span>
              <div className="flex items-center gap-1">
                <ShieldCheck
                  className={`w-4 h-4 ${
                    bank.is_malicious ? 'text-danger-600' : 'text-success-600'
                  }`}
                />
                <span
                  className={`font-medium ${
                    bank.is_malicious ? 'text-danger-600' : 'text-success-600'
                  }`}
                >
                  {bank.is_malicious ? 'INVALID' : 'VERIFIED'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Reputation Scores */}
      <div className="mb-6">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Bank Reputation Scores
        </h3>
        <div className="space-y-2">
          {status.banks.map((bank) => {
            const reputationPercent = bank.reputation * 100
            const reputationColor =
              reputationPercent >= 80
                ? 'bg-success-500'
                : reputationPercent >= 50
                ? 'bg-warning-500'
                : 'bg-danger-500'

            return (
              <div key={bank.bank_id} className="flex items-center gap-2">
                <span className="text-xs text-gray-600 dark:text-gray-400 w-16">
                  Bank {bank.bank_id}
                </span>
                <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                  <div
                    className={`h-3 rounded-full transition-all ${reputationColor}`}
                    style={{ width: `${reputationPercent}%` }}
                  />
                </div>
                <span className="text-xs font-semibold text-gray-900 dark:text-white w-10 text-right">
                  {reputationPercent.toFixed(0)}%
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Attack Controls (for demo) */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Attack Simulation (Demo)
        </h3>
        <div className="space-y-2">
          <button
            onClick={() => injectAttack(2, 'sign_flip')}
            className="w-full px-4 py-2 bg-danger-100 dark:bg-danger-900/30 text-danger-700 dark:text-danger-300 rounded-lg hover:bg-danger-200 dark:hover:bg-danger-900/50 transition-colors text-sm font-medium"
          >
            Inject Malicious Bank (Bank 2)
          </button>
          <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
            Simulates a Byzantine attack for demonstration
          </p>
        </div>
      </div>
    </Card>
  )
}
