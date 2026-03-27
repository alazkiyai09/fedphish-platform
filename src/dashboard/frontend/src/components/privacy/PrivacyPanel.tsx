/**
 * Privacy Metrics Panel
 */

import Card from '../common/Card'
import { useWebSocket } from '../../hooks/useWebSocket'
import { Lock, Shield, Eye } from 'lucide-react'

interface PrivacyPanelProps {
  ws: ReturnType<typeof useWebSocket>
}

export default function PrivacyPanel({ ws }: PrivacyPanelProps) {
  const { status, updatePrivacy } = ws

  if (!status) return null

  const handlePrivacyLevelChange = async (level: number) => {
    await updatePrivacy(level, 1.0)
  }

  return (
    <Card title="Privacy Mechanisms">
      {/* Privacy Level Indicator */}
      <div className="mb-6">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Privacy Level
        </h3>
        <div className="flex items-center gap-2">
          {[1, 2, 3].map((level) => (
            <button
              key={level}
              onClick={() => handlePrivacyLevelChange(level)}
              className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${
                level === 1
                  ? 'bg-warning-100 text-warning-700 border-2 border-warning-500 dark:bg-warning-900/20 dark:text-warning-300'
                  : level === 2
                  ? 'bg-primary-100 text-primary-700 border-2 border-primary-500 dark:bg-primary-900/20 dark:text-primary-300'
                  : 'bg-success-100 text-success-700 border-2 border-success-500 dark:bg-success-900/20 dark:text-success-300'
              }`}
            >
              Level {level}
            </button>
          ))}
        </div>
        <div className="mt-3 text-sm text-gray-600 dark:text-gray-400 space-y-1">
          <p>Level 1: Local DP only</p>
          <p>Level 2: DP + Homomorphic Encryption</p>
          <p>Level 3: DP + HE + Trusted Execution</p>
        </div>
      </div>

      {/* Privacy Budget */}
      <div className="mb-6">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Privacy Budget (ε)
        </h3>
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-success-500 to-warning-500 transition-all"
                style={{
                  width: `${Math.min(
                    100,
                    (status.privacy_epsilon_spent / 20) * 100
                  )}%`
                }}
              />
            </div>
          </div>
          <span className="text-sm font-semibold text-gray-900 dark:text-white">
            {status.privacy_epsilon_spent.toFixed(1)}
          </span>
        </div>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Budget used: {status.privacy_epsilon_spent.toFixed(1)} / 20.0
        </p>
      </div>

      {/* Encryption Status */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Encryption Status
        </h3>

        <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <div className="flex items-center gap-2">
            <Lock className="w-5 h-5 text-success-600" />
            <span className="text-sm text-gray-900 dark:text-white">
              Homomorphic Encryption
            </span>
          </div>
          <div className={`text-xs font-medium ${
            status.is_running ? 'text-success-600' : 'text-gray-500'
          }`}>
            {status.is_running ? 'ACTIVE' : 'INACTIVE'}
          </div>
        </div>

        <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <div className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-primary-600" />
            <span className="text-sm text-gray-900 dark:text-white">
              Zero-Knowledge Proofs
            </span>
          </div>
          <div className={`text-xs font-medium ${
            status.is_running ? 'text-success-600' : 'text-gray-500'
          }`}>
            {status.is_running ? 'ACTIVE' : 'INACTIVE'}
          </div>
        </div>

        <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <div className="flex items-center gap-2">
            <Eye className="w-5 h-5 text-warning-600" />
            <span className="text-sm text-gray-900 dark:text-white">
              Trusted Execution Environment
            </span>
          </div>
          <div className="text-xs font-medium text-gray-500">
            OPTIONAL
          </div>
        </div>
      </div>
    </Card>
  )
}
