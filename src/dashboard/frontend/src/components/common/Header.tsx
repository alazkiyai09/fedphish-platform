/**
 * Header component
 */

import { Play, Pause, RotateCcw, Moon, Sun } from 'lucide-react'

interface HeaderProps {
  scenario: string
  onScenarioChange: (scenario: string) => void
  theme: 'light' | 'dark'
  onToggleTheme: () => void
}

const scenarios = [
  { value: 'happy_path', label: 'Happy Path' },
  { value: 'non_iid', label: 'Non-IID Challenge' },
  { value: 'attack_scenario', label: 'Attack Scenario' },
  { value: 'privacy_mode', label: 'Privacy Mode' },
]

export default function Header({ scenario, onScenarioChange, theme, onToggleTheme }: HeaderProps) {
  return (
    <header className="bg-white dark:bg-gray-800 shadow-md border-b border-gray-200 dark:border-gray-700">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo & Title */}
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-700 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-xl">FP</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                FedPhish Demo Dashboard
              </h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Federated Phishing Detection for Financial Institutions
              </p>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-4">
            {/* Scenario Selector */}
            <select
              value={scenario}
              onChange={(e) => onScenarioChange(e.target.value)}
              className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              {scenarios.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>

            {/* Theme Toggle */}
            <button
              onClick={onToggleTheme}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              aria-label="Toggle theme"
            >
              {theme === 'light' ? (
                <Moon className="w-5 h-5" />
              ) : (
                <Sun className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>
      </div>
    </header>
  )
}
