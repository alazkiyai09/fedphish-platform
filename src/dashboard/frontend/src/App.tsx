/**
 * FedPhish Demo Dashboard - Main App Component
 */

import { useState } from 'react'
import { useTheme } from './hooks/useTheme'
import { useWebSocket } from './hooks/useWebSocket'
import Header from './components/common/Header'
import DashboardLayout from './layouts/DashboardLayout'

function App() {
  const { theme, toggleTheme } = useTheme()
  const [scenario, setScenario] = useState('happy_path')

  const ws = useWebSocket(scenario)

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
      <Header
        scenario={scenario}
        onScenarioChange={setScenario}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <main className="container mx-auto px-4 py-6">
        {ws.connected ? (
          <DashboardLayout ws={ws} scenario={scenario} />
        ) : (
          <div className="flex items-center justify-center h-[60vh]">
            <div className="text-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary-600 mx-auto mb-4"></div>
              <p className="text-gray-600 dark:text-gray-400">
                {ws.error ? ws.error : 'Connecting to server...'}
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
