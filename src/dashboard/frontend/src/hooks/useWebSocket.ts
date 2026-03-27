/**
 * WebSocket hook for real-time updates
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import { WebSocketMessage, FederationStatus } from '../types/federation'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8001/ws/federation'

export function useWebSocket(scenario: string = 'happy_path') {
  const [status, setStatus] = useState<FederationStatus | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const messageHandlersRef = useRef<Map<string, (msg: any) => void>>(new Map())

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    }
  }, [])

  const on = useCallback((eventType: string, handler: (msg: any) => void) => {
    messageHandlersRef.current.set(eventType, handler)
    return () => {
      messageHandlersRef.current.delete(eventType)
    }
  }, [])

  useEffect(() => {
    let reconnectTimeout: NodeJS.Timeout

    const connect = () => {
      try {
        const ws = new WebSocket(WS_URL)
        wsRef.current = ws

        ws.onopen = () => {
          console.log('WebSocket connected')
          setConnected(true)
          setError(null)

          // Subscribe to scenario
          ws.send(JSON.stringify({
            action: 'subscribe',
            scenario,
          }))
        }

        ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data)

            // Handle message types
            if (message.type === 'subscribed') {
              setStatus(message.status)
            } else if (message.type === 'round_update') {
              setStatus((prev) => ({
                ...prev,
                round: message.round,
                is_running: true,
                global_accuracy: message.global_accuracy,
                privacy_epsilon_spent: message.privacy.epsilon_spent,
                scenario: message.scenario.name,
                total_rounds: message.scenario.total_rounds,
                is_paused: false,
                num_banks: message.banks.length,
                banks: message.banks.map((b) => ({
                  bank_id: b.bank_id,
                  name: `Bank ${b.bank_id}`,
                  location: '',
                  lat: 0,
                  lon: 0,
                  data_distribution: '',
                  phishing_types: [],
                  is_malicious: b.bank_id === message.security.malicious_bank_id,
                })),
              }))
            }

            // Call registered handlers
            const handler = messageHandlersRef.current.get(message.type)
            if (handler) {
              handler(message)
            }
          } catch (err) {
            console.error('Error parsing WebSocket message:', err)
          }
        }

        ws.onclose = () => {
          console.log('WebSocket disconnected')
          setConnected(false)
          // Reconnect after 3 seconds
          reconnectTimeout = setTimeout(connect, 3000)
        }

        ws.onerror = (error) => {
          console.error('WebSocket error:', error)
          setError('WebSocket connection error')
        }
      } catch (err) {
        console.error('Failed to create WebSocket:', err)
        setError('Failed to connect to server')
      }
    }

    connect()

    return () => {
      clearTimeout(reconnectTimeout)
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [scenario])

  const start = useCallback(() => {
    sendMessage({ action: 'start', scenario })
  }, [sendMessage, scenario])

  const pause = useCallback(() => {
    sendMessage({ action: 'pause', scenario })
  }, [sendMessage, scenario])

  const resume = useCallback(() => {
    sendMessage({ action: 'resume', scenario })
  }, [sendMessage, scenario])

  const reset = useCallback(() => {
    sendMessage({ action: 'reset', scenario })
  }, [sendMessage, scenario])

  const injectAttack = useCallback((bankId: number, attackType: string = 'sign_flip') => {
    sendMessage({ action: 'inject_attack', scenario, bank_id, attack_type: attackType })
  }, [sendMessage, scenario])

  const updatePrivacy = useCallback((level: number, epsilon: number) => {
    sendMessage({ action: 'update_privacy', scenario, privacy_level: level, epsilon })
  }, [sendMessage, scenario])

  return {
    status,
    connected,
    error,
    sendMessage,
    start,
    pause,
    resume,
    reset,
    injectAttack,
    updatePrivacy,
    on,
  }
}
