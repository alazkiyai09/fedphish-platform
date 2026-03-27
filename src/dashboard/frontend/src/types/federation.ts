/**
 * Federation-related types
 */

export interface Bank {
  bank_id: number
  name: string
  location: string
  lat: number
  lon: number
  data_distribution: string
  phishing_types: string[]
  is_malicious: boolean
}

export interface BankState {
  bank_id: number
  name: string
  location: string
  accuracy: number
  loss: number
  samples: number
  status: 'idle' | 'training' | 'complete' | 'malicious'
  gradient_norm: number
  reputation: number
  is_malicious: boolean
}

export interface RoundUpdate {
  type: 'round_update'
  round: number
  is_complete: boolean
  global_accuracy: number
  global_loss: number
  banks: BankState[]
  privacy: PrivacyState
  security: SecurityState
  scenario: ScenarioInfo
}

export interface PrivacyState {
  epsilon_spent: number
  epsilon_limit: number
  privacy_level: number
  encryption_active: boolean
  tee_mode: boolean
}

export interface SecurityState {
  attack_detected: boolean
  defense_activated: boolean
  malicious_bank_id: number | null
}

export interface ScenarioInfo {
  name: string
  total_rounds: number
}

export interface FederationStatus {
  scenario: string
  round: number
  total_rounds: number
  is_running: boolean
  is_paused: boolean
  global_accuracy: number
  privacy_epsilon_spent: number
  num_banks: number
  banks: Bank[]
}

export type WebSocketMessage =
  | RoundUpdate
  | { type: 'subscribed'; scenario: string; status: FederationStatus }
  | { type: 'started'; scenario: string }
  | { type: 'paused'; scenario: string }
  | { type: 'resumed'; scenario: string }
  | { type: 'reset'; scenario: string; status: FederationStatus }
  | { type: 'training_complete'; scenario: string; final_accuracy: number }
  | { type: 'attack_injected'; scenario: string; bank_id: number; attack_type: string }
  | { type: 'privacy_updated'; scenario: string; privacy_level: number }
  | { type: 'error'; message: string }
