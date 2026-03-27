/**
 * Demo mode types
 */

export interface EmailAnalysis {
  text: string
  prediction: 'phishing' | 'legitimate'
  confidence: number
  probabilities: {
    legitimate: number
    phishing: number
  }
  features: ExtractedFeature[]
  explanation: string
}

export interface ExtractedFeature {
  name: string
  value: string | number
  importance: number
  category: 'url' | 'email' | 'content' | 'metadata'
}

export interface AttentionData {
  tokens: string[]
  attention_weights: number[][]
  layer: number
  head: number
}

export interface SampleEmail {
  id: string
  text: string
  label: 'phishing' | 'legitimate'
  category: string
  description: string
}
