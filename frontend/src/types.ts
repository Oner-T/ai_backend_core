export interface Source {
  bolum: number | null
  madde: number | null
  madde_title: string | null
  document_name?: string
  content: string
}

export type Regime = 'tr' | 'eu'

export interface QueryResponse {
  answer: string
  model: string
  regime: Regime
  sources: Source[]
  session_id: number
}

export interface Message {
  role: 'user' | 'assistant'
  text: string
  sources?: Source[]
  regime?: Regime
}

export interface HistoryMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface FeedbackPayload {
  question: string
  answer: string
  sources: Source[]
  rating: 'good' | 'bad'
  comment?: string
}

export interface AuthTokens {
  access: string
  refresh: string
  email: string
}

export interface ChatSession {
  id: number
  title: string
  created_at: string
  updated_at: string
}

export interface ChatMessageRecord {
  role: 'user' | 'assistant'
  content: string
  sources: Source[]
  created_at: string
}
