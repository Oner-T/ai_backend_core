export interface Source {
  bolum: number | null
  madde: number | null
  madde_title: string | null
  document_name?: string
  content: string
}

export interface QueryResponse {
  answer: string
  model: string
  sources: Source[]
}

export interface Message {
  role: 'user' | 'assistant'
  text: string
  sources?: Source[]
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
