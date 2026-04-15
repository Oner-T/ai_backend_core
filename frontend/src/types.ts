export interface Source {
  bolum: number | null
  madde: number | null
  madde_title: string | null
  is_cross_reference: boolean
  content: string
}

export interface QueryResponse {
  answer: string
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
