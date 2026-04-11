export interface Source {
  bolum: number | null
  madde: number | null
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
