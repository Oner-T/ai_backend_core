import axios from 'axios'
import type { QueryResponse, HistoryMessage } from './types'

export async function sendQuestion(question: string, history: HistoryMessage[]): Promise<QueryResponse> {
  const { data } = await axios.post<QueryResponse>('/api/query/', { question, history })
  return data
}
