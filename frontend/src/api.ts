import axios from 'axios'
import type { QueryResponse, HistoryMessage, FeedbackPayload } from './types'

export async function sendQuestion(question: string, history: HistoryMessage[]): Promise<QueryResponse> {
  const { data } = await axios.post<QueryResponse>('/api/query/', { question, history })
  return data
}

export async function sendFeedback(payload: FeedbackPayload): Promise<void> {
  await axios.post('/api/feedback/', payload)
}
