import axios from 'axios'
import type { QueryResponse } from './types'

export async function sendQuestion(question: string): Promise<QueryResponse> {
  const { data } = await axios.post<QueryResponse>('/api/query/', { question })
  return data
}
