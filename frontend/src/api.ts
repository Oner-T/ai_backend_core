import axios from 'axios'
import type { QueryResponse, HistoryMessage, FeedbackPayload, AuthTokens, ChatSession, ChatMessageRecord, Regime } from './types'

const api = axios.create({ baseURL: '/api' })

// ── Token helpers ─────────────────────────────────────────────────────────────

export function getAccessToken()  { return localStorage.getItem('access') }
export function getRefreshToken() { return localStorage.getItem('refresh') }
export function getEmail()        { return localStorage.getItem('email') }

export function saveTokens({ access, refresh, email }: AuthTokens) {
  localStorage.setItem('access',  access)
  localStorage.setItem('refresh', refresh)
  localStorage.setItem('email',   email)
}

export function clearTokens() {
  localStorage.removeItem('access')
  localStorage.removeItem('refresh')
  localStorage.removeItem('email')
}

// ── Attach Bearer token to every request ─────────────────────────────────────

api.interceptors.request.use((config) => {
  const token = getAccessToken()
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// ── On 401: try to refresh, retry once, then force logout ────────────────────

let isRefreshing = false
let failedQueue: Array<{ resolve: (v: string) => void; reject: (e: unknown) => void }> = []

function processQueue(error: unknown, token: string | null) {
  failedQueue.forEach((p) => (error ? p.reject(error) : p.resolve(token!)))
  failedQueue = []
}

api.interceptors.response.use(
  (res) => res,
  async (error) => {
    const original = error.config
    if (error.response?.status === 401 && !original._retry) {
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject })
        }).then((token) => {
          original.headers.Authorization = `Bearer ${token}`
          return api(original)
        })
      }

      original._retry  = true
      isRefreshing     = true
      const refresh    = getRefreshToken()

      if (!refresh) {
        clearTokens()
        window.location.href = '/'
        return Promise.reject(error)
      }

      try {
        const { data } = await axios.post('/api/auth/refresh/', { refresh })
        localStorage.setItem('access', data.access)
        if (data.refresh) localStorage.setItem('refresh', data.refresh)
        processQueue(null, data.access)
        original.headers.Authorization = `Bearer ${data.access}`
        return api(original)
      } catch (err) {
        processQueue(err, null)
        clearTokens()
        window.location.href = '/'
        return Promise.reject(err)
      } finally {
        isRefreshing = false
      }
    }
    return Promise.reject(error)
  }
)

// ── Auth ──────────────────────────────────────────────────────────────────────

export async function register(email: string, password: string, password2: string): Promise<AuthTokens> {
  const { data } = await api.post<AuthTokens>('/auth/register/', { email, password, password2 })
  return data
}

export async function login(email: string, password: string): Promise<AuthTokens> {
  const { data } = await api.post<AuthTokens>('/auth/login/', { email, password })
  return data
}

// ── Chat ──────────────────────────────────────────────────────────────────────

export async function sendQuestion(
  question: string,
  history: HistoryMessage[],
  session_id?: number | null,
  regime: Regime = 'tr',
): Promise<QueryResponse> {
  const { data } = await api.post<QueryResponse>('/query/', { question, history, session_id, regime })
  return data
}

export async function sendFeedback(payload: FeedbackPayload): Promise<void> {
  await api.post('/feedback/', payload)
}

// ── Sessions ──────────────────────────────────────────────────────────────────

export async function fetchSessions(): Promise<ChatSession[]> {
  const { data } = await api.get<ChatSession[]>('/sessions/')
  return data
}

export async function fetchSessionMessages(sessionId: number): Promise<ChatMessageRecord[]> {
  const { data } = await api.get<ChatMessageRecord[]>(`/sessions/${sessionId}/messages/`)
  return data
}
