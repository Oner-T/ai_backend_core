import { useState, useRef, useEffect } from 'react'
import type { KeyboardEvent, ReactNode } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { sendQuestion, sendFeedback, fetchSessions, fetchSessionMessages } from './api'
import { useAuth } from './AuthContext'
import type { Message, Source, HistoryMessage, ChatSession, Regime } from './types'

const REGIME_CONFIG = {
  tr: {
    label:       'KVKK',
    flag:        '🇹🇷',
    title:       'KVKK Asistanı',
    placeholder: 'KVKK hakkında bir soru sorun…',
    emptyTitle:  'KVKK hakkında soru sorun',
    emptySubtitle: 'Yanıtlar yalnızca KVKK mevzuatı içeriğine dayalıdır.',
    suggestions: [
      'Kişisel veri nedir?',
      'Veri sorumlusunun yükümlülükleri nelerdir?',
      'Açık rıza şartları nelerdir?',
      'Veri ihlali durumunda ne yapılmalıdır?',
    ],
  },
  eu: {
    label:       'EU Regulations',
    flag:        '🇪🇺',
    title:       'EU Data & AI Law',
    placeholder: 'Ask about GDPR, ePrivacy Directive, or the EU AI Act…',
    emptyTitle:  'Ask about EU data protection & AI law',
    emptySubtitle: 'Answers are based strictly on GDPR, ePrivacy Directive, and EU AI Act.',
    suggestions: [
      'What are the lawful bases for processing under GDPR?',
      'What is a data breach notification requirement?',
      'What are high-risk AI systems under the EU AI Act?',
      'What rights do data subjects have under GDPR?',
    ],
  },
} as const

function renderAnswerWithCitations(text: string, sources: Source[], regime: Regime): ReactNode {
  const verifiedMaddes = new Set(sources.map((s) => s.madde).filter(Boolean))

  if (regime === 'eu') {
    const parts = text.split(/(Article\s+\d{1,3})/gi)
    return parts.map((part, i) => {
      const match = part.match(/^Article\s+(\d{1,3})$/i)
      if (!match) return part
      const num = parseInt(match[1])
      const isVerified = verifiedMaddes.has(num)
      return isVerified ? (
        <mark key={i} className="bg-blue-100 text-blue-700 rounded px-0.5 not-italic font-medium">
          {part}
        </mark>
      ) : (
        <mark
          key={i}
          title="This article could not be verified in the sources"
          className="bg-amber-100 text-amber-700 rounded px-0.5 not-italic font-medium inline-flex items-center gap-0.5"
        >
          {part}
          <svg className="w-3 h-3 shrink-0 inline" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
          </svg>
        </mark>
      )
    })
  }

  // Turkish regime: highlight "Madde X"
  const parts = text.split(/(Madde\s+\d{1,3})/gi)
  return parts.map((part, i) => {
    const match = part.match(/^Madde\s+(\d{1,3})$/i)
    if (!match) return part
    const num = parseInt(match[1])
    const isVerified = verifiedMaddes.has(num)
    return isVerified ? (
      <mark key={i} className="bg-blue-100 text-blue-700 rounded px-0.5 not-italic font-medium">
        {part}
      </mark>
    ) : (
      <mark
        key={i}
        title="Bu madde kaynaklarda doğrulanamadı"
        className="bg-amber-100 text-amber-700 rounded px-0.5 not-italic font-medium inline-flex items-center gap-0.5"
      >
        {part}
        <svg className="w-3 h-3 shrink-0 inline" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
        </svg>
      </mark>
    )
  })
}

function formatDate(iso: string) {
  const d = new Date(iso)
  return d.toLocaleDateString('tr-TR', { day: 'numeric', month: 'short' })
}

function sourceLabel(s: Source, regime: Regime): string {
  const docPrefix = s.document_name && s.document_name !== 'KVKK' ? `${s.document_name} · ` : ''
  if (regime === 'eu') {
    return `${docPrefix}Article ${s.madde}${s.madde_title ? ` — ${s.madde_title}` : ''}`
  }
  return `${docPrefix}Madde ${s.madde}${s.madde_title ? ` — ${s.madde_title}` : ''}`
}

export default function App() {
  const { email, logout } = useAuth()
  const queryClient = useQueryClient()

  const [regime, setRegime]             = useState<Regime>('tr')
  const [messages, setMessages]         = useState<Message[]>([])
  const [input, setInput]               = useState('')
  const [expandedSource, setExpandedSource] = useState<string | null>(null)
  const [activeModel, setActiveModel]   = useState<string | null>(null)
  const [ratings, setRatings]           = useState<Map<number, 'good' | 'bad'>>(new Map())
  const [sessionId, setSessionId]       = useState<number | null>(null)
  const [sidebarOpen, setSidebarOpen]   = useState(true)

  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef  = useRef<HTMLTextAreaElement>(null)

  const cfg = REGIME_CONFIG[regime]

  const { data: sessions = [] } = useQuery<ChatSession[]>({
    queryKey: ['sessions'],
    queryFn:  fetchSessions,
  })

  const mutation = useMutation({
    mutationFn: ({ question, history }: { question: string; history: HistoryMessage[] }) =>
      sendQuestion(question, history, sessionId, regime),
    onSuccess: (data) => {
      setActiveModel(data.model)
      setSessionId(data.session_id)
      setMessages((prev) => [...prev, { role: 'assistant', text: data.answer, sources: data.sources, regime: data.regime }])
      queryClient.invalidateQueries({ queryKey: ['sessions'] })
    },
    onError: () => {
      setMessages((prev) => [...prev, { role: 'assistant', text: regime === 'eu' ? 'An error occurred. Please try again.' : 'Bir hata oluştu. Lütfen tekrar deneyin.', sources: [], regime }])
    },
  })

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, mutation.isPending])

  function switchRegime(next: Regime) {
    if (next === regime) return
    setRegime(next)
    // Start fresh when switching regime — context would be mismatched otherwise
    setMessages([])
    setSessionId(null)
    setExpandedSource(null)
    setInput('')
    setRatings(new Map())
    inputRef.current?.focus()
  }

  async function loadSession(session: ChatSession) {
    const msgs = await fetchSessionMessages(session.id)
    setSessionId(session.id)
    setMessages(msgs.map((m) => ({ role: m.role, text: m.content, sources: m.sources, regime })))
    setRatings(new Map())
    setExpandedSource(null)
  }

  function newConversation() {
    setMessages([])
    setSessionId(null)
    setExpandedSource(null)
    setInput('')
    setRatings(new Map())
    inputRef.current?.focus()
  }

  function rate(assistantIndex: number, rating: 'good' | 'bad') {
    if (ratings.has(assistantIndex)) return
    const assistantMsg = messages[assistantIndex]
    const userMsg      = messages[assistantIndex - 1]
    if (!assistantMsg || !userMsg || userMsg.role !== 'user') return

    setRatings((prev) => new Map(prev).set(assistantIndex, rating))
    sendFeedback({
      question: userMsg.text,
      answer:   assistantMsg.text,
      sources:  assistantMsg.sources ?? [],
      rating,
    }).catch(() => {
      setRatings((prev) => { const m = new Map(prev); m.delete(assistantIndex); return m })
    })
  }

  function submit(question: string) {
    const q = question.trim()
    if (!q || mutation.isPending) return
    const history: HistoryMessage[] = messages.map((m) => ({ role: m.role, content: m.text }))
    setMessages((prev) => [...prev, { role: 'user', text: q, regime }])
    setInput('')
    mutation.mutate({ question: q, history })
    inputRef.current?.focus()
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit(input)
    }
  }

  return (
    <div className="flex h-screen bg-slate-50 overflow-hidden">

      {/* ── Sidebar ────────────────────────────────────────────────────────── */}
      <aside className={`${sidebarOpen ? 'w-64' : 'w-0'} shrink-0 transition-all duration-200 overflow-hidden flex flex-col bg-slate-900 text-white`}>

        {/* Sidebar header */}
        <div className="px-4 pt-5 pb-3 flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-blue-500 flex items-center justify-center shrink-0">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 0 0 2.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 0 0-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75 2.25 2.25 0 0 0-.1-.664m-5.8 0A2.251 2.251 0 0 1 13.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25Z" />
            </svg>
          </div>
          <span className="text-sm font-semibold text-white truncate">{cfg.title}</span>
        </div>

        {/* Regime toggle */}
        <div className="px-3 pb-3">
          <div className="flex rounded-lg bg-slate-800 p-0.5 gap-0.5">
            {(['tr', 'eu'] as Regime[]).map((r) => (
              <button
                key={r}
                onClick={() => switchRegime(r)}
                className={`flex-1 text-xs py-1.5 rounded-md font-medium transition-colors cursor-pointer ${
                  regime === r
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {REGIME_CONFIG[r].flag} {REGIME_CONFIG[r].label}
              </button>
            ))}
          </div>
        </div>

        {/* New chat button */}
        <div className="px-3 pb-3">
          <button
            onClick={newConversation}
            className="w-full flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-slate-300 hover:bg-slate-800 hover:text-white transition-colors cursor-pointer"
          >
            <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
            </svg>
            {regime === 'eu' ? 'New Conversation' : 'Yeni Konuşma'}
          </button>
        </div>

        <div className="mx-3 border-t border-slate-700 mb-2" />

        {/* Session list */}
        <div className="flex-1 overflow-y-auto px-2 flex flex-col gap-0.5">
          {sessions.length === 0 && (
            <p className="text-xs text-slate-500 px-3 py-2">
              {regime === 'eu' ? 'No conversations yet' : 'Henüz konuşma yok'}
            </p>
          )}
          {sessions.map((s) => (
            <button
              key={s.id}
              onClick={() => loadSession(s)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors cursor-pointer group ${
                s.id === sessionId
                  ? 'bg-slate-700 text-white'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-white'
              }`}
            >
              <p className="truncate text-xs font-medium leading-snug">{s.title || 'Conversation'}</p>
              <p className="text-xs text-slate-500 mt-0.5">{formatDate(s.updated_at)}</p>
            </button>
          ))}
        </div>

        {/* User + logout */}
        <div className="border-t border-slate-700 px-3 py-3 flex items-center gap-2">
          <div className="w-7 h-7 rounded-full bg-blue-500 flex items-center justify-center text-xs font-bold text-white shrink-0">
            {email?.[0]?.toUpperCase() ?? '?'}
          </div>
          <span className="text-xs text-slate-400 truncate flex-1">{email}</span>
          <button
            onClick={logout}
            title="Çıkış yap"
            className="text-slate-500 hover:text-red-400 transition-colors cursor-pointer"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 9V5.25A2.25 2.25 0 0 0 13.5 3h-6a2.25 2.25 0 0 0-2.25 2.25v13.5A2.25 2.25 0 0 0 7.5 21h6a2.25 2.25 0 0 0 2.25-2.25V15M12 9l-3 3m0 0 3 3m-3-3h12.75" />
            </svg>
          </button>
        </div>
      </aside>

      {/* ── Main ──────────────────────────────────────────────────────────── */}
      <div className="flex flex-col flex-1 min-w-0">

        {/* Header */}
        <header className="shrink-0 bg-white border-b border-slate-200 px-4 py-3 flex items-center gap-3 shadow-sm">
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            className="text-slate-400 hover:text-slate-600 transition-colors cursor-pointer"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
            </svg>
          </button>
          <h1 className="text-sm font-semibold text-slate-800">{cfg.flag} {cfg.title}</h1>
          <div className="ml-auto flex items-center gap-3">
            <span className="flex items-center gap-1.5 text-xs text-emerald-600 font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              {activeModel ?? 'Gemini · 2.5 Flash'}
            </span>
          </div>
        </header>

        {/* Messages */}
        <main className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-2xl mx-auto flex flex-col gap-4">

            {messages.length === 0 && (
              <div className="flex flex-col items-center gap-6 mt-12">
                <div className="w-14 h-14 rounded-2xl bg-blue-600 flex items-center justify-center shadow-lg">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 0 0 2.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 0 0-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75 2.25 2.25 0 0 0-.1-.664m-5.8 0A2.251 2.251 0 0 1 13.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25Z" />
                  </svg>
                </div>
                <div className="text-center">
                  <h2 className="text-lg font-semibold text-slate-700">{cfg.emptyTitle}</h2>
                  <p className="text-sm text-slate-400 mt-1">{cfg.emptySubtitle}</p>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full">
                  {cfg.suggestions.map((s) => (
                    <button
                      key={s}
                      onClick={() => submit(s)}
                      className="text-left text-sm text-slate-600 bg-white border border-slate-200 rounded-xl px-4 py-3 hover:border-blue-400 hover:text-blue-600 transition-colors shadow-sm cursor-pointer"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {msg.role === 'assistant' && (
                  <div className="shrink-0 w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center mt-0.5">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09Z" />
                    </svg>
                  </div>
                )}
                <div className="flex flex-col gap-1.5 max-w-[80%]">
                  <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                    msg.role === 'user'
                      ? 'bg-blue-600 text-white rounded-tr-sm'
                      : 'bg-white text-slate-700 border border-slate-200 shadow-sm rounded-tl-sm'
                  }`}>
                    {msg.role === 'assistant' && msg.sources
                      ? renderAnswerWithCitations(msg.text, msg.sources, msg.regime ?? regime)
                      : msg.text}
                  </div>

                  {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                    <div className="flex flex-col gap-2 px-1">
                      <div className="flex flex-wrap gap-1.5">
                        {msg.sources.map((s, idx) => {
                          const key   = `${i}-${idx}`
                          const isOpen = expandedSource === key
                          const label  = sourceLabel(s, msg.regime ?? regime)
                          return (
                            <button
                              key={idx}
                              onClick={() => setExpandedSource(isOpen ? null : key)}
                              className="text-xs bg-blue-50 text-blue-600 border border-blue-100 rounded-full px-2.5 py-0.5 font-medium flex items-center gap-1 hover:bg-blue-100 transition-colors cursor-pointer"
                            >
                              {label}
                              <svg className={`w-3 h-3 shrink-0 transition-transform ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" d="m19 9-7 7-7-7" />
                              </svg>
                            </button>
                          )
                        })}
                      </div>
                      {msg.sources.map((s, idx) => {
                        const key = `${i}-${idx}`
                        if (expandedSource !== key) return null
                        return (
                          <div key={idx} className="text-xs text-slate-600 bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 leading-relaxed whitespace-pre-wrap">
                            <p className="font-semibold text-slate-400 mb-1.5">
                              {sourceLabel(s, msg.regime ?? regime)}
                            </p>
                            {s.content}
                          </div>
                        )
                      })}
                    </div>
                  )}

                  {msg.role === 'assistant' && (
                    <div className="flex items-center gap-1 px-1 mt-0.5">
                      {ratings.get(i) ? (
                        <span className="text-xs text-slate-400">
                          {ratings.get(i) === 'good'
                            ? (regime === 'eu' ? '👍 Thanks!' : '👍 Teşekkürler!')
                            : (regime === 'eu' ? '👎 Feedback received' : '👎 Geri bildirim alındı')}
                        </span>
                      ) : (
                        <>
                          <button onClick={() => rate(i, 'good')} title={regime === 'eu' ? 'Answer is correct and helpful' : 'Yanıt doğru ve yararlı'} className="p-1 rounded text-slate-300 hover:text-emerald-500 hover:bg-emerald-50 transition-colors cursor-pointer">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14Zm-7 13H5a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h2" />
                            </svg>
                          </button>
                          <button onClick={() => rate(i, 'bad')} title={regime === 'eu' ? 'Answer is wrong or incomplete' : 'Yanıt yanlış veya eksik'} className="p-1 rounded text-slate-300 hover:text-red-400 hover:bg-red-50 transition-colors cursor-pointer">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10Zm7-13h2a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-2" />
                            </svg>
                          </button>
                        </>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {mutation.isPending && (
              <div className="flex gap-3 justify-start">
                <div className="shrink-0 w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center mt-0.5">
                  <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09Z" />
                  </svg>
                </div>
                <div className="bg-white border border-slate-200 shadow-sm rounded-2xl rounded-tl-sm px-4 py-3.5 flex items-center gap-1.5">
                  <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
                  <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
                  <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" />
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        </main>

        {/* Input bar */}
        <div className="shrink-0 bg-white border-t border-slate-200 px-4 py-4">
          <form
            onSubmit={(e) => { e.preventDefault(); submit(input) }}
            className="max-w-2xl mx-auto flex items-end gap-2"
          >
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              placeholder={cfg.placeholder}
              disabled={mutation.isPending}
              className="flex-1 resize-none rounded-xl border border-slate-300 bg-slate-50 px-4 py-3 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 max-h-36 overflow-y-auto"
            />
            <button
              type="submit"
              disabled={!input.trim() || mutation.isPending}
              className="shrink-0 w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors cursor-pointer"
            >
              <svg className="w-4 h-4 rotate-90" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L5.999 12Zm0 0h7.5" />
              </svg>
            </button>
          </form>
          <p className="text-center text-xs text-slate-400 mt-2">
            {cfg.flag} {cfg.label} · Shift+Enter {regime === 'eu' ? 'for new line' : 'yeni satır'}
          </p>
        </div>

      </div>
    </div>
  )
}
