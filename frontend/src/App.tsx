import { useState, useRef, useEffect } from 'react'
import type { KeyboardEvent, ReactNode } from 'react'
import { useMutation } from '@tanstack/react-query'
import { sendQuestion } from './api'
import type { Message, Source, HistoryMessage } from './types'

function renderAnswerWithCitations(text: string, sources: Source[]): ReactNode {
  const verifiedMaddes = new Set(sources.map((s) => s.madde).filter(Boolean))
  const parts = text.split(/(Madde\s+\d{1,3})/gi)

  return parts.map((part, i) => {
    const match = part.match(/^Madde\s+(\d{1,3})$/i)
    if (!match) return part

    const num = parseInt(match[1])
    const isVerified = verifiedMaddes.has(num)

    return isVerified ? (
      <mark
        key={i}
        className="bg-blue-100 text-blue-700 rounded px-0.5 not-italic font-medium"
      >
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

const SUGGESTIONS = [
  'Kişisel veri nedir?',
  'Veri sorumlusunun yükümlülükleri nelerdir?',
  'Açık rıza şartları nelerdir?',
  'Veri ihlali durumunda ne yapılmalıdır?',
]


export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [expandedSource, setExpandedSource] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const mutation = useMutation({
    mutationFn: ({ question, history }: { question: string; history: HistoryMessage[] }) =>
      sendQuestion(question, history),
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: data.answer, sources: data.sources },
      ])
    },
    onError: () => {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: 'Bir hata oluştu. Lütfen tekrar deneyin.', sources: [] },
      ])
    },
  })

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, mutation.isPending])

  function clearConversation() {
    setMessages([])
    setExpandedSource(null)
    setInput('')
    inputRef.current?.focus()
  }

  function submit(question: string) {
    const q = question.trim()
    if (!q || mutation.isPending) return
    const history: HistoryMessage[] = messages.map((m) => ({ role: m.role, content: m.text }))
    setMessages((prev) => [...prev, { role: 'user', text: q }])
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
    <div className="flex flex-col h-screen bg-slate-50">

      {/* Header */}
      <header className="shrink-0 bg-white border-b border-slate-200 px-6 py-4 flex items-center gap-3 shadow-sm">
        <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center">
          <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 0 0 2.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 0 0-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75 2.25 2.25 0 0 0-.1-.664m-5.8 0A2.251 2.251 0 0 1 13.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25Z" />
          </svg>
        </div>
        <div>
          <h1 className="text-base font-semibold text-slate-800 leading-tight">KVKK Asistanı</h1>
          <p className="text-xs text-slate-400">Kişisel Verilerin Korunması Kanunu</p>
        </div>
        <div className="ml-auto flex items-center gap-3">
          {messages.length > 0 && (
            <button
              onClick={clearConversation}
              title="Yeni konuşma başlat"
              className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-600 transition-colors cursor-pointer"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99" />
              </svg>
              Yeni Konuşma
            </button>
          )}
          <span className="flex items-center gap-1.5 text-xs text-emerald-600 font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            Ollama · llama3.2
          </span>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-2xl mx-auto flex flex-col gap-4">

          {/* Empty state */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center gap-6 mt-12">
              <div className="w-14 h-14 rounded-2xl bg-blue-600 flex items-center justify-center shadow-lg">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 0 0 2.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 0 0-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75 2.25 2.25 0 0 0-.1-.664m-5.8 0A2.251 2.251 0 0 1 13.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25Z" />
                </svg>
              </div>
              <div className="text-center">
                <h2 className="text-lg font-semibold text-slate-700">KVKK hakkında soru sorun</h2>
                <p className="text-sm text-slate-400 mt-1">Yanıtlar yalnızca KVKK belgesi içeriğine dayalıdır.</p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full">
                {SUGGESTIONS.map((s) => (
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

          {/* Message list */}
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
                <div
                  className={`rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                    msg.role === 'user'
                      ? 'bg-blue-600 text-white rounded-tr-sm'
                      : 'bg-white text-slate-700 border border-slate-200 shadow-sm rounded-tl-sm'
                  }`}
                >
                  {msg.role === 'assistant' && msg.sources
                    ? renderAnswerWithCitations(msg.text, msg.sources)
                    : msg.text}
                </div>

                {/* Source badges */}
                {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                  <div className="flex flex-col gap-2 px-1">
                    <div className="flex flex-wrap gap-1.5">
                      {msg.sources.map((s, idx) => {
                        const key = `${i}-${idx}`
                        const isOpen = expandedSource === key
                        const label = `Bölüm ${s.bolum} · Madde ${s.madde}${s.madde_title ? ` — ${s.madde_title}` : ''}`
                        return s.is_cross_reference ? (
                          <button
                            key={idx}
                            onClick={() => setExpandedSource(isOpen ? null : key)}
                            title="Bu madde, birincil sonuçlardan çapraz referans olarak eklendi"
                            className="text-xs bg-amber-50 text-amber-600 border border-amber-200 rounded-full px-2.5 py-0.5 font-medium flex items-center gap-1 hover:bg-amber-100 transition-colors cursor-pointer"
                          >
                            <svg className="w-3 h-3 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" d="M13.19 8.688a4.5 4.5 0 0 1 1.242 7.244l-4.5 4.5a4.5 4.5 0 0 1-6.364-6.364l1.757-1.757m13.35-.622 1.757-1.757a4.5 4.5 0 0 0-6.364-6.364l-4.5 4.5a4.5 4.5 0 0 0 1.242 7.244" />
                            </svg>
                            {label}
                            <svg className={`w-3 h-3 shrink-0 transition-transform ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" d="m19 9-7 7-7-7" />
                            </svg>
                          </button>
                        ) : (
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

                    {/* Expanded snippet panel */}
                    {msg.sources.map((s, idx) => {
                      const key = `${i}-${idx}`
                      if (expandedSource !== key) return null
                      return (
                        <div key={idx} className="text-xs text-slate-600 bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 leading-relaxed whitespace-pre-wrap">
                          <p className="font-semibold text-slate-400 mb-1.5">
                            Bölüm {s.bolum} · Madde {s.madde}{s.madde_title ? ` — ${s.madde_title}` : ''}
                          </p>
                          {s.content}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Typing indicator */}
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
            placeholder="KVKK hakkında bir soru sorun…"
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
          Yanıtlar yalnızca KVKK belgesi içeriğine dayalıdır · Shift+Enter yeni satır
        </p>
      </div>

    </div>
  )
}
