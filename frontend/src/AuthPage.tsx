import { useState } from 'react'
import { useAuth } from './AuthContext'

export default function AuthPage() {
  const { login, register } = useAuth()
  const [mode, setMode]         = useState<'login' | 'signup'>('login')
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [password2, setPassword2] = useState('')
  const [error, setError]       = useState('')
  const [loading, setLoading]   = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      if (mode === 'login') {
        await login(email, password)
      } else {
        await register(email, password, password2)
      }
    } catch (err: unknown) {
      const raw = (err as { response?: { data?: { error?: unknown } } })?.response?.data?.error
      const msg = typeof raw === 'string' ? raw : 'Bir hata oluştu. Lütfen tekrar deneyin.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  function switchMode() {
    setMode(mode === 'login' ? 'signup' : 'login')
    setError('')
    setPassword('')
    setPassword2('')
  }

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center px-4">
      <div className="w-full max-w-sm">

        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-12 h-12 rounded-2xl bg-blue-600 flex items-center justify-center shadow-lg mb-3">
            <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 0 0 2.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 0 0-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75 2.25 2.25 0 0 0-.1-.664m-5.8 0A2.251 2.251 0 0 1 13.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25Z" />
            </svg>
          </div>
          <h1 className="text-xl font-semibold text-slate-800">KVKK Asistanı</h1>
          <p className="text-sm text-slate-400 mt-0.5">Kişisel Verilerin Korunması Kanunu</p>
        </div>

        {/* Card */}
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
          <h2 className="text-base font-semibold text-slate-700 mb-5">
            {mode === 'login' ? 'Giriş Yap' : 'Hesap Oluştur'}
          </h2>

          <form onSubmit={handleSubmit} className="flex flex-col gap-4">

            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium text-slate-500">E-posta</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="ornek@email.com"
                required
                className="rounded-lg border border-slate-300 bg-slate-50 px-3 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium text-slate-500">Şifre</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
                className="rounded-lg border border-slate-300 bg-slate-50 px-3 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {mode === 'signup' && (
              <div className="flex flex-col gap-1">
                <label className="text-xs font-medium text-slate-500">Şifre Tekrar</label>
                <input
                  type="password"
                  value={password2}
                  onChange={(e) => setPassword2(e.target.value)}
                  placeholder="••••••••"
                  required
                  className="rounded-lg border border-slate-300 bg-slate-50 px-3 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            )}

            {error && (
              <p className="text-xs text-red-500 bg-red-50 border border-red-100 rounded-lg px-3 py-2">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="mt-1 w-full rounded-lg bg-blue-600 text-white text-sm font-medium py-2.5 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer"
            >
              {loading
                ? 'Lütfen bekleyin…'
                : mode === 'login' ? 'Giriş Yap' : 'Hesap Oluştur'}
            </button>
          </form>
        </div>

        {/* Switch mode */}
        <p className="text-center text-sm text-slate-400 mt-4">
          {mode === 'login' ? 'Hesabınız yok mu?' : 'Zaten hesabınız var mı?'}{' '}
          <button
            onClick={switchMode}
            className="text-blue-600 font-medium hover:underline cursor-pointer"
          >
            {mode === 'login' ? 'Kayıt ol' : 'Giriş yap'}
          </button>
        </p>

      </div>
    </div>
  )
}
