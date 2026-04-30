import { createContext, useContext, useState } from 'react'
import type { ReactNode } from 'react'
import { login as apiLogin, register as apiRegister, saveTokens, clearTokens, getAccessToken, getEmail } from './api'

interface AuthContextType {
  isAuthenticated: boolean
  email: string | null
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, password2: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextType | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(() => !!getAccessToken())
  const [email, setEmail] = useState<string | null>(() => getEmail())

  async function login(email: string, password: string) {
    const tokens = await apiLogin(email, password)
    saveTokens(tokens)
    setIsAuthenticated(true)
    setEmail(tokens.email)
  }

  async function register(email: string, password: string, password2: string) {
    const tokens = await apiRegister(email, password, password2)
    saveTokens(tokens)
    setIsAuthenticated(true)
    setEmail(tokens.email)
  }

  function logout() {
    clearTokens()
    setIsAuthenticated(false)
    setEmail(null)
  }

  return (
    <AuthContext.Provider value={{ isAuthenticated, email, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider')
  return ctx
}
