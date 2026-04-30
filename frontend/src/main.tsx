import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './index.css'
import App from './App.tsx'
import AuthPage from './AuthPage.tsx'
import { AuthProvider, useAuth } from './AuthContext.tsx'

const queryClient = new QueryClient()

function Root() {
  const { isAuthenticated } = useAuth()
  return isAuthenticated ? <App /> : <AuthPage />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <Root />
      </AuthProvider>
    </QueryClientProvider>
  </StrictMode>,
)
