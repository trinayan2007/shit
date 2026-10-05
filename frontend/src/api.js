// API client: every value rendered by the UI comes from these calls.
const BASE = import.meta.env.VITE_API_URL || ''

async function get(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}`)
  return res.json()
}

async function post(path) {
  const res = await fetch(`${BASE}${path}`, { method: 'POST' })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    const msg = body?.detail || `HTTP ${res.status}`
    throw new Error(msg)
  }
  return body
}

export const api = {
  health: () => get('/api/health'),
  dashboard: () => get('/api/dashboard'),
  events: (params = {}) => {
    const q = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
    )
    return get(`/api/events?${q}`)
  },
  event: (id) => get(`/api/events/${id}`),
  threats: (params = {}) => {
    const q = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
    )
    return get(`/api/threats?${q}`)
  },
  threat: (id) => get(`/api/threats/${id}`),
  block: (id) => post(`/api/threats/${id}/block`),
  users: (params = {}) => {
    const q = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
    )
    return get(`/api/users?${q}`)
  },
}
