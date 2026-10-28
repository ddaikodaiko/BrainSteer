import { create } from 'zustand'
import { ClientAction, type AttentionGeometry, type EndReason, type SaeInfo, type TelemetryMode } from '../lib/protocol'
import { send } from '../lib/transport'
import { useCircuitStore } from './useCircuitStore'

export type ConnectionStatus = 'connecting' | 'open' | 'closed'

export type MessageStatus = 'pending' | 'queued' | 'streaming' | 'done' | 'cancelled' | 'error'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  text: string
  requestId?: string
  status: MessageStatus
  stats?: { tokens: number; tokensPerS: number | null; reason: EndReason; promptTokens?: number }
}

export interface GenerateParams {
  maxNewTokens: number
  temperature: number
  chat: boolean
  telemetry: TelemetryMode
}

interface SessionState {
  url: string
  connection: ConnectionStatus
  reconnectAttempt: number
  model: string | null
  geometry: AttentionGeometry | null
  sae: SaeInfo | null
  latencyMs: number | null
  messages: ChatMessage[]
  activeRequestId: string | null
  params: GenerateParams

  setConnection: (status: ConnectionStatus, attempt?: number) => void
  setModel: (model: string, geometry: AttentionGeometry, sae?: SaeInfo | null) => void
  setLatency: (ms: number) => void
  setParams: (patch: Partial<GenerateParams>) => void

  generate: (prompt: string) => boolean
  cancel: () => void
  clearChat: () => void

  // Mutaciones dirigidas por eventos del servidor.
  markQueued: (requestId: string, queued: boolean) => void
  startStream: (requestId: string, promptTokens: number) => void
  appendChunk: (requestId: string, chunk: string) => void
  endStream: (requestId: string, reason: EndReason, tokens: number, tokensPerS: number | null) => void
  pushSystem: (text: string, requestId?: string) => void
}

const MAX_MESSAGES = 200
let msgCounter = 0
const msgId = () => `m${(msgCounter++).toString(36)}`
const newRequestId = () =>
  (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}${Math.random()}`).replace(/-/g, '').slice(0, 12)

const updateByRequest = (messages: ChatMessage[], requestId: string, fn: (m: ChatMessage) => ChatMessage) =>
  messages.map((m) => (m.role === 'assistant' && m.requestId === requestId ? fn(m) : m))

export const useSessionStore = create<SessionState>()((set, get) => ({
  url: '',
  connection: 'connecting',
  reconnectAttempt: 0,
  model: null,
  geometry: null,
  sae: null,
  latencyMs: null,
  messages: [],
  activeRequestId: null,
  params: { maxNewTokens: 160, temperature: 0, chat: true, telemetry: 'risky' },

  setConnection: (connection, reconnectAttempt = 0) => {
    set({ connection, reconnectAttempt })
    if (connection === 'closed' && get().activeRequestId) {
      const id = get().activeRequestId!
      set((s) => ({
        activeRequestId: null,
        messages: updateByRequest(s.messages, id, (m) => ({ ...m, status: 'error' })),
      }))
      get().pushSystem('Conexión perdida: la generación en curso se ha interrumpido.')
    }
  },
  setModel: (model, geometry, sae = null) => set({ model, geometry, sae }),
  setLatency: (latencyMs) => set({ latencyMs }),
  setParams: (patch) => set((s) => ({ params: { ...s.params, ...patch } })),

  generate: (prompt) => {
    const text = prompt.trim()
    if (!text || get().activeRequestId) return false
    // El circuito pendiente de debounce sale antes que la generación.
    useCircuitStore.getState().flushSync()
    const requestId = newRequestId()
    const { params } = get()
    const ok = send({
      action: ClientAction.GENERATE,
      prompt: text,
      request_id: requestId,
      max_new_tokens: params.maxNewTokens,
      temperature: params.temperature,
      chat: params.chat,
      telemetry: params.telemetry,
    })
    if (!ok) return false
    set((s) => ({
      activeRequestId: requestId,
      messages: [
        ...s.messages,
        { id: msgId(), role: 'user' as const, text, status: 'done' as const },
        { id: msgId(), role: 'assistant' as const, text: '', requestId, status: 'pending' as const },
      ].slice(-MAX_MESSAGES),
    }))
    return true
  },

  cancel: () => {
    if (get().activeRequestId) send({ action: ClientAction.CANCEL })
  },

  clearChat: () => set({ messages: [] }),

  markQueued: (requestId, queued) =>
    set((s) => ({
      messages: updateByRequest(s.messages, requestId, (m) => ({ ...m, status: queued ? 'queued' : m.status })),
    })),

  startStream: (requestId, promptTokens) =>
    set((s) => ({
      messages: updateByRequest(s.messages, requestId, (m) => ({
        ...m,
        status: 'streaming',
        stats: { tokens: 0, tokensPerS: null, reason: 'stop', promptTokens },
      })),
    })),

  appendChunk: (requestId, chunk) => {
    if (!chunk) return
    set((s) => ({ messages: updateByRequest(s.messages, requestId, (m) => ({ ...m, text: m.text + chunk })) }))
  },

  endStream: (requestId, reason, tokens, tokensPerS) =>
    set((s) => ({
      activeRequestId: s.activeRequestId === requestId ? null : s.activeRequestId,
      messages: updateByRequest(s.messages, requestId, (m) => ({
        ...m,
        status: reason === 'cancelled' ? 'cancelled' : reason === 'error' ? 'error' : 'done',
        stats: { ...m.stats, tokens, tokensPerS, reason },
      })),
    })),

  pushSystem: (text, requestId) =>
    set((s) => ({
      messages: [...s.messages, { id: msgId(), role: 'system' as const, text, requestId, status: 'done' as const }].slice(
        -MAX_MESSAGES,
      ),
    })),
}))
