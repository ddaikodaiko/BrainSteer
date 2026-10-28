import { useEffect, useRef } from 'react'
import { handleServerMessage, notePingSent } from '../lib/dispatch'
import { ClientAction, DEFAULT_WS_URL, parseServerMessage } from '../lib/protocol'
import { setSender } from '../lib/transport'
import { useSessionStore } from '../store/useSessionStore'

export interface UseWebSocketOptions {
  /** Intervalo del ping de latencia/keep-alive. */
  heartbeatMs?: number
  /** Backoff exponencial de reconexión: base y techo. */
  retryBaseMs?: number
  retryMaxMs?: number
}

/** URL efectiva: `?ws=` en la URL > `VITE_WS_URL` > `ws://127.0.0.1:8000/ws/stream`. */
export function resolveWsUrl(): string {
  const fromQuery = typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('ws') : null
  return fromQuery || import.meta.env.VITE_WS_URL || DEFAULT_WS_URL
}

/**
 * Conexión persistente con `/ws/stream`: reconexión con backoff, heartbeat y
 * enrutado de eventos. Los `STREAM_CHUNK` pintan el chat (store de sesión) y
 * alimentan el buffer de telemetría que lee el osciloscopio.
 * Debe montarse una sola vez (en `App`).
 */
export function useWebSocket(url: string = resolveWsUrl(), options: UseWebSocketOptions = {}) {
  const { heartbeatMs = 10_000, retryBaseMs = 500, retryMaxMs = 8_000 } = options
  const status = useSessionStore((s) => s.connection)
  const attempt = useSessionStore((s) => s.reconnectAttempt)
  const socketRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let disposed = false
    let retries = 0
    let retryTimer: ReturnType<typeof setTimeout> | undefined
    let heartbeat: ReturnType<typeof setInterval> | undefined
    const session = useSessionStore.getState()
    useSessionStore.setState({ url })

    const connect = () => {
      session.setConnection('connecting', retries)
      const ws = new WebSocket(url)
      socketRef.current = ws

      ws.onopen = () => {
        retries = 0
        setSender((message) => {
          if (ws.readyState !== WebSocket.OPEN) return false
          if (message.action === ClientAction.PING) notePingSent()
          ws.send(JSON.stringify(message))
          return true
        })
        session.setConnection('open')
        heartbeat = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            notePingSent()
            ws.send(JSON.stringify({ action: ClientAction.PING }))
          }
        }, heartbeatMs)
      }

      ws.onmessage = (event) => {
        const msg = parseServerMessage(event.data)
        if (msg) handleServerMessage(msg)
        else console.warn('[latentflow] mensaje desconocido del servidor', event.data)
      }

      ws.onclose = () => {
        clearInterval(heartbeat)
        if (socketRef.current === ws) {
          socketRef.current = null
          setSender(null)
        }
        if (disposed) return
        session.setConnection('closed', retries)
        const delay = Math.min(retryMaxMs, retryBaseMs * 2 ** retries) * (0.8 + Math.random() * 0.4)
        retries++
        retryTimer = setTimeout(connect, delay)
      }

      // `onerror` siempre va seguido de `onclose`, que gestiona la reconexión.
      ws.onerror = () => {}
    }

    connect()
    return () => {
      disposed = true
      clearTimeout(retryTimer)
      clearInterval(heartbeat)
      setSender(null)
      socketRef.current?.close()
      socketRef.current = null
    }
  }, [url, heartbeatMs, retryBaseMs, retryMaxMs])

  return { status, attempt, url }
}
