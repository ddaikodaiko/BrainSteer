import { useCircuitStore } from '../store/useCircuitStore'
import { useSessionStore } from '../store/useSessionStore'
import { ServerEvent, type ServerMessage } from './protocol'
import { telemetryBuffer } from './telemetryBuffer'

/** Errores del servidor que invalidan el último `UPDATE_CIRCUIT` enviado. */
const CIRCUIT_ERRORS = new Set([
  'invalid_layer',
  'not_calibrated',
  'empty_update',
  'feature_not_found',
  'feature_invalid',
  'circuit_apply_failed',
])

let pingSentAt: number | null = null
export function notePingSent(): void {
  pingSentAt = performance.now()
}

/** Enruta cada evento del servidor a los stores y al buffer de telemetría. */
export function handleServerMessage(msg: ServerMessage): void {
  const session = useSessionStore.getState()
  const circuit = useCircuitStore.getState()

  switch (msg.type) {
    case ServerEvent.READY:
      session.setModel(msg.model, msg.geometry, msg.sae)
      circuit.setModelInfo(msg.geometry.num_layers, msg.layers)
      circuit.setServerCircuit(msg.circuit, false)
      // El circuito del servidor es global: tras (re)conectar, el grafo local manda.
      if (msg.circuit.calibrated) circuit.flushSync(true)
      else circuit.markSync('error', 'El servidor no está calibrado')
      return

    case ServerEvent.ACCEPTED:
      session.markQueued(msg.request_id, msg.queued)
      return

    case ServerEvent.START:
      telemetryBuffer.reset(msg.request_id)
      session.startStream(msg.request_id, msg.prompt_tokens)
      circuit.setServerCircuit(msg.circuit, false)
      return

    case ServerEvent.STREAM_CHUNK:
      if (msg.request_id === session.activeRequestId) {
        session.appendChunk(msg.request_id, msg.token)
        telemetryBuffer.push(msg.index, msg.telemetry)
      }
      return

    case ServerEvent.END:
      session.endStream(msg.request_id, msg.reason, msg.num_tokens, msg.tokens_per_s)
      return

    case ServerEvent.UPDATE_ACCEPTED:
      if (circuit.sync.state === 'sent') circuit.markSync('acked')
      return

    case ServerEvent.CIRCUIT:
      circuit.setServerCircuit(msg.circuit, true)
      if (msg.applied_at_index !== null && msg.request_id === telemetryBuffer.requestId) {
        telemetryBuffer.mark(`v${msg.circuit.version}`)
      }
      return

    case ServerEvent.CALIBRATED:
      circuit.setServerCircuit(msg.circuit, false)
      circuit.flushSync(true)
      return

    case ServerEvent.STATUS:
      circuit.setServerCircuit(msg.circuit, false)
      return

    case ServerEvent.PONG:
      if (pingSentAt !== null) session.setLatency(Math.round(performance.now() - pingSentAt))
      pingSentAt = null
      return

    case ServerEvent.ERROR:
      if (CIRCUIT_ERRORS.has(msg.code)) circuit.markSync('error', msg.message)
      if (msg.code === 'idle') return // cancelar sin generación activa: inocuo
      session.pushSystem(`⚠ ${msg.code}: ${msg.message}`, msg.request_id)
      return
  }
}
