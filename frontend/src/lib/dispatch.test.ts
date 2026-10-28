import { beforeEach, describe, expect, it } from 'vitest'
import { useCircuitStore } from '../store/useCircuitStore'
import { useSessionStore } from '../store/useSessionStore'
import { handleServerMessage } from './dispatch'
import { parseServerMessage, ServerEvent, type CircuitSnapshot, type ClientMessage } from './protocol'
import { telemetryBuffer } from './telemetryBuffer'
import { setSender } from './transport'

const snapshot = (over: Partial<CircuitSnapshot> = {}): CircuitSnapshot => ({
  version: 1,
  calibrated: true,
  steering: true,
  skop_filter: true,
  alpha: 1,
  gamma_energy: 0.9,
  risk_percentile: 0.8,
  normalize_rayleigh: true,
  risk_threshold: 0.2,
  num_risky_heads: 3,
  risky_heads: { '12': [1, 4] },
  active_layers: [12],
  layer_plan: [{ layer: 12, alpha: 2.5, mode: 'orthogonal' }],
  ...over,
})

let sent: ClientMessage[]

beforeEach(() => {
  sent = []
  setSender((m) => {
    sent.push(m)
    return true
  })
  useSessionStore.setState({ messages: [], activeRequestId: null })
})

describe('parseServerMessage', () => {
  it('acepta eventos conocidos y descarta basura', () => {
    expect(parseServerMessage('{"type":"pong","server_time":1}')).toEqual({ type: 'pong', server_time: 1 })
    expect(parseServerMessage('{"type":"zzz"}')).toBeNull()
    expect(parseServerMessage('no json')).toBeNull()
    expect(parseServerMessage(new ArrayBuffer(2))).toBeNull()
  })
})

describe('handleServerMessage', () => {
  it('ready fija la geometría y fuerza el envío del circuito local', () => {
    handleServerMessage({
      type: ServerEvent.READY,
      model: 'google/gemma-2-2b-it',
      geometry: { num_layers: 26, num_heads: 8, num_kv_heads: 4, head_dim: 256 },
      layers: Array.from({ length: 26 }, (_, i) => i),
      hidden_size: 2304,
      sae: { kind: 'gemma-scope', root: '/sae', hf_cache: true },
      circuit: snapshot(),
    })
    expect(useCircuitStore.getState().numLayers).toBe(26)
    expect(useSessionStore.getState().model).toBe('google/gemma-2-2b-it')
    expect(useSessionStore.getState().sae?.kind).toBe('gemma-scope')
    expect(sent.filter((m) => m.action === 'update')).toHaveLength(1)
  })

  it('STREAM_CHUNK pinta el chat y alimenta la telemetría del request activo', () => {
    expect(useSessionStore.getState().generate('hola')).toBe(true)
    const requestId = useSessionStore.getState().activeRequestId!
    handleServerMessage({ type: ServerEvent.START, request_id: requestId, prompt_tokens: 3, circuit: snapshot() })
    for (const [i, tok] of ['Ho', 'la', '!'].entries()) {
      handleServerMessage({
        type: ServerEvent.STREAM_CHUNK,
        request_id: requestId,
        index: i,
        token: tok,
        token_id: 100 + i,
        circuit_version: 1,
        telemetry: {
          available: true,
          summary: { mean: 0.1, max: 0.3, mean_unsteered: 0.08, argmax: { layer: 12, head: 1 }, risk_threshold: 0.2 },
          heads: [{ layer: 12, head: 1, rayleigh: 0.3, unsteered: 0.2, baseline: 0.25, risky: true, steered: true, projected: true }],
        },
      })
    }
    // Chunk de otro request: se ignora.
    handleServerMessage({
      type: ServerEvent.STREAM_CHUNK,
      request_id: 'otro',
      index: 0,
      token: 'XXX',
      token_id: 1,
      circuit_version: 1,
      telemetry: null,
    })
    handleServerMessage({ type: ServerEvent.CIRCUIT, changes: { alpha: 2 }, applied_at_index: 2, request_id: requestId, circuit: snapshot({ version: 2 }) })
    handleServerMessage({ type: ServerEvent.END, request_id: requestId, reason: 'length', num_tokens: 3, text: 'Hola!', elapsed_s: 0.1, tokens_per_s: 30 })

    const assistant = useSessionStore.getState().messages.find((m) => m.role === 'assistant')!
    expect(assistant.text).toBe('Hola!')
    expect(assistant.status).toBe('done')
    expect(useSessionStore.getState().activeRequestId).toBeNull()
    expect(telemetryBuffer.samples).toBe(3)
    expect(telemetryBuffer.series.get('12:1')!.last).toBeCloseTo(0.3)
    expect(telemetryBuffer.markers).toEqual([{ sample: 2, label: 'v2' }])
  })

  it('un error de circuito marca la sincronización como rechazada', () => {
    handleServerMessage({ type: ServerEvent.ERROR, code: 'invalid_layer', message: 'Capas fuera del rango' })
    expect(useCircuitStore.getState().sync).toMatchObject({ state: 'error', error: 'Capas fuera del rango' })
  })

  it('invalid_message de un generate libera el chat en vez de dejarlo esperando', () => {
    expect(useSessionStore.getState().generate('hola')).toBe(true)
    const requestId = useSessionStore.getState().activeRequestId!
    useCircuitStore.getState().markSync('acked')
    handleServerMessage({
      type: ServerEvent.ERROR,
      code: 'invalid_message',
      message: 'Mensaje no válido (generate.max_new_tokens: Input should be less than or equal to 4096)',
      action: 'generate',
      request_id: requestId,
    })
    const session = useSessionStore.getState()
    expect(session.activeRequestId).toBeNull()
    expect(session.messages.find((m) => m.requestId === requestId)!.status).toBe('error')
    expect(session.messages.at(-1)!.text).toContain('max_new_tokens')
    expect(useCircuitStore.getState().sync.state).toBe('acked') // no era un update
    expect(session.generate('otra vez')).toBe(true)
  })

  it('invalid_message de un update marca el circuito como rechazado', () => {
    handleServerMessage({ type: ServerEvent.ERROR, code: 'invalid_message', message: 'Mensaje no válido (update.layers)', action: 'update' })
    expect(useCircuitStore.getState().sync).toMatchObject({ state: 'error', error: 'Mensaje no válido (update.layers)' })
    expect(useSessionStore.getState().activeRequestId).toBeNull()
  })
})
