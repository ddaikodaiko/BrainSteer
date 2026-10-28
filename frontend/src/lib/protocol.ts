/**
 * Protocolo de `/ws/stream`: espejo tipado de `latentflow/api/server.py`.
 *
 * La UI usa nombres semánticos (`UPDATE_CIRCUIT`, `STREAM_CHUNK`); los valores
 * son los literales reales del cable. Si el servidor cambia, sólo cambia aquí.
 */

export const DEFAULT_WS_URL = 'ws://127.0.0.1:8000/ws/stream'

export const ClientAction = {
  GENERATE: 'generate',
  UPDATE_CIRCUIT: 'update',
  CANCEL: 'cancel',
  CALIBRATE: 'calibrate',
  STATUS: 'status',
  PING: 'ping',
} as const

export const ServerEvent = {
  READY: 'ready',
  ACCEPTED: 'accepted',
  START: 'start',
  STREAM_CHUNK: 'token',
  END: 'end',
  UPDATE_ACCEPTED: 'update_accepted',
  CIRCUIT: 'circuit',
  CALIBRATED: 'calibrated',
  STATUS: 'status',
  PONG: 'pong',
  ERROR: 'error',
} as const

// ---------------------------------------------------------------- dominio
export type SteeringMode = 'additive' | 'orthogonal'
export type TelemetryMode = 'none' | 'summary' | 'risky' | 'all'

/** Componente de la dirección de una capa: feature de SAE o, con `null`, el r_q calibrado. */
export interface FeatureSpec {
  feature_id: string | null
  alpha: number
}

/**
 * Inyección en una capa. Con `features`, la dirección es `alpha · Σ_f alpha_f · r_q(f)`,
 * con `r_q(f) = q_proj(input_layernorm(W_dec[f]))` calculado en el servidor.
 */
export interface LayerSpec {
  layer: number
  alpha: number
  mode: SteeringMode
  features?: FeatureSpec[]
}

export interface ResolvedFeatureInfo extends FeatureSpec {
  /** `dev-hash` (simulada), `gemma-scope:<ruta>` o `calibrated`. */
  source: string
  sae_layer?: number | null
}

export interface LayerPlanEntry extends Omit<LayerSpec, 'features'> {
  features?: ResolvedFeatureInfo[]
}

export interface SaeInfo {
  kind: 'dev-hash' | 'gemma-scope' | string
  root?: string | null
  hf_cache?: boolean
}

export interface AttentionGeometry {
  num_layers: number
  num_heads: number
  num_kv_heads: number
  head_dim: number
}

export interface CircuitSnapshot {
  version: number
  calibrated: boolean
  steering: boolean
  skop_filter: boolean
  alpha: number
  gamma_energy: number
  risk_percentile: number
  normalize_rayleigh: boolean
  risk_threshold: number | null
  num_risky_heads: number
  risky_heads: Record<string, number[]>
  active_layers: number[]
  layer_plan: LayerPlanEntry[] | null
}

export interface HeadTelemetry {
  layer: number
  head: number
  /** R(q) de la query actual, ya steerada: energía absoluta (≥ 0), o en [0, 1] si `normalize_rayleigh`. */
  rayleigh: number
  /** R(q − δ): la misma query sin el vector inyectado en su capa. Misma escala que `rayleigh`. */
  unsteered: number
  /** R de la dirección de steering de la capa (escala del ranking de riesgo, no de las queries). */
  baseline: number
  risky: boolean
  steered: boolean
  projected: boolean
}

export interface TelemetrySummary {
  mean: number
  max: number
  /** Media de R(q − δ) sobre las mismas cabezas que `mean`. */
  mean_unsteered: number
  argmax: { layer: number; head: number }
  /** Umbral del ranking (escala de `baseline`, no comparable con R(q)). */
  risk_threshold: number | null
}

export type Telemetry =
  | { available: false }
  | { available: true; summary: TelemetrySummary; heads?: HeadTelemetry[] }

// ---------------------------------------------------- cliente -> servidor
export interface GenerateMessage {
  action: typeof ClientAction.GENERATE
  prompt: string
  request_id?: string
  max_new_tokens?: number
  temperature?: number
  top_p?: number
  chat?: boolean
  telemetry?: TelemetryMode
}

export interface UpdateCircuitMessage {
  action: typeof ClientAction.UPDATE_CIRCUIT
  alpha?: number
  steering?: boolean
  skop_filter?: boolean
  gamma_energy?: number
  risk_percentile?: number
  layers?: LayerSpec[]
  reset_layers?: boolean
}

export type ClientMessage =
  | GenerateMessage
  | UpdateCircuitMessage
  | { action: typeof ClientAction.CANCEL }
  /** `utility` estima Σ_ΔK (Focus/Tail); `positive`/`negative`, la dirección r_q. */
  | { action: typeof ClientAction.CALIBRATE; utility: string[]; positive: string[]; negative: string[] }
  | { action: typeof ClientAction.STATUS }
  | { action: typeof ClientAction.PING }

// ---------------------------------------------------- servidor -> cliente
export interface ReadyEvent {
  type: typeof ServerEvent.READY
  model: string
  geometry: AttentionGeometry
  layers: number[]
  hidden_size: number
  sae: SaeInfo
  circuit: CircuitSnapshot
}
export interface AcceptedEvent {
  type: typeof ServerEvent.ACCEPTED
  request_id: string
  queued: boolean
}
export interface StartEvent {
  type: typeof ServerEvent.START
  request_id: string
  prompt_tokens: number
  circuit: CircuitSnapshot
}
export interface StreamChunkEvent {
  type: typeof ServerEvent.STREAM_CHUNK
  request_id: string
  index: number
  token: string
  token_id: number
  circuit_version: number
  telemetry: Telemetry | null
}
export type EndReason = 'eos' | 'length' | 'cancelled' | 'stop' | 'error'
export interface EndEvent {
  type: typeof ServerEvent.END
  request_id: string
  reason: EndReason
  num_tokens: number
  text: string
  elapsed_s: number
  tokens_per_s: number | null
}
export interface UpdateAcceptedEvent {
  type: typeof ServerEvent.UPDATE_ACCEPTED
  changes: Record<string, unknown>
  deferred: boolean
  features?: { feature_id: string; source: string; sae_layer: number | null }[]
}
export interface CircuitEvent {
  type: typeof ServerEvent.CIRCUIT
  changes: Record<string, unknown>
  applied_at_index: number | null
  request_id: string | null
  circuit: CircuitSnapshot
}
export interface CalibratedEvent {
  type: typeof ServerEvent.CALIBRATED
  num_pairs: number
  num_utility: number
  circuit: CircuitSnapshot
}
export interface StatusEvent {
  type: typeof ServerEvent.STATUS
  circuit: CircuitSnapshot
  busy: boolean
  active_request_id: string | null
  sessions: number
}
export interface PongEvent {
  type: typeof ServerEvent.PONG
  server_time: number
}
export interface ErrorEvent {
  type: typeof ServerEvent.ERROR
  code: string
  message: string
  request_id?: string
  /** En `invalid_message`: la acción del mensaje rechazado, si se pudo leer. */
  action?: string
  details?: unknown
}

export type ServerMessage =
  | ReadyEvent
  | AcceptedEvent
  | StartEvent
  | StreamChunkEvent
  | EndEvent
  | UpdateAcceptedEvent
  | CircuitEvent
  | CalibratedEvent
  | StatusEvent
  | PongEvent
  | ErrorEvent

const KNOWN_EVENTS = new Set<string>(Object.values(ServerEvent))

/** Parsea un frame del servidor; devuelve `null` si no es un evento conocido. */
export function parseServerMessage(raw: unknown): ServerMessage | null {
  if (typeof raw !== 'string') return null
  let value: unknown
  try {
    value = JSON.parse(raw)
  } catch {
    return null
  }
  if (typeof value !== 'object' || value === null) return null
  const type = (value as { type?: unknown }).type
  return typeof type === 'string' && KNOWN_EVENTS.has(type) ? (value as ServerMessage) : null
}
