import type { HeadTelemetry, Telemetry } from './protocol'

export interface HeadSeries {
  key: string
  layer: number
  head: number
  /** Ring buffer alineado con `TelemetryBuffer.tokens`; NaN = sin dato. */
  values: Float32Array
  /** R(q − δ) de la misma cabeza sin el vector inyectado, alineado con `values`. */
  unsteered: Float32Array
  baseline: number
  steered: boolean
  projected: boolean
  last: number
}

export interface ScopeMarker {
  /** Contador absoluto de muestra (no índice del anillo). */
  sample: number
  label: string
}

/**
 * Historia token a token del cociente de Rayleigh de las cabezas de riesgo.
 * Vive fuera de React: el WebSocket escribe y el osciloscopio lee en su bucle
 * `requestAnimationFrame`, comparando `version` para redibujar sólo si hay
 * datos nuevos. Ninguna llegada de token provoca un render de React.
 */
export class TelemetryBuffer {
  readonly capacity: number
  /** Nº total de muestras escritas desde el último `reset()`. */
  samples = 0
  version = 0
  requestId: string | null = null
  threshold: number | null = null
  lastTokenIndex: number | null = null
  readonly tokens: Int32Array
  readonly mean: Float32Array
  readonly max: Float32Array
  /** Máximo de R(q − δ) entre las series de cada muestra (para la escala Y). */
  readonly unsteeredMax: Float32Array
  readonly series = new Map<string, HeadSeries>()
  markers: ScopeMarker[] = []

  constructor(capacity = 256) {
    this.capacity = capacity
    this.tokens = new Int32Array(capacity)
    this.mean = new Float32Array(capacity).fill(Number.NaN)
    this.max = new Float32Array(capacity).fill(Number.NaN)
    this.unsteeredMax = new Float32Array(capacity).fill(Number.NaN)
  }

  reset(requestId: string | null = null): void {
    this.samples = 0
    this.requestId = requestId
    this.threshold = null
    this.lastTokenIndex = null
    this.mean.fill(Number.NaN)
    this.max.fill(Number.NaN)
    this.unsteeredMax.fill(Number.NaN)
    this.series.clear()
    this.markers = []
    this.version++
  }

  /** Nº de muestras válidas en el anillo (≤ capacity). */
  get length(): number {
    return Math.min(this.samples, this.capacity)
  }

  /** Posición en el anillo de la muestra absoluta `sample`. */
  slot(sample: number): number {
    return ((sample % this.capacity) + this.capacity) % this.capacity
  }

  push(tokenIndex: number, telemetry: Telemetry | null): void {
    const slot = this.slot(this.samples)
    this.tokens[slot] = tokenIndex
    this.lastTokenIndex = tokenIndex
    const seen = new Set<string>()
    let unsteeredMax = Number.NaN

    if (telemetry?.available) {
      this.mean[slot] = telemetry.summary.mean
      this.max[slot] = telemetry.summary.max
      this.threshold = telemetry.summary.risk_threshold
      for (const head of telemetry.heads ?? []) {
        if (!head.risky) continue
        const s = this.ensureSeries(head)
        s.values[slot] = head.rayleigh
        s.unsteered[slot] = head.unsteered
        if (Number.isFinite(head.unsteered) && !(head.unsteered <= unsteeredMax)) unsteeredMax = head.unsteered
        s.last = head.rayleigh
        s.baseline = head.baseline
        s.steered = head.steered
        s.projected = head.projected
        seen.add(s.key)
      }
    } else {
      this.mean[slot] = Number.NaN
      this.max[slot] = Number.NaN
    }
    for (const s of this.series.values()) {
      if (!seen.has(s.key)) {
        s.values[slot] = Number.NaN
        s.unsteered[slot] = Number.NaN
        s.last = Number.NaN
      }
    }
    this.unsteeredMax[slot] = unsteeredMax
    this.samples++
    // Descarta marcadores que ya salieron de la ventana.
    const oldest = this.samples - this.capacity
    if (this.markers.length && this.markers[0]!.sample < oldest) {
      this.markers = this.markers.filter((m) => m.sample >= oldest)
    }
    this.version++
  }

  /** Marca la última muestra (p. ej. un hot-swap aplicado en esa frontera de token). */
  mark(label: string): void {
    if (this.samples === 0) return
    this.markers.push({ sample: this.samples - 1, label })
    this.version++
  }

  /** Las `n` series con mayor valor en la última muestra. */
  top(n: number): HeadSeries[] {
    return [...this.series.values()]
      .filter((s) => Number.isFinite(s.last))
      .sort((a, b) => b.last - a.last)
      .slice(0, n)
  }

  private ensureSeries(head: HeadTelemetry): HeadSeries {
    const key = `${head.layer}:${head.head}`
    let s = this.series.get(key)
    if (!s) {
      s = {
        key,
        layer: head.layer,
        head: head.head,
        values: new Float32Array(this.capacity).fill(Number.NaN),
        unsteered: new Float32Array(this.capacity).fill(Number.NaN),
        baseline: head.baseline,
        steered: head.steered,
        projected: head.projected,
        last: Number.NaN,
      }
      this.series.set(key, s)
    }
    return s
  }
}

/** Instancia compartida entre el WebSocket y el osciloscopio. */
export const telemetryBuffer = new TelemetryBuffer(256)
