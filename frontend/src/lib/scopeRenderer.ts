import type { HeadSeries, TelemetryBuffer } from './telemetryBuffer'

export type ScopeScale = 'auto' | 'unit'

export interface ScopeView {
  width: number
  height: number
  dpr: number
  scale: ScopeScale
  showMean: boolean
  showBaseline: boolean
}

const PAD = { l: 46, r: 14, t: 30, b: 22 }
const BG = '#04070a'
const GRID = 'rgba(57, 245, 180, 0.06)'
const GRID_MAJOR = 'rgba(57, 245, 180, 0.14)'
const TEXT = '#5d6b80'
const SIGNAL = '#39f5b4'
const THRESHOLD = '#fbbf24'
const MARKER = '#a78bfa'
const FONT = '10px "JetBrains Mono", ui-monospace, Menlo, monospace'
const NICE_STEPS = [0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 1]

/** Menor tope "redondo" ≥ x (rango del eje Y; el cociente normalizado vive en [0, 1]). */
export function niceCeil(x: number): number {
  if (!Number.isFinite(x) || x <= 0) return NICE_STEPS[0]!
  return NICE_STEPS.find((s) => s >= x) ?? 1
}

/** Primera muestra visible: la ventana es la capacidad del anillo. */
export function windowStart(buf: TelemetryBuffer): number {
  return Math.max(0, buf.samples - buf.capacity)
}

export function yMaxFor(buf: TelemetryBuffer, scale: ScopeScale): number {
  if (scale === 'unit') return 1
  let max = buf.threshold ?? 0
  for (let s = windowStart(buf); s < buf.samples; s++) {
    const v = buf.max[buf.slot(s)]!
    if (v > max) max = v
  }
  return niceCeil(max * 1.15)
}

/** Color por familia: azul = proyección SKOP, rosa = aditivo, gris = capa sin steering. */
export function seriesColor(series: HeadSeries, rank: number, alpha = 0.9): string {
  const light = 52 + ((rank * 7) % 24)
  if (series.projected) return `hsla(199, 92%, ${light}%, ${alpha})`
  if (series.steered) return `hsla(330, 85%, ${light + 4}%, ${alpha})`
  return `hsla(215, 12%, ${light - 6}%, ${alpha * 0.6})`
}

/** Dibuja un frame completo. Sin asignaciones por punto: recorre los Float32Array. */
export function drawScope(ctx: CanvasRenderingContext2D, buf: TelemetryBuffer, view: ScopeView): void {
  const { width, height, dpr } = view
  const plotW = Math.max(1, width - PAD.l - PAD.r)
  const plotH = Math.max(1, height - PAD.t - PAD.b)
  const yMax = yMaxFor(buf, view.scale)
  const s0 = windowStart(buf)
  const span = Math.max(1, buf.capacity - 1)
  const xOf = (sample: number) => PAD.l + ((sample - s0) / span) * plotW
  const yOf = (v: number) => PAD.t + plotH - (Math.min(v, yMax) / yMax) * plotH

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.fillStyle = BG
  ctx.fillRect(0, 0, width, height)

  // Retícula de osciloscopio: 10 × 5 divisiones con subdivisiones.
  ctx.lineWidth = 1
  for (let i = 0; i <= 10; i++) {
    const x = Math.round(PAD.l + (plotW * i) / 10) + 0.5
    ctx.strokeStyle = i % 5 === 0 ? GRID_MAJOR : GRID
    line(ctx, x, PAD.t, x, PAD.t + plotH)
  }
  ctx.font = FONT
  ctx.textBaseline = 'middle'
  ctx.textAlign = 'right'
  for (let j = 0; j <= 5; j++) {
    const y = Math.round(PAD.t + (plotH * j) / 5) + 0.5
    ctx.strokeStyle = j === 5 ? GRID_MAJOR : GRID
    line(ctx, PAD.l, y, PAD.l + plotW, y)
    ctx.fillStyle = TEXT
    ctx.fillText(formatY(yMax * (1 - j / 5), yMax), PAD.l - 6, y)
  }

  if (buf.samples === 0) {
    ctx.textAlign = 'center'
    ctx.fillStyle = TEXT
    ctx.font = '11px "JetBrains Mono", ui-monospace, monospace'
    ctx.fillText('esperando telemetría · STREAM_CHUNK', PAD.l + plotW / 2, PAD.t + plotH / 2)
    return
  }

  // Marcadores de hot-swap (evento `circuit` aplicado en una frontera de token).
  ctx.setLineDash([3, 4])
  ctx.strokeStyle = MARKER
  ctx.fillStyle = MARKER
  ctx.textAlign = 'left'
  ctx.textBaseline = 'top'
  for (const m of buf.markers) {
    const x = Math.round(xOf(m.sample)) + 0.5
    line(ctx, x, PAD.t, x, PAD.t + plotH)
    ctx.fillText(m.label, x + 3, PAD.t + 2)
  }

  // Umbral de riesgo (percentil de calibración, misma escala normalizada).
  if (buf.threshold !== null && buf.threshold <= yMax) {
    const y = Math.round(yOf(buf.threshold)) + 0.5
    ctx.strokeStyle = THRESHOLD
    ctx.globalAlpha = 0.75
    line(ctx, PAD.l, y, PAD.l + plotW, y)
    ctx.globalAlpha = 1
    ctx.fillStyle = THRESHOLD
    ctx.textAlign = 'right'
    ctx.textBaseline = 'bottom'
    ctx.fillText(`τ ${buf.threshold.toFixed(3)}`, PAD.l + plotW - 2, y - 2)
  }
  ctx.setLineDash([])

  // Trazas por cabeza de riesgo.
  const ranked = buf.top(Number.POSITIVE_INFINITY)
  const rankOf = new Map(ranked.map((s, i) => [s.key, i]))
  // Orden de pintado: capas sin steering debajo; cabezas activas encima y más gruesas.
  const ordered = [...buf.series.values()].sort((a, b) => Number(a.steered) - Number(b.steered))
  ctx.lineJoin = 'round'
  for (const s of ordered) {
    ctx.lineWidth = s.steered ? 1.6 : 1
    ctx.strokeStyle = seriesColor(s, rankOf.get(s.key) ?? 0, s.steered ? 0.95 : 0.75)
    tracePath(ctx, buf, s.values, s0, xOf, yOf)
    if (view.showBaseline && Number.isFinite(s.baseline) && s.baseline <= yMax) {
      // Referencia de calibración r_q: marca en el borde derecho.
      const y = yOf(s.baseline)
      ctx.strokeStyle = seriesColor(s, rankOf.get(s.key) ?? 0, 0.5)
      line(ctx, PAD.l + plotW + 2, y, PAD.l + plotW + 8, y)
    }
  }

  if (view.showMean) {
    ctx.setLineDash([4, 3])
    ctx.strokeStyle = 'rgba(227, 232, 239, 0.55)'
    ctx.lineWidth = 1
    tracePath(ctx, buf, buf.mean, s0, xOf, yOf)
    ctx.setLineDash([])
  }

  // Envolvente máxima con brillo de fósforo.
  ctx.save()
  ctx.shadowColor = SIGNAL
  ctx.shadowBlur = 8
  ctx.strokeStyle = SIGNAL
  ctx.lineWidth = 2
  tracePath(ctx, buf, buf.max, s0, xOf, yOf)
  const last = buf.max[buf.slot(buf.samples - 1)]!
  if (Number.isFinite(last)) {
    ctx.fillStyle = SIGNAL
    ctx.beginPath()
    ctx.arc(xOf(buf.samples - 1), yOf(last), 3, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.restore()

  // Lecturas: cabecera izquierda y top-4 cabezas a la derecha.
  ctx.font = FONT
  ctx.textBaseline = 'middle'
  ctx.textAlign = 'left'
  const meanLast = buf.mean[buf.slot(buf.samples - 1)]!
  ctx.fillStyle = SIGNAL
  ctx.fillText(`max ${fmt(last)}`, PAD.l, 14)
  ctx.fillStyle = 'rgba(227, 232, 239, 0.75)'
  ctx.fillText(`mean ${fmt(meanLast)}`, PAD.l + 78, 14)
  ctx.fillStyle = TEXT
  const info = `tok #${buf.lastTokenIndex ?? '—'} · ${buf.series.size} cabezas`
  ctx.fillText(info, PAD.l + 162, 14)
  const leftEnd = PAD.l + 162 + ctx.measureText(info).width + 16

  // Top-4 cabezas por valor actual, de derecha a izquierda mientras quepan.
  ctx.textAlign = 'right'
  let x = width - PAD.r
  for (const s of ranked.slice(0, 4)) {
    const label = `L${s.layer}·H${s.head} ${fmt(s.last)}`
    const w = ctx.measureText(label).width
    if (x - w < leftEnd) break
    ctx.fillStyle = seriesColor(s, rankOf.get(s.key) ?? 0, 1)
    ctx.fillText(label, x, 14)
    x -= w + 12
  }

  // Eje X: índices de token en los extremos de la ventana.
  ctx.fillStyle = TEXT
  ctx.textBaseline = 'top'
  ctx.textAlign = 'left'
  ctx.fillText(String(buf.tokens[buf.slot(s0)]), PAD.l, PAD.t + plotH + 6)
  ctx.textAlign = 'right'
  ctx.fillText(String(buf.lastTokenIndex ?? ''), xOf(buf.samples - 1), PAD.t + plotH + 6)
}

function tracePath(
  ctx: CanvasRenderingContext2D,
  buf: TelemetryBuffer,
  values: Float32Array,
  s0: number,
  xOf: (s: number) => number,
  yOf: (v: number) => number,
): void {
  ctx.beginPath()
  let pen = false
  for (let s = s0; s < buf.samples; s++) {
    const v = values[buf.slot(s)]!
    if (!Number.isFinite(v)) {
      pen = false // hueco: no interpola sobre tokens sin dato
      continue
    }
    const x = xOf(s)
    const y = yOf(v)
    if (pen) ctx.lineTo(x, y)
    else ctx.moveTo(x, y)
    pen = true
  }
  ctx.stroke()
}

function line(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number): void {
  ctx.beginPath()
  ctx.moveTo(x0, y0)
  ctx.lineTo(x1, y1)
  ctx.stroke()
}

const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '—')
const formatY = (v: number, yMax: number) => v.toFixed(yMax < 0.05 ? 3 : 2)
