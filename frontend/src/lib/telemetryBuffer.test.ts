import { describe, expect, it } from 'vitest'
import type { HeadTelemetry, Telemetry } from './protocol'
import { niceCeil, windowStart, yMaxFor } from './scopeRenderer'
import { TelemetryBuffer } from './telemetryBuffer'

const head = (layer: number, h: number, rayleigh: number, risky = true): HeadTelemetry => ({
  layer,
  head: h,
  rayleigh,
  baseline: 0.1,
  risky,
  steered: true,
  projected: risky,
})

const tel = (heads: HeadTelemetry[], threshold = 0.2): Telemetry => ({
  available: true,
  summary: {
    mean: heads.reduce((a, b) => a + b.rayleigh, 0) / Math.max(1, heads.length),
    max: Math.max(0, ...heads.map((x) => x.rayleigh)),
    argmax: { layer: 0, head: 0 },
    risk_threshold: threshold,
    live_over_threshold: 0,
  },
  heads,
})

describe('TelemetryBuffer', () => {
  it('guarda sólo cabezas de riesgo y alinea huecos con NaN', () => {
    const buf = new TelemetryBuffer(8)
    buf.push(0, tel([head(1, 2, 0.3), head(1, 3, 0.9, false)]))
    buf.push(1, tel([head(4, 0, 0.5)]))
    expect([...buf.series.keys()].sort()).toEqual(['1:2', '4:0'])
    const s12 = buf.series.get('1:2')!
    expect(s12.values[0]).toBeCloseTo(0.3)
    expect(Number.isNaN(s12.values[1]!)).toBe(true)
    expect(Number.isNaN(buf.series.get('4:0')!.values[0]!)).toBe(true)
    expect(buf.threshold).toBe(0.2)
  })

  it('funciona como anillo: la ventana avanza y descarta marcadores viejos', () => {
    const buf = new TelemetryBuffer(4)
    for (let i = 0; i < 3; i++) buf.push(i, tel([head(0, 0, i / 10)]))
    buf.mark('v2')
    for (let i = 3; i < 10; i++) buf.push(i, tel([head(0, 0, i / 10)]))
    expect(buf.length).toBe(4)
    expect(windowStart(buf)).toBe(6)
    expect(buf.tokens[buf.slot(6)]).toBe(6)
    expect(buf.series.get('0:0')!.values[buf.slot(9)]).toBeCloseTo(0.9)
    expect(buf.markers).toEqual([])
  })

  it('telemetría no disponible avanza el tiempo sin datos', () => {
    const buf = new TelemetryBuffer(4)
    buf.push(0, { available: false })
    buf.push(1, null)
    expect(buf.samples).toBe(2)
    expect(Number.isNaN(buf.max[0]!)).toBe(true)
  })

  it('version cambia en cada escritura (señal de redibujado)', () => {
    const buf = new TelemetryBuffer(4)
    const v0 = buf.version
    buf.push(0, tel([head(0, 0, 0.1)]))
    buf.mark('x')
    buf.reset('r2')
    expect(buf.version).toBe(v0 + 3)
    expect(buf.samples).toBe(0)
    expect(buf.requestId).toBe('r2')
  })

  it('top() ordena por el último valor', () => {
    const buf = new TelemetryBuffer(4)
    buf.push(0, tel([head(0, 0, 0.1), head(0, 1, 0.7), head(2, 5, 0.4)]))
    expect(buf.top(2).map((s) => s.key)).toEqual(['0:1', '2:5'])
  })
})

describe('escala del osciloscopio', () => {
  it('niceCeil redondea hacia arriba a pasos legibles', () => {
    expect(niceCeil(0.003)).toBe(0.005)
    expect(niceCeil(0.11)).toBe(0.2)
    expect(niceCeil(0.9)).toBe(1)
    expect(niceCeil(Number.NaN)).toBe(0.005)
  })

  it('el eje Y auto cubre el máximo de la ventana y el umbral', () => {
    const buf = new TelemetryBuffer(8)
    buf.push(0, tel([head(0, 0, 0.04)], 0.08))
    expect(yMaxFor(buf, 'auto')).toBe(0.1)
    expect(yMaxFor(buf, 'unit')).toBe(1)
  })
})
