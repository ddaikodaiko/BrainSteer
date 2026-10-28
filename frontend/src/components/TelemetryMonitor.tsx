import { useEffect, useRef, useState, type ReactNode } from 'react'
import { cx } from '../lib/cx'
import { drawScope, type ScopeScale, type ScopeView } from '../lib/scopeRenderer'
import { telemetryBuffer, type TelemetryBuffer } from '../lib/telemetryBuffer'

interface TelemetryMonitorProps {
  buffer?: TelemetryBuffer
  className?: string
}

/**
 * Osciloscopio del cociente de Rayleigh normalizado de las cabezas de riesgo.
 *
 * Canvas 2D puro: un bucle `requestAnimationFrame` compara `buffer.version` y
 * sólo redibuja si llegó un token nuevo (o cambió el tamaño/las opciones). Los
 * tokens no pasan por el estado de React, así que el streaming no re-renderiza
 * este componente.
 */
export function TelemetryMonitor({ buffer = telemetryBuffer, className }: TelemetryMonitorProps) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [scale, setScale] = useState<ScopeScale>('auto')
  const [showMean, setShowMean] = useState(true)
  const [showBaseline, setShowBaseline] = useState(true)
  const optionsRef = useRef({ scale, showMean, showBaseline, dirty: true })
  optionsRef.current = { scale, showMean, showBaseline, dirty: true }

  useEffect(() => {
    const wrap = wrapRef.current
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!wrap || !canvas || !ctx) return

    const view: ScopeView = { width: 0, height: 0, dpr: 1, scale: 'auto', showMean: true, showBaseline: true }
    let lastVersion = -1
    let raf = 0

    const resize = () => {
      const rect = wrap.getBoundingClientRect()
      view.dpr = window.devicePixelRatio || 1
      view.width = Math.max(1, Math.floor(rect.width))
      view.height = Math.max(1, Math.floor(rect.height))
      canvas.width = Math.round(view.width * view.dpr)
      canvas.height = Math.round(view.height * view.dpr)
      canvas.style.width = `${view.width}px`
      canvas.style.height = `${view.height}px`
      lastVersion = -1
    }
    const observer = new ResizeObserver(resize)
    observer.observe(wrap)
    resize()

    const frame = () => {
      raf = requestAnimationFrame(frame)
      const opts = optionsRef.current
      if (buffer.version === lastVersion && !opts.dirty) return
      opts.dirty = false
      lastVersion = buffer.version
      view.scale = opts.scale
      view.showMean = opts.showMean
      view.showBaseline = opts.showBaseline
      drawScope(ctx, buffer, view)
    }
    raf = requestAnimationFrame(frame)

    return () => {
      cancelAnimationFrame(raf)
      observer.disconnect()
    }
  }, [buffer])

  return (
    <section className={cx('flex min-h-0 flex-col rounded-2xl border border-ink-700 bg-ink-900/80', className)}>
      <header className="flex flex-wrap items-center gap-2 border-b border-ink-700/80 px-4 py-2.5">
        <span className="size-1.5 rounded-full bg-signal shadow-[0_0_8px_var(--color-signal)]" />
        <h2 className="text-[13px] font-semibold text-ink-100">Osciloscopio de riesgo</h2>
        <span className="font-mono text-[10.5px] text-ink-400">R̂(q) = qᵀΣ̂q / qᵀq · Σ̂ = Σ/tr(Σ)</span>
        <div className="ml-auto flex items-center gap-1">
          <Chip active={scale === 'auto'} onClick={() => setScale(scale === 'auto' ? 'unit' : 'auto')}>
            {scale === 'auto' ? 'Y auto' : 'Y [0,1]'}
          </Chip>
          <Chip active={showMean} onClick={() => setShowMean(!showMean)}>
            media
          </Chip>
          <Chip active={showBaseline} onClick={() => setShowBaseline(!showBaseline)}>
            r_q
          </Chip>
        </div>
      </header>
      <div ref={wrapRef} className="relative min-h-[180px] flex-1">
        <canvas
          ref={canvasRef}
          className="absolute inset-0"
          role="img"
          aria-label="Osciloscopio del cociente de Rayleigh normalizado por token de las cabezas de riesgo"
        />
      </div>
      <footer className="flex flex-wrap gap-x-4 gap-y-1 border-t border-ink-700/80 px-4 py-2 font-mono text-[10px] text-ink-400">
        <Legend color="var(--color-signal)">máx. global</Legend>
        <Legend color="var(--color-ortho)">cabeza proyectada (SKOP)</Legend>
        <Legend color="var(--color-additive)">cabeza aditiva</Legend>
        <Legend color="var(--color-ink-500)">capa sin steering</Legend>
        <Legend color="var(--color-warn)" dashed>
          umbral de riesgo τ
        </Legend>
        <Legend color="#a78bfa" dashed>
          hot-swap
        </Legend>
      </footer>
    </section>
  )
}

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cx(
        'rounded-md border px-2 py-0.5 font-mono text-[10.5px] transition-colors',
        active ? 'border-signal/40 bg-signal/10 text-signal' : 'border-ink-700 text-ink-400 hover:text-ink-200',
      )}
    >
      {children}
    </button>
  )
}

function Legend({ color, dashed, children }: { color: string; dashed?: boolean; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="inline-block h-0 w-3.5 border-t-2" style={{ borderColor: color, borderStyle: dashed ? 'dashed' : 'solid' }} />
      {children}
    </span>
  )
}
