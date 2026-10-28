import { memo, useEffect, useRef, useState, type CSSProperties, type KeyboardEvent } from 'react'
import { Handle, Position, type NodeProps } from 'reactflow'
import {
  ALPHA_MAX,
  ALPHA_MIN,
  ALPHA_STEP,
  featureSource,
  neuronpediaUrl,
  parseGemmaScopeId,
  type GemmaScopeData,
} from '../../lib/circuit'
import { useCircuitStore } from '../../store/useCircuitStore'
import { FieldLabel, NodeShell } from './NodeShell'

const ACCENT = '#38bdf8'

function GemmaScopeNodeImpl({ id, data, selected }: NodeProps<GemmaScopeData>) {
  const updateNodeData = useCircuitStore((s) => s.updateNodeData<GemmaScopeData>)
  const removeNode = useCircuitStore((s) => s.removeNode)
  const active = useCircuitStore((s) => s.serialized.derived[id]?.active ?? false)
  const source = useCircuitStore((s) => featureSource(s.serverCircuit, data.featureId))

  // El ID se edita en un borrador local y sólo llega al circuito (y al servidor) al
  // pulsar Enter o al salir del campo: nada de UPDATE_CIRCUIT con IDs a medio escribir.
  const [draft, setDraft] = useState(data.featureId)
  const discardDraft = useRef(false)
  useEffect(() => setDraft(data.featureId), [data.featureId])
  const commitDraft = () => {
    if (discardDraft.current) {
      discardDraft.current = false
      return
    }
    const next = draft.trim()
    if (next !== data.featureId) updateNodeData(id, { featureId: next })
    else setDraft(next)
  }
  const onFeatureKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') e.currentTarget.blur()
    if (e.key === 'Escape') {
      discardDraft.current = true
      setDraft(data.featureId)
      e.currentTarget.blur()
    }
  }
  const editing = draft.trim() !== data.featureId
  const status = editing
    ? { label: '↵ para aplicar', tone: 'text-ink-300', title: 'Enter o salir del campo: aplicar · Esc: descartar' }
    : resolutionStatus(data.featureId, source, active)

  const url = neuronpediaUrl(data.featureId)
  const zero = ((0 - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)) * 100
  const pos = ((data.alpha - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)) * 100
  const negative = data.alpha < 0
  const sliderStyle = {
    '--fill-from': `${Math.min(zero, pos)}%`,
    '--fill-to': `${Math.max(zero, pos)}%`,
    '--fill-color': negative ? 'var(--color-danger)' : 'var(--color-ortho)',
  } as CSSProperties

  return (
    <NodeShell
      kicker="Gemma Scope · feature"
      title={data.featureId.trim() ? `#${data.featureId.split('/').pop()}` : 'Feature sin ID'}
      accent={ACCENT}
      active={active}
      enabled={data.enabled}
      selected={selected}
      onToggleEnabled={() => updateNodeData(id, { enabled: !data.enabled })}
      onRemove={() => removeNode(id)}
    >
      <div className="space-y-1.5">
        <FieldLabel htmlFor={`${id}-feature`}>Feature ID · Neuronpedia</FieldLabel>
        <input
          id={`${id}-feature`}
          className="nodrag w-full rounded-lg border border-ink-700 bg-ink-950/70 px-2.5 py-1.5 font-mono text-[11.5px] text-ink-100 placeholder:text-ink-600 focus:border-ortho/70 focus:outline-none"
          placeholder="gemma-2-2b/20-gemmascope-res-16k/12082"
          value={draft}
          spellCheck={false}
          enterKeyHint="done"
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commitDraft}
          onKeyDown={onFeatureKeyDown}
        />
        <div className="flex h-4 items-center justify-between text-[10.5px]">
          {url ? (
            <a href={url} target="_blank" rel="noreferrer" className="nodrag text-ortho/90 hover:text-ortho">
              Abrir en Neuronpedia ↗
            </a>
          ) : (
            <span className="text-ink-500">Formato: modelo/sae/índice</span>
          )}
          <span className={status.tone} title={status.title}>
            {status.label}
          </span>
        </div>
      </div>

      <div className="space-y-1">
        <div className="flex items-baseline justify-between">
          <FieldLabel htmlFor={`${id}-alpha`}>Intensidad α</FieldLabel>
          <button
            type="button"
            className="nodrag font-mono text-[15px] font-semibold tabular-nums"
            style={{ color: negative ? 'var(--color-danger)' : data.alpha === 0 ? 'var(--color-ink-300)' : ACCENT }}
            title="Doble clic: volver a 0"
            onDoubleClick={() => updateNodeData(id, { alpha: 0 })}
          >
            {data.alpha >= 0 ? '+' : '−'}
            {Math.abs(data.alpha).toFixed(2)}
          </button>
        </div>
        <input
          id={`${id}-alpha`}
          type="range"
          className="alpha-slider nodrag nowheel"
          style={sliderStyle}
          min={ALPHA_MIN}
          max={ALPHA_MAX}
          step={ALPHA_STEP}
          value={data.alpha}
          onChange={(e) => updateNodeData(id, { alpha: Number(e.target.value) })}
          aria-valuetext={`alpha ${data.alpha.toFixed(2)}`}
        />
        <div className="flex justify-between font-mono text-[9.5px] text-ink-400">
          <span>{ALPHA_MIN.toFixed(1)}</span>
          <span>0</span>
          <span>+{ALPHA_MAX.toFixed(1)}</span>
        </div>
      </div>

      <Handle type="source" position={Position.Right} style={{ background: active ? ACCENT : undefined }} />
    </NodeShell>
  )
}

/** Cómo resolvió el servidor el decoder de la feature (ver `featureSource`). */
function resolutionStatus(featureId: string, source: string | null, active: boolean) {
  if (!featureId.trim()) {
    return { label: 'r_q calibrado', tone: 'text-ink-400', title: 'Sin Feature ID: usa la dirección r_q calibrada por SKOP' }
  }
  if (parseGemmaScopeId(featureId)?.site === 'att') {
    return { label: 'error: att no soportado', tone: 'text-danger', title: 'Los SAEs de atención no viven en el residual stream' }
  }
  if (source === 'dev-hash') {
    return { label: 'simulada (dev)', tone: 'text-warn', title: 'Dev backend: vector determinista derivado del hash del ID' }
  }
  if (source?.startsWith('gemma-scope:')) {
    return { label: 'W_dec ✓', tone: 'text-signal', title: `Decoder leído de ${source.slice('gemma-scope:'.length)}` }
  }
  return active
    ? { label: 'pendiente', tone: 'text-ink-300', title: 'Esperando a que el servidor resuelva el decoder' }
    : { label: 'sin aplicar', tone: 'text-ink-500', title: 'Conéctala a un inyector de capa habilitado' }
}

export const GemmaScopeNode = memo(GemmaScopeNodeImpl)
