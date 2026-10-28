import { memo, type CSSProperties } from 'react'
import { Handle, Position, type NodeProps } from 'reactflow'
import { ALPHA_MAX, ALPHA_MIN, ALPHA_STEP, neuronpediaUrl, type GemmaScopeData } from '../../lib/circuit'
import { useCircuitStore } from '../../store/useCircuitStore'
import { FieldLabel, NodeShell } from './NodeShell'

const ACCENT = '#38bdf8'

function GemmaScopeNodeImpl({ id, data, selected }: NodeProps<GemmaScopeData>) {
  const updateNodeData = useCircuitStore((s) => s.updateNodeData<GemmaScopeData>)
  const removeNode = useCircuitStore((s) => s.removeNode)
  const active = useCircuitStore((s) => s.serialized.derived[id]?.active ?? false)

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
          value={data.featureId}
          spellCheck={false}
          onChange={(e) => updateNodeData(id, { featureId: e.target.value })}
        />
        <div className="flex h-4 items-center justify-between text-[10.5px]">
          {url ? (
            <a href={url} target="_blank" rel="noreferrer" className="nodrag text-ortho/90 hover:text-ortho">
              Abrir en Neuronpedia ↗
            </a>
          ) : (
            <span className="text-ink-500">Formato: modelo/sae/índice</span>
          )}
          <span className="text-ink-400" title="El servidor aplica el vector r_q calibrado; el ID viaja como metadato del DAG">
            metadato
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

export const GemmaScopeNode = memo(GemmaScopeNodeImpl)
