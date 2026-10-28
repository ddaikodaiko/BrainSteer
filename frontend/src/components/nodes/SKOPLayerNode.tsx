import { memo } from 'react'
import { Handle, Position, type NodeProps } from 'reactflow'
import { FALLBACK_MAX_LAYER, type SKOPLayerData } from '../../lib/circuit'
import { cx } from '../../lib/cx'
import { useCircuitStore } from '../../store/useCircuitStore'
import { FieldLabel, NodeShell } from './NodeShell'

const MODE_META = {
  additive: { label: '[Additive Steering]', formula: "q' = q + α·r_q", color: 'var(--color-additive)', hex: '#f472b6' },
  orthogonal: {
    label: '[Orthogonal Projection]',
    formula: "q' = q + α·P_ΔK·r_q",
    color: 'var(--color-ortho)',
    hex: '#38bdf8',
  },
} as const

function SKOPLayerNodeImpl({ id, data, selected }: NodeProps<SKOPLayerData>) {
  const updateNodeData = useCircuitStore((s) => s.updateNodeData<SKOPLayerData>)
  const removeNode = useCircuitStore((s) => s.removeNode)
  const numLayers = useCircuitStore((s) => s.numLayers)
  const derived = useCircuitStore((s) => s.serialized.derived[id])
  const riskyHeads = useCircuitStore((s) => s.serverCircuit?.risky_heads[String(data.layer)]?.length ?? null)
  const applied = useCircuitStore(
    (s) => s.serverCircuit?.steering === true && (s.serverCircuit.layer_plan?.some((p) => p.layer === data.layer) ?? false),
  )

  const maxLayer = numLayers ? numLayers - 1 : FALLBACK_MAX_LAYER
  const meta = MODE_META[data.mode]
  const active = derived?.active ?? false
  const setLayer = (value: number) => updateNodeData(id, { layer: value })
  const toggleMode = () => updateNodeData(id, { mode: data.mode === 'orthogonal' ? 'additive' : 'orthogonal' })

  return (
    <NodeShell
      kicker="SKOP · inyector"
      title={`Capa ${data.layer}`}
      accent={meta.hex}
      active={active}
      enabled={data.enabled}
      selected={selected}
      onToggleEnabled={() => updateNodeData(id, { enabled: !data.enabled })}
      onRemove={() => removeNode(id)}
    >
      <Handle type="target" position={Position.Left} style={{ background: active ? meta.hex : undefined }} />

      <div className="space-y-1.5">
        <FieldLabel htmlFor={`${id}-layer`}>
          Capa del modelo · 0–{maxLayer}
          {numLayers === null && <span className="ml-1 normal-case text-ink-500">(sin geometría)</span>}
        </FieldLabel>
        <div className="flex items-stretch overflow-hidden rounded-lg border border-ink-700 bg-ink-950/70">
          <button
            type="button"
            className="nodrag w-9 text-lg text-ink-300 hover:bg-ink-800 disabled:opacity-30"
            onClick={() => setLayer(data.layer - 1)}
            disabled={data.layer <= 0}
            aria-label="Capa anterior"
          >
            −
          </button>
          <input
            id={`${id}-layer`}
            type="number"
            className="no-spin nodrag nowheel min-w-0 flex-1 bg-transparent text-center font-mono text-lg font-semibold tabular-nums text-ink-100 focus:outline-none"
            min={0}
            max={maxLayer}
            step={1}
            value={data.layer}
            onChange={(e) => e.target.value !== '' && setLayer(Number(e.target.value))}
          />
          <button
            type="button"
            className="nodrag w-9 text-lg text-ink-300 hover:bg-ink-800 disabled:opacity-30"
            onClick={() => setLayer(data.layer + 1)}
            disabled={data.layer >= maxLayer}
            aria-label="Capa siguiente"
          >
            +
          </button>
        </div>
      </div>

      <div className="space-y-1.5">
        <FieldLabel>Modo de inyección</FieldLabel>
        <button
          type="button"
          role="switch"
          aria-checked={data.mode === 'orthogonal'}
          onClick={toggleMode}
          className="nodrag relative grid w-full grid-cols-2 rounded-lg border border-ink-700 bg-ink-950/70 p-0.5 font-mono text-[9.5px] whitespace-nowrap"
          title="Alternar entre steering aditivo y proyección ortogonal SKOP"
        >
          <span
            className="absolute inset-y-0.5 w-[calc(50%-2px)] rounded-md transition-[left,background] duration-200"
            style={{
              left: data.mode === 'additive' ? '2px' : 'calc(50%)',
              background: `color-mix(in srgb, ${meta.color} 22%, transparent)`,
              boxShadow: `inset 0 0 0 1px color-mix(in srgb, ${meta.color} 60%, transparent)`,
            }}
          />
          {(['additive', 'orthogonal'] as const).map((mode) => (
            <span
              key={mode}
              className={cx('relative py-1.5 text-center transition-colors', data.mode === mode ? '' : 'text-ink-400')}
              style={{ color: data.mode === mode ? MODE_META[mode].color : undefined }}
            >
              {MODE_META[mode].label}
            </span>
          ))}
        </button>
        <div className="text-center font-mono text-[10.5px] text-ink-300">{meta.formula}</div>
      </div>

      <dl className="grid grid-cols-3 gap-1.5 text-center">
        <Stat label="α Σ" value={derived?.alpha !== undefined ? derived.alpha.toFixed(2) : '—'} />
        <Stat label="riesgo" value={riskyHeads === null ? '—' : String(riskyHeads)} />
        <Stat
          label="servidor"
          value={applied ? 'aplicada' : active ? 'pendiente' : 'inactiva'}
          tone={applied ? 'ok' : active ? 'warn' : 'muted'}
        />
      </dl>
    </NodeShell>
  )
}

function Stat({ label, value, tone = 'default' }: { label: string; value: string; tone?: 'default' | 'ok' | 'warn' | 'muted' }) {
  const color = {
    default: 'text-ink-100',
    ok: 'text-signal',
    warn: 'text-warn',
    muted: 'text-ink-400',
  }[tone]
  return (
    <div className="rounded-md border border-ink-700/70 bg-ink-900/60 px-1 py-1.5">
      <dt className="font-mono text-[8.5px] tracking-[0.12em] text-ink-400 uppercase">{label}</dt>
      <dd className={cx('font-mono text-[12px] font-semibold tabular-nums', color)}>{value}</dd>
    </div>
  )
}

export const SKOPLayerNode = memo(SKOPLayerNodeImpl)
