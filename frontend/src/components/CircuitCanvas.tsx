import { useMemo, useState } from 'react'
import ReactFlow, { Background, BackgroundVariant, Controls, MiniMap, Panel, type Edge } from 'reactflow'
import { cx } from '../lib/cx'
import { useCircuitStore } from '../store/useCircuitStore'
import { nodeTypes } from './nodes'

const MINIMAP_COLORS: Record<string, string> = { gemmaScope: '#38bdf8', skopLayer: '#f472b6' }

export function CircuitCanvas({ className }: { className?: string }) {
  const nodes = useCircuitStore((s) => s.nodes)
  const edges = useCircuitStore((s) => s.edges)
  const derived = useCircuitStore((s) => s.serialized.derived)
  const onNodesChange = useCircuitStore((s) => s.onNodesChange)
  const onEdgesChange = useCircuitStore((s) => s.onEdgesChange)
  const onConnect = useCircuitStore((s) => s.onConnect)
  const isValidConnection = useCircuitStore((s) => s.isValidConnection)
  const addGemmaScopeNode = useCircuitStore((s) => s.addGemmaScopeNode)
  const addLayerNode = useCircuitStore((s) => s.addLayerNode)

  // Resalta las aristas que forman parte del circuito activo.
  const styledEdges = useMemo<Edge[]>(
    () =>
      edges.map((e) => {
        const live = derived[e.source]?.active && derived[e.target]?.active
        return { ...e, animated: Boolean(live), className: live ? 'active-edge' : undefined }
      }),
    [edges, derived],
  )

  const spawn = (kind: 'feature' | 'layer') => {
    const offset = (nodes.length % 6) * 28
    if (kind === 'feature') addGemmaScopeNode({ x: 40 + offset, y: 60 + offset })
    else addLayerNode({ x: 420 + offset, y: 60 + offset })
  }

  return (
    <section className={cx('relative flex min-h-0 flex-col overflow-hidden rounded-2xl border border-ink-700', className)}>
      <div className="min-h-0 flex-1">
        <ReactFlow
          nodes={nodes}
          edges={styledEdges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          isValidConnection={isValidConnection}
          deleteKeyCode={['Backspace', 'Delete']}
          fitView
          fitViewOptions={{ padding: 0.2, maxZoom: 1 }}
          minZoom={0.3}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={22} size={1.2} color="#1c2533" />
          <Controls showInteractive={false} position="bottom-left" />
          <MiniMap
            position="bottom-right"
            pannable
            zoomable
            maskColor="rgb(5 7 11 / 0.7)"
            nodeColor={(n) => MINIMAP_COLORS[n.type ?? ''] ?? '#5d6b80'}
            nodeStrokeWidth={0}
            style={{ width: 150, height: 96 }}
          />
          <Panel position="top-left" className="flex gap-2">
            <ToolButton onClick={() => spawn('feature')} color="var(--color-ortho)">
              + Feature Gemma Scope
            </ToolButton>
            <ToolButton onClick={() => spawn('layer')} color="var(--color-additive)">
              + Inyector de capa SKOP
            </ToolButton>
          </Panel>
        </ReactFlow>
      </div>
      <CircuitInspector />
    </section>
  )
}

function ToolButton({ onClick, color, children }: { onClick: () => void; color: string; children: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-lg border border-ink-700 bg-ink-850/90 px-3 py-1.5 text-[12px] font-medium text-ink-100 backdrop-blur hover:border-ink-500"
      style={{ boxShadow: `inset 3px 0 0 ${color}` }}
    >
      {children}
    </button>
  )
}

type InspectorTab = 'diagnostics' | 'payload' | 'dag'

/** Panel inferior: diagnósticos de serialización, payload enviado y DAG exportable. */
function CircuitInspector() {
  const serialized = useCircuitStore((s) => s.serialized)
  const sync = useCircuitStore((s) => s.sync)
  const [tab, setTab] = useState<InspectorTab>('diagnostics')
  const [open, setOpen] = useState(true)

  const json = tab === 'payload' ? serialized.update : serialized.dag
  const issues = serialized.diagnostics.filter((d) => d.level !== 'info').length

  const exportDag = () => {
    const blob = new Blob([JSON.stringify(serialized.dag, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = Object.assign(document.createElement('a'), { href: url, download: 'latentflow-circuit.json' })
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="border-t border-ink-700 bg-ink-900/95">
      <div className="flex items-center gap-1 px-3 py-1.5">
        {(
          [
            ['diagnostics', `Diagnóstico${issues ? ` · ${issues}` : ''}`],
            ['payload', 'Payload UPDATE_CIRCUIT'],
            ['dag', 'DAG'],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            onClick={() => {
              setTab(id)
              setOpen(true)
            }}
            className={cx(
              'rounded-md px-2.5 py-1 font-mono text-[10.5px]',
              open && tab === id ? 'bg-ink-700 text-ink-100' : 'text-ink-400 hover:text-ink-200',
              id === 'diagnostics' && issues > 0 && 'text-warn',
            )}
          >
            {label}
          </button>
        ))}
        <span className="ml-auto font-mono text-[10px] text-ink-400">
          {sync.sentCount} envíos · debounce 100 ms
        </span>
        <button type="button" onClick={exportDag} className="rounded-md px-2 py-1 font-mono text-[10.5px] text-ink-300 hover:bg-ink-800">
          Exportar ↓
        </button>
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="rounded-md px-2 py-1 text-[11px] text-ink-400 hover:bg-ink-800"
          aria-label={open ? 'Plegar panel' : 'Desplegar panel'}
        >
          {open ? '▾' : '▴'}
        </button>
      </div>
      {open && (
        <div className="scrollbar-thin max-h-40 overflow-auto px-3 pb-3">
          {tab === 'diagnostics' ? (
            <ul className="space-y-1">
              {serialized.diagnostics.map((d, i) => (
                <li key={`${d.code}-${i}`} className="flex gap-2 text-[11.5px]">
                  <span
                    className={cx(
                      'mt-[3px] h-2 w-2 shrink-0 rounded-full',
                      d.level === 'error' ? 'bg-danger' : d.level === 'warning' ? 'bg-warn' : 'bg-ink-500',
                    )}
                  />
                  <span className="text-ink-200">{d.message}</span>
                </li>
              ))}
              {sync.error && (
                <li className="flex gap-2 text-[11.5px]">
                  <span className="mt-[3px] h-2 w-2 shrink-0 rounded-full bg-danger" />
                  <span className="text-danger">Servidor: {sync.error}</span>
                </li>
              )}
              {serialized.diagnostics.length === 0 && !sync.error && <li className="text-[11.5px] text-ink-400">Sin incidencias.</li>}
            </ul>
          ) : (
            <pre className="font-mono text-[10.5px] leading-relaxed text-ink-200">{JSON.stringify(json, null, 2)}</pre>
          )}
        </div>
      )}
    </div>
  )
}
