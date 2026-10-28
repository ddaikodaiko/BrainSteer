import { cx } from '../lib/cx'
import { useCircuitStore, type SyncState } from '../store/useCircuitStore'
import { useSessionStore } from '../store/useSessionStore'

const SYNC_META: Record<SyncState, { label: string; tone: string }> = {
  idle: { label: 'sin enviar', tone: 'text-ink-400' },
  pending: { label: 'debounce…', tone: 'text-warn' },
  sent: { label: 'enviado', tone: 'text-ortho' },
  acked: { label: 'aceptado', tone: 'text-ortho' },
  applied: { label: 'aplicado', tone: 'text-signal' },
  offline: { label: 'sin conexión', tone: 'text-danger' },
  error: { label: 'rechazado', tone: 'text-danger' },
}

export function AppHeader() {
  const connection = useSessionStore((s) => s.connection)
  const attempt = useSessionStore((s) => s.reconnectAttempt)
  const url = useSessionStore((s) => s.url)
  const model = useSessionStore((s) => s.model)
  const latency = useSessionStore((s) => s.latencyMs)
  const sae = useSessionStore((s) => s.sae)
  const server = useCircuitStore((s) => s.serverCircuit)
  const sync = useCircuitStore((s) => s.sync)
  const syncMeta = SYNC_META[sync.state]

  return (
    <header className="flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-ink-800 bg-ink-900/70 px-5 py-3 backdrop-blur">
      <div className="flex items-center gap-2.5">
        <Logo />
        <div className="leading-tight">
          <div className="text-[14px] font-semibold tracking-tight text-ink-100">BrainSteer</div>
          <div className="font-mono text-[9.5px] tracking-[0.2em] text-ink-400 uppercase">LatentFlow · SKOP</div>
        </div>
      </div>

      <div className="flex items-center gap-2 rounded-full border border-ink-700 bg-ink-950/60 px-3 py-1" title={url}>
        <span
          className={cx(
            'size-2 rounded-full',
            connection === 'open' ? 'bg-signal shadow-[0_0_8px_var(--color-signal)]' : connection === 'connecting' ? 'animate-pulse bg-warn' : 'bg-danger',
          )}
        />
        <span className="font-mono text-[11px] text-ink-200">
          {connection === 'open' ? 'conectado' : connection === 'connecting' ? `conectando${attempt ? ` (#${attempt})` : '…'}` : 'desconectado'}
        </span>
        {connection === 'open' && latency !== null && <span className="font-mono text-[10.5px] text-ink-400">{latency} ms</span>}
      </div>

      {model && <span className="font-mono text-[11px] text-ink-300">{model}</span>}
      {sae && (
        <span className="font-mono text-[10.5px] text-ink-400" title={sae.root ? `LATENTFLOW_SAE_DIR=${sae.root}` : undefined}>
          SAE: {sae.kind === 'dev-hash' ? 'simulado (dev)' : sae.kind === 'gemma-scope' ? 'Gemma Scope local' : sae.kind}
        </span>
      )}

      <div className="ml-auto flex flex-wrap items-center gap-2 font-mono text-[10.5px]">
        {server ? (
          <>
            <Pill label="circuito" value={`v${server.version}`} />
            <Pill label="steering" value={server.steering ? 'ON' : 'OFF'} tone={server.steering ? 'text-signal' : 'text-ink-400'} />
            <Pill
              label="capas"
              value={server.steering ? (server.layer_plan ? server.layer_plan.map((l) => l.layer).join(', ') || '—' : 'todas') : '—'}
            />
            <Pill label="cabezas riesgo" value={String(server.num_risky_heads)} />
            {!server.calibrated && <Pill label="estado" value="sin calibrar" tone="text-danger" />}
          </>
        ) : (
          <span className="text-ink-500">sin estado del servidor</span>
        )}
        <Pill label="sync" value={syncMeta.label} tone={syncMeta.tone} />
      </div>
    </header>
  )
}

function Pill({ label, value, tone = 'text-ink-100' }: { label: string; value: string; tone?: string }) {
  return (
    <span className="flex items-center gap-1.5 rounded-md border border-ink-700 bg-ink-950/50 px-2 py-0.5">
      <span className="text-ink-400">{label}</span>
      <span className={cx('font-semibold', tone)}>{value}</span>
    </span>
  )
}

function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="size-8" aria-hidden>
      <defs>
        <linearGradient id="lf-g" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#38bdf8" />
          <stop offset="1" stopColor="#39f5b4" />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="30" height="30" rx="9" fill="#0e131c" stroke="#283345" />
      <path d="M6 20 C 10 20, 10 10, 14 10 S 18 22, 22 22 S 25 14, 26 14" fill="none" stroke="url(#lf-g)" strokeWidth="2.2" strokeLinecap="round" />
      <circle cx="14" cy="10" r="2" fill="#38bdf8" />
      <circle cx="22" cy="22" r="2" fill="#39f5b4" />
    </svg>
  )
}
