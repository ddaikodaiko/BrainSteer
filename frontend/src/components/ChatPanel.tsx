import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { cx } from '../lib/cx'
import { useSessionStore, type ChatMessage } from '../store/useSessionStore'

/** Chat en tiempo real: cada `STREAM_CHUNK` se añade al mensaje del asistente activo. */
export function ChatPanel({ className }: { className?: string }) {
  const messages = useSessionStore((s) => s.messages)
  const connection = useSessionStore((s) => s.connection)
  const activeRequestId = useSessionStore((s) => s.activeRequestId)
  const params = useSessionStore((s) => s.params)
  const setParams = useSessionStore((s) => s.setParams)
  const generate = useSessionStore((s) => s.generate)
  const cancel = useSessionStore((s) => s.cancel)
  const clearChat = useSessionStore((s) => s.clearChat)
  const [draft, setDraft] = useState('')
  const listRef = useRef<HTMLDivElement>(null)
  const stickToBottom = useRef(true)

  const online = connection === 'open'
  const busy = activeRequestId !== null

  // Auto-scroll mientras el usuario esté abajo del todo.
  useEffect(() => {
    const el = listRef.current
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight
  }, [messages])

  const submit = (e?: FormEvent) => {
    e?.preventDefault()
    if (busy) return cancel()
    if (generate(draft)) {
      setDraft('')
      stickToBottom.current = true
    }
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      if (!busy) submit()
    }
  }

  return (
    <section className={cx('flex min-h-0 flex-col rounded-2xl border border-ink-700 bg-ink-900/80', className)}>
      <header className="flex items-center gap-2 border-b border-ink-700/80 px-4 py-2.5">
        <h2 className="text-[13px] font-semibold text-ink-100">Chat</h2>
        <span className="font-mono text-[10.5px] text-ink-400">streaming token a token</span>
        <button
          type="button"
          onClick={clearChat}
          className="ml-auto rounded-md px-2 py-0.5 text-[11px] text-ink-400 hover:bg-ink-800 hover:text-ink-200"
        >
          Limpiar
        </button>
      </header>

      <div
        ref={listRef}
        onScroll={(e) => {
          const el = e.currentTarget
          stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
        }}
        className="scrollbar-thin min-h-0 flex-1 space-y-3 overflow-y-auto px-4 py-3"
        aria-live="polite"
      >
        {messages.length === 0 && (
          <p className="pt-8 text-center text-[12.5px] text-ink-400">
            Diseña el circuito a la izquierda y lanza un prompt.
            <br />
            Los cambios del grafo se aplican en caliente, incluso a mitad de la generación.
          </p>
        )}
        {messages.map((m) => (
          <Message key={m.id} message={m} />
        ))}
      </div>

      <form onSubmit={submit} className="space-y-2 border-t border-ink-700/80 p-3">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          rows={2}
          disabled={!online}
          placeholder={online ? 'Escribe un prompt… (Enter para enviar, Shift+Enter salto de línea)' : 'Sin conexión con el servidor'}
          className="scrollbar-thin w-full resize-none rounded-xl border border-ink-700 bg-ink-950/70 px-3 py-2 text-[13px] text-ink-100 placeholder:text-ink-500 focus:border-ortho/60 focus:outline-none disabled:opacity-50"
        />
        <div className="flex flex-wrap items-center gap-3 text-[11px] text-ink-300">
          <label className="flex items-center gap-1.5">
            <span className="font-mono text-ink-400">max</span>
            <input
              type="number"
              min={1}
              max={4096}
              value={params.maxNewTokens}
              onChange={(e) => setParams({ maxNewTokens: Math.max(1, Math.min(4096, Number(e.target.value) || 1)) })}
              className="no-spin w-14 rounded-md border border-ink-700 bg-ink-950/70 px-1.5 py-0.5 text-center font-mono tabular-nums focus:outline-none"
            />
          </label>
          <label className="flex items-center gap-1.5">
            <span className="font-mono text-ink-400">T</span>
            <input
              type="range"
              min={0}
              max={1.5}
              step={0.05}
              value={params.temperature}
              onChange={(e) => setParams({ temperature: Number(e.target.value) })}
              className="w-20 accent-[var(--color-ortho)]"
            />
            <span className="w-8 font-mono tabular-nums">{params.temperature === 0 ? 'greedy' : params.temperature.toFixed(2)}</span>
          </label>
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={params.chat}
              onChange={(e) => setParams({ chat: e.target.checked })}
              className="accent-[var(--color-ortho)]"
            />
            plantilla chat
          </label>
          <button
            type="submit"
            disabled={!online || (!busy && !draft.trim())}
            className={cx(
              'ml-auto rounded-lg px-4 py-1.5 text-[12px] font-semibold transition-colors disabled:opacity-40',
              busy ? 'bg-danger/15 text-danger ring-1 ring-danger/40 hover:bg-danger/25' : 'bg-signal text-ink-950 hover:bg-signal/85',
            )}
          >
            {busy ? 'Detener' : 'Generar'}
          </button>
        </div>
      </form>
    </section>
  )
}

const STATUS_LABEL: Record<ChatMessage['status'], string> = {
  pending: 'enviado',
  queued: 'en cola (GPU ocupada)',
  streaming: 'generando',
  done: '',
  cancelled: 'cancelado',
  error: 'error',
}

function Message({ message: m }: { message: ChatMessage }) {
  if (m.role === 'system') {
    return <div className="rounded-lg border border-warn/25 bg-warn/5 px-3 py-1.5 font-mono text-[11px] text-warn/90">{m.text}</div>
  }
  const user = m.role === 'user'
  const live = m.status === 'pending' || m.status === 'queued' || m.status === 'streaming'
  return (
    <div className={cx('flex flex-col gap-1', user ? 'items-end' : 'items-start')}>
      <div
        className={cx(
          'max-w-[92%] rounded-2xl px-3.5 py-2 text-[13px] leading-relaxed whitespace-pre-wrap [overflow-wrap:anywhere]',
          user ? 'rounded-br-md bg-ortho/15 text-ink-100 ring-1 ring-ortho/25' : 'rounded-bl-md bg-ink-800 text-ink-100 ring-1 ring-ink-700',
        )}
      >
        <span className={cx(live && 'caret')}>{m.text || (live ? '' : '∅')}</span>
      </div>
      {!user && (
        <div className="flex gap-2 px-1 font-mono text-[10px] text-ink-400">
          {STATUS_LABEL[m.status] && <span className={cx(m.status === 'error' && 'text-danger')}>{STATUS_LABEL[m.status]}</span>}
          {m.stats && m.status !== 'pending' && (
            <>
              <span>{m.stats.tokens} tok</span>
              {m.stats.tokensPerS !== null && <span>{m.stats.tokensPerS.toFixed(1)} tok/s</span>}
              {!live && <span>fin: {m.stats.reason}</span>}
            </>
          )}
        </div>
      )}
    </div>
  )
}
