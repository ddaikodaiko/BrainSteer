import type { ReactNode } from 'react'
import { cx } from '../../lib/cx'

interface NodeShellProps {
  kicker: string
  title: string
  accent: string
  active: boolean
  enabled: boolean
  selected: boolean
  onToggleEnabled: () => void
  onRemove: () => void
  children: ReactNode
}

/** Tarjeta común de los nodos del circuito: cabecera, LED de actividad y acciones. */
export function NodeShell({
  kicker,
  title,
  accent,
  active,
  enabled,
  selected,
  onToggleEnabled,
  onRemove,
  children,
}: NodeShellProps) {
  return (
    <div
      className={cx(
        'w-[300px] rounded-2xl border bg-ink-850/95 shadow-[0_12px_40px_-12px_rgb(0_0_0/0.8)] backdrop-blur',
        'transition-[border-color,opacity] duration-150',
        selected ? 'border-ink-300/60' : 'border-ink-700',
        !enabled && 'opacity-55',
      )}
      style={{ boxShadow: active ? `0 0 0 1px ${accent}33, 0 12px 40px -12px ${accent}55` : undefined }}
    >
      <div className="flex items-center gap-2 border-b border-ink-700/80 px-3.5 py-2.5">
        <span
          className="size-2 shrink-0 rounded-full"
          style={{ background: active ? accent : 'var(--color-ink-600)', boxShadow: active ? `0 0 8px ${accent}` : 'none' }}
          title={active ? 'Activo en el circuito' : 'Inactivo (deshabilitado o sin conectar)'}
        />
        <div className="min-w-0 flex-1 leading-tight">
          <div className="truncate font-mono text-[9.5px] tracking-[0.16em] text-ink-400 uppercase">{kicker}</div>
          <div className="truncate text-[13px] font-semibold text-ink-100">{title}</div>
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={enabled}
          aria-label={enabled ? 'Deshabilitar nodo' : 'Habilitar nodo'}
          onClick={onToggleEnabled}
          className={cx(
            'nodrag relative h-[18px] w-8 rounded-full border transition-colors',
            enabled ? 'border-transparent' : 'border-ink-600 bg-ink-800',
          )}
          style={{ background: enabled ? accent : undefined }}
        >
          <span
            className={cx(
              'absolute top-[2px] size-3 rounded-full bg-ink-950 transition-[left] duration-150',
              enabled ? 'left-[16px]' : 'left-[2px] bg-ink-400',
            )}
          />
        </button>
        <button
          type="button"
          onClick={onRemove}
          aria-label="Eliminar nodo"
          className="nodrag grid size-6 place-items-center rounded-md text-ink-400 hover:bg-ink-700 hover:text-danger"
        >
          ×
        </button>
      </div>
      <div className="space-y-3 px-3.5 py-3">{children}</div>
    </div>
  )
}

export function FieldLabel({ children, htmlFor }: { children: ReactNode; htmlFor?: string }) {
  return (
    <label htmlFor={htmlFor} className="block font-mono text-[10px] tracking-[0.14em] text-ink-400 uppercase">
      {children}
    </label>
  )
}
