export interface Debounced<A extends unknown[]> {
  (...args: A): void
  /** Ejecuta ya la llamada pendiente, si la hay. */
  flush(): void
  cancel(): void
  pending(): boolean
}

/** Debounce trailing: sólo se ejecuta la última llamada tras `wait` ms de calma. */
export function debounce<A extends unknown[]>(fn: (...args: A) => void, wait: number): Debounced<A> {
  let timer: ReturnType<typeof setTimeout> | null = null
  let lastArgs: A | null = null

  const invoke = () => {
    timer = null
    const args = lastArgs
    lastArgs = null
    if (args) fn(...args)
  }

  const debounced = ((...args: A) => {
    lastArgs = args
    if (timer !== null) clearTimeout(timer)
    timer = setTimeout(invoke, wait)
  }) as Debounced<A>

  debounced.flush = () => {
    if (timer === null) return
    clearTimeout(timer)
    invoke()
  }
  debounced.cancel = () => {
    if (timer !== null) clearTimeout(timer)
    timer = null
    lastArgs = null
  }
  debounced.pending = () => timer !== null
  return debounced
}
