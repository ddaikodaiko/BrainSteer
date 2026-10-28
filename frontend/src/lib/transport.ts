import type { ClientMessage } from './protocol'

/**
 * Canal de salida único hacia el servidor. `useWebSocket` lo registra al abrir
 * la conexión y lo retira al cerrarla; los stores sólo llaman a `send()`, lo que
 * los mantiene desacoplados del socket (y testeables con un sender falso).
 */
type Sender = (message: ClientMessage) => boolean

let sender: Sender | null = null

export function setSender(next: Sender | null): void {
  sender = next
}

export function isConnected(): boolean {
  return sender !== null
}

/** Devuelve `false` si no hay conexión abierta. */
export function send(message: ClientMessage): boolean {
  return sender !== null && sender(message)
}
