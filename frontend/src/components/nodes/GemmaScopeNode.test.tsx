// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { ReactFlowProvider, type NodeProps } from 'reactflow'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { GemmaScopeData } from '../../lib/circuit'
import type { ClientMessage } from '../../lib/protocol'
import { setSender } from '../../lib/transport'
import { initialEdges, initialNodes, resetSyncState, SYNC_DEBOUNCE_MS, useCircuitStore } from '../../store/useCircuitStore'
import { GemmaScopeNode } from './GemmaScopeNode'

const NODE = 'feat-honesty'
const ORIGINAL = 'gemma-2-2b/20-gemmascope-res-16k/12082'

/** El nodo con sus `data` leídos del store, como en el canvas. */
function Harness() {
  const data = useCircuitStore((s) => s.nodes.find((n) => n.id === NODE)!.data) as GemmaScopeData
  const props = { id: NODE, data, selected: false } as NodeProps<GemmaScopeData>
  return <GemmaScopeNode {...props} />
}

let sent: ClientMessage[]
const updates = () => sent.filter((m) => m.action === 'update')
const featureId = () => (useCircuitStore.getState().nodes.find((n) => n.id === NODE)!.data as GemmaScopeData).featureId
const input = () => screen.getByLabelText(/Feature ID/) as HTMLInputElement
const settle = () => act(() => vi.advanceTimersByTime(SYNC_DEBOUNCE_MS * 3))

beforeEach(() => {
  vi.useFakeTimers()
  sent = []
  setSender((m) => {
    sent.push(m)
    return true
  })
  useCircuitStore.getState().setGraph(initialNodes, initialEdges)
  useCircuitStore.setState({ numLayers: 26 })
  resetSyncState()
  render(
    <ReactFlowProvider>
      <Harness />
    </ReactFlowProvider>,
  )
})

afterEach(() => {
  cleanup()
  setSender(null)
  vi.useRealTimers()
})

describe('GemmaScopeNode · Feature ID', () => {
  it('teclear no toca el circuito ni envía UPDATE_CIRCUIT', () => {
    input().focus()
    for (const partial of ['gemma-2-2b/2', 'gemma-2-2b/20-gemmascope-res-16k/', 'gemma-2-2b/20-gemmascope-res-16k/77']) {
      fireEvent.change(input(), { target: { value: partial } })
      settle()
    }
    expect(featureId()).toBe(ORIGINAL)
    expect(updates()).toHaveLength(0)
    expect(screen.getByText('↵ para aplicar')).toBeTruthy()
  })

  it('Enter aplica el ID (recortado) y envía un único update', () => {
    input().focus()
    fireEvent.change(input(), { target: { value: '  gemma-2-2b/20-gemmascope-res-16k/77 ' } })
    fireEvent.keyDown(input(), { key: 'Enter' })
    expect(featureId()).toBe('gemma-2-2b/20-gemmascope-res-16k/77')
    expect(document.activeElement).not.toBe(input())
    settle()
    expect(updates()).toHaveLength(1)
    expect(JSON.stringify(updates()[0])).toContain('gemma-2-2b/20-gemmascope-res-16k/77')
    expect(input().value).toBe('gemma-2-2b/20-gemmascope-res-16k/77')
  })

  it('perder el foco también aplica el ID', () => {
    input().focus()
    fireEvent.change(input(), { target: { value: 'gemma-2-2b/20-gemmascope-res-16k/5' } })
    fireEvent.blur(input())
    expect(featureId()).toBe('gemma-2-2b/20-gemmascope-res-16k/5')
    settle()
    expect(updates()).toHaveLength(1)
  })

  it('Escape descarta el borrador sin enviar nada', () => {
    input().focus()
    fireEvent.change(input(), { target: { value: 'basura a medio escribir' } })
    fireEvent.keyDown(input(), { key: 'Escape' })
    settle()
    expect(featureId()).toBe(ORIGINAL)
    expect(input().value).toBe(ORIGINAL)
    expect(updates()).toHaveLength(0)
  })

  it('salir sin cambios no envía nada', () => {
    input().focus()
    fireEvent.blur(input())
    settle()
    expect(updates()).toHaveLength(0)
  })

  it('un cambio externo del ID (p. ej. cargar un grafo) actualiza el campo', () => {
    act(() => useCircuitStore.getState().updateNodeData(NODE, { featureId: 'gemma-2-2b/6-gemmascope-res-16k/1' }))
    expect(input().value).toBe('gemma-2-2b/6-gemmascope-res-16k/1')
  })
})
