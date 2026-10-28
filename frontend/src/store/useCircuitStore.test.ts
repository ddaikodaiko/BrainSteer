import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ClientMessage } from '../lib/protocol'
import { setSender } from '../lib/transport'
import { initialEdges, initialNodes, resetSyncState, SYNC_DEBOUNCE_MS, useCircuitStore } from './useCircuitStore'

let sent: ClientMessage[]

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
})

afterEach(() => {
  setSender(null)
  vi.useRealTimers()
})

const store = () => useCircuitStore.getState()
const updates = () => sent.filter((m) => m.action === 'update')

describe('debounce de UPDATE_CIRCUIT', () => {
  it('un barrido rápido del slider produce un único envío con el valor final', () => {
    for (let i = 0; i <= 40; i++) {
      store().updateNodeData('feat-honesty', { alpha: -5 + i * 0.25 })
      vi.advanceTimersByTime(10) // 10 ms entre eventos: nunca 100 ms de calma
    }
    expect(updates()).toHaveLength(0)
    expect(store().sync.state).toBe('pending')

    vi.advanceTimersByTime(SYNC_DEBOUNCE_MS - 11)
    expect(updates()).toHaveLength(0)
    vi.advanceTimersByTime(1)
    expect(updates()).toHaveLength(1)
    expect(updates()[0]).toMatchObject({
      action: 'update',
      layers: expect.arrayContaining([
        { layer: 21, alpha: 1, mode: 'orthogonal', features: [{ feature_id: 'gemma-2-2b/20-gemmascope-res-16k/12082', alpha: 5 }] },
      ]),
    })
    expect(store().sync.state).toBe('sent')
  })

  it('no reenvía un payload idéntico al último enviado', () => {
    store().updateNodeData('feat-honesty', { alpha: 1 })
    vi.advanceTimersByTime(SYNC_DEBOUNCE_MS)
    store().updateNodeData('feat-honesty', { alpha: 1.5 })
    store().updateNodeData('feat-honesty', { alpha: 1 }) // vuelve al valor enviado
    vi.advanceTimersByTime(SYNC_DEBOUNCE_MS)
    expect(updates()).toHaveLength(1)
  })

  it('mover nodos no envía nada; borrar un nodo sí', () => {
    store().onNodesChange([{ type: 'position', id: 'layer-a', position: { x: 999, y: 1 }, dragging: true }])
    vi.advanceTimersByTime(SYNC_DEBOUNCE_MS * 3)
    expect(updates()).toHaveLength(0)

    store().onNodesChange([{ type: 'remove', id: 'layer-a' }])
    vi.advanceTimersByTime(SYNC_DEBOUNCE_MS)
    expect(updates()).toEqual([
      { action: 'update', steering: true, layers: [{ layer: 6, alpha: 1, mode: 'additive', features: [{ feature_id: null, alpha: -1 }] }] },
    ])
  })

  it('el toggle de modo y el cambio de capa pasan por el mismo debounce', () => {
    store().updateNodeData('layer-a', { mode: 'additive' })
    store().updateNodeData('layer-a', { layer: 22 })
    vi.advanceTimersByTime(SYNC_DEBOUNCE_MS)
    expect(updates()).toHaveLength(1)
    expect((updates()[0] as { layers: unknown[] }).layers).toContainEqual({
      layer: 22,
      alpha: 1,
      mode: 'additive',
      features: [{ feature_id: 'gemma-2-2b/20-gemmascope-res-16k/12082', alpha: 2.5 }],
    })
  })

  it('flushSync envía de inmediato y cancela el debounce pendiente', () => {
    store().updateNodeData('feat-honesty', { alpha: 3 })
    expect(store().flushSync()).toBe(true)
    vi.advanceTimersByTime(SYNC_DEBOUNCE_MS * 2)
    expect(updates()).toHaveLength(1)
  })

  it('sin conexión queda offline y se reenvía con force al reconectar', () => {
    setSender(null)
    store().updateNodeData('feat-honesty', { alpha: 4 })
    vi.advanceTimersByTime(SYNC_DEBOUNCE_MS)
    expect(store().sync.state).toBe('offline')

    setSender((m) => {
      sent.push(m)
      return true
    })
    expect(store().flushSync(true)).toBe(true)
    expect(updates()).toHaveLength(1)
  })
})

describe('edición del grafo', () => {
  it('clampa alpha a [-5, 5] y la capa a la geometría del modelo', () => {
    store().updateNodeData('feat-honesty', { alpha: 42 })
    store().updateNodeData('layer-a', { layer: 99 })
    const nodes = store().nodes
    expect(nodes.find((n) => n.id === 'feat-honesty')!.data).toMatchObject({ alpha: 5 })
    expect(nodes.find((n) => n.id === 'layer-a')!.data).toMatchObject({ layer: 25 })
  })

  it('sólo acepta conexiones feature -> capa y sin duplicados', () => {
    const { isValidConnection } = store()
    expect(isValidConnection({ source: 'feat-formal', target: 'layer-a', sourceHandle: null, targetHandle: null })).toBe(true)
    expect(isValidConnection({ source: 'layer-a', target: 'feat-formal', sourceHandle: null, targetHandle: null })).toBe(false)
    expect(isValidConnection({ source: 'feat-honesty', target: 'layer-a', sourceHandle: null, targetHandle: null })).toBe(false)
  })

  it('setModelInfo re-ajusta capas fuera de rango', () => {
    store().setModelInfo(4, [0, 1, 2, 3])
    const layers = store().nodes.filter((n) => n.type === 'skopLayer').map((n) => (n.data as { layer: number }).layer)
    expect(layers.every((l) => l <= 3)).toBe(true)
    expect(store().serialized.diagnostics.some((d) => d.code === 'layer_out_of_range')).toBe(false)
  })
})
