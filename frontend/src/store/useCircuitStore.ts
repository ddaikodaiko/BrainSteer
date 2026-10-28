import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type NodeChange,
  type XYPosition,
} from 'reactflow'
import { create } from 'zustand'
import {
  clampAlpha,
  clampLayer,
  isFeatureNode,
  isLayerNode,
  serializeCircuit as serializeGraph,
  type CircuitNode,
  type GemmaScopeData,
  type SerializedCircuit,
  type SKOPLayerData,
} from '../lib/circuit'
import { debounce } from '../lib/debounce'
import type { CircuitSnapshot, UpdateCircuitMessage } from '../lib/protocol'
import { send } from '../lib/transport'

/** Espera tras el último cambio antes de enviar `UPDATE_CIRCUIT`. */
export const SYNC_DEBOUNCE_MS = 100

export type SyncState =
  | 'idle' // nada enviado todavía
  | 'pending' // cambios locales esperando al debounce
  | 'sent' // enviado, sin respuesta
  | 'acked' // `update_accepted` recibido
  | 'applied' // `circuit` recibido: el servidor ya lo aplica
  | 'offline' // no se pudo enviar (sin conexión); se reenvía al reconectar
  | 'error'

export interface SyncInfo {
  state: SyncState
  lastPayload: UpdateCircuitMessage | null
  lastSentAt: number | null
  sentCount: number
  error: string | null
}

interface CircuitState {
  nodes: CircuitNode[]
  edges: Edge[]
  numLayers: number | null
  calibratedLayers: number[] | null
  serverCircuit: CircuitSnapshot | null
  /** Cache de `serializeCircuit()` (selector estable para los componentes). */
  serialized: SerializedCircuit
  sync: SyncInfo
  lastSentKey: string | null

  onNodesChange: (changes: NodeChange[]) => void
  onEdgesChange: (changes: EdgeChange[]) => void
  onConnect: (connection: Connection) => void
  isValidConnection: (connection: Connection) => boolean
  addGemmaScopeNode: (position?: XYPosition, data?: Partial<GemmaScopeData>) => string
  addLayerNode: (position?: XYPosition, data?: Partial<SKOPLayerData>) => string
  updateNodeData: <T extends GemmaScopeData | SKOPLayerData>(id: string, patch: Partial<T>) => void
  removeNode: (id: string) => void
  setGraph: (nodes: CircuitNode[], edges: Edge[]) => void

  /** Grafo visual -> DAG + payload `UPDATE_CIRCUIT` + diagnósticos. */
  serializeCircuit: () => SerializedCircuit
  /** Programa un envío con debounce de `SYNC_DEBOUNCE_MS`. */
  scheduleSync: () => void
  /** Envía ya (cancela el debounce). Sin `force`, no reenvía un payload idéntico. */
  flushSync: (force?: boolean) => boolean

  setModelInfo: (numLayers: number, calibratedLayers: number[]) => void
  setServerCircuit: (snapshot: CircuitSnapshot, applied: boolean) => void
  markSync: (state: SyncState, error?: string | null) => void
}

let nodeCounter = 0
const nextId = (prefix: string) => `${prefix}-${Date.now().toString(36)}-${(nodeCounter++).toString(36)}`

const EDGE_STYLE = { animated: true, style: { strokeWidth: 2 } }

export const initialNodes: CircuitNode[] = [
  {
    id: 'feat-honesty',
    type: 'gemmaScope',
    position: { x: 40, y: 40 },
    data: { featureId: 'gemma-2-2b/20-gemmascope-res-16k/12082', alpha: 2.5, enabled: true },
  },
  {
    id: 'feat-formal',
    type: 'gemmaScope',
    position: { x: 40, y: 380 },
    data: { featureId: '', alpha: -1, enabled: true },
  },
  {
    id: 'layer-a',
    type: 'skopLayer',
    position: { x: 440, y: 20 },
    // SAE residual de la capa 20 -> su entrada natural son las queries de la capa 21.
    data: { layer: 21, mode: 'orthogonal', enabled: true },
  },
  {
    id: 'layer-b',
    type: 'skopLayer',
    position: { x: 440, y: 380 },
    data: { layer: 6, mode: 'additive', enabled: true },
  },
]

export const initialEdges: Edge[] = [
  { id: 'e-honesty-a', source: 'feat-honesty', target: 'layer-a', ...EDGE_STYLE },
  { id: 'e-formal-b', source: 'feat-formal', target: 'layer-b', ...EDGE_STYLE },
]

/** Cambios de React Flow que no alteran el circuito lógico (arrastrar, seleccionar...). */
const isCosmetic = (change: NodeChange | EdgeChange) =>
  change.type === 'position' || change.type === 'dimensions' || change.type === 'select'

const debouncedFlush = debounce(() => {
  useCircuitStore.getState().flushSync()
}, SYNC_DEBOUNCE_MS)

const emptySync: SyncInfo = { state: 'idle', lastPayload: null, lastSentAt: null, sentCount: 0, error: null }

export const useCircuitStore = create<CircuitState>()((set, get) => {
  /** Aplica un cambio de grafo, recalcula la serialización y, si procede, programa el envío. */
  const commit = (patch: Partial<Pick<CircuitState, 'nodes' | 'edges' | 'numLayers'>>, sync: boolean) => {
    const next = { ...get(), ...patch }
    set({ ...patch, serialized: serializeGraph(next.nodes, next.edges, { numLayers: next.numLayers }) })
    if (sync) get().scheduleSync()
  }

  return {
    nodes: initialNodes,
    edges: initialEdges,
    numLayers: null,
    calibratedLayers: null,
    serverCircuit: null,
    serialized: serializeGraph(initialNodes, initialEdges, { numLayers: null }),
    sync: emptySync,
    lastSentKey: null,

    onNodesChange: (changes) => {
      const nodes = applyNodeChanges<GemmaScopeData | SKOPLayerData>(changes, get().nodes) as CircuitNode[]
      if (changes.every(isCosmetic)) set({ nodes })
      else commit({ nodes }, true)
    },

    onEdgesChange: (changes) => {
      const edges = applyEdgeChanges(changes, get().edges)
      if (changes.every(isCosmetic)) set({ edges })
      else commit({ edges }, true)
    },

    isValidConnection: (connection) => {
      const { nodes, edges } = get()
      const source = nodes.find((n) => n.id === connection.source)
      const target = nodes.find((n) => n.id === connection.target)
      if (!isFeatureNode(source) || !isLayerNode(target)) return false
      return !edges.some((e) => e.source === source.id && e.target === target.id)
    },

    onConnect: (connection) => {
      if (!get().isValidConnection(connection)) return
      commit({ edges: addEdge({ ...connection, ...EDGE_STYLE }, get().edges) }, true)
    },

    addGemmaScopeNode: (position = { x: 60, y: 60 }, data = {}) => {
      const id = nextId('feat')
      const node: CircuitNode = {
        id,
        type: 'gemmaScope',
        position,
        data: { featureId: '', alpha: 1, enabled: true, ...data },
      }
      commit({ nodes: [...get().nodes, node] }, true)
      return id
    },

    addLayerNode: (position = { x: 440, y: 60 }, data = {}) => {
      const id = nextId('layer')
      const node: CircuitNode = {
        id,
        type: 'skopLayer',
        position,
        data: { layer: 0, mode: 'orthogonal', enabled: true, ...data },
      }
      commit({ nodes: [...get().nodes, node] }, true)
      return id
    },

    updateNodeData: (id, patch) => {
      const { numLayers } = get()
      const nodes = get().nodes.map((n) => {
        if (n.id !== id) return n
        if (isFeatureNode(n)) {
          const data = { ...n.data, ...(patch as Partial<GemmaScopeData>) }
          return { ...n, data: { ...data, alpha: clampAlpha(data.alpha) } }
        }
        const data = { ...n.data, ...(patch as Partial<SKOPLayerData>) }
        return { ...n, data: { ...data, layer: clampLayer(data.layer, numLayers) } }
      }) as CircuitNode[]
      commit({ nodes }, true)
    },

    removeNode: (id) => {
      commit(
        {
          nodes: get().nodes.filter((n) => n.id !== id),
          edges: get().edges.filter((e) => e.source !== id && e.target !== id),
        },
        true,
      )
    },

    setGraph: (nodes, edges) => commit({ nodes, edges }, true),

    serializeCircuit: () => get().serialized,

    scheduleSync: () => {
      set((s) => ({ sync: { ...s.sync, state: 'pending', error: null } }))
      debouncedFlush()
    },

    flushSync: (force = false) => {
      debouncedFlush.cancel()
      const { update } = get().serialized
      const key = JSON.stringify(update)
      if (!force && key === get().lastSentKey) {
        // Nada nuevo que enviar (p. ej. el slider volvió al valor ya enviado).
        set((s) => ({ sync: { ...s.sync, state: s.sync.sentCount ? 'applied' : 'idle' } }))
        return false
      }
      if (!send(update)) {
        set((s) => ({ sync: { ...s.sync, state: 'offline' } }))
        return false
      }
      set((s) => ({
        lastSentKey: key,
        sync: { state: 'sent', lastPayload: update, lastSentAt: Date.now(), sentCount: s.sync.sentCount + 1, error: null },
      }))
      return true
    },

    setModelInfo: (numLayers, calibratedLayers) => {
      // Re-ajusta capas fuera de rango a la geometría real del modelo.
      const nodes = get().nodes.map((n) =>
        isLayerNode(n) && n.data.layer >= numLayers ? { ...n, data: { ...n.data, layer: numLayers - 1 } } : n,
      ) as CircuitNode[]
      set({ calibratedLayers })
      commit({ numLayers, nodes }, false)
    },

    setServerCircuit: (snapshot, applied) => {
      set((s) => ({
        serverCircuit: snapshot,
        sync: applied && (s.sync.state === 'sent' || s.sync.state === 'acked') ? { ...s.sync, state: 'applied' } : s.sync,
      }))
    },

    markSync: (state, error = null) => set((s) => ({ sync: { ...s.sync, state, error } })),
  }
})

/** Para tests y para forzar el reenvío completo tras una reconexión. */
export function resetSyncState(): void {
  debouncedFlush.cancel()
  useCircuitStore.setState({ lastSentKey: null, sync: emptySync })
}
