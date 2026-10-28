import type { Edge, Node } from 'reactflow'
import {
  ClientAction,
  type CircuitSnapshot,
  type LayerSpec,
  type SteeringMode,
  type UpdateCircuitMessage,
} from './protocol'

// ------------------------------------------------------------------ nodos
export const ALPHA_MIN = -5
export const ALPHA_MAX = 5
export const ALPHA_STEP = 0.05
/** Tope del selector de capa mientras no se conoce la geometría del modelo. */
export const FALLBACK_MAX_LAYER = 255

export interface GemmaScopeData {
  /** ID de Neuronpedia / Gemma Scope, p. ej. `gemma-2-2b/20-gemmascope-res-16k/12082`. */
  featureId: string
  alpha: number
  enabled: boolean
}

export interface SKOPLayerData {
  layer: number
  mode: SteeringMode
  enabled: boolean
}

export type GemmaScopeFlowNode = Node<GemmaScopeData, 'gemmaScope'>
export type SKOPLayerFlowNode = Node<SKOPLayerData, 'skopLayer'>
export type CircuitNode = GemmaScopeFlowNode | SKOPLayerFlowNode

export const isFeatureNode = (n: CircuitNode | undefined): n is GemmaScopeFlowNode => n?.type === 'gemmaScope'
export const isLayerNode = (n: CircuitNode | undefined): n is SKOPLayerFlowNode => n?.type === 'skopLayer'

export function clampAlpha(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.min(ALPHA_MAX, Math.max(ALPHA_MIN, value))
}

export function clampLayer(value: number, numLayers: number | null): number {
  const max = numLayers ? numLayers - 1 : FALLBACK_MAX_LAYER
  if (!Number.isFinite(value)) return 0
  return Math.min(max, Math.max(0, Math.round(value)))
}

const NEURONPEDIA_ID = /^[\w.-]+\/[\w.-]+\/\d+$/
const GEMMA_SCOPE_ID = /^([\w.-]+)\/(\d+)-gemmascope-(res|mlp|att)-(\d+[km])\/(\d+)$/

export interface GemmaScopeFeature {
  model: string
  /** Capa del SAE: su decoder describe el residual tras el bloque `layer`. */
  layer: number
  site: 'res' | 'mlp' | 'att'
  width: string
  index: number
}

/** `gemma-2-2b/20-gemmascope-res-16k/12082` -> partes; `null` si no es un ID de Gemma Scope. */
export function parseGemmaScopeId(featureId: string): GemmaScopeFeature | null {
  const m = GEMMA_SCOPE_ID.exec(featureId.trim())
  if (!m) return null
  return { model: m[1]!, layer: Number(m[2]), site: m[3] as GemmaScopeFeature['site'], width: m[4]!, index: Number(m[5]) }
}

/** Origen con el que el servidor resolvió una feature en el circuito aplicado, si está. */
export function featureSource(snapshot: CircuitSnapshot | null, featureId: string): string | null {
  const id = featureId.trim()
  if (!id || !snapshot?.steering) return null
  for (const entry of snapshot.layer_plan ?? []) {
    const hit = entry.features?.find((f) => f.feature_id === id)
    if (hit) return hit.source
  }
  return null
}

/** URL de Neuronpedia si el ID tiene la forma `modelo/sae/índice`. */
export function neuronpediaUrl(featureId: string): string | null {
  const id = featureId.trim()
  return NEURONPEDIA_ID.test(id) ? `https://www.neuronpedia.org/${id}` : null
}

// --------------------------------------------------------- serialización
export const DAG_SCHEMA = 'latentflow.circuit/v1'

export interface DagLayer extends LayerSpec {
  /** Nodos SKOPLayer que se fusionaron en esta capa. */
  node_ids: string[]
  sources: { node_id: string; feature_id: string | null; alpha: number }[]
}

export interface CircuitDAG {
  schema: typeof DAG_SCHEMA
  active_nodes: string[]
  nodes: { id: string; type: string; active: boolean; data: GemmaScopeData | SKOPLayerData }[]
  edges: { id: string; source: string; target: string }[]
  layers: DagLayer[]
}

export type DiagnosticLevel = 'info' | 'warning' | 'error'

export interface Diagnostic {
  level: DiagnosticLevel
  code: string
  message: string
  nodeIds?: string[]
}

export interface NodeDerived {
  active: boolean
  /** Para capas: alpha efectivo (suma de las features conectadas). */
  alpha?: number
}

export interface SerializedCircuit {
  dag: CircuitDAG
  /** Payload exacto de `UPDATE_CIRCUIT` (`{"action": "update", ...}`). */
  update: UpdateCircuitMessage
  diagnostics: Diagnostic[]
  derived: Record<string, NodeDerived>
}

export interface SerializeOptions {
  /** Nº de capas del modelo (de `ready.geometry`); `null` si aún no se conoce. */
  numLayers: number | null
}

const round3 = (x: number) => Math.round(x * 1000) / 1000

/**
 * Grafo visual -> DAG del circuito + payload para el servidor.
 *
 * Semántica:
 * - Las aristas válidas van de una feature (GemmaScope) a una capa (SKOPLayer).
 * - Una capa está activa si está habilitada y recibe al menos una feature habilitada.
 * - Su dirección es la combinación lineal de sus features, `Σ α_f · r_q(f)`: con
 *   Feature ID, el decoder del SAE llevado al espacio de queries de esa capa; sin
 *   ID, el r_q calibrado. El `alpha` mostrado en la capa es `Σ α_f`.
 * - Varias capas con el mismo índice se fusionan: alphas sumados y, si los modos
 *   difieren, gana `orthogonal` (la opción conservadora), con aviso.
 */
export function serializeCircuit(
  nodes: readonly CircuitNode[],
  edges: readonly Edge[],
  { numLayers }: SerializeOptions,
): SerializedCircuit {
  const byId = new Map(nodes.map((n) => [n.id, n]))
  const diagnostics: Diagnostic[] = []
  const derived: Record<string, NodeDerived> = {}

  const validEdges = edges.filter((e) => isFeatureNode(byId.get(e.source)) && isLayerNode(byId.get(e.target)))
  if (validEdges.length !== edges.length) {
    diagnostics.push({
      level: 'warning',
      code: 'invalid_edge',
      message: 'Se ignoran conexiones que no van de una feature a una capa.',
    })
  }

  // Los SAEs de atención codifican salidas de cabezas, no el residual: no se pueden
  // llevar al espacio de queries y se excluyen (el servidor los rechazaría).
  const unsupported = new Set<string>()
  for (const node of nodes) {
    if (isFeatureNode(node) && node.data.enabled && parseGemmaScopeId(node.data.featureId)?.site === 'att') {
      unsupported.add(node.id)
    }
  }
  if (unsupported.size) {
    diagnostics.push({
      level: 'error',
      code: 'attention_sae_unsupported',
      message: 'Las features de SAEs de atención (att) no viven en el residual stream; se excluyen. Usa res o mlp.',
      nodeIds: [...unsupported],
    })
  }

  // Features habilitadas que alimentan cada capa habilitada.
  const inputs = new Map<string, GemmaScopeFlowNode[]>()
  for (const edge of validEdges) {
    const source = byId.get(edge.source) as GemmaScopeFlowNode
    const target = byId.get(edge.target) as SKOPLayerFlowNode
    if (!source.data.enabled || !target.data.enabled || unsupported.has(source.id)) continue
    const list = inputs.get(target.id) ?? []
    if (!list.includes(source)) list.push(source)
    inputs.set(target.id, list)
  }

  // Agrupa capas activas por índice de capa.
  const groups = new Map<number, { nodes: SKOPLayerFlowNode[]; features: GemmaScopeFlowNode[] }>()
  const activeFeatures = new Set<string>()
  for (const node of nodes) {
    if (!isLayerNode(node)) continue
    const feats = inputs.get(node.id) ?? []
    const layer = node.data.layer
    const inRange = Number.isInteger(layer) && layer >= 0 && (numLayers === null || layer < numLayers)
    if (node.data.enabled && feats.length > 0 && !inRange) {
      diagnostics.push({
        level: 'error',
        code: 'layer_out_of_range',
        message: `La capa ${layer} no existe en el modelo (0–${(numLayers ?? 1) - 1}); se excluye.`,
        nodeIds: [node.id],
      })
    }
    const active = node.data.enabled && feats.length > 0 && inRange
    derived[node.id] = { active, alpha: round3(feats.reduce((acc, f) => acc + f.data.alpha, 0)) }
    if (!active) continue
    const group = groups.get(layer) ?? { nodes: [], features: [] }
    group.nodes.push(node)
    for (const f of feats) {
      group.features.push(f)
      activeFeatures.add(f.id)
    }
    groups.set(layer, group)
  }

  for (const node of nodes) {
    if (isFeatureNode(node)) derived[node.id] = { active: activeFeatures.has(node.id) }
  }

  const layers: DagLayer[] = []
  for (const [layer, group] of [...groups.entries()].sort(([a], [b]) => a - b)) {
    const modes = new Set(group.nodes.map((n) => n.data.mode))
    if (group.nodes.length > 1) {
      diagnostics.push({
        level: modes.size > 1 ? 'warning' : 'info',
        code: modes.size > 1 ? 'mixed_modes' : 'merged_layer',
        message:
          modes.size > 1
            ? `Capa ${layer}: inyectores con modos distintos; se aplica Orthogonal Projection.`
            : `Capa ${layer}: ${group.nodes.length} inyectores fusionados (alphas sumados).`,
        nodeIds: group.nodes.map((n) => n.id),
      })
    }
    layers.push({
      layer,
      alpha: round3(group.features.reduce((acc, f) => acc + f.data.alpha, 0)),
      mode: modes.has('orthogonal') ? 'orthogonal' : 'additive',
      node_ids: group.nodes.map((n) => n.id),
      sources: group.features.map((f) => ({
        node_id: f.id,
        feature_id: f.data.featureId.trim() || null,
        alpha: f.data.alpha,
      })),
    })
  }

  // Un SAE de la capa L describe el residual tras el bloque L: su entrada natural son
  // las queries de la capa L + 1. Otra capa es válida, pero conviene saberlo.
  const mismatched: string[] = []
  for (const l of layers) {
    for (const s of l.sources) {
      const parsed = s.feature_id ? parseGemmaScopeId(s.feature_id) : null
      if (parsed && parsed.layer + 1 !== l.layer) mismatched.push(`${s.feature_id} → capa ${l.layer} (natural: ${parsed.layer + 1})`)
    }
  }
  if (mismatched.length) {
    diagnostics.push({
      level: 'info',
      code: 'sae_layer_mismatch',
      message: `Features inyectadas fuera de la capa siguiente a su SAE: ${mismatched.join('; ')}.`,
    })
  }
  if (layers.length === 0) {
    diagnostics.push({
      level: 'info',
      code: 'no_active_layers',
      message: 'Ninguna capa activa: el steering queda desactivado.',
    })
  }

  // Cada capa envía sus features: el servidor compone la dirección Σ α_f · r_q(f)
  // (feature_id null = r_q calibrado) y `alpha` actúa como ganancia (1).
  const spec: LayerSpec[] = layers.map(({ layer, mode, sources }) => ({
    layer,
    alpha: 1,
    mode,
    features: sources.map((s) => ({ feature_id: s.feature_id, alpha: round3(s.alpha) })),
  }))
  const update: UpdateCircuitMessage = {
    action: ClientAction.UPDATE_CIRCUIT,
    steering: spec.length > 0,
    layers: spec,
  }

  const dag: CircuitDAG = {
    schema: DAG_SCHEMA,
    active_nodes: nodes.filter((n) => derived[n.id]?.active).map((n) => n.id),
    nodes: nodes.map((n) => ({ id: n.id, type: n.type ?? 'unknown', active: derived[n.id]?.active ?? false, data: n.data })),
    edges: validEdges.map((e) => ({ id: e.id, source: e.source, target: e.target })),
    layers,
  }

  return { dag, update, diagnostics, derived }
}
