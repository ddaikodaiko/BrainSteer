import type { Edge } from 'reactflow'
import { describe, expect, it } from 'vitest'
import { clampLayer, featureSource, neuronpediaUrl, parseGemmaScopeId, serializeCircuit, type CircuitNode } from './circuit'
import type { CircuitSnapshot } from './protocol'

const feat = (id: string, alpha: number, featureId = '', enabled = true): CircuitNode => ({
  id,
  type: 'gemmaScope',
  position: { x: 0, y: 0 },
  data: { featureId, alpha, enabled },
})
const layer = (id: string, n: number, mode: 'additive' | 'orthogonal', enabled = true): CircuitNode => ({
  id,
  type: 'skopLayer',
  position: { x: 0, y: 0 },
  data: { layer: n, mode, enabled },
})
const edge = (source: string, target: string): Edge => ({ id: `${source}->${target}`, source, target })

describe('serializeCircuit', () => {
  it('produce el payload exacto de UPDATE_CIRCUIT con capas ordenadas', () => {
    const nodes = [feat('f1', 2.5, 'gemma-2-2b/20-gemmascope-res-16k/12082'), feat('f2', -1), layer('l12', 12, 'orthogonal'), layer('l6', 6, 'additive')]
    const edges = [edge('f1', 'l12'), edge('f2', 'l6')]
    const { update, dag, derived } = serializeCircuit(nodes, edges, { numLayers: 26 })

    expect(update).toEqual({
      action: 'update',
      steering: true,
      layers: [
        { layer: 6, alpha: 1, mode: 'additive', features: [{ feature_id: null, alpha: -1 }] },
        { layer: 12, alpha: 1, mode: 'orthogonal', features: [{ feature_id: 'gemma-2-2b/20-gemmascope-res-16k/12082', alpha: 2.5 }] },
      ],
    })
    expect(dag.schema).toBe('latentflow.circuit/v1')
    expect(dag.active_nodes.sort()).toEqual(['f1', 'f2', 'l12', 'l6'])
    expect(dag.layers[1]!.sources).toEqual([{ node_id: 'f1', feature_id: 'gemma-2-2b/20-gemmascope-res-16k/12082', alpha: 2.5 }])
    expect(derived.l12).toEqual({ active: true, alpha: 2.5 })
  })

  it('suma los alphas de varias features en una capa (superposición) y redondea', () => {
    const nodes = [feat('a', 0.1), feat('b', 0.2), layer('l', 3, 'additive')]
    const { update, dag } = serializeCircuit(nodes, [edge('a', 'l'), edge('b', 'l')], { numLayers: 26 })
    expect(update.layers).toEqual([
      { layer: 3, alpha: 1, mode: 'additive', features: [{ feature_id: null, alpha: 0.1 }, { feature_id: null, alpha: 0.2 }] },
    ])
    expect(dag.layers[0]!.alpha).toBe(0.3)
  })

  it('excluye nodos deshabilitados, capas sin entrada y features sueltas', () => {
    const nodes = [feat('on', 1), feat('off', 4, '', false), feat('loose', 2), layer('l1', 1, 'additive'), layer('l2', 2, 'additive'), layer('l3', 3, 'orthogonal', false)]
    const edges = [edge('on', 'l1'), edge('off', 'l2'), edge('on', 'l3')]
    const { update, derived, diagnostics } = serializeCircuit(nodes, edges, { numLayers: 26 })
    expect(update.layers).toEqual([{ layer: 1, alpha: 1, mode: 'additive', features: [{ feature_id: null, alpha: 1 }] }])
    expect(derived.loose!.active).toBe(false)
    expect(derived.off!.active).toBe(false)
    expect(derived.l2!.active).toBe(false)
    expect(derived.l3!.active).toBe(false)
    expect(diagnostics.some((d) => d.level === 'error')).toBe(false)
  })

  it('fusiona capas repetidas; con modos mixtos gana orthogonal y avisa', () => {
    const nodes = [feat('a', 1), feat('b', 2), layer('x', 5, 'additive'), layer('y', 5, 'orthogonal')]
    const { update, diagnostics } = serializeCircuit(nodes, [edge('a', 'x'), edge('b', 'y')], { numLayers: 26 })
    expect(update.layers).toEqual([
      { layer: 5, alpha: 1, mode: 'orthogonal', features: [{ feature_id: null, alpha: 1 }, { feature_id: null, alpha: 2 }] },
    ])
    expect(diagnostics.find((d) => d.code === 'mixed_modes')?.nodeIds).toEqual(['x', 'y'])
  })

  it('marca como error las capas fuera de la geometría del modelo', () => {
    const nodes = [feat('a', 1), layer('l', 30, 'additive')]
    const { update, diagnostics, derived } = serializeCircuit(nodes, [edge('a', 'l')], { numLayers: 26 })
    expect(update).toEqual({ action: 'update', steering: false, layers: [] })
    expect(diagnostics.find((d) => d.code === 'layer_out_of_range')?.level).toBe('error')
    expect(derived.l!.active).toBe(false)
  })

  it('sin capas activas desactiva el steering y avisa', () => {
    const { update, diagnostics } = serializeCircuit([feat('a', 1)], [], { numLayers: null })
    expect(update).toEqual({ action: 'update', steering: false, layers: [] })
    expect(diagnostics.map((d) => d.code)).toContain('no_active_layers')
  })

  it('ignora aristas inválidas (capa -> feature, a nodos inexistentes)', () => {
    const nodes = [feat('a', 1), layer('l', 1, 'additive')]
    const { update, diagnostics, dag } = serializeCircuit(nodes, [edge('l', 'a'), edge('a', 'ghost')], { numLayers: 26 })
    expect(update.layers).toEqual([])
    expect(dag.edges).toEqual([])
    expect(diagnostics.map((d) => d.code)).toContain('invalid_edge')
  })

  it('compone varias features (SAE y r_q calibrado) en la dirección de una capa', () => {
    const nodes = [feat('sae', 2, ' gemma-2-2b/20-gemmascope-res-16k/7 '), feat('cal', -0.5), layer('l', 21, 'orthogonal')]
    const { update, diagnostics } = serializeCircuit(nodes, [edge('sae', 'l'), edge('cal', 'l')], { numLayers: 26 })
    expect(update.layers).toEqual([
      {
        layer: 21,
        alpha: 1,
        mode: 'orthogonal',
        features: [
          { feature_id: 'gemma-2-2b/20-gemmascope-res-16k/7', alpha: 2 },
          { feature_id: null, alpha: -0.5 },
        ],
      },
    ])
    expect(diagnostics.map((d) => d.code)).not.toContain('sae_layer_mismatch') // 20 -> 21: capa natural
  })

  it('avisa si una feature se inyecta fuera de la capa siguiente a su SAE', () => {
    const nodes = [feat('a', 1, 'gemma-2-2b/20-gemmascope-res-16k/1'), layer('l', 6, 'additive')]
    const d = serializeCircuit(nodes, [edge('a', 'l')], { numLayers: 26 }).diagnostics.find((x) => x.code === 'sae_layer_mismatch')
    expect(d?.level).toBe('info')
    expect(d?.message).toContain('natural: 21')
  })

  it('excluye features de SAEs de atención (no viven en el residual stream)', () => {
    const nodes = [feat('att', 3, 'gemma-2-2b/20-gemmascope-att-16k/5'), feat('res', 1, 'gemma-2-2b/5-gemmascope-res-16k/9'),
      layer('l1', 21, 'additive'), layer('l2', 6, 'additive')]
    const { update, derived, diagnostics } = serializeCircuit(nodes, [edge('att', 'l1'), edge('res', 'l2')], { numLayers: 26 })
    expect(update.layers!.map((l) => l.layer)).toEqual([6])
    expect(derived.att!.active).toBe(false)
    expect(derived.l1!.active).toBe(false)
    expect(diagnostics.find((x) => x.code === 'attention_sae_unsupported')).toMatchObject({ level: 'error', nodeIds: ['att'] })
  })
})

describe('utilidades', () => {
  it('parseGemmaScopeId reconoce IDs de Neuronpedia de Gemma Scope', () => {
    expect(parseGemmaScopeId('gemma-2-2b/20-gemmascope-res-16k/12082')).toEqual({
      model: 'gemma-2-2b', layer: 20, site: 'res', width: '16k', index: 12082,
    })
    expect(parseGemmaScopeId('gemma-2-2b/custom/1')).toBeNull()
  })

  it('featureSource lee cómo resolvió el servidor cada feature', () => {
    const fid = 'gemma-2-2b/20-gemmascope-res-16k/1'
    const snapshot = {
      steering: true,
      layer_plan: [{ layer: 21, alpha: 1, mode: 'orthogonal', features: [{ feature_id: fid, alpha: 2, source: 'dev-hash' }] }],
    } as unknown as CircuitSnapshot
    expect(featureSource(snapshot, ` ${fid} `)).toBe('dev-hash')
    expect(featureSource(snapshot, 'otra/sae/1')).toBeNull()
    expect(featureSource({ ...snapshot, steering: false }, fid)).toBeNull()
    expect(featureSource(null, fid)).toBeNull()
  })

  it('clampLayer respeta la geometría o el tope de respaldo', () => {
    expect(clampLayer(40, 26)).toBe(25)
    expect(clampLayer(-3, 26)).toBe(0)
    expect(clampLayer(2.6, 26)).toBe(3)
    expect(clampLayer(999, null)).toBe(255)
  })

  it('neuronpediaUrl sólo enlaza IDs bien formados', () => {
    expect(neuronpediaUrl('gemma-2-2b/20-gemmascope-res-16k/12082')).toBe(
      'https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/12082',
    )
    expect(neuronpediaUrl('12082')).toBeNull()
    expect(neuronpediaUrl('a/b/c')).toBeNull()
  })
})
