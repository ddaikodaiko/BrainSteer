import type { NodeTypes } from 'reactflow'
import { GemmaScopeNode } from './GemmaScopeNode'
import { SKOPLayerNode } from './SKOPLayerNode'

/** Definido a nivel de módulo: React Flow exige una referencia estable. */
export const nodeTypes: NodeTypes = {
  gemmaScope: GemmaScopeNode,
  skopLayer: SKOPLayerNode,
}

export { GemmaScopeNode, SKOPLayerNode }
