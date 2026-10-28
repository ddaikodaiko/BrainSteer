# BrainSteer · LatentFlow — cliente web

Cabina de control de `latentflow/api/server.py`: editor de circuitos (React Flow),
chat en streaming y osciloscopio de riesgo SKOP (canvas 2D).

```bash
npm install
npm run dev        # http://127.0.0.1:5173 (servidor en ws://127.0.0.1:8000/ws/stream)
npm test           # vitest
npm run build      # tsc estricto + vite build
```

La URL del WebSocket se puede cambiar con `VITE_WS_URL` (ver `.env.example`) o con
`?ws=ws://host:puerto/ws/stream` en la URL.

## Estructura

| Ruta | Rol |
| --- | --- |
| `src/lib/protocol.ts` | Tipos del protocolo; `UPDATE_CIRCUIT` = `"update"`, `STREAM_CHUNK` = `"token"` |
| `src/lib/circuit.ts` | `serializeCircuit()`: grafo → DAG + payload `update` + diagnósticos |
| `src/store/useCircuitStore.ts` | Nodos/aristas (Zustand) y sincronización con debounce de 100 ms |
| `src/store/useSessionStore.ts` | Conexión, chat y generación |
| `src/hooks/useWebSocket.ts` | Conexión persistente, reconexión con backoff, heartbeat |
| `src/lib/dispatch.ts` | Enruta eventos del servidor a stores y telemetría |
| `src/lib/telemetryBuffer.ts` | Ring buffer de telemetría, fuera de React |
| `src/lib/scopeRenderer.ts` · `src/components/TelemetryMonitor.tsx` | Osciloscopio en canvas |
| `src/components/nodes/` | `GemmaScopeNode`, `SKOPLayerNode` |

## Semántica del circuito

Feature (Gemma Scope) → Capa (SKOP). Una capa está activa si está habilitada y
recibe al menos una feature habilitada; su α es la suma de las features conectadas.
El payload enviado es `{"action": "update", "steering": bool, "layers": [{layer, alpha, mode}]}`.
Los Feature IDs viajan en el DAG exportado como metadatos: el servidor aplica el
vector `r_q` calibrado por SKOP, no el decoder del SAE.
