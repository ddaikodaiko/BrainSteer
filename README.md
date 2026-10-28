# BrainSteer · LatentFlow

IDE de steering de atención para LLMs locales (Gemma-2-2B-IT, Llama-3.1-8B en 4 bit):

- **`latentflow/core/steering.py`** — motor SKOP. Calibración offline de
  `Σ_ΔK = E[Δk Δkᵀ]` en CPU, proyector ortogonal `P = I − U_k U_kᵀ` por energía
  espectral y filtrado de cabezas de riesgo por cociente de Rayleigh normalizado
  por traza. Inyección `q' = q + α·P·r_q` (o aditiva) mediante hooks en `q_proj`.
- **`latentflow/api/server.py`** — FastAPI + WebSocket (`/ws/stream`): streaming
  token a token con telemetría de riesgo, hot-swap del circuito en frontera de
  token y plan por capa.
- **`frontend/`** — cabina web en React Flow: editor de circuitos, chat en
  streaming y osciloscopio en canvas. Ver [`frontend/README.md`](frontend/README.md).

## Puesta en marcha

```bash
pip install -e ".[server,models,test]"
```

Servidor con un modelo real (GPU con ~8 GB; necesita una calibración SKOP guardada
con `SKOPEngine.save()`, o la acción `calibrate` desde el cliente):

```bash
LATENTFLOW_MODEL_ID=google/gemma-2-2b-it LATENTFLOW_CALIBRATION=calib/gemma2.pt latentflow-server
```

Sin GPU, el **backend de desarrollo** ejecuta el mismo servidor sobre un Llama
tiny de 26 capas con pesos aleatorios (el texto es ruido; protocolo, streaming,
hot-swap y telemetría son los reales):

```bash
latentflow-dev                       # o: python -m latentflow.tools.dev_backend --layers 4 --delay 0
cd frontend && npm install && npm run dev
```

## Seguridad

El WebSocket valida el header `Origin` en el handshake. Por defecto sólo acepta el
frontend local (Vite `:5173` / `:4173` en `localhost` y `127.0.0.1`, y Tauri), y
responde HTTP 403 a cualquier otro origen, de modo que una web abierta en el
navegador no puede tomar el control del modelo (Cross-Site WebSocket Hijacking).
El `CORSMiddleware` de FastAPI no sirve para esto, porque no se aplica a WebSockets.

| Variable | Por defecto | Efecto |
| --- | --- | --- |
| `LATENTFLOW_ALLOWED_ORIGINS` | frontend local | Orígenes permitidos, separados por comas (`*` desactiva la protección) |
| `LATENTFLOW_ALLOW_MISSING_ORIGIN` | `1` | Acepta clientes sin `Origin` (scripts/CLI; los navegadores siempre lo envían) |

El servidor escucha en `127.0.0.1`. No lo expongas en la red sin autenticación.

## Tests

```bash
python -m pytest -q          # motor, servidor, seguridad y dev backend
cd frontend && npm test      # serialización, debounce, telemetría, despachador
```
