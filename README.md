# BrainSteer · LatentFlow

[![CI](https://github.com/ddaikodaiko/BrainSteer/actions/workflows/ci.yml/badge.svg)](https://github.com/ddaikodaiko/BrainSteer/actions/workflows/ci.yml)

IDE de steering de atención para LLMs locales (Gemma-2-2B-IT, Llama-3.1-8B en 4 bit):

- **`latentflow/core/steering.py`** — motor SKOP. Σ_ΔK = E[Δk Δkᵀ] se estima en
  CPU/float64 sobre un *utility set* genérico con el modelo base, con
  `Δk = k_i − k_j`, `i` en el *Focus Set* (keys con atención > `tau_high`) y `j` en
  el *Tail Set*. Proyector ortogonal `P = I − U_k U_kᵀ` por energía espectral
  (γ = 0.90) y filtrado de cabezas de riesgo por cociente de Rayleigh normalizado
  por traza. La dirección conceptual `r_q` es externa (pares contrastivos de
  concepto o vectores propios). Inyección `q' = q + α·P·r_q` (o aditiva) con hooks
  en `q_proj`.
- **`latentflow/api/server.py`** — FastAPI + WebSocket (`/ws/stream`): streaming
  token a token con telemetría de riesgo, hot-swap del circuito en frontera de
  token y plan por capa.
- **`latentflow/core/sae.py`** — resolución de features de Gemma Scope
  (IDs de Neuronpedia) a su decoder `W_dec[f]` desde pesos locales, sin red.
- **`latentflow/tools/initializer.py`** — CLI `latentflow-calibrate`: genera el
  artefacto de calibración a partir de datasets locales.
- **`frontend/`** — cabina web en React Flow: editor de circuitos, chat en
  streaming y osciloscopio en canvas. Ver [`frontend/README.md`](frontend/README.md).

## Puesta en marcha

```bash
pip install -e ".[server,models,test]"
```

### 1. Calibración (una vez por modelo y concepto)

`latentflow-calibrate` no descarga nada: recibe un *utility set* (para Σ_ΔK) y un
*concept set* de pares `{"positive", "negative"}` (para `r_q`) como ficheros
`.jsonl`/`.json` locales. Formatos y ejemplos en
[`examples/calibration/`](examples/calibration/README.md).

```bash
latentflow-calibrate \
    --utility data/alpaca.jsonl --utility data/gsm8k.jsonl \
    --concept data/formal_vs_informal.jsonl \
    -o calib/gemma2_skop.pt              # Gemma-2-2B-IT en 4 bit; --quant none sin GPU
```

La calibración de Σ_ΔK usa las probabilidades de atención reales, así que el
modelo se carga con `attn_implementation="eager"`. Opciones: `--tau-high`,
`--gamma`, `--layers 0-25`, `--max-utility`… (`latentflow-calibrate --help`).

### 2. Servidor

Con una GPU de ~8 GB, el servidor carga el artefacto en frío:

```bash
LATENTFLOW_MODEL_ID=google/gemma-2-2b-it LATENTFLOW_CALIBRATION=calib/gemma2_skop.pt latentflow-server
```

Sin GPU, el **backend de desarrollo** ejecuta el mismo servidor sobre un Llama
tiny de 26 capas con pesos aleatorios (el texto es ruido; protocolo, streaming,
hot-swap y telemetría son los reales):

```bash
latentflow-dev                       # o: python -m latentflow.tools.dev_backend --layers 4 --delay 0
cd frontend && npm install && npm run dev
```

El ciclo completo también funciona offline con el modelo tiny:

```bash
latentflow-calibrate --dev --utility examples/calibration/utility.example.jsonl \
    --concept examples/calibration/concept.example.jsonl -o calib/dev_skop.pt
LATENTFLOW_CALIBRATION=calib/dev_skop.pt latentflow-dev
```

## Features de Gemma Scope en el canvas

Un nodo *Gemma Scope* con un Feature ID de Neuronpedia
(`gemma-2-2b/20-gemmascope-res-16k/12082`) conectado a un inyector de capa aporta
la dirección del decoder de esa feature. El decoder vive en el residual stream
(`d_model`), así que el motor lo lleva al espacio de queries de la capa con
`r_q(f) = q_proj(input_layernorm(W_dec[f]))`. La dirección de la capa es
`Σ_f α_f · r_q(f)`: un nodo sin ID aporta el `r_q` calibrado. El riesgo de
Rayleigh se recalcula con esa dirección y P_ΔK se aplica sólo a sus cabezas de
alto riesgo.

- La entrada natural de un SAE residual de la capa *L* son las queries de la capa
  *L + 1*; el canvas avisa si inyectas en otra.
- Los SAEs de atención (`att`) no se pueden mapear a queries y se excluyen.
- Los pesos se leen de `LATENTFLOW_SAE_DIR` (árbol de Gemma Scope o ficheros
  `.safetensors`/`.npz`/`.pt` con `W_dec`) o de la caché local de Hugging Face:

  ```bash
  huggingface-cli download google/gemma-scope-2b-pt-res-canonical --include "layer_20/width_16k/canonical/*"
  ```

- En `latentflow-dev` cada ID se simula con un vector determinista (sha256 del ID).
- Gemma Scope está entrenado sobre `gemma-2-2b` (base); usarlo con `-it` es una
  transferencia habitual, no una garantía.

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

## Licencia

[MIT](LICENSE) © 2026 ddaikodaiko / BrainSteer Team.

## Tests

```bash
python -m pytest -q          # motor, servidor, seguridad, dev backend y calibración
cd frontend && npm test      # serialización, debounce, telemetría, despachador
```

El CI de GitHub Actions (`.github/workflows/ci.yml`) ejecuta ambas suites y el build
del frontend en cada push a `main` y en cada pull request.
