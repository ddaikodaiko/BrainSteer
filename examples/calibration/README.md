# Datos de ejemplo para `latentflow-calibrate`

Ficheros mínimos que **documentan el formato**; no son datasets de calibración reales.

- `utility.example.jsonl` — utility set para Σ_ΔK (Focus/Tail). Admite registros
  `{"text"}`, estilo Alpaca `{"instruction", "input", "output"}`, estilo GSM8K
  `{"question", "answer"}` o cadenas sueltas.
- `concept.example.jsonl` — pares `{"positive", "negative"}` para la dirección
  `r_q` (aquí, registro formal frente a informal).
- `formal_informal.jsonl` — 60 pares formal ↔ informal en inglés con el mismo
  contenido: un concept set listo para una primera calibración real.

Para una calibración real, usa unos cientos de muestras de utilidad (por ejemplo
Alpaca + GSM8K exportados con `datasets`, ver `latentflow/tools/initializer.py`) y
pares de concepto que aíslen el comportamiento que quieres dirigir.

```bash
latentflow-calibrate --utility examples/calibration/utility.example.jsonl \
    --concept examples/calibration/concept.example.jsonl --dev -o /tmp/skop-dev.pt
```
