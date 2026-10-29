# Vídeo de lanzamiento (Remotion)

Proyecto del primer vídeo de BrainSteer (62 s, 1080p, 30 fps), hecho con
[Remotion](https://www.remotion.dev/) y los helpers de cámara y audio de
[launch-video-kit](https://github.com/borjaperfra/launch-video-kit) (MIT, ver
`src/kit/LICENSE`).

> **Borrador, no publicado.** Se montó antes de probar el proyecto con pesos reales.
> Sirve de base para el vídeo con resultados medidos; no lo publiques tal cual.

```bash
cd media/launch-video
npm ci
python scripts/compose_music.py public/music.wav   # pista original, 120 BPM (necesita numpy)
npm run studio                                    # previsualizar
npm run render                                    # out/brainsteer-launch.mp4
```

- `src/Launch.tsx` encadena las escenas de `src/scenes/` sobre una rejilla de 60
  frames (un compás de la música).

## Vídeo de teoría (`BrainSteerTheory`)

Vídeo de 62 s para LinkedIn que explica el steering, por qué rompe el modelo y el
método SKOP, y enseña la cabina funcionando con pesos reales. **No afirma
resultados**: termina con lo que falta medir. Usa la misma pista y la misma rejilla.

```bash
npm run render:theory                            # out/brainsteer-teoria.mp4
```

- `src/Theory.tsx` y `src/theory/`: las escenas.
- Las salidas del modelo de la escena del steering son reales (`latentflow-probe`,
  Gemma-2-2B-IT en 4 bit, greedy).
- `public/ui-real.png`: captura real de la cabina con Gemma-2-2B-IT en una RTX
  3060 Ti, recortada a la app y centrada en 1920×1080. Los encuadres de la cámara
  evitan la esquina inferior derecha.
- `public/ui-full.png` y `public/ui-boxes.json` son la captura de la cabina y las
  cajas de sus zonas, para los primeros planos de la cámara virtual.
