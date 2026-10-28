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
- `public/ui-full.png` y `public/ui-boxes.json` son la captura de la cabina y las
  cajas de sus zonas, para los primeros planos de la cámara virtual.
