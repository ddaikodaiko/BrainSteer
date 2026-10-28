import type { ReactNode } from 'react';
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from 'remotion';
import { moves, useCamera } from '../kit/camera';
import { C, MONO, SANS } from '../theme';

const DESIGN = { width: 1920, height: 1080 };

// Centros (px CSS del viewport 1920×1080) medidos con Playwright sobre la captura: ui-boxes.json.
export const FRAMINGS = [
  { at: 0, fx: 632, fy: 350, s: 2.0 },     // feature #12082 → Capa 21
  { at: 100, fx: 632, fy: 700, s: 2.0 },   // feature #42 → Capa 6 en Orthogonal Projection
  { at: 190, fx: 1586, fy: 868, s: 2.0 },  // osciloscopio
  { at: 280, fx: 960, fy: 540, s: 1.0 },   // plano general
];

const CAPTIONS: { from: number; to: number; text: ReactNode }[] = [
  { from: 18, to: 96, text: <>Una feature de Gemma Scope → el inyector de la capa 21</> },
  { from: 104, to: 186, text: <><span style={{ color: C.ortho }}>Orthogonal Projection</span>: P<sub>ΔK</sub> sólo en las cabezas de riesgo</> },
  { from: 194, to: 276, text: <>Rayleigh por cabeza, token a token, sin re-renders de React</> },
  { from: 284, to: 362, text: <>Mueve el slider a mitad del stream: hot-swap en la frontera de token</> },
];

/** La cabina real (captura a 2×) bajo la cámara virtual de launch-video-kit. */
export const Cockpit = ({ duration }: { duration: number }) => {
  const f = useCurrentFrame();
  const cam = useCamera(moves(FRAMINGS), f, DESIGN);
  const opacity = interpolate(f, [0, 16, duration - 8, duration], [0, 1, 1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  const badge = interpolate(f, [FRAMINGS[3]!.at - 4, FRAMINGS[3]!.at + 6], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  return (
    <AbsoluteFill style={{ opacity, background: C.ink950 }}>
      <div style={{ ...DESIGN, transform: cam.transform, transformOrigin: '0 0' }}>
        <Img src={staticFile('ui-full.png')} style={{ width: DESIGN.width, height: DESIGN.height }} />
      </div>
      {/* Sólo en el plano general, donde se lee el texto del chat (generado con pesos aleatorios). */}
      <div
        style={{
          position: 'absolute', top: 78, right: 40, fontFamily: MONO, fontSize: 22, color: C.warn,
          padding: '10px 18px', borderRadius: 12, background: `${C.ink950}e6`, border: `1px solid ${C.warn}55`,
          opacity: badge,
        }}
      >
        backend de desarrollo · modelo tiny con pesos aleatorios
      </div>
      {CAPTIONS.map((c) => {
        const o = interpolate(f, [c.from, c.from + 8, c.to - 8, c.to], [0, 1, 1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
        if (o <= 0) return null;
        return (
          <div key={c.from} style={{ position: 'absolute', left: 0, right: 0, bottom: 0, height: 300, opacity: o }}>
            <div
              style={{
                position: 'absolute', inset: 0,
                background: 'linear-gradient(to top, rgba(5,7,12,0.94) 0%, rgba(5,7,12,0.8) 45%, rgba(5,7,12,0) 100%)',
              }}
            />
            <div
              style={{
                position: 'absolute', left: '8%', right: '8%', bottom: 66, textAlign: 'center',
                fontFamily: SANS, fontSize: 50, fontWeight: 600, color: C.ink100, letterSpacing: '-0.015em',
                transform: `translateY(${(1 - o) * 14}px)`,
              }}
            >
              {c.text}
            </div>
          </div>
        );
      })}
    </AbsoluteFill>
  );
};
