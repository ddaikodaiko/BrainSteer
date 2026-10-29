import type { ReactNode } from 'react';
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from 'remotion';
import { moves, useCamera } from '../kit/camera';
import { C, MONO, SANS } from '../theme';

const DESIGN = { width: 1920, height: 1080 };

// ui-real.png: la captura real de la cabina (Gemma-2-2B-IT, RTX 3060 Ti) recortada a la app y
// centrada en 1920×1080 (y de la captura + 43). Los encuadres evitan la esquina inferior derecha,
// donde está la marca de agua de Windows.
const FRAMINGS = [
  { at: 0, fx: 400, fy: 300, s: 2.4 },     // cabecera: logo, modelo y "SAE: Gemma Scope local"
  { at: 100, fx: 440, fy: 361, s: 2.3 },   // feature #12082 con W_dec ✓
  { at: 190, fx: 631, fy: 355, s: 1.55 },  // feature → inyector de la capa 21
  { at: 280, fx: 662, fy: 560, s: 1.45 },  // el canvas completo
];

const CAPTIONS: { from: number; to: number; text: ReactNode }[] = [
  { from: 18, to: 96, text: <>Gemma-2-2B-IT en 4 bit, en una <span style={{ color: C.signal }}>RTX 3060 Ti de 8 GB</span></> },
  { from: 104, to: 186, text: <>Una feature real de Gemma Scope: el decoder <span style={{ color: C.signal }}>W_dec</span> se lee del SAE descargado</> },
  { from: 194, to: 276, text: <>Cada nodo inyecta su dirección en una capa: aditiva o con la <span style={{ color: C.ortho }}>proyección SKOP</span></> },
  { from: 284, to: 362, text: <>Se ajusta en caliente mientras el modelo genera</> },
];

/** BrainSteer funcionando con pesos reales, bajo la cámara virtual de launch-video-kit. */
export const RealCockpit = ({ duration }: { duration: number }) => {
  const f = useCurrentFrame();
  const cam = useCamera(moves(FRAMINGS), f, DESIGN);
  const opacity = interpolate(f, [0, 16, duration - 8, duration], [0, 1, 1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  // Desde el segundo encuadre: en el primero taparía "SAE: Gemma Scope local" en la cabecera.
  const badge = interpolate(f, [FRAMINGS[1]!.at - 4, FRAMINGS[1]!.at + 10], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  return (
    <AbsoluteFill style={{ opacity, background: C.ink950 }}>
      <div style={{ ...DESIGN, transform: cam.transform, transformOrigin: '0 0' }}>
        <Img src={staticFile('ui-real.png')} style={{ width: DESIGN.width, height: DESIGN.height }} />
      </div>
      <div
        style={{
          position: 'absolute', top: 40, right: 40, fontFamily: MONO, fontSize: 22, color: C.signal,
          padding: '10px 18px', borderRadius: 12, background: `${C.ink950}e6`, border: `1px solid ${C.signal}55`,
          opacity: badge,
        }}
      >
        captura real · Gemma-2-2B-IT · RTX 3060 Ti
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
                position: 'absolute', left: '7%', right: '7%', bottom: 66, textAlign: 'center',
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
