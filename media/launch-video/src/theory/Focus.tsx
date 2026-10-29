import { AbsoluteFill, interpolate, useCurrentFrame } from 'remotion';
import { C, MONO, SANS } from '../theme';
import { Backdrop, Body, Heading, Kicker, progress, Rise, useSceneFade } from '../ui/primitives';

const TOKENS = ['¿', 'Dónde', 'está', 'la', 'torre', 'Eiffel', '?'];
const BASE = [0.03, 0.05, 0.04, 0.03, 0.36, 0.46, 0.03];   // el Focus Set se lleva el 82 %
const STEERED = [0.2, 0.07, 0.18, 0.22, 0.08, 0.09, 0.16]; // reenrutada hacia la cola
const FOCUS = new Set([4, 5]);

/** Por qué se rompe: el Focus Set pierde atención (esquema ilustrativo). */
export const Focus = ({ duration }: { duration: number }) => {
  const f = useCurrentFrame();
  const fade = useSceneFade(duration);
  const mix = progress(f, 120, 160);
  return (
    <AbsoluteFill style={{ opacity: fade }}>
      <Backdrop glow={C.warn} />
      <AbsoluteFill style={{ padding: '90px 150px', gap: 22 }}>
        <Rise><Kicker color={C.warn}>Por qué pasa</Kicker></Rise>
        <Rise at={6}><Heading size={68}>La atención se escapa de los tokens importantes</Heading></Rise>
        <Rise at={16}>
          <Body size={32}>
            En cada cabeza, unos pocos tokens concentran el 80 % de la atención: el <span style={{ color: C.signal }}>Focus Set</span>.
            El steering desvía esa masa hacia el resto, el <span style={{ color: C.additive }}>Tail Set</span>.
          </Body>
        </Rise>
        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'center', gap: 34, height: 360, marginTop: 24 }}>
          {TOKENS.map((tok, i) => {
            const w = interpolate(mix, [0, 1], [BASE[i]!, STEERED[i]!]);
            const color = FOCUS.has(i) ? C.signal : mix > 0.5 ? C.additive : C.ink400;
            const grow = progress(f, 30 + i * 3, 56 + i * 3);
            return (
              <div key={tok} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14, width: 150 }}>
                <div style={{ fontFamily: MONO, fontSize: 26, color: C.ink300 }}>{w.toFixed(2)}</div>
                <div style={{ width: 92, height: 600 * w * grow, borderRadius: 14, background: color, boxShadow: `0 0 28px ${color}55` }} />
                <div
                  style={{
                    fontFamily: SANS, fontSize: 34, fontWeight: 600, padding: '10px 18px', borderRadius: 12,
                    color: FOCUS.has(i) ? C.signal : C.ink200, border: `1px solid ${FOCUS.has(i) ? `${C.signal}88` : C.ink700}`,
                    background: C.ink900,
                  }}
                >
                  {tok}
                </div>
              </div>
            );
          })}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 10 }}>
          <Body size={32} color={mix < 0.5 ? C.signal : C.additive}>
            {mix < 0.5 ? 'Modelo base: el 82 % de la atención, en el Focus Set' : 'Con steering: la masa se va a la cola'}
          </Body>
          <div style={{ fontFamily: MONO, fontSize: 22, color: C.ink400 }}>esquema ilustrativo</div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
