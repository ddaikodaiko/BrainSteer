import { AbsoluteFill, interpolate, useCurrentFrame } from 'remotion';
import { C, MONO, SANS } from '../theme';
import { Backdrop, Body, Heading, Kicker, progress, Rise, useSceneFade } from '../ui/primitives';

const TOKENS = ['¿', 'Dónde', 'está', 'la', 'torre', 'Eiffel', '?'];
const BASE = [0.04, 0.08, 0.05, 0.05, 0.3, 0.42, 0.06];      // atención en el Focus Set
const NAIVE = [0.2, 0.07, 0.18, 0.22, 0.08, 0.09, 0.16];     // reenrutada hacia la cola
const FOCUS = new Set([4, 5]);

export const Problem = ({ duration }: { duration: number }) => {
  const f = useCurrentFrame();
  const fade = useSceneFade(duration);
  const mix = progress(f, 70, 110);
  return (
    <AbsoluteFill style={{ opacity: fade }}>
      <Backdrop glow={C.additive} />
      <AbsoluteFill style={{ padding: '110px 150px', gap: 26 }}>
        <Rise><Kicker color={C.additive}>El problema</Kicker></Rise>
        <Rise at={6}><Heading>El steering en queries desvía la atención</Heading></Rise>
        <Rise at={14}>
          <Body>Sumar α·r a la query cambia el emparejamiento Q·K: la atención abandona los tokens que importan.</Body>
        </Rise>
        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'center', gap: 34, height: 380, marginTop: 30 }}>
          {TOKENS.map((tok, i) => {
            const w = interpolate(mix, [0, 1], [BASE[i]!, NAIVE[i]!]);
            const color = FOCUS.has(i) ? C.signal : mix > 0.5 ? C.additive : C.ink400;
            const grow = progress(f, 20 + i * 3, 44 + i * 3);
            return (
              <div key={tok} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16, width: 150 }}>
                <div style={{ fontFamily: MONO, fontSize: 26, color: C.ink300 }}>{w.toFixed(2)}</div>
                <div style={{ width: 92, height: 640 * w * grow, borderRadius: 14, background: color, boxShadow: `0 0 28px ${color}55` }} />
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
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 8 }}>
          <Body size={32} color={mix < 0.5 ? C.signal : C.additive}>
            {mix < 0.5 ? 'Modelo base: atención en el Focus Set' : 'Steering ingenuo: la atención se va a la cola (Tail Set)'}
          </Body>
          <div style={{ fontFamily: MONO, fontSize: 22, color: C.ink400 }}>esquema ilustrativo</div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
