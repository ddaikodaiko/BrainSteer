import { AbsoluteFill } from 'remotion';
import { C, MONO, SANS } from '../theme';
import { Backdrop, Body, Card, Heading, Kicker, Rise, useSceneFade } from '../ui/primitives';

const METRICS = [
  { at: 30, title: 'Steering', what: 'Preguntas A/B de las Model-Written Evaluations de Anthropic', color: C.additive },
  { at: 54, title: 'Utilidad', what: 'Acierto en preguntas de ciencia (ARC-Easy)', color: C.ortho },
  { at: 78, title: 'Mecanismo', what: 'Cuánta atención pierde el Focus Set', color: C.signal },
];

/** Lo que falta: medir. Sin resultados todavía. */
export const Measure = ({ duration }: { duration: number }) => {
  const fade = useSceneFade(duration);
  return (
    <AbsoluteFill style={{ opacity: fade }}>
      <Backdrop glow={C.warn} />
      <AbsoluteFill style={{ padding: '100px 150px', gap: 22 }}>
        <Rise><Kicker color={C.warn}>Lo siguiente</Kicker></Rise>
        <Rise at={6}><Heading size={72}>Ahora toca medir: vanilla frente a SKOP</Heading></Rise>
        <div style={{ display: 'flex', gap: 28, marginTop: 34 }}>
          {METRICS.map((m) => (
            <Rise key={m.title} at={m.at} style={{ flex: 1, display: 'flex' }}>
              <Card accent={m.color} style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 16, minHeight: 250 }}>
                <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 58, color: m.color, letterSpacing: '-0.02em' }}>{m.title}</div>
                <Body size={30} color={C.ink200}>{m.what}</Body>
              </Card>
            </Rise>
          ))}
        </div>
        <Rise at={110} style={{ marginTop: 26 }}>
          <div style={{ fontFamily: MONO, fontSize: 28, color: C.ink300 }}>barrido de α · las mismas preguntas en los dos modos</div>
        </Rise>
        <Rise at={170} style={{ marginTop: 26 }}>
          <Body size={46} color={C.ink100} style={{ fontWeight: 700 }}>
            Las primeras pruebas son prometedoras.
          </Body>
        </Rise>
        <Rise at={215}>
          <Body size={46} color={C.warn} style={{ fontWeight: 700 }}>
            Sin números no hay conclusiones: resultados en unos días.
          </Body>
        </Rise>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
