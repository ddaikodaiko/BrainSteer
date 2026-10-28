import { AbsoluteFill } from 'remotion';
import { C } from '../theme';
import { Tex } from '../ui/Tex';
import { Backdrop, Body, Card, Heading, Kicker, Rise, useSceneFade } from '../ui/primitives';

const STEPS = [
  {
    at: 24,
    tex: String.raw`\Sigma_{\Delta K}=\mathbb{E}\big[(k_i-k_j)(k_i-k_j)^{\top}\big],\quad i\in\mathcal{F}_t,\; j\in\mathcal{T}_t`,
    note: 'Geometría de keys del modelo base: tokens que atiende (Focus) frente al resto (Tail)',
    color: C.ortho,
  },
  {
    at: 84,
    tex: String.raw`P_{\Delta K}=I-U_kU_k^{\top}`,
    note: 'Se retira el subespacio que concentra el 90 % de esa energía',
    color: C.signal,
  },
  {
    at: 144,
    tex: String.raw`q'=q+\alpha\,P_{\Delta K}\,r_q`,
    note: 'Sólo en las cabezas de mayor riesgo según su cociente de Rayleigh',
    color: C.warn,
  },
];

export const Skop = ({ duration }: { duration: number }) => {
  const fade = useSceneFade(duration);
  return (
    <AbsoluteFill style={{ opacity: fade }}>
      <Backdrop glow={C.ortho} />
      <AbsoluteFill style={{ padding: '100px 150px', gap: 22 }}>
        <Rise><Kicker color={C.ortho}>La solución · SKOP</Kicker></Rise>
        <Rise at={6}><Heading>Steering que respeta el patrón de atención</Heading></Rise>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 26, marginTop: 34 }}>
          {STEPS.map((s) => (
            <Rise key={s.tex} at={s.at}>
              <Card accent={s.color} style={{ display: 'flex', flexDirection: 'column', gap: 12, padding: '24px 40px' }}>
                <Tex tex={s.tex} size={50} color={C.ink100} />
                <Body size={28}>{s.note}</Body>
              </Card>
            </Rise>
          ))}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
