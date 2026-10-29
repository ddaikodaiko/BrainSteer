import { AbsoluteFill } from 'remotion';
import { C } from '../theme';
import { Tex } from '../ui/Tex';
import { Backdrop, Body, Card, Heading, Kicker, Rise, useSceneFade } from '../ui/primitives';

const STEPS = [
  {
    at: 30,
    tex: String.raw`\Sigma_{\Delta K}=\mathbb{E}_t\,\mathbb{E}_{(i,j)\sim\mathcal{U}(\mathcal{F}_t\times\mathcal{T}_t)}\big[\Delta k\,\Delta k^{\top}\big]`,
    note: '1 · En texto genérico, mide en qué direcciones se diferencian las keys del Focus Set de las del resto',
    color: C.ortho,
  },
  {
    at: 140,
    tex: String.raw`P_{\Delta K}=I-U_kU_k^{\top}`,
    note: '2 · Quita del vector de steering el subespacio que concentra el 90 % de esa energía',
    color: C.signal,
  },
  {
    at: 250,
    tex: String.raw`q'=q+\alpha\,P_{\Delta K}\,r_q`,
    note: '3 · Y solo en el 20 % de cabezas con más riesgo, según su cociente de Rayleigh',
    color: C.warn,
  },
];

/** El método SKOP en tres pasos (la notación del README). */
export const SkopTheory = ({ duration }: { duration: number }) => {
  const fade = useSceneFade(duration);
  return (
    <AbsoluteFill style={{ opacity: fade }}>
      <Backdrop glow={C.ortho} />
      <AbsoluteFill style={{ padding: '90px 150px', gap: 20 }}>
        <Rise><Kicker color={C.ortho}>La solución · SKOP · arXiv:2605.06342</Kicker></Rise>
        <Rise at={6}><Heading size={70}>Steering que respeta el Focus Set</Heading></Rise>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24, marginTop: 30 }}>
          {STEPS.map((s) => (
            <Rise key={s.tex} at={s.at}>
              <Card accent={s.color} style={{ display: 'flex', flexDirection: 'column', gap: 12, padding: '24px 40px' }}>
                <Tex tex={s.tex} size={46} color={C.ink100} />
                <Body size={29}>{s.note}</Body>
              </Card>
            </Rise>
          ))}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
