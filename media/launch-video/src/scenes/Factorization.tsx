import { AbsoluteFill, useCurrentFrame } from 'remotion';
import { C, MONO } from '../theme';
import { Tex } from '../ui/Tex';
import { Backdrop, Body, Card, Heading, Kicker, progress, Rise, useSceneFade } from '../ui/primitives';

export const Factorization = ({ duration }: { duration: number }) => {
  const f = useCurrentFrame();
  const fade = useSceneFade(duration, 2);
  const strike = progress(f, 96, 122);
  return (
    <AbsoluteFill style={{ opacity: fade }}>
      <Backdrop glow={C.signal} />
      <AbsoluteFill style={{ padding: '100px 150px', gap: 24 }}>
        <Rise><Kicker>Factorización exacta</Kicker></Rise>
        <Rise at={4}><Heading>Todos los pares Focus × Tail en una forma cuadrática</Heading></Rise>
        <Rise at={18} style={{ display: 'flex', justifyContent: 'center', marginTop: 34 }}>
          <Card accent={C.signal} style={{ padding: '40px 70px' }}>
            <Tex
              tex={String.raw`\sum_{t}\sum_{i\in\mathcal{F}_t,\,j\in\mathcal{T}_t}(k_i-k_j)(k_i-k_j)^{\top}=K^{\top}G\,K`}
              size={66}
              color={C.ink100}
            />
          </Card>
        </Rise>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 70, marginTop: 50 }}>
          <Rise at={60} style={{ position: 'relative' }}>
            <Tex tex={String.raw`O(T^3\,d^2)`} size={74} color={C.danger} />
            <div
              style={{
                position: 'absolute', left: -10, right: -10, top: '52%', height: 6, borderRadius: 3,
                background: C.danger, transform: `scaleX(${strike})`, transformOrigin: 'left center',
              }}
            />
          </Rise>
          <Rise at={110}><div style={{ fontFamily: MONO, fontSize: 64, color: C.ink400 }}>→</div></Rise>
          <Rise at={122}>
            <Tex tex={String.raw`O(T^3+T^2d+Td^2)`} size={74} color={C.signal} />
          </Rise>
        </div>
        <Rise at={150} style={{ display: 'flex', justifyContent: 'center', marginTop: 34 }}>
          <Body size={34} color={C.ink200}>
            ≈ 10<sup>11</sup> → ≈ 5·10<sup>7</sup> operaciones por cabeza
            <span style={{ color: C.ink400 }}> · peor caso, T = d = 256 · validada contra fuerza bruta</span>
          </Body>
        </Rise>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
