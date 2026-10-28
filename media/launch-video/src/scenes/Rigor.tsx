import { AbsoluteFill, interpolate, useCurrentFrame } from 'remotion';
import { C, MONO, SANS } from '../theme';
import { Backdrop, Card, Heading, Kicker, Rise, useSceneFade } from '../ui/primitives';

const Stat = ({ value, label, color, at }: { value: string; label: string; color: string; at: number }) => (
  <Rise at={at} style={{ flex: 1 }}>
    <Card accent={color} style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'flex-start', minHeight: 230 }}>
      <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 110, lineHeight: 1, color, letterSpacing: '-0.03em' }}>{value}</div>
      <div style={{ fontFamily: MONO, fontSize: 26, color: C.ink300 }}>{label}</div>
    </Card>
  </Rise>
);

export const Rigor = ({ duration }: { duration: number }) => {
  const f = useCurrentFrame();
  const fade = useSceneFade(duration);
  const count = (to: number, a: number) =>
    String(Math.round(interpolate(f, [a, a + 30], [0, to], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' })));
  return (
    <AbsoluteFill style={{ opacity: fade }}>
      <Backdrop glow={C.signal} />
      <AbsoluteFill style={{ padding: '110px 150px', gap: 24, justifyContent: 'center' }}>
        <Rise><Kicker>Rigor</Kicker></Rise>
        <Rise at={4}><Heading>Probado en cada push</Heading></Rise>
        <div style={{ display: 'flex', gap: 28, marginTop: 40 }}>
          <Stat at={10} value={count(144, 12)} label="tests de Python" color={C.signal} />
          <Stat at={16} value={count(34, 18)} label="tests de Vitest" color={C.ortho} />
          <Stat at={22} value="CI ✓" label="GitHub Actions" color={C.ink100} />
          <Stat at={28} value="MIT" label="código abierto" color={C.warn} />
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
