import { AbsoluteFill, useCurrentFrame } from 'remotion';
import { C, MONO, SANS } from '../theme';
import { Backdrop, Body, Chip, Logo, progress, Rise, useSceneFade } from '../ui/primitives';

export const Title = ({ duration }: { duration: number }) => {
  const f = useCurrentFrame();
  const fade = useSceneFade(duration, 1);
  return (
    <AbsoluteFill style={{ opacity: fade }}>
      <Backdrop glow={C.signal} />
      <AbsoluteFill style={{ alignItems: 'center', justifyContent: 'center', gap: 34 }}>
        <Logo size={190} draw={progress(f, 4, 46)} />
        <Rise at={18}>
          <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 150, letterSpacing: '-0.04em', color: C.ink100, lineHeight: 1 }}>
            BrainSteer
          </div>
        </Rise>
        <Rise at={28}>
          <div style={{ fontFamily: MONO, fontSize: 28, letterSpacing: '0.3em', color: C.ink400, textTransform: 'uppercase' }}>
            powered by LatentFlow
          </div>
        </Rise>
        <Rise at={44}>
          <Body size={44} color={C.ink200} style={{ textAlign: 'center', marginTop: 16 }}>
            Steering de atención en local, con features de Gemma Scope
          </Body>
        </Rise>
        <Rise at={64} style={{ display: 'flex', gap: 18, marginTop: 10 }}>
          <Chip color={C.signal}>SKOP</Chip>
          <Chip color={C.ortho}>Gemma Scope</Chip>
          <Chip>FastAPI · WebSocket</Chip>
          <Chip color={C.additive}>React Flow</Chip>
        </Rise>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
