import { AbsoluteFill, useCurrentFrame } from 'remotion';
import { C, MONO, SANS } from '../theme';
import { Backdrop, Body, Chip, Logo, progress, Rise } from '../ui/primitives';

export const Outro = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill>
      <Backdrop glow={C.signal} />
      <AbsoluteFill style={{ alignItems: 'center', justifyContent: 'center', gap: 30 }}>
        <Logo size={150} draw={progress(f, 0, 30)} />
        <Rise at={8}>
          <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 120, letterSpacing: '-0.04em', color: C.ink100, lineHeight: 1 }}>BrainSteer</div>
        </Rise>
        <Rise at={18}>
          <div
            style={{
              fontFamily: MONO, fontSize: 46, color: C.signal, padding: '18px 34px', borderRadius: 18,
              border: `1px solid ${C.signal}66`, background: `${C.ink900}cc`, boxShadow: `0 0 40px ${C.signal}33`,
            }}
          >
            github.com/ddaikodaiko/BrainSteer
          </div>
        </Rise>
        <Rise at={30} style={{ display: 'flex', gap: 18 }}>
          <Chip color={C.warn}>MIT · open source</Chip>
          <Chip>Guía del piloto local en el README</Chip>
        </Rise>
        <Rise at={40}>
          <Body size={32} color={C.ink300}>Clónalo, conéctalo a tu GPU y audita la atención de tus modelos.</Body>
        </Rise>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
