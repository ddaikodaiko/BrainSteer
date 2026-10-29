import { AbsoluteFill, useCurrentFrame } from 'remotion';
import { C, MONO, SANS } from '../theme';
import { Backdrop, Body, Chip, Logo, progress, Rise } from '../ui/primitives';

export const TheoryOutro = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill>
      <Backdrop glow={C.signal} />
      <AbsoluteFill style={{ alignItems: 'center', justifyContent: 'center', gap: 30 }}>
        <Logo size={140} draw={progress(f, 0, 30)} />
        <Rise at={8}>
          <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 116, letterSpacing: '-0.04em', color: C.ink100, lineHeight: 1 }}>BrainSteer</div>
        </Rise>
        <Rise at={18}>
          <div
            style={{
              fontFamily: MONO, fontSize: 44, color: C.signal, padding: '18px 34px', borderRadius: 18,
              border: `1px solid ${C.signal}66`, background: `${C.ink900}cc`, boxShadow: `0 0 40px ${C.signal}33`,
            }}
          >
            github.com/ddaikodaiko/BrainSteer
          </div>
        </Rise>
        <Rise at={28} style={{ display: 'flex', gap: 18 }}>
          <Chip color={C.warn}>MIT · open source</Chip>
          <Chip color={C.ortho}>Gemma-2-2B · GPU de 8 GB</Chip>
        </Rise>
        <Rise at={40}>
          <Body size={30} color={C.ink300} style={{ textAlign: 'center' }}>
            Método SKOP: Luo, Espinosa Zarlenga y Jamnik · arXiv:2605.06342
          </Body>
        </Rise>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
