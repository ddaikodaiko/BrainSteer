import { AbsoluteFill, useCurrentFrame } from 'remotion';
import { C, MONO, SANS } from '../theme';
import { Backdrop, Kicker, Logo, progress, Rise, useSceneFade } from '../ui/primitives';

/** Gancho: la pregunta que responde el vídeo. */
export const TheoryTitle = ({ duration }: { duration: number }) => {
  const f = useCurrentFrame();
  const fade = useSceneFade(duration, 1);
  return (
    <AbsoluteFill style={{ opacity: fade }}>
      <Backdrop glow={C.signal} />
      <AbsoluteFill style={{ alignItems: 'center', justifyContent: 'center', gap: 40, padding: '0 160px' }}>
        <Rise at={2}><Kicker>Interpretabilidad · steering de LLMs</Kicker></Rise>
        <Rise at={10}>
          <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 108, lineHeight: 1.05, letterSpacing: '-0.035em', color: C.ink100, textAlign: 'center' }}>
            ¿Se puede dirigir un LLM <span style={{ color: C.signal }}>sin romperlo</span>?
          </div>
        </Rise>
        <Rise at={46} style={{ display: 'flex', alignItems: 'center', gap: 22 }}>
          <Logo size={64} draw={progress(f, 46, 80)} />
          <div style={{ fontFamily: MONO, fontSize: 30, color: C.ink300, letterSpacing: '0.04em' }}>
            BrainSteer · implementación open source de SKOP
          </div>
        </Rise>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
