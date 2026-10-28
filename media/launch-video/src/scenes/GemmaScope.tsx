import { AbsoluteFill, useCurrentFrame } from 'remotion';
import { C, MONO, SANS } from '../theme';
import { Tex } from '../ui/Tex';
import { Backdrop, Body, Heading, Kicker, progress, Rise, useSceneFade } from '../ui/primitives';

/** Barra que representa un vector: `cells` celdas con un patrón determinista. */
const VectorBar = ({ cells, color, reveal, height = 300, width = 70 }: { cells: number; color: string; reveal: number; height?: number; width?: number }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 3, height, width, overflow: 'hidden', borderRadius: 10 }}>
    {Array.from({ length: cells }, (_, i) => {
      const v = Math.abs(Math.sin(i * 12.9898) * 43758.5453) % 1;
      return <div key={i} style={{ flex: 1, background: color, opacity: (0.25 + 0.75 * v) * (i / cells < reveal ? 1 : 0.06) }} />;
    })}
  </div>
);

const Stage = ({ label, sub, active }: { label: string; sub: string; active: number }) => (
  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, opacity: 0.25 + 0.75 * active }}>
    <div
      style={{
        fontFamily: MONO, fontSize: 30, color: C.ink100, padding: '18px 26px', borderRadius: 16,
        border: `1px solid ${active > 0.5 ? C.ortho : C.ink700}`, background: C.ink850,
        boxShadow: active > 0.5 ? `0 0 30px ${C.ortho}44` : 'none',
      }}
    >
      {label}
    </div>
    <div style={{ fontFamily: SANS, fontSize: 22, color: C.ink400 }}>{sub}</div>
  </div>
);

const Arrow = ({ p }: { p: number }) => (
  <div style={{ width: 90, height: 4, background: `linear-gradient(90deg, ${C.ortho} ${p * 100}%, ${C.ink700} ${p * 100}%)`, borderRadius: 2 }} />
);

export const GemmaScope = ({ duration }: { duration: number }) => {
  const f = useCurrentFrame();
  const fade = useSceneFade(duration, 10, 14);
  return (
    <AbsoluteFill style={{ opacity: fade }}>
      <Backdrop glow={C.ortho} />
      <AbsoluteFill style={{ padding: '90px 150px', gap: 18 }}>
        <Rise><Kicker color={C.ortho}>Gemma Scope</Kicker></Rise>
        <Rise at={4}><Heading size={70}>Una feature del SAE, llevada al espacio de queries</Heading></Rise>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 34, marginTop: 40 }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14 }}>
            <VectorBar cells={48} color={C.ortho} reveal={progress(f, 8, 30)} />
            <Tex tex={String.raw`W_{\text{dec}}[f]\in\mathbb{R}^{2304}`} size={34} color={C.ink200} />
            <div style={{ fontFamily: SANS, fontSize: 22, color: C.ink400 }}>residual stream</div>
          </div>
          <Arrow p={progress(f, 28, 40)} />
          <Stage label="input_layernorm" sub="RMSNorm de la capa" active={progress(f, 36, 44)} />
          <Arrow p={progress(f, 44, 56)} />
          <Stage label="q_proj" sub="W_Q real, sin hooks" active={progress(f, 52, 60)} />
          <Arrow p={progress(f, 60, 72)} />
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14 }}>
            <div style={{ display: 'flex', gap: 8 }}>
              {Array.from({ length: 8 }, (_, h) => (
                <VectorBar key={h} cells={16} color={C.signal} width={26} reveal={progress(f, 66 + h * 2, 80 + h * 2)} />
              ))}
            </div>
            <Tex tex={String.raw`r_q\in\mathbb{R}^{8\times 256}`} size={34} color={C.ink200} />
            <div style={{ fontFamily: SANS, fontSize: 22, color: C.ink400 }}>8 cabezas × 256</div>
          </div>
        </div>
        <Rise at={74} style={{ display: 'flex', justifyContent: 'center', marginTop: 30 }}>
          <Body size={30} color={C.ink300}>
            <Tex tex={String.raw`r_q=q_{\text{proj}}\big(\text{input\_layernorm}(W_{\text{dec}}[f])\big)`} size={40} color={C.ink100} />
          </Body>
        </Rise>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
