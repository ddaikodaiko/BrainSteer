import { AbsoluteFill, useCurrentFrame } from 'remotion';
import { C, MONO } from '../theme';
import { Tex } from '../ui/Tex';
import { Backdrop, Body, Card, Heading, Kicker, Rise, useSceneFade } from '../ui/primitives';

// Salidas reales de `latentflow-probe` (Gemma-2-2B-IT en 4 bit, RTX 3060 Ti, greedy).
const PROMPT = 'Write a short email to my boss asking for a day off.';
const BASE = ['Subject: Day Off Request - [Your Name]', "Hi [Boss's Name],", 'This email is to request a day off on [Date]…'];
const BROKEN = [
  "You're asking for a short and concise response to a request for a job interview.",
  "Here's a breakdown of the email.",
  "Here's a breakdown of the email.",
  "Here's a breakdown of the email.",
];

const Output = ({ title, lines, color }: { title: string; lines: string[]; color: string }) => (
  <Card accent={color} style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 14, padding: '26px 34px' }}>
    <div style={{ fontFamily: MONO, fontSize: 26, color, letterSpacing: '0.06em' }}>{title}</div>
    {lines.map((line, i) => (
      <div key={i} style={{ fontFamily: MONO, fontSize: 28, lineHeight: 1.32, color: C.ink200 }}>{line}</div>
    ))}
  </Card>
);

/** Qué es el steering y su problema, con salidas reales del modelo. */
export const Steering = ({ duration }: { duration: number }) => {
  const f = useCurrentFrame();
  const fade = useSceneFade(duration);
  return (
    <AbsoluteFill style={{ opacity: fade }}>
      <Backdrop glow={C.additive} />
      <AbsoluteFill style={{ padding: '80px 130px', gap: 20, justifyContent: 'center' }}>
        <Rise><Kicker color={C.additive}>Activation steering</Kicker></Rise>
        <Rise at={6} style={{ display: 'flex', alignItems: 'baseline', gap: 48 }}>
          <Heading size={64}>Empujar un comportamiento sin reentrenar</Heading>
          <Tex tex={String.raw`q' = q + \alpha\, r_q`} size={52} color={C.additive} />
        </Rise>
        <Rise at={18}>
          <Body size={30}>Se suma una dirección r_q a las queries de atención: más formal, más complaciente…</Body>
        </Rise>
        <Rise at={70} style={{ marginTop: 18 }}>
          <div style={{ fontFamily: MONO, fontSize: 24, color: C.ink400 }}>
            prompt · <span style={{ color: C.ink200 }}>{PROMPT}</span>
          </div>
        </Rise>
        <div style={{ display: 'flex', gap: 30 }}>
          <Rise at={84} style={{ flex: 1, display: 'flex' }}><Output title="SIN STEERING" lines={BASE} color={C.signal} /></Rise>
          <Rise at={112} style={{ flex: 1, display: 'flex' }}><Output title="STEERING VANILLA · α = −3" lines={BROKEN} color={C.danger} /></Rise>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 6 }}>
          <Rise at={150}>
            <Body size={34} color={C.danger} style={{ fontWeight: 600 }}>Con α alto, el modelo se rompe: repite y se sale del tema.</Body>
          </Rise>
          <div style={{ fontFamily: MONO, fontSize: 20, color: C.ink400, opacity: f > 84 ? 1 : 0 }}>
            salidas reales · Gemma-2-2B-IT · greedy
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
