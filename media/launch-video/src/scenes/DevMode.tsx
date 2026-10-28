import { AbsoluteFill, useCurrentFrame } from 'remotion';
import { C, MONO } from '../theme';
import { Backdrop, Body, Card, Heading, Kicker, Rise, useSceneFade } from '../ui/primitives';

const LINES: { at: number; text: string; color: string }[] = [
  { at: 12, text: '$ latentflow-dev', color: C.ink100 },
  { at: 34, text: 'INFO latentflow.dev: dev/tiny-llama-26L (pesos aleatorios)', color: C.ink300 },
  { at: 40, text: '     en ws://127.0.0.1:8000/ws/stream', color: C.ink300 },
  { at: 58, text: '$ cd frontend && npm run dev', color: C.ink100 },
  { at: 78, text: '  ➜  Local:   http://127.0.0.1:5173/', color: C.signal },
];

export const DevMode = ({ duration }: { duration: number }) => {
  const f = useCurrentFrame();
  const fade = useSceneFade(duration);
  return (
    <AbsoluteFill style={{ opacity: fade }}>
      <Backdrop glow={C.violet} />
      <AbsoluteFill style={{ padding: '100px 150px', gap: 22, justifyContent: 'center' }}>
        <Rise><Kicker color={C.violet}>Sin GPU</Kicker></Rise>
        <Rise at={4}><Heading>El servidor real sobre un modelo tiny</Heading></Rise>
        <Rise at={8}><Body>Prototipa la cabina en cualquier portátil. Las features de Gemma Scope se simulan con un vector determinista (SHA256 del ID).</Body></Rise>
        <Rise at={10} style={{ marginTop: 26 }}>
          <Card accent={C.violet} style={{ padding: '30px 40px', minHeight: 300 }}>
            {LINES.map((l) => {
              const chars = Math.max(0, Math.min(l.text.length, Math.floor((f - l.at) * 2.4)));
              return (
                <div key={l.at} style={{ fontFamily: MONO, fontSize: 32, lineHeight: 1.6, color: l.color, whiteSpace: 'pre' }}>
                  {l.text.slice(0, chars)}
                </div>
              );
            })}
          </Card>
        </Rise>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
