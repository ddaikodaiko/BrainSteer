import { AbsoluteFill } from 'remotion';
import { C, MONO } from '../theme';
import { Backdrop, Body, Card, Heading, Kicker, Rise, useSceneFade } from '../ui/primitives';

const Row = ({ at, origin, status, ok }: { at: number; origin: string; status: string; ok: boolean }) => (
  <Rise at={at}>
    <Card accent={ok ? C.signal : C.danger} style={{ display: 'flex', alignItems: 'center', gap: 30, padding: '24px 36px' }}>
      <span style={{ fontFamily: MONO, fontSize: 34, color: C.ink300 }}>Origin:</span>
      <span style={{ fontFamily: MONO, fontSize: 34, color: C.ink100, flex: 1 }}>{origin}</span>
      <span style={{ fontFamily: MONO, fontSize: 34, fontWeight: 600, color: ok ? C.signal : C.danger }}>{status}</span>
    </Card>
  </Rise>
);

export const Security = ({ duration }: { duration: number }) => {
  const fade = useSceneFade(duration, 2);
  return (
    <AbsoluteFill style={{ opacity: fade }}>
      <Backdrop glow={C.danger} />
      <AbsoluteFill style={{ padding: '110px 150px', gap: 24, justifyContent: 'center' }}>
        <Rise><Kicker color={C.danger}>Seguridad</Kicker></Rise>
        <Rise at={4}><Heading>WebSocket blindado ante CSWSH</Heading></Rise>
        <Rise at={10}><Body>El Origin se valida antes de accept(). El CORSMiddleware de FastAPI no cubre WebSockets.</Body></Rise>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 22, marginTop: 30 }}>
          <Row at={20} origin="http://127.0.0.1:5173" status="101 · conectado" ok />
          <Row at={36} origin="https://evil.example" status="403 Forbidden" ok={false} />
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
