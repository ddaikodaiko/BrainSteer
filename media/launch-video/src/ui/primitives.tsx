import type { CSSProperties, ReactNode } from 'react';
import { AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { C, MONO, SANS } from '../theme';

const clamp = { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' } as const;

/** Fondo de la cabina: degradado radial + rejilla de puntos de React Flow. */
export const Backdrop = ({ glow = C.ortho }: { glow?: string }) => (
  <AbsoluteFill style={{ background: `radial-gradient(1400px 800px at 25% 0%, #0d1520 0%, ${C.ink950} 62%)` }}>
    <AbsoluteFill
      style={{
        backgroundImage: `radial-gradient(${C.ink700} 1.3px, transparent 1.3px)`,
        backgroundSize: '28px 28px',
        opacity: 0.55,
      }}
    />
    <AbsoluteFill style={{ background: `radial-gradient(900px 500px at 80% 110%, ${glow}14 0%, transparent 70%)` }} />
  </AbsoluteFill>
);

/** Opacidad de entrada/salida de una escena (frames relativos a la Sequence). */
export const useSceneFade = (duration: number, fadeIn = 10, fadeOut = 8) => {
  const f = useCurrentFrame();
  return interpolate(f, [0, fadeIn, duration - fadeOut, duration], [0, 1, 1, 0], clamp);
};

/** Aparición con muelle: sube 24 px y gana opacidad. */
export const Rise = ({ at = 0, children, style }: { at?: number; children: ReactNode; style?: CSSProperties }) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = spring({ frame: f - at, fps, config: { damping: 200, mass: 0.6 } });
  return <div style={{ opacity: p, transform: `translateY(${(1 - p) * 24}px)`, ...style }}>{children}</div>;
};

export const Kicker = ({ children, color = C.signal }: { children: ReactNode; color?: string }) => (
  <div style={{ fontFamily: MONO, fontSize: 24, letterSpacing: '0.22em', textTransform: 'uppercase', color }}>{children}</div>
);

export const Heading = ({ children, size = 76 }: { children: ReactNode; size?: number }) => (
  <div style={{ fontFamily: SANS, fontWeight: 700, fontSize: size, lineHeight: 1.08, letterSpacing: '-0.025em', color: C.ink100 }}>
    {children}
  </div>
);

export const Body = ({ children, size = 36, color = C.ink300, style }: { children: ReactNode; size?: number; color?: string; style?: CSSProperties }) => (
  <div style={{ fontFamily: SANS, fontWeight: 500, fontSize: size, lineHeight: 1.35, color, ...style }}>{children}</div>
);

export const Chip = ({ children, color = C.ink300 }: { children: ReactNode; color?: string }) => (
  <span
    style={{
      fontFamily: MONO, fontSize: 24, color, padding: '10px 20px', borderRadius: 12,
      border: `1px solid ${C.ink600}`, background: `${C.ink900}cc`, whiteSpace: 'nowrap',
    }}
  >
    {children}
  </span>
);

/** Tarjeta con el estilo de los nodos de la cabina. */
export const Card = ({ children, accent = C.ortho, style }: { children: ReactNode; accent?: string; style?: CSSProperties }) => (
  <div
    style={{
      background: `${C.ink850}f2`, border: `1px solid ${C.ink700}`, borderRadius: 28, padding: '34px 42px',
      boxShadow: `0 0 0 1px ${accent}22, 0 30px 80px -30px ${accent}66`, ...style,
    }}
  >
    {children}
  </div>
);

export const progress = (f: number, a: number, b: number, ease = Easing.inOut(Easing.cubic)) =>
  interpolate(f, [a, b], [0, 1], { ...clamp, easing: ease });

/** Logo de BrainSteer (el de la cabecera de la app), con trazo animado. */
export const Logo = ({ size = 180, draw = 1 }: { size?: number; draw?: number }) => (
  <svg viewBox="0 0 32 32" width={size} height={size}>
    <defs>
      <linearGradient id="lf-g" x1="0" x2="1" y1="0" y2="1">
        <stop offset="0" stopColor={C.ortho} />
        <stop offset="1" stopColor={C.signal} />
      </linearGradient>
    </defs>
    <rect x="1" y="1" width="30" height="30" rx="9" fill={C.ink850} stroke={C.ink600} />
    <path
      d="M6 20 C 10 20, 10 10, 14 10 S 18 22, 22 22 S 25 14, 26 14"
      fill="none" stroke="url(#lf-g)" strokeWidth="2.2" strokeLinecap="round"
      pathLength={1} strokeDasharray={1} strokeDashoffset={1 - draw}
    />
    <circle cx="14" cy="10" r="2" fill={C.ortho} opacity={draw > 0.45 ? 1 : 0} />
    <circle cx="22" cy="22" r="2" fill={C.signal} opacity={draw > 0.8 ? 1 : 0} />
  </svg>
);
