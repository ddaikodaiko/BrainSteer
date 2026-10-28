import './fonts';
import { AbsoluteFill, Audio, Sequence, staticFile } from 'remotion';
import { DEFAULT_DUCK, duckVolume } from './kit/audio';
import { Cockpit } from './scenes/Cockpit';
import { DevMode } from './scenes/DevMode';
import { Factorization } from './scenes/Factorization';
import { GemmaScope } from './scenes/GemmaScope';
import { Outro } from './scenes/Outro';
import { Problem } from './scenes/Problem';
import { Rigor } from './scenes/Rigor';
import { Security } from './scenes/Security';
import { Skop } from './scenes/Skop';
import { Title } from './scenes/Title';
import { BAR, C } from './theme';

// Montaje sobre la rejilla de compases (1 compás = 60 frames). Drops en los compases 10 y 22.
const TIMELINE = [
  { from: 0, bars: 3, Scene: Title },
  { from: 3, bars: 3, Scene: Problem },
  { from: 6, bars: 4, Scene: Skop },
  { from: 10, bars: 4, Scene: Factorization },   // drop 1: la factorización
  { from: 14, bars: 2, Scene: GemmaScope },
  { from: 22, bars: 2, Scene: Security },        // drop final
  { from: 24, bars: 2, Scene: Rigor },
  { from: 26, bars: 2, Scene: DevMode },
] as const;

// La cabina entra solapada con el final de Gemma Scope y ocupa el bajón (compases 18-22).
const COCKPIT = { from: 16 * BAR - 10, to: 22 * BAR };
const OUTRO = { from: 28 * BAR, bars: 3 };
export const TOTAL_FRAMES = 31 * BAR;

// Sin locución: la música no se agacha bajo ninguna voz; sólo un hueco antes del drop final.
const HUSH = [{ at: 22 * BAR - 14, frames: 12, level: 0.25 }];

export const Launch = () => (
  <AbsoluteFill style={{ background: C.ink950 }}>
    <Audio
      src={staticFile('music.wav')}
      volume={(f) => duckVolume(f, [], DEFAULT_DUCK, { total: TOTAL_FRAMES, hush: HUSH, fadeOut: 40 })}
    />
    {TIMELINE.map(({ from, bars, Scene }) => (
      <Sequence key={from} from={from * BAR} durationInFrames={bars * BAR}>
        <Scene duration={bars * BAR} />
      </Sequence>
    ))}
    <Sequence from={COCKPIT.from} durationInFrames={COCKPIT.to - COCKPIT.from}>
      <Cockpit duration={COCKPIT.to - COCKPIT.from} />
    </Sequence>
    <Sequence from={OUTRO.from} durationInFrames={OUTRO.bars * BAR}>
      <Outro />
    </Sequence>
  </AbsoluteFill>
);
