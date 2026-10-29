import './fonts';
import { AbsoluteFill, Audio, Sequence, staticFile } from 'remotion';
import { DEFAULT_DUCK, duckVolume } from './kit/audio';
import { BAR, C } from './theme';
import { Focus } from './theory/Focus';
import { Measure } from './theory/Measure';
import { RealCockpit } from './theory/RealCockpit';
import { SkopTheory } from './theory/SkopTheory';
import { Steering } from './theory/Steering';
import { TheoryOutro } from './theory/TheoryOutro';
import { TheoryTitle } from './theory/TheoryTitle';

// Vídeo de teoría para LinkedIn: explica SKOP y lo que está construido, sin afirmar resultados.
// Misma pista y misma rejilla que el primer vídeo (1 compás = 60 frames; drops en 10 y 22).
const TIMELINE = [
  { from: 0, bars: 2, Scene: TheoryTitle },
  { from: 2, bars: 4, Scene: Steering },
  { from: 6, bars: 4, Scene: Focus },
  { from: 10, bars: 6, Scene: SkopTheory },   // drop 1: el método
  { from: 22, bars: 6, Scene: Measure },      // drop final: lo que falta medir
] as const;

// La cabina real entra solapada con el final de SKOP y ocupa el bajón (compases 16-22).
const COCKPIT = { from: 16 * BAR - 10, to: 22 * BAR };
const OUTRO = { from: 28 * BAR, bars: 3 };
export const THEORY_FRAMES = 31 * BAR;

const HUSH = [{ at: 22 * BAR - 14, frames: 12, level: 0.25 }];

export const Theory = () => (
  <AbsoluteFill style={{ background: C.ink950 }}>
    <Audio
      src={staticFile('music.wav')}
      volume={(f) => duckVolume(f, [], DEFAULT_DUCK, { total: THEORY_FRAMES, hush: HUSH, fadeOut: 40 })}
    />
    {TIMELINE.map(({ from, bars, Scene }) => (
      <Sequence key={from} from={from * BAR} durationInFrames={bars * BAR}>
        <Scene duration={bars * BAR} />
      </Sequence>
    ))}
    <Sequence from={COCKPIT.from} durationInFrames={COCKPIT.to - COCKPIT.from}>
      <RealCockpit duration={COCKPIT.to - COCKPIT.from} />
    </Sequence>
    <Sequence from={OUTRO.from} durationInFrames={OUTRO.bars * BAR}>
      <TheoryOutro />
    </Sequence>
  </AbsoluteFill>
);
