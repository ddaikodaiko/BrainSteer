import { Composition } from 'remotion';
import { Launch, TOTAL_FRAMES } from './Launch';
import { Theory, THEORY_FRAMES } from './Theory';

export const Root = () => (
  <>
    <Composition id="BrainSteerLaunch" component={Launch} durationInFrames={TOTAL_FRAMES} fps={30} width={1920} height={1080} />
    <Composition id="BrainSteerTheory" component={Theory} durationInFrames={THEORY_FRAMES} fps={30} width={1920} height={1080} />
  </>
);
