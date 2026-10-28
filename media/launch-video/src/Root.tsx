import { Composition } from 'remotion';
import { Launch, TOTAL_FRAMES } from './Launch';

export const Root = () => (
  <Composition
    id="BrainSteerLaunch"
    component={Launch}
    durationInFrames={TOTAL_FRAMES}
    fps={30}
    width={1920}
    height={1080}
  />
);
