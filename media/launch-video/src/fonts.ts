import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/inter/700.css';
import '@fontsource/inter/800.css';
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/jetbrains-mono/600.css';
import 'katex/dist/katex.min.css';
import { continueRender, delayRender } from 'remotion';

// Espera a que todas las fuentes (Inter, JetBrains Mono y las de KaTeX) estén cargadas
// antes de renderizar ningún fotograma: si no, los primeros frames salen con la fuente de reserva.
const handle = delayRender('fuentes');
const faces = [
  '400 40px Inter', '600 40px Inter', '700 40px Inter', '800 40px Inter',
  '400 30px "JetBrains Mono"', '600 30px "JetBrains Mono"',
  '40px KaTeX_Main', 'italic 40px KaTeX_Math', '40px KaTeX_AMS', '40px KaTeX_Caligraphic', '40px KaTeX_Size1', '40px KaTeX_Size2',
];
Promise.all(faces.map((f) => document.fonts.load(f)))
  .then(() => document.fonts.ready)
  .then(() => continueRender(handle))
  .catch(() => continueRender(handle));
