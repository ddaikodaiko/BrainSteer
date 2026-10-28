import katex from 'katex';
import type { CSSProperties } from 'react';

/** Fórmula renderizada con KaTeX (la misma notación que el README). */
export const Tex = ({ tex, size = 48, color, style }: { tex: string; size?: number; color?: string; style?: CSSProperties }) => (
  <span
    // nowrap: en modo inline KaTeX permite saltos tras '=' y partiría las fórmulas largas.
    style={{ fontSize: size, color, whiteSpace: 'nowrap', ...style }}
    dangerouslySetInnerHTML={{ __html: katex.renderToString(tex, { displayMode: false, throwOnError: true, output: 'html' }) }}
  />
);
