// Tokens de color de la cabina (frontend/src/index.css), para que el vídeo y la app sean la misma marca.
export const C = {
  ink950: '#05070b',
  ink900: '#0a0e15',
  ink850: '#0e131c',
  ink800: '#131a25',
  ink700: '#1c2533',
  ink600: '#283345',
  ink400: '#5d6b80',
  ink300: '#8a97ab',
  ink200: '#b8c2d1',
  ink100: '#e3e8ef',
  signal: '#39f5b4',
  ortho: '#38bdf8',
  additive: '#f472b6',
  warn: '#fbbf24',
  danger: '#fb7185',
  violet: '#a78bfa',
} as const;

export const SANS = 'Inter, system-ui, sans-serif';
export const MONO = '"JetBrains Mono", ui-monospace, monospace';

/** 1 compás = 2 s = 60 frames a 30 fps (pista a 120 BPM medida, no estimada). */
export const BAR = 60;
