/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Puertos fijos: el servidor sólo acepta WebSockets desde estos orígenes
  // (DEFAULT_ALLOWED_ORIGINS en latentflow/api/server.py). Si el puerto está
  // ocupado, Vite falla en vez de saltar a otro que el servidor rechazaría.
  server: { host: '127.0.0.1', port: 5173, strictPort: true },
  preview: { host: '127.0.0.1', port: 4173, strictPort: true },
  test: { environment: 'node', include: ['src/**/*.test.{ts,tsx}'] },
})
