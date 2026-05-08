import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    dedupe: ['react', 'react-dom', 'motion', 'motion/react'],
  },
  server: {
    port: 5177,
    proxy: {
      // Real squish FastAPI server (default 11435).
      '/v1':       { target: 'http://localhost:11435', changeOrigin: true },
      '/health':   { target: 'http://localhost:11435', changeOrigin: true },
      // Ollama-compatible chat (same port).
      '/api/chat': { target: 'http://localhost:11435', changeOrigin: true },
      // KV-cache demo endpoints live on the demo server.
      '/api':      { target: 'http://localhost:8001',  changeOrigin: true },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: true,
  },
})
