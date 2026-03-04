import { defineConfig } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

// BUG 8 FIX: Added server.proxy so:
// - WebSocket ws://localhost:5173/cognitive-stream → ws://127.0.0.1:8000/cognitive-stream
// - Video http://localhost:5173/video_feed         → http://127.0.0.1:8000/video_feed
// This means you can use relative URLs in components instead of hardcoded
// 127.0.0.1:8000, and CORS stops being a cross-origin concern in dev.
// Note: you can keep the hardcoded URLs for now — the proxy is a safety net.
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/video_feed': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/cognitive-stream': {
        target: 'ws://127.0.0.1:8000',
        ws: true,             // ← critical: tells Vite to proxy WebSocket
        changeOrigin: true,
      },
    },
  },
  assetsInclude: ['**/*.svg', '**/*.csv'],
})