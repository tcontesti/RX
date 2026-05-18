import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  server: {
    host: true,
    port: 5175,
    proxy: {
      '/api': {
        target: 'http://localhost:9020',
        changeOrigin: true,
      },
    },
  },
})
