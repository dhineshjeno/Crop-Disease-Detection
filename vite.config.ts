import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    watch: {
      ignored: [
        '../backend/data/**',   // ignore dataset folder
        '../backend/**',        // ignore backend files
        'node_modules/**'
      ]
    }
  }
})
