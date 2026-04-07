import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: mode === 'production' ? '/assets/apps/openarxiv/' : '/',
  define: mode === 'production' ? {
    'import.meta.env.VITE_DATA_SOURCE': JSON.stringify('remote'),
  } : {},
}))
