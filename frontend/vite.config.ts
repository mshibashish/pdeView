import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// VITE_STATIC=1 builds the static GitHub Pages site (docs/static-site.md); VITE_BASE is its path.
export default defineConfig({
  base: process.env.VITE_BASE || "/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  worker: { format: "es" },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { outDir: "dist", chunkSizeWarningLimit: 900 },
});
