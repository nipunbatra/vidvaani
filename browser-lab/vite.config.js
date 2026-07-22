import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  build: {
    outDir: "../lab",
    emptyOutDir: true,
    sourcemap: false,
    target: "es2022",
    rollupOptions: {
      output: {
        entryFileNames: "assets/[name]-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]"
      }
    }
  },
  worker: {
    format: "es"
  }
});
