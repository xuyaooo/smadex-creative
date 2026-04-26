import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      // Asset PNGs come from FastAPI's StaticFiles mount at :8000/assets
      "/assets": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
