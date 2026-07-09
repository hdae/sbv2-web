import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";
import { defineConfig } from "vite";
import { comlink } from "vite-plugin-comlink";

export default defineConfig({
  plugins: [comlink(), react(), tailwindcss()],
  worker: {
    format: "es",
    plugins: () => [comlink()],
  },
  resolve: {
    alias: [
      {
        find: "@hdae/yomi/browser",
        replacement: resolve(
          __dirname,
          "node_modules/@hdae/yomi/src/browser/mod.js",
        ),
      },
      {
        find: "@hdae/yomi",
        replacement: resolve(__dirname, "node_modules/@hdae/yomi/src/mod.js"),
      },
      {
        find: "onnxruntime-web/webgpu",
        replacement: resolve(
          __dirname,
          "node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs",
        ),
      },
      {
        find: "onnxruntime-web",
        replacement: resolve(
          __dirname,
          "node_modules/onnxruntime-web/dist/ort.bundle.min.mjs",
        ),
      },
      { find: "@", replacement: resolve(__dirname, "src") },
    ],
  },
  server: {
    fs: {
      allow: ["../.."],
    },
  },
});
