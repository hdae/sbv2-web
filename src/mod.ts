// @hdae/sbv2-web: browser-primary entry. Re-exports the runtime-agnostic core
// together with the onnxruntime-web backend (`Sbv2ModelAdapter`). For a native
// (server / CLI) process use `@hdae/sbv2-web/node` instead.

export * from "./core.ts";
export { Sbv2ModelAdapter } from "./web/mod.ts";
