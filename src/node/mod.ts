// @hdae/sbv2-web/node: native (server / CLI) entry. Re-exports the runtime-agnostic
// core together with the onnxruntime-node backend (`Sbv2NodeModelAdapter`), so a Deno
// or Node process gets everything from here without pulling in onnxruntime-web. GPU
// via WebGPU (Dawn) / DirectML / CUDA.

export * from "../core.ts";
export { type NodeDevice, Sbv2NodeModelAdapter } from "./model_adapter.ts";
