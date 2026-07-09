# Known Issues

Open or being-worked problems. Intentional by-design constraints live in
[limitations.md](limitations.md) instead.

## WebGPU provider is not usable yet

Loading the acoustic model with the WebGPU execution provider fails inside the
stochastic duration predictor:

```
[WebGPU] Kernel "[GatherND] /sdp/flows.7/GatherND" failed. Error: Unsupported data type: 7
```

Data type 7 is INT64. This is an `onnxruntime-web` WebGPU limitation (the op is
claimed by the WebGPU EP and then throws at shader build), not a bug in this
repo. Listing `["webgpu", "wasm"]` does not help, and 1.27.0 is the latest
release. The browser lab therefore defaults to WASM. See
[limitations.md](limitations.md#webgpu) for the remediation options (offline
int64→int32 rewrite, or a split-provider run).

## Load-progress callback is not released

The browser lab passes an `onProgress` callback into the worker via Comlink
`proxy(...)`. Comlink keeps a `MessagePort` open for each proxied callback and
there is no clean handle to release an argument proxy, so each load leaks one
port. This is negligible for a lab tool (loads are rare) but should be revisited
if load is called in a loop.

## Node backend GPU providers are not verified in CI

`@hdae/sbv2-web/node` (onnxruntime-node, pinned to 1.27.0 to match
onnxruntime-web) drives synthesis on a chosen device. Verified status:

- `cpu` — works with the int4 DeBERTa (correct non-silent audio).
- `webgpu` — ORT's native WebGPU EP (Dawn). Works with the int4 DeBERTa on GPU
  hardware. This is a **different** implementation from the browser's
  `onnxruntime-web` WebGPU and runs the acoustic model where the browser path
  fails on int64. Requires a real GPU adapter (Vulkan/D3D/Metal). **Recommended
  GPU device.**
- `dml` (Windows DirectML) — works with an **fp16** DeBERTa, but **fails with
  the int4 (MatMulNBits) DeBERTa** on longer inputs. DirectML's quantized-MatMul
  path is the culprit (the CPU and WebGPU paths handle int4 fine, and the CPU
  path is length-agnostic across 1–10 character inputs). For DML, use an fp16
  DeBERTa, or prefer `webgpu`.
- `cuda` — **not bundled** in the stock package (`listSupportedBackends()`
  reports `cuda: bundled=false`); needs a CUDA-enabled build plus the CUDA
  runtime.

Operational notes: the native addon needs `libstdc++` on the loader path (set
`LD_LIBRARY_PATH` in nix/devbox shells); the node type-check is a separate task
(`deno task check:node`), kept out of the default `deno task check` so CI does
not download the heavy native package.

## Bench harness defaults reference locally generated assets

`tools/bench/run_matrix.ts` and `tools/bench/smoke_one.ts` default to model/dict
directories that are produced by the Python tooling and live under the
gitignored `data/`, so they do not run out of the box on a fresh checkout —
regenerate the assets with `tools/model-tools` first. `smoke_one.ts` accepts
explicit paths via flags (`--acoustic`/`--deberta`/`--tokenizer`/`--dict`);
`run_matrix.ts` has a hardcoded combo matrix and must be edited in source.
