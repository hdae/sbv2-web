# 0002. Runtime-agnostic core with injected ONNX Runtime backends

- Status: accepted
- Date: 2026-07-09

## Context

The package must serve two very different ONNX Runtime targets:
`onnxruntime-web` (browsers, Deno) and `onnxruntime-node` (server / CLI, with
GPU execution providers — WebGPU/DirectML/CUDA). The first cut shipped a single
`Sbv2ModelAdapter` bound to `onnxruntime-web`, exported from the package root.
That had two problems: importing the root pulled `onnxruntime-web` even for
consumers that only wanted the tokenizer or the node backend, and a second
(node) adapter would duplicate the session-driving logic, risking silent numeric
divergence between the two paths.

## Decision

Two public entry points, browser-primary:

- `.` — the browser/web entry: the runtime-agnostic core **plus**
  `Sbv2ModelAdapter` (injecting `onnxruntime-web`).
- `./node` — the native entry: the same core **plus** `Sbv2NodeModelAdapter`
  (injecting `onnxruntime-node`; device cpu / webgpu / dml / cuda).

The runtime-agnostic core (`src/core.ts`: tokenizer, `word2ph`, tensor assembly,
`synthesizeText`, the unified `Sbv2Adapter`) pulls in no ONNX Runtime and is
shared internally, re-exported by both entries. A node process imports
everything from `./node` and never pulls `onnxruntime-web`, while the common
browser import stays the short `@hdae/sbv2-web`.

The adapter logic lives once in `Sbv2Adapter` (`src/runtime/adapter_core.ts`),
parameterized by an injected `OrtBackend` that supplies only what differs
between runtimes: `int64` / `float32` tensor factories and `createSession`.
Backend types are written against `onnxruntime-common`, which both
`onnxruntime-web` and `onnxruntime-node` re-export, so their `Tensor` /
`InferenceSession` unify. Tensor assembly (`tensor_build.ts`) and the
model-agnostic types (`adapter_types.ts`) are already shared.

## Consequences

- Each entry pulls exactly one ONNX Runtime; a node process (importing from
  `./node`) never pulls in `onnxruntime-web`.
- The numeric path exists once; adding a backend means supplying an
  `OrtBackend`, not re-implementing synthesis — the invariant holds
  structurally, per the project rule against second hand-written paths.
- Browser-primary: `Sbv2ModelAdapter` stays at `.`, so the common web import is
  unchanged; `./node` is additive. The core is shared internally, not offered as
  its own public entry.
- Device coverage is documented in [known-issues.md](../known-issues.md): cpu
  and native WebGPU (Dawn) run the int4 model; DirectML needs an fp16 DeBERTa.

See also [aivmx-interface.md §6.2](../aivmx-interface.md) and
[ADR-0001](0001-frontend-synth-responsibility-split.md).
