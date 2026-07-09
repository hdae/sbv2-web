# Active Design

Current design focus and a pitfalls index, so a reviewer or planner never starts
cold. Keep this to a screenful.

## Recently landed (2026-07-08 → 07-09)

- **yomi v0.2.0 follow-up.** `@hdae/yomi` narrowed its surface to an
  `analyzeWithWords` facade and dropped the `/sbv2` subpath. The SBV2
  `given_phone`/`given_tone` bridge (`toSbv2PhoneTone`) moved into
  `src/text/phone_tone.ts`, reusing yomi's public building materials
  (`moraToPhones`/`moraTones`/`pausePunct`) instead of duplicating them.
  `synthesizeText` now calls `analyzeWithWords` (result + words from one analysis).
  CLI consolidated to node-only at `examples/cli/` (single-shot or REPL); the web
  path is exercised in the browser lab.
- **Browser lab overhaul.** Worker lifecycle fixed (was terminating the Comlink
  worker on every `audioUrl` change / StrictMode mount → 2nd-synth hang + frozen
  load status); load flow reduced from 6 file inputs to 1 (DeBERTa streams from HF,
  tokenizer bundled, dict auto-fetched from HF, 6-file path behind an Advanced
  toggle); shadcn moved from Radix to Base UI (`base-nova`), Tailwind v3→v4;
  provider is a Base UI Select.
- **Two published entry points, browser-primary** (ADR-0002): `.` = core +
  `Sbv2ModelAdapter` (onnxruntime-web); `./node` = same core + `Sbv2NodeModelAdapter`
  (onnxruntime-node). ORT-agnostic core is `src/core.ts`, re-exported by both. The
  adapter logic lives once in `src/runtime/adapter_core.ts`, parameterized by an
  injected `OrtBackend` (int64/float32 factories + createSession); backend types via
  `onnxruntime-common`.
- **Node device support verified on real hardware:** cpu ✓, native `webgpu` (Dawn)
  ✓ with the int4 model, `dml` works only with an fp16 DeBERTa (int4 MatMulNBits
  breaks DirectML), `cuda` not bundled. See docs/known-issues.md.
- **Release/bump toolkit (yomi-aligned).** Version single-source: `deno.json`
  `version` ↔ baked `VERSION` (`src/core.ts`), synced by `deno task bump` in one
  commit; drift is fail-loud via `scripts/version_sync.test.ts` (in `deno task
  check`) + `scripts/verify_tag.ts` at release. GitHub Release on a `v<version>` tag
  → JSR publish (`release-jsr.yml`). README rewritten in Japanese (yomi style);
  `deno fmt` set to `proseWrap: preserve` so JP prose isn't broken mid-phrase.

## Pitfalls index

- **Worker teardown MUST live in an empty-dependency effect** and null the ref in
  cleanup; never key `release()`/`terminate()` on `audioUrl` or other state
  (`examples/browser/src/App.tsx`).
- **DeBERTa auto-fetch relies on HuggingFace CORS + range.** The dictionary is
  fetched from HuggingFace by yomi's `getDictionary()` (v0.2.0, version-matched +
  Cache-API-cached), so no same-origin file is needed; the gitignored
  `public/dict/naist-jdic.jtd` is now unused (the Advanced path still accepts an
  uploaded dict).
- **Browser WebGPU (onnxruntime-web JSEP) still fails on int64 GatherND**; native
  ORT WebGPU (Dawn, via `./node`) is a different impl and does run it. A browser fix
  needs an OFFLINE int64→int32 rewrite (replace GatherND/ScatterND, saturate the two
  ±(2^63−1) Slice sentinels, re-attach AIVM metadata) — large + unproven payoff.
- **Treat vocab.txt as LF/CRLF-agnostic** (`src/text/deberta_tokenizer.ts`).
- **Browser tsc needs explicit `onnxruntime-common`/`onnxruntime-web` paths**
  (`examples/browser/tsconfig.json`) to resolve the `/repo/src` imports.
- **Two hand-written adapters would diverge** — keep synthesis in the single
  `Sbv2Adapter` core; a backend only supplies an `OrtBackend`.

## Next / resume point

- Manual checks DONE (user-verified): browser lab (HF dict auto-fetch, repeated
  synth, Base UI Select) and node CLI (single-shot + REPL) both OK; `--device
  webgpu/dml` verified on GPU hardware earlier.
- Publish-prep candidate: the browser offline int64→int32 conversion (validation
  spike first). `public/dict/naist-jdic.jtd` stays as-is (gitignored; vanishes at
  the planned init squash).
- Deferred to post-release: a lightweight Deno server over the node backend.

## Anchors

- Model I/O contract: [../docs/aivmx-interface.md](../docs/aivmx-interface.md).
- Layer split: [../docs/decisions/0001-frontend-synth-responsibility-split.md](../docs/decisions/0001-frontend-synth-responsibility-split.md).
- Backend split: [../docs/decisions/0002-runtime-agnostic-core-injected-backends.md](../docs/decisions/0002-runtime-agnostic-core-injected-backends.md).
