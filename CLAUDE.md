# @hdae/sbv2-web

Browser-oriented Deno/TypeScript runtime for Style-Bert-VITS2 / AivisSpeech
JP-Extra ONNX inference. This file is the project entry point for contributors
and assistants; it takes precedence over any global conventions.

## Layout

- `src/` — the published JSR package (`@hdae/sbv2-web`), two entry points
  (browser-primary):
  - `.` (`src/mod.ts`) — browser/web: the shared core + `Sbv2ModelAdapter`
    (onnxruntime-web).
  - `./node` (`src/node/mod.ts`) — native: the same core +
    `Sbv2NodeModelAdapter` (onnxruntime-node; GPU via WebGPU/DirectML/CUDA). The
    runtime-agnostic core (`src/core.ts`: tokenizer, word2ph, tensor assembly,
    text→audio glue, and the unified `Sbv2Adapter`) is re-exported by both
    entries. The adapter logic lives once in `src/runtime/adapter_core.ts`; each
    backend only supplies `int64`/`float32` tensor factories and `createSession`
    (an `OrtBackend`), so numeric behavior stays structurally in one place.
- `examples/browser/` — a Vite + React + shadcn (Base UI) verification lab. Uses
  **pnpm**, runs ONNX Runtime Web in a Comlink worker (WASM / WebGPU
  selectable).
- `examples/cli/` — onnxruntime-node synthesis CLI (single-shot or REPL).
- `tools/model-tools/` — **uv**-managed Python tools for model conversion,
  quantization, and HuggingFace packaging. Not part of the JSR package.
- `tools/bench/` — Deno bench harness.
- `data/` — local model / dictionary assets. **gitignored**, never published.

## Commands

- `deno task check` — fmt (check) + lint + `deno check` + test. Clear before
  moving on.
- `deno task cli -- --aivmx path --device cpu|dml|cuda|webgpu` —
  onnxruntime-node synthesis CLI. `--text` で単発、省略で REPL。DeBERTa・辞書は
  既定で HuggingFace 自動取得（`--deberta`+`--tokenizer` / `--dict` で上書き）。
- `deno task check:node` — type-checks the node backend (kept out of the default
  check to avoid downloading the native package in CI).
- `deno task bench:smoke` / `deno task bench:matrix` — benchmark harness.
- Browser lab: `pnpm -C examples/browser dev` (or `build`). shadcn components
  are added via the CLI (`pnpm dlx shadcn@latest add <name>`), never
  hand-written.

## Architecture

The text frontend (`@hdae/yomi`) is model-agnostic and zero-dependency: it emits
`given_phone` / `given_tone` and word alignment only. All SBV2 JP-Extra
specifics (symbol table, tone +6, `add_blank`, BERT tiling, style vector, ONNX
I/O) are confined to the `Sbv2Adapter` core (driven by an injected ORT backend).
`synthesizeText` is the glue that assembles the model-agnostic `SynthInput`
(`SynthInput` is a public contract — ADR-0003; `release()` lifecycle —
ADR-0004). Asset acquisition: `getDeberta()` (`src/assets/deberta.ts`) fetches
the quantized DeBERTa + tokenizer set from HuggingFace, SHA-pinned and verified,
via `@hdae/fetch-cache`; the dictionary comes from yomi's `getDictionary()`. See
[ADR-0001](docs/decisions/0001-frontend-synth-responsibility-split.md).

## Docs (what goes where)

- [docs/aivmx-interface.md](docs/aivmx-interface.md) — the acoustic/DeBERTa
  model I/O contract ("how the model is driven").
- [docs/decisions/](docs/decisions/) — ADRs (the "why" behind choices).
- [docs/known-issues.md](docs/known-issues.md) — open / being-worked problems.
- [docs/limitations.md](docs/limitations.md) — intentional by-design
  constraints.
- [docs/benchmark.md](docs/benchmark.md) — quantization/perf matrix.
- [docs/license-audit.md](docs/license-audit.md) — package vs model/dict
  licensing.
- [docs/migration-0.2.md](docs/migration-0.2.md) — 0.1.0 → 0.2.0 API migration
  guide.
- [docs/migration-0.3.md](docs/migration-0.3.md) — 0.2.0 → 0.3.0 migration
  (real punctuation in given_phone; yomi ^0.4.0 / fetch-cache ^0.2.0).
- [.claude/ACTIVE_DESIGN.md](.claude/ACTIVE_DESIGN.md) — current design focus
  and a pitfalls index. Read it before reviewing or planning.

## Conventions

- **Fail loudly.** No silent coercion of broken/old data. **Released since
  0.2.0 (JSR)** — breaking changes now need a migration note
  (`docs/migration-<ver>.md`) and a deliberate version bump; do not break the
  published API casually.
- Ship tests in the same commit as the change. `src/` numeric paths are faithful
  ports of `tools/model-tools/scripts/synth_aivmx.py`; keep that parity.
- Keep inline comments about "why". Larger rationale goes to an ADR or a design
  doc, not into untraceable inline references.
