# Active Design

Current design focus and a pitfalls index, so a reviewer or planner never starts
cold. Keep this to a screenful.

## Recently landed (2026-07-10, the 0.2.0 batch)

- **Review fixes.** styleVector rejects non-integer styleId / non-finite weight
  (was silent NaN vectors); `fromOnnx` releases the acoustic session on partial
  init failure; browser lab buttons are gated by a machine `busy` state (the
  old string-match guard was broken by progress text); node `withDevice` no
  longer clobbers caller `executionProviders` (device+EP together throws).
- **Consumer API batch (light-sbv2 requests).** Per-call `SynthInput.scalars`
  (ADR-0003); `release()` contract — idempotent, waits for in-flight, throws
  after (ADR-0004); typed `readAivmxManifest` (default `stripAssets` drops the
  multi-MB data-URL icons/voice samples); typed `Sbv2HyperParameters`
  (spk2id/n_speakers/…) + adapter accessors; `fromAivmx({ metadata })` reuse;
  `validateSynthInput`; `padSilence`/`concatWithSilence` +
  `preSilenceSec`/`postSilenceSec`. Silent 44100 fallback removed (fromOnnx
  sampleRate required).
- **`@hdae/fetch-cache` (new sibling repo)** — generic Cache-API fetch layer
  (`.`) + HF layer (`./hf`: revision→SHA resolve, named-spec parallel fetch,
  size/sha256 validate with self-heal). sbv2-web's `getDeberta()` fetches the
  DeBERTa 4-file set SHA-pinned; browser lab dropped bundled tokenizer assets
  and the hand-rolled mutable-URL fetch; CLI runs with just `--aivmx`
  (deberta/dict auto-fetch, verified end-to-end incl. cache hit).

## Pitfalls index

- **Publish order: fetch-cache BEFORE sbv2-web.** sbv2-web imports
  `jsr:@hdae/fetch-cache@^0.1.0`; until it exists on JSR, local dev needs
  `"links": ["../fetch-cache"]` in deno.json (removed for CI — a missing links
  dir is a hard error). Examples/browser aliases fetch-cache to the sibling
  repo source; switch to the pnpm jsr dep after publish (TODO in vite.config).
- **getDeberta pins.** `DEBERTA_REVISION` + `PINNED_FILES` (sizes, model
  sha256) are baked for the default revision only; replacing the HF model
  means updating both (`src/assets/deberta.ts`).
- **Deno lacks `Cache.keys()`** (2.8): fetch-cache `listCachedUrls` throws on
  Deno rather than lying with `[]`; browsers are fine.
- **Worker teardown MUST live in an empty-dependency effect**; UI actions are
  gated by `busy`, and the adapter's release contract (ADR-0004) is the second
  line of defense.
- **Browser WebGPU (onnxruntime-web JSEP) still fails on int64 GatherND**;
  native ORT WebGPU (Dawn, via `./node`) does run it. A browser fix needs an
  OFFLINE int64→int32 rewrite — large + unproven payoff.
- **Treat vocab.txt as LF/CRLF-agnostic** (`src/text/deberta_tokenizer.ts`).
- **Two hand-written adapters would diverge** — keep synthesis in the single
  `Sbv2Adapter` core; a backend only supplies an `OrtBackend`.
- **Manifest strip is the default** — `/speakers`-style catalogs must not
  round-trip icons through `readAivmxManifest()` defaults; pass
  `{ stripAssets: false }` only where icons/samples are actually served.

## Next / resume point

- **Pending user actions:** publish `@hdae/fetch-cache` v0.1.0 (push + GitHub
  Release), then push sbv2-web (CI resolves the jsr dep) and cut v0.2.0.
  After publish: switch examples/browser to the pnpm jsr dep (vite TODO), and
  re-add local `links` only while co-developing.
- Deferred: assist_text (BERT-space emotion reference — needs a second DeBERTa
  pass); pitch/intonation post-processing (needs a WORLD port, quality loss);
  browser offline int64→int32 conversion (validation spike first).
- `public/dict/naist-jdic.jtd` stays gitignored-but-tracked until the init
  squash.

## Anchors

- Model I/O contract: [../docs/aivmx-interface.md](../docs/aivmx-interface.md).
- Layer split: [../docs/decisions/0001-frontend-synth-responsibility-split.md](../docs/decisions/0001-frontend-synth-responsibility-split.md).
- Backend split: [../docs/decisions/0002-runtime-agnostic-core-injected-backends.md](../docs/decisions/0002-runtime-agnostic-core-injected-backends.md).
- SynthInput contract: [../docs/decisions/0003-synthinput-public-contract.md](../docs/decisions/0003-synthinput-public-contract.md).
- release lifecycle: [../docs/decisions/0004-release-lifecycle-contract.md](../docs/decisions/0004-release-lifecycle-contract.md).
- Migration: [../docs/migration-0.2.md](../docs/migration-0.2.md).
