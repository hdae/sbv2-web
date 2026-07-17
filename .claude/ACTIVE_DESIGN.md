# Active Design

Current design focus and a pitfalls index, so a reviewer or planner never starts
cold. Keep this to a screenful.

## Recently landed (2026-07-11, the 0.4.0 batch — released on JSR)

- **Shared DeBERTa (`DebertaExtractor`, ADR-0005).** BERT session + tokenizer +
  phone-level tiling split out of `Sbv2Adapter`
  (`src/runtime/deberta_extractor.ts`); adapters take a `BertSource` union —
  `bertOnnxBytes`+`tokenizer` (owned, released with the adapter, fully backward
  compatible) or `deberta` (shared, released ONLY by its creator).
  `createDeberta` on both wrappers. Measured (cpu / ort-node): 1 session
  ≈ 490MB; light-sbv2 4-model residency RSS 2986→1874MB (−1.1GB); model load
  2.5s→1.7s.

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

- **Sibling deps resolve from JSR** (`@hdae/fetch-cache@^0.3.0` → 0.3.0,
  `@hdae/yomi@^0.4.0` → 0.4.1; both published and the lock is converged). To
  co-develop unpublished sibling changes, temporarily add
  `"links": ["../fetch-cache"]` etc. to deno.json, but keep links OUT of commits
  — a missing links dir is a hard error on CI.
- **`toSbv2PhoneTone` MUST mirror yomi's `wordPhoneAlignment` order**
  (leadingPunctuations + per-phrase moras + punctuations); any divergence
  breaks `sum(word2ph) === phones.length` and synthesis throws.
  Examples/browser maps the library source's bare imports via vite alias (JS)
  plus tsconfig paths (`_dist` d.ts) into `node_modules/@hdae/*` — same pattern
  for yomi and fetch-cache.
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
- **A shared `DebertaExtractor` MUST outlive its adapters** — adapters never
  release it; the creator releases it after all adapters are done. Releasing
  it early makes later synthesis throw (fail loud) by design (ADR-0005). With
  a shared extractor, `sessionOptions` on adapter factories affects only the
  acoustic session.

## Next / resume point

- **v0.4.0 released on JSR (2026-07-11, tag v0.4.0)** — shared
  `DebertaExtractor` (ADR-0005, additive). light-sbv2 can now pin the published
  0.4.0 and drop its links-based wiring.
- **v0.3.0 released on JSR (2026-07-10)** — the yomi v0.4.0
  follow-up: `toSbv2PhoneTone` now packs REAL punctuation
  (`leadingPunctuations` + per-phrase `punctuations`, canonical `! ? … , . ' -`)
  instead of synthesizing `,`/`.` from pause classes (`pausePunct` was deleted
  upstream); question marks now reach the model = better sentence-final
  intonation. Examples import `@hdae/yomi/loader` (was `./browser`). Dep floors:
  yomi `^0.4.0`, fetch-cache `^0.2.0`. See
  [docs/migration-0.3.md](../docs/migration-0.3.md).
- **v0.2.0 released on JSR (2026-07-10)** with fetch-cache v0.1.0 and yomi
  v0.3.0; that batch is closed
  ([docs/migration-0.2.md](../docs/migration-0.2.md) went to light-sbv2).
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
