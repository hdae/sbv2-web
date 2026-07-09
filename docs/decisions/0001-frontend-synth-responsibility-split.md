# 0001. Frontend / synth responsibility split

- Status: accepted
- Date: 2026-07-08

## Context

Style-Bert-VITS2 JP-Extra inference has two distinct concerns: a text frontend
(Japanese analysis → `given_phone` / `given_tone` → word-to-phone alignment) and
model inference (acoustic ONNX + DeBERTa BERT features). Mixing the
SBV2-specific model details into the frontend would make the frontend
un-reusable and couple text analysis to one model family.

## Decision

1. **The frontend is model-agnostic and zero-dependency.** It lives in
   `@hdae/yomi` and emits only a model-neutral intermediate (`given_phone` /
   `given_tone` and word alignment). All SBV2 JP-Extra specifics — the
   112-symbol table, tone +6, `add_blank`, BERT tiling, style vectors, scalar
   parameters, ONNX I/O — are confined to `Sbv2ModelAdapter`. The text→audio
   glue (`synthesizeText`) sits on the synth side and only assembles the
   model-agnostic `SynthInput`.

2. **`word2ph` is produced as the pre-`add_blank` base form**, with `[1]`
   sentinels at both ends (for the leading/trailing `_`). The `add_blank`
   adjustment (double each entry, +1 on the head) is the adapter's job, not the
   frontend's.

## Consequences

- The frontend is reusable across acoustic models; a second adapter can reuse
  the same analysis output.
- Two invariants are checked at the boundary and fail loudly:
  `len(word2ph) == Σ tokenize(surface) + 2` and
  `sum(word2ph) == given_phone length`.
- The glue must never re-run analysis: `analyzeToNodes` is called exactly once
  and its result is shared by `buildResult` / `toSbv2PhoneTone` /
  `wordPhoneAlignment`.

See also [aivmx-interface.md](../aivmx-interface.md) for the model I/O contract.
