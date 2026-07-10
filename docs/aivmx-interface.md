# AIVMX Interface

The runtime I/O contract that `src/runtime/*` implements for the
Style-Bert-VITS2 JP-Extra acoustic model (AIVMX / plain ONNX). This reconstructs
the "how the model is driven" reference. Ground truth:
`tools/model-tools/scripts/synth_aivmx.py` (the Python reference implementation)
and behavior measured on the mao / kohaku AIVMX models. For the "why" of the
layer split, see
[ADR-0001](decisions/0001-frontend-synth-responsibility-split.md).

## 1. AIVMX Container

### 1.2 Metadata

An AIVMX file is an ONNX model whose metadata entries carry base64-encoded
assets:

- `aivm_manifest` — JSON manifest.
- `aivm_hyper_parameters` — JSON hyper-parameters (holds `sampling_rate`).
- `aivm_style_vectors` — a `.npy` matrix, `[num_styles, 256]` float32.

`src/runtime/aivmx_meta.ts` extracts these; `src/runtime/npy.ts` parses the
style matrix. Plain ONNX acoustic models carry no AIVM metadata, so the style
vectors and sample rate must be supplied separately.

## 2. Acoustic Graph

### 2.1 Input Binding

Feeds are bound by the graph's declared input names (`session.inputNames`),
never by a hardcoded order. Observed input names: `x_tst`, `x_tst_lengths`,
`sid`, `tones`, `language`, `bert`, `style_vec`, `length_scale`, `sdp_ratio`,
`noise_scale`, `noise_scale_w`. A missing binding fails loudly.

### 2.3 Output

The primary waveform output name is `output` (`[1, 1, N]` float32, reshaped to a
flat `Float32Array`). No int16 peak-normalization round trip is applied,
matching the Python reference.

### 2.4 Symbols, Tone Offset, add_blank

A stable 112-symbol phoneme table (`src/text/symbols.ts`) maps phones to IDs.
The full table is used for indexing; indexing with only the JP subset would
shift IDs. JP tones are offset by +6, JP language ID is 1. `add_blank`
intersperses `0` between and around every element, giving length `2n+1`
(equivalent to SBV2 `commons.intersperse`).

### 2.5 Default Scalars

`length_scale` 1.0, `sdp_ratio` 0.2, `noise_scale` 0.6, `noise_scale_w` 0.8 —
the AivisSpeech operating values.

### 2.6 Style Vector

From the style matrix `[N, 256]`:
`style_vec = mean + (row[styleId] - mean) * weight`, a length-256 float32
vector. The matrix is parsed from the `.npy` in the AIVM metadata and must be
`[N, 256]` float32 or it fails loudly. The per-column mean is taken over all
rows, so selecting the mean row leaves the vector weight-invariant.

## 3. BERT Features

### 3.1 word2ph and Tiling

DeBERTa produces per-character hidden states `[seq_len, 1024]`. These are tiled
to the phone level using `word2ph` and transposed to `[1024, T]` where
`T = sum(word2ph)`. The base `word2ph` (pre-`add_blank`, with `[1]` sentinels at
both ends) is produced by `src/text/word2ph.ts`; the adapter doubles each entry
and adds 1 to the head to match `add_blank`. The DeBERTa input text is the
concatenation of the word-alignment surfaces (not the normalized text directly),
because `word2ph` is built from `Σ tokenize(surface)`.

## 6. Adapter Layer

### 6.2 Responsibility Boundary

The text frontend (analysis, phone/tone, word alignment) is model-agnostic and
zero-dependency; it lives in `@hdae/yomi`. All SBV2 JP-Extra specifics — the
symbol table, tone +6, `add_blank`, BERT tiling, style vector, scalar
parameters, and ONNX input/output wiring — are confined to `Sbv2ModelAdapter`.
The text→audio glue (`synthesizeText`) lives on the synth side and only
assembles the model-agnostic `SynthInput`. See
[ADR-0001](decisions/0001-frontend-synth-responsibility-split.md).

### 6.3 SynthInput as a Public Contract

`SynthInput` is a stable public contract so consumers (e.g. a VOICEVOX-style
server synthesizing from user-edited accent phrases) can build the
`given_phone` / `given_tone` path themselves
([ADR-0003](decisions/0003-synthinput-public-contract.md)). Invariants:

- `phones` — SBV2 symbols including both `_` pads and punctuation, pre
  `add_blank`; `tones` — same length, values 0/1 (pre +6).
- `baseWord2ph` — both-end sentinels are `1`, every entry a non-negative
  integer (`0` = "assign no phones to this char"; it legitimately occurs: `"…"`
  is one phone but NFKC-expands to the 3 char tokens `"..."`, so
  `distributePhone(1, 3) = [1, 0, 0]`), `sum(baseWord2ph) === phones.length`,
  and `baseWord2ph.length === tokenize(bertText).length + 2` (DeBERTa char
  tokens).
- `scalars` — optional per-call partial override merged over the adapter
  defaults; non-finite values throw.

`validateSynthInput(input, tokenizer?)` checks all of the above up front (the
DeBERTa-token check only when a tokenizer is passed); the adapter re-checks the
deep invariants during synthesis regardless. `release()` follows
[ADR-0004](decisions/0004-release-lifecycle-contract.md): idempotent, waits for
in-flight synthesis, and rejects further calls loudly.
