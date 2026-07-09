# License Audit

This note separates the MIT-licensed JSR package from model and dictionary
assets used for local testing and distribution.

## JSR Package

The TypeScript source published as @hdae/sbv2-web is intended to be MIT licensed
under the repository LICENSE file.

The JSR publish allowlist includes README.md, LICENSE, deno.json, docs, and src.
It excludes tests and does not include data, model files, dictionaries, Python
tools, or generated HuggingFace packages.

No GPL, LGPL, or AGPL dependency is imported by the published TypeScript
entrypoints (`.` and `./node`). The runtime dependencies are @hdae/yomi (MIT),
onnxruntime-web for `.`, onnxruntime-node for `./node`, and onnxruntime-common
as the shared type layer (all ONNX Runtime packages are MIT).

## SYMBOLS Table

src/text/symbols.ts contains a small JP-Extra phoneme compatibility table and
related numeric constants. Public source comments avoid presenting this as
copied upstream implementation code. The table is treated as interoperability
data required to address compatible acoustic model inputs.

## @hdae/yomi And Dictionary Data

@hdae/yomi is used for Japanese text analysis and SBV2 phone/tone conversion.
Its source code is MIT licensed. The naist-jdic dictionary data used for tests
and examples is BSD-3-Clause licensed and is not bundled in the JSR package.

## Model Assets

AIVMX and acoustic ONNX models are separate assets and are not covered by this
package license. AIVMX models may use ACML terms or other model-specific terms.

Quantized DeBERTa ONNX models are derivatives of:

- ku-nlp/deberta-v2-large-japanese-char-wwm
- tsukumijima/deberta-v2-large-japanese-char-wwm-onnx

Those derivatives are distributed separately under CC BY-SA 4.0 with LICENSE and
NOTICE files.

## Tooling

Python/uv tooling under tools/model-tools is not included in the JSR package. It
is repository tooling for local conversion, benchmarking, and HuggingFace
packaging. Generated model packages must carry their own model license, NOTICE,
and manifest.

## Current Conclusion

The source package can be published as MIT as long as model files, dictionaries,
and generated HuggingFace packages remain outside the JSR package and keep their
own license notices.

This is an engineering audit note, not legal advice.
