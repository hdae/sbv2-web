# Limitations

## Model Assets

This package does not bundle model files. Faithful AivisSpeech /
Style-Bert-VITS2 JP-Extra inference needs:

- an acoustic AIVMX or ONNX model;
- a DeBERTa ONNX model for Japanese BERT features;
- tokenizer assets for the DeBERTa model;
- a JTD dictionary for `@hdae/yomi`.

Model licenses are separate from this package. AIVMX models may use ACML terms,
and DeBERTa derivatives inherit CC BY-SA 4.0 obligations from the original model
family.

See also: [License audit](license-audit.md).

## AIVMX Int8 Is Lossy

Static int8 AIVMX is useful for memory and speed, but it is not transparent.
Listening tests found a small degradation that was barely distinguishable in AB
comparison. Keep fp32 AIVMX / ONNX available for maximum quality.

## Browser Runtime Quantization

Deno/browser `onnxruntime-web@1.27.0` can execute quantized ONNX graphs when the
model already contains quantized operators such as QLinearConv, ConvInteger,
MatMulNBits, and DequantizeLinear.

It does not expose public JavaScript APIs for static quantization,
calibration-data collection, shape inference, or graph rewriting comparable to
Python ONNX Runtime's `quantize_static` and `quant_pre_process`. This was
checked with `tools/bench/check_ortweb_quantization.ts`.

For browser delivery, pre-quantize offline and download or cache the resulting
static int8 model.

## WebGPU

The browser lab defaults to the WASM execution provider. Loading the acoustic
model with the WebGPU provider currently fails inside the stochastic duration
predictor with `[GatherND] ... Unsupported data type: 7` (INT64): the WebGPU
execution provider claims the op and then throws at shader build. This is an
`onnxruntime-web` limitation, not a bug in this package — listing
`["webgpu", "wasm"]` does not fall back per kernel, and 1.27.0 is the latest
release.

Remediation options, none adopted yet (tracked in
[known-issues.md](known-issues.md)):

- rewrite the acoustic ONNX offline so the op is fed int32 instead of int64
  (ONNX Runtime's recommended workaround), then re-host the model;
- run a split-provider setup (DeBERTa on WebGPU, acoustic on WASM) — but the
  heavy HiFi-GAN vocoder stays on WASM, so the speedup is uncertain and
  DeBERTa's own int64 gather ops are untested on WebGPU.

## DeBERTa Lower Than Int4

Int2 NBits DeBERTa was not usable. The generated model failed in Python ONNX
Runtime and Deno `onnxruntime-web` WASM with
`MatMulNBits<MLFloat16> ... nbits_ == 8 was false`.

In the tested stack, int4 is the practical lower bound for DeBERTa weight-only
quantization.

## Zero-BERT

Zero-BERT is only an ablation path. JP-Extra models need real DeBERTa features
for faithful emotion and prosody.
