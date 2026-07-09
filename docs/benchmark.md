# Benchmarks

Benchmarks were run locally with Deno and `onnxruntime-web` WASM unless
otherwise noted. The input text was `こんにちは、今日はいい天気ですね。`; every
successful row generated non-silent WAV output.

## Main Matrix

| acoustic          |             DeBERTa | acoustic bytes | DeBERTa bytes |     peak RSS | synth ms |
| ----------------- | ------------------: | -------------: | ------------: | -----------: | -------: |
| fp32 AIVMX        |                fp16 |    258,037,076 |   653,075,699 | 4,205,988 KB |    4,827 |
| fp32 AIVMX        |        dynamic int8 |    258,037,076 |   397,495,820 | 2,152,772 KB |    4,801 |
| fp32 AIVMX        |          NBits int8 |    258,037,076 |   402,237,376 | 2,290,072 KB |    5,311 |
| fp32 AIVMX        |  NBits int4 RTN b32 |    258,037,076 |   259,499,945 | 1,827,232 KB |    5,620 |
| fp32 AIVMX        |  NBits int4 HQQ b32 |    258,037,076 |   272,472,752 | 1,920,780 KB |    5,437 |
| static int8 AIVMX |  NBits int4 RTN b32 |     83,104,930 |   259,499,945 | 1,556,192 KB |    4,241 |
| fp32 AIVMX        | NBits int4 RTN b256 |    258,037,076 |   240,576,205 | 1,763,824 KB |    5,675 |
| static int8 AIVMX | NBits int4 RTN b256 |     83,104,930 |   240,576,205 | 1,429,904 KB |    4,182 |

Best measured RAM/speed balance: `static int8 AIVMX + DeBERTa int4 RTN b256`.

Quality-sensitive default candidate: `fp32 AIVMX + DeBERTa int4 RTN b32` or
`fp32 AIVMX + DeBERTa int4 HQQ b32`, because AIVMX int8 is lossy and larger
DeBERTa block sizes reduce BERT feature similarity.

## AIVMX Quantization

| acoustic           |      DeBERTa | acoustic bytes |     peak RSS | synth ms | note               |
| ------------------ | -----------: | -------------: | -----------: | -------: | ------------------ |
| fp32 AIVMX         | int4 RTN b32 |    258,037,076 | 1,827,232 KB |    5,620 | baseline           |
| static int8 AIVMX  | int4 RTN b32 |     83,104,930 | 1,556,192 KB |    4,241 | smaller and faster |
| dynamic int8 AIVMX | int4 RTN b32 |     81,172,722 | 1,530,620 KB |   32,464 | too slow on WASM   |
| dynamic int8 AIVMX | int8 qlinear |     81,172,722 | 1,823,628 KB |   31,095 | too slow on WASM   |

Static int8 uses QLinearConv and is the only AIVMX quantization variant that was
both smaller and faster on WASM. Dynamic int8 uses ConvInteger and was about 7x
slower in this setup.

Listening tests found that static int8 AIVMX has a small quality drop that is
barely distinguishable in AB comparison. It is probably practical, but it should
be treated as a lossy quality/performance option rather than a transparent
replacement for fp32.

## DeBERTa Quantization

BERT feature cosine similarity was measured against fp16 ONNX outputs on five
Japanese sentences.

| DeBERTa        |       bytes | avg cosine | min cosine |
| -------------- | ----------: | ---------: | ---------: |
| int8 qlinear   | 397,495,820 |   0.977453 |   0.974748 |
| int8 NBits b32 | 402,237,376 |   0.999904 |   0.999829 |
| int4 RTN b32   | 259,499,945 |   0.974084 |   0.963577 |
| int4 RTN b128  | 243,279,697 |   0.959083 |   0.940689 |
| int4 RTN b256  | 240,576,205 |   0.951447 |   0.932897 |
| int4 HQQ b32   | 272,472,752 |   0.977996 |   0.972342 |

`int2 RTN b128` produced a 173,532,833 byte model, but both Python ONNX Runtime
and Deno `onnxruntime-web` WASM failed at runtime with
`MatMulNBits<MLFloat16> ... nbits_ == 8 was false`. It is not browser-usable in
the current stack.

`block_size=512` is not supported by the ONNX Runtime NBits quantizer used here.
The quantizer reported that only 16, 32, 64, 128, and 256 are supported.

## Runtime Static Quantization

Converting a raw AIVMX to static int8 immediately before inference was tested
with Python/uv tooling, one calibration sentence, and DeBERTa int4 RTN b256.

| phase                                           |           result |
| ----------------------------------------------- | ---------------: |
| quantization elapsed time                       |           74.1 s |
| quantization peak RSS                           |     2,640,284 KB |
| output static int8 model size                   | 83,104,945 bytes |
| generated-model inference peak RSS              |     1,458,576 KB |
| fp32 AIVMX inference peak RSS with same DeBERTa |     1,763,824 KB |

Runtime static quantization only helps after the generated model is cached and
reused. It is not a startup memory optimization.

## WebGPU Status

Deno WebGPU could not be measured in this environment. `onnxruntime-web`
reported `webgpu backend not found`, even with `--unstable-webgpu`. Validate
provider behavior and VRAM separately in the browser lab (`examples/browser`,
WASM / WebGPU selectable), or via the node CLI with `--device webgpu` (ORT's
native WebGPU EP) on a WebGPU-capable machine.
