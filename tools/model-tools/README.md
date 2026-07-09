# model-tools

uv-managed Python tools for model conversion, quantization, and HuggingFace
packaging.

## DeBERTa quantization

```sh
uv run python scripts/quantize_deberta.py --out-dir ../../data/models/deberta-int8
uv run python scripts/quantize_deberta_int4.py --bits 8 --algo rtn --out-dir ../../data/models/deberta-int8-rtn
uv run python scripts/quantize_deberta_int4.py --bits 4 --algo rtn --out-dir ../../data/models/deberta-int4-rtn
uv run python scripts/quantize_deberta_int4.py --bits 4 --algo hqq --out-dir ../../data/models/deberta-int4-hqq
```

`quantize_deberta.py` uses fp16 -> fp32 -> dynamic QInt8.
`quantize_deberta_int4.py` uses ONNX Runtime's `MatMulNBitsQuantizer` directly
on fp16 ONNX for weight-only 4/8-bit models.

## AIVMX acoustic quantization

```sh
uv run python scripts/quantize_aivmx.py \
  --aivmx ../../data/models/aivmx/mao.aivmx \
  --out-dir ../../data/models/aivmx-int8-static
```

This produces static int8 `QLinearConv` acoustic ONNX and copies AIVMX metadata
back into the quantized model.

## HuggingFace package preparation

```sh
uv run python scripts/prepare_hf_deberta.py \
  --model-dir ../../data/models/deberta-int4-rtn \
  --tokenizer-dir ../../data/models/deberta-tokenizer \
  --out-dir ../../data/hf-packages/deberta-int4-rtn
```

Add `--repo-id owner/name --upload` to create/upload with `huggingface_hub`. A
token must be configured in the environment for upload.
