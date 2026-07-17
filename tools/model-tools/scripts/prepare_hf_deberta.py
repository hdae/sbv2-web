"""Prepare or upload a quantized DeBERTa ONNX folder for HuggingFace Hub.

This script packages a quantized model directory produced by quantize_deberta.py or
quantize_deberta_int4.py with tokenizer assets, README metadata, LICENSE, NOTICE,
and a package manifest. It can optionally upload the folder to a private or
public HuggingFace Hub model repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi

DEBERTA_BASE_REPO = "ku-nlp/deberta-v2-large-japanese-char-wwm"
DEBERTA_ONNX_REPO = "tsukumijima/deberta-v2-large-japanese-char-wwm-onnx"

# Keys mirror src/assets/deberta.ts PINNED_FILES. Tokenizer asset names are fixed;
# the model uses the packaged filename (default model.onnx).
_PIN_TOKENIZER_FILES = {
    "vocab": "vocab.txt",
    "cleanRanges": "clean_ranges.json",
    "meta": "meta.json",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_pins(out_dir: Path, model_filename: str) -> dict[str, dict[str, object]]:
    """Compute (expectedBytes, sha256) for all four PINNED_FILES entries at once.

    Emitting the whole set from one place makes a partial pin update — e.g. bumping
    only model.onnx while a tokenizer asset silently changes — hard to do by hand.
    Pins are taken from the prepared out_dir, which is exactly what gets uploaded.
    """
    files = {"model": model_filename, **_PIN_TOKENIZER_FILES}
    pins: dict[str, dict[str, object]] = {}
    for key, name in files.items():
        p = out_dir / name
        if not p.exists():
            raise FileNotFoundError(f"cannot pin missing file: {p}")
        pins[key] = {
            "path": name,
            "expectedBytes": p.stat().st_size,
            "sha256": _sha256_file(p),
        }
    return pins


def _format_pins_ts(pins: dict[str, dict[str, object]]) -> str:
    """Render pins as a src/assets/deberta.ts PINNED_FILES block for copy-paste."""
    lines = ["const PINNED_FILES = {"]
    for key, pin in pins.items():
        lines.append(f"  {key}: {{")
        lines.append(f'    path: "{pin["path"]}",')
        lines.append(f"    expectedBytes: {pin['expectedBytes']},")
        lines.append(f'    sha256: "{pin["sha256"]}",')
        lines.append("  },")
    lines.append("} as const;")
    return "\n".join(lines)


def _copy_required(src: Path, dst: Path, names: list[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        p = src / name
        if not p.exists():
            raise FileNotFoundError(f"required file missing: {p}")
        shutil.copy2(p, dst / name)


def _find_model(model_dir: Path) -> Path:
    candidates = sorted(model_dir.glob("model_int*.onnx")) + sorted(model_dir.glob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"no ONNX model found in {model_dir}")
    return candidates[0]


def _variant_info(model_dir: Path) -> dict[str, str | float | int | None]:
    name = model_dir.name
    if name == "deberta-int8-rtn":
        return {
            "variant": "int8-nbits-rtn-b32",
            "quantization": "ONNX Runtime MatMulNBits weight-only quantization, 8-bit RTN, block_size=32",
            "recommended_use": "High-fidelity int8 baseline for JP-Extra browser and Deno inference.",
            "avg_cosine": 0.999904,
            "min_cosine": 0.999829,
            "wasm_synth_ms": 5311,
            "wasm_peak_rss_kb": 2290072,
        }
    if name == "deberta-int4-rtn":
        return {
            "variant": "int4-rtn-b32",
            "quantization": "ONNX Runtime MatMulNBits weight-only quantization, 4-bit RTN, block_size=32",
            "recommended_use": "Quality-sensitive int4 default for JP-Extra browser and Deno inference.",
            "avg_cosine": 0.974084,
            "min_cosine": 0.963577,
            "wasm_synth_ms": 5620,
            "wasm_peak_rss_kb": 1827232,
        }
    if name == "deberta-int4-rtn-b256":
        return {
            "variant": "int4-rtn-b256",
            "quantization": "ONNX Runtime MatMulNBits weight-only quantization, 4-bit RTN, block_size=256",
            "recommended_use": "Lowest measured RAM option; BERT feature similarity is lower than b32.",
            "avg_cosine": 0.951447,
            "min_cosine": 0.932897,
            "wasm_synth_ms": 5675,
            "wasm_peak_rss_kb": 1763824,
        }
    if name == "deberta-int4-hqq":
        return {
            "variant": "int4-hqq-b32",
            "quantization": "ONNX Runtime MatMulNBits weight-only quantization, 4-bit HQQ, block_size=32",
            "recommended_use": "Experimental int4 variant with higher cosine than RTN b32 but larger model size in local tests.",
            "avg_cosine": 0.977996,
            "min_cosine": 0.972342,
            "wasm_synth_ms": 5437,
            "wasm_peak_rss_kb": 1920780,
        }
    if name == "deberta-int8":
        return {
            "variant": "int8-dynamic-qlinear",
            "quantization": "ONNX Runtime dynamic QInt8 quantization",
            "recommended_use": "Fast int8 comparison point; feature similarity is lower than NBits int8.",
            "avg_cosine": 0.977453,
            "min_cosine": 0.974748,
            "wasm_synth_ms": 4801,
            "wasm_peak_rss_kb": 2152772,
        }
    return {
        "variant": name,
        "quantization": "Quantized ONNX DeBERTa derivative",
        "recommended_use": "Experimental quantized DeBERTa model.",
        "avg_cosine": None,
        "min_cosine": None,
        "wasm_synth_ms": None,
        "wasm_peak_rss_kb": None,
    }


def _write_notice(out_dir: Path, *, model_file: str, info: dict[str, object]) -> None:
    notice = f"""NOTICE

This repository contains {model_file}, a quantized derivative of the ONNX DeBERTa model used for Style-Bert-VITS2 / AivisSpeech JP-Extra BERT feature extraction.

Upstream sources:
- Original model: {DEBERTA_BASE_REPO}
  https://huggingface.co/{DEBERTA_BASE_REPO}
- ONNX source: {DEBERTA_ONNX_REPO} (model_fp16.onnx)
  https://huggingface.co/{DEBERTA_ONNX_REPO}

License:
- Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
  https://creativecommons.org/licenses/by-sa/4.0/

Modifications:
- Quantization: {info["quantization"]}
- Packaged for browser/Deno ONNX Runtime use with @hdae/sbv2-web.
- Included tokenizer assets are copied from the ONNX source repository for runtime compatibility.

This derivative is distributed under the same CC BY-SA 4.0 license. See LICENSE for the full license text.
"""
    (out_dir / "NOTICE").write_text(notice, encoding="utf-8")


def _write_readme(out_dir: Path, *, model_file: str, repo_id: str | None, info: dict[str, object]) -> None:
    readme = f"""---
license: cc-by-sa-4.0
library_name: onnxruntime
tags:
- deberta
- japanese
- onnx
- quantized
- style-bert-vits2
- aivisspeech
---

# Quantized DeBERTa v2 large Japanese char WWM ONNX ({info["variant"]})

This repository contains {model_file}, a quantized derivative of the ONNX DeBERTa model used by Style-Bert-VITS2 / AivisSpeech JP-Extra inference.

- Original model: [{DEBERTA_BASE_REPO}](https://huggingface.co/{DEBERTA_BASE_REPO})
- ONNX source: [{DEBERTA_ONNX_REPO}](https://huggingface.co/{DEBERTA_ONNX_REPO})
- Quantization: {info["quantization"]}
- License: CC BY-SA 4.0

The tokenizer assets (vocab.txt, clean_ranges.json, meta.json) are included for browser/Deno runtimes that use @hdae/sbv2-web.

## Intended Use

{info["recommended_use"]}

Load {model_file} as bertOnnxBytes when creating Sbv2ModelAdapter from @hdae/sbv2-web.

## Local Benchmark Snapshot

Benchmarks were measured with onnxruntime-web WASM and fp32 AIVMX unless otherwise noted.

| metric | value |
| --- | ---: |
| average cosine vs fp16 | {info["avg_cosine"]} |
| minimum cosine vs fp16 | {info["min_cosine"]} |
| synthesis time | {info["wasm_synth_ms"]} ms |
| peak RSS | {info["wasm_peak_rss_kb"]} KB |

See the sbv2-web benchmark document for the full matrix and caveats.

## Files

- {model_file}: quantized ONNX model
- vocab.txt, clean_ranges.json, meta.json: tokenizer/runtime assets
- LICENSE: CC BY-SA 4.0 license text
- NOTICE: attribution and modification notice
- package_manifest.json: packaging metadata
"""
    if repo_id:
        readme += f"\nTarget Hub repo: {repo_id}\n"
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def prepare(
    model_dir: Path,
    tokenizer_dir: Path,
    out_dir: Path,
    repo_id: str | None,
    model_filename: str,
) -> dict[str, object]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = _find_model(model_dir)
    info = _variant_info(model_dir)
    shutil.copy2(model_path, out_dir / model_filename)
    _copy_required(tokenizer_dir, out_dir, ["vocab.txt", "clean_ranges.json", "meta.json"])

    license_src = model_dir / "LICENSE"
    if license_src.exists():
        shutil.copy2(license_src, out_dir / "LICENSE")
    else:
        raise FileNotFoundError(f"required file missing: {license_src}")

    _write_notice(out_dir, model_file=model_filename, info=info)
    _write_readme(out_dir, model_file=model_filename, repo_id=repo_id, info=info)

    # 4 ファイル分のピン（bytes + sha256）を一括で計算し、片手落ちの更新をしにくくする。
    pins = _compute_pins(out_dir, model_filename)

    manifest = {
        "variant": info["variant"],
        "quantization": info["quantization"],
        "source_model_dir": str(model_dir),
        "source_model_file": model_path.name,
        "model_file": model_filename,
        "model_bytes": model_path.stat().st_size,
        "tokenizer_dir": str(tokenizer_dir),
        "repo_id": repo_id,
        "license": "CC-BY-SA-4.0",
        "base_repo": DEBERTA_BASE_REPO,
        "onnx_source_repo": DEBERTA_ONNX_REPO,
        "pins": pins,
        "benchmark": {
            "avg_cosine": info["avg_cosine"],
            "min_cosine": info["min_cosine"],
            "wasm_synth_ms": info["wasm_synth_ms"],
            "wasm_peak_rss_kb": info["wasm_peak_rss_kb"],
        },
    }
    (out_dir / "package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("../../data/models/deberta-tokenizer"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-filename", default="model.onnx")
    parser.add_argument("--repo-id", default=None, help="Optional HuggingFace repo id to upload to")
    parser.add_argument("--private", action="store_true", help="Create/upload as a private repo")
    parser.add_argument("--upload", action="store_true", help="Upload prepared folder to HuggingFace Hub")
    args = parser.parse_args()

    result = prepare(args.model_dir, args.tokenizer_dir, args.out_dir, args.repo_id, args.model_filename)
    if args.upload:
        if not args.repo_id:
            raise ValueError("--upload requires --repo-id")
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(args.out_dir), repo_id=args.repo_id, repo_type="model")
        result["uploaded"] = True
    else:
        result["uploaded"] = False
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # DEBERTA_REVISION 更新時に src/assets/deberta.ts へそのまま貼れる 4 ファイル分のピン。
    print("\n// paste into src/assets/deberta.ts (all four pins at once):")
    print(_format_pins_ts(result["pins"]))


if __name__ == "__main__":
    main()
