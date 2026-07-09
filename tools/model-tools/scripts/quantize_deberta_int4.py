"""DeBERTa（ja_bert）の weight-only NBits 量子化（int4 主体・int8 も可）。

aivmx + 量子化 DeBERTa の構成を「large を保ったまま容量を詰める」方向で検証する
ためのツール。int8（quantize_deberta.py・quantize_dynamic QDQ）に対し、本スクリプトは
onnxruntime の MatMulNBitsQuantizer で **重みだけを 4/8-bit** にブロック量子化する
（線形層 = 定数 B を持つ MatMul のみが対象。attention の activation×activation は非対象）。

## fp16 を「直接」量子化する（int8 QDQ とは手順が違う。実測で確定）

quantize_deberta.py（int8 QDQ）は fp16→fp32 変換が必須だった（QDQ の DequantizeLinear の
weight_scale が fp16 だと onnxruntime ロードで INVALID_GRAPH になるため）。一方 NBits は
**fp16 モデルを直接量子化して CPU で動く**（4/8-bit）。これが重要で、非量子化重み
（埋め込み表・LayerNorm 等）が **fp16 のまま残る**ため、量子化後サイズが「実配置サイズ」
に一致する（fp32 経由だと非量子化部が 2 倍に膨れて過大評価になる）。実測（10 文）:

    bits=8 block=32 → ~402MB  cos≈1.000（ほぼ無損失）
    bits=4 block=32 → ~260MB  cos≈0.94（平均）/ 0.89（最悪文）
    bits=4 block=128→ ~243MB  cos≈0.93
    bits=2          → CPU の MatMulNBits(fp16) カーネルが nbits!=8 で失敗／fp32 経由でも cos≈0.55 で崩壊

このため本スクリプトは **4/8-bit を対象**とする（2-bit は品質・CPU カーネル両面で非対応。
指定時は生成後の CPU ロード検証（fail loudly）でエラーを surface する）。

## 帰属義務（docs/license-audit.md 参照）

DeBERTa は CC-BY-SA-4.0（原著 ku-nlp、ONNX 化 tsukumijima）。量子化派生は表示(BY)+継承(SA)
の下で LICENSE 全文と NOTICE（改変明示）を出力先に同梱する（LICENSE 取得は _quant_common）。

使い方:
    uv run python scripts/quantize_deberta_int4.py                        # int4 RTN block32
    uv run python scripts/quantize_deberta_int4.py --bits 4 --algo hqq --block-size 32
    uv run python scripts/quantize_deberta_int4.py --bits 8 --block-size 32   # NBits int8
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path

# import 副作用で HF_HOME 等を data/hf-cache に固定する（他 HF import より前）。
from _paths import MODELS_DIR
from _quant_common import (
    DEBERTA_BASE_REPO,
    DEBERTA_ONNX_REPO,
    fetch_fp16_model_path,
    write_cc_by_sa_license,
)

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto
from onnxruntime.quantization.matmul_nbits_quantizer import (
    HQQWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)

# 量子化のノード単位ログ・プログレスバーは stdout を汚す（summary JSON 専用にしたい）ため
# stderr へ退避する。ログレベルも WARNING に落とす。
logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)

# CPU ロード検証で流す固定文（実文で「動く」ことまで確認する。感情等は問わない）。
SMOKE_TEXT = "こんにちは、今日はいい天気ですね。感情を込めて話します。"


def _quantized_bytes_breakdown(model_path: Path) -> dict[str, int]:
    """保存済みモデルの initializer を量子化重み(uint8/int8)と非量子化(その他)で分ける。"""
    graph = onnx.load(str(model_path)).graph
    quant_bytes = 0
    nonquant_bytes = 0
    for init in graph.initializer:
        nbytes = len(init.raw_data) if init.raw_data else 0
        if init.data_type in (TensorProto.UINT8, TensorProto.INT8):
            quant_bytes += nbytes
        else:
            nonquant_bytes += nbytes
    return {"quantized_weight_bytes": quant_bytes, "nonquantized_bytes": nonquant_bytes}


def _count_op(model_path: Path, op_type: str) -> int:
    graph = onnx.load(str(model_path)).graph
    return sum(1 for node in graph.node if node.op_type == op_type)


def _write_notice(out_dir: Path, *, model_file: str, bits: int, algo: str, block_size: int) -> Path:
    """NBits 量子化派生の NOTICE（改変明示）を書く。手順は bits/algo で変わるため個別実装。"""
    notice_path = out_dir / "NOTICE"
    notice_path.write_text(
        f"NOTICE — DeBERTa int{bits} 量子化モデルの出典とライセンス\n"
        "\n"
        f"本モデル（{model_file}）は以下の原著作物の派生物です。\n"
        "\n"
        f"- 原著モデル: {DEBERTA_BASE_REPO}\n"
        f"  https://huggingface.co/{DEBERTA_BASE_REPO}\n"
        f"- ONNX 化: {DEBERTA_ONNX_REPO}（model_fp16.onnx）\n"
        f"  https://huggingface.co/{DEBERTA_ONNX_REPO}\n"
        "- ライセンス: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)\n"
        "  https://creativecommons.org/licenses/by-sa/4.0/\n"
        "\n"
        "## 改変の明示\n"
        "\n"
        "上記 fp16 ONNX を次の手順で改変しました:\n"
        "1. onnxruntime.quantization の MatMulNBitsQuantizer による weight-only "
        f"{bits}-bit ブロック量子化（algo={algo}, block_size={block_size}）。\n"
        "   線形層（定数重みを持つ MatMul）のみを量子化し、非量子化重みは fp16 のまま残す。\n"
        "\n"
        f"この派生物（{model_file}）も同じ CC BY-SA 4.0 の下で提供します。\n"
        "同梱の LICENSE ファイルにライセンス全文を掲載しています。\n",
        encoding="utf-8",
    )
    return notice_path


def _smoke_run(model_path: Path) -> dict[str, object]:
    """生成モデルを CPU EP でロードし、実文 1 本を forward する（fail loudly）。

    int2 の fp16 CPU カーネル非対応など「量子化は成功するが実行できない」失敗を
    ここで例外として surface する（対症フォールバックしない）。
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_ONNX_REPO)
    enc = tokenizer(SMOKE_TEXT, return_tensors="np")

    session = ort.InferenceSession(
        str(model_path), sess_options=ort.SessionOptions(), providers=["CPUExecutionProvider"]
    )
    input_names = [i.name for i in session.get_inputs()]
    feed = {name: enc[name].astype(np.int64) for name in input_names if name in enc}
    missing = set(input_names) - set(feed)
    if missing:
        raise ValueError(
            f"量子化モデルの入力 {missing} をトークナイザ出力から供給できない"
            f"（tokenizer keys={list(enc.keys())}, model inputs={input_names}）"
        )
    (out,) = session.run(None, feed)  # [seq_len, 1024]
    if out.ndim != 2 or out.shape[1] != 1024:
        raise ValueError(f"量子化モデルの出力 shape が想定外: {out.shape}（[seq_len, 1024] を期待）")
    return {
        "smoke_output_shape": list(out.shape),
        "smoke_output_finite": bool(np.isfinite(out).all()),
        "model_input_names": input_names,
        "model_output_names": [o.name for o in session.get_outputs()],
    }


def quantize(
    out_dir: Path, *, bits: int, block_size: int, algo: str, symmetric: bool, accuracy_level: int
) -> dict[str, object]:
    """fp16 ONNX を直接 NBits 量子化し、CPU ロード検証まで通す。結果メタデータを返す。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    fp16_path = fetch_fp16_model_path()
    fp16_bytes = fp16_path.stat().st_size

    t0 = time.time()
    model = onnx.load(str(fp16_path))

    # algo_config を渡すと constructor の bits/block_size より algo 側の既定が優先される罠が
    # あるため、HQQ は明示的に bits/block_size を渡し、RTN は algo_config=None（＝constructor
    # の bits/block_size を尊重する既定量子化器）にする。
    if algo == "hqq":
        algo_config: HQQWeightOnlyQuantConfig | None = HQQWeightOnlyQuantConfig(
            block_size=block_size, bits=bits
        )
    elif algo == "rtn":
        algo_config = None
    else:
        raise ValueError(f"未知の --algo: {algo}（rtn か hqq）")

    quantizer = MatMulNBitsQuantizer(
        model,
        bits=bits,
        block_size=block_size,
        is_symmetric=symmetric,
        accuracy_level=accuracy_level,
        algo_config=algo_config,
    )
    # プログレスバー・ノードログは stderr へ（stdout は summary JSON 専用）。
    with contextlib.redirect_stdout(sys.stderr):
        quantizer.process()

    model_file = f"model_int{bits}.onnx"
    out_path = out_dir / model_file
    quantizer.model.save_model_to_file(str(out_path), use_external_data_format=False)
    quantize_sec = time.time() - t0

    out_bytes = out_path.stat().st_size
    breakdown = _quantized_bytes_breakdown(out_path)
    matmul_nbits_nodes = _count_op(out_path, "MatMulNBits")
    remaining_matmul = _count_op(out_path, "MatMul")

    smoke = _smoke_run(out_path)

    write_cc_by_sa_license(out_dir)
    _write_notice(out_dir, model_file=model_file, bits=bits, algo=algo, block_size=block_size)

    return {
        "fp16_model_path": str(fp16_path),
        "fp16_file_bytes": fp16_bytes,
        "quantized_model_path": str(out_path),
        "quantized_file_bytes": out_bytes,
        "size_ratio_over_fp16": round(out_bytes / fp16_bytes, 4),
        "size_reduction_percent": round((1 - out_bytes / fp16_bytes) * 100, 2),
        "bits": bits,
        "block_size": block_size,
        "algo": algo,
        "is_symmetric": symmetric,
        "accuracy_level": accuracy_level,
        "matmulnbits_node_count": matmul_nbits_nodes,
        "remaining_plain_matmul_count": remaining_matmul,
        **breakdown,
        "quantize_sec": round(quantize_sec, 2),
        "method": (
            "onnxruntime.quantization.matmul_nbits_quantizer.MatMulNBitsQuantizer "
            f"(weight-only {bits}-bit, algo={algo}, block_size={block_size}, "
            f"symmetric={symmetric}, accuracy_level={accuracy_level}); fp16 を直接量子化"
        ),
        **smoke,
        "license_path": str(out_dir / "LICENSE"),
        "notice_path": str(out_dir / "NOTICE"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, default=4, choices=(4, 8), help="量子化ビット数（既定: 4）")
    parser.add_argument("--block-size", type=int, default=32, help="ブロックサイズ（既定: 32）")
    parser.add_argument(
        "--algo", default="rtn", choices=("rtn", "hqq"), help="量子化アルゴリズム（既定: rtn）"
    )
    parser.add_argument(
        "--symmetric", action="store_true", help="対称量子化（既定: 非対称。非対称の方が高精度）"
    )
    parser.add_argument(
        "--accuracy-level",
        type=int,
        default=1,
        help="MatMulNBits の計算精度（1=fp32 accumulate。重み量子化誤差だけを見るため既定 1）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="ONNX・LICENSE・NOTICE の出力先（既定: data/models/deberta-int<bits>-<algo>）",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir or (MODELS_DIR / f"deberta-int{args.bits}-{args.algo}")

    try:
        result = quantize(
            out_dir,
            bits=args.bits,
            block_size=args.block_size,
            algo=args.algo,
            symmetric=args.symmetric,
            accuracy_level=args.accuracy_level,
        )
    except Exception as error:  # noqa: BLE001 - 失敗内容を summary で surface する
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        json.dump(failure, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        raise

    summary = {"status": "ok", **result}
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
