"""DeBERTa（ja_bert）の int8 動的量子化と軽量化検討。

本構成は fp16 DeBERTa（tsukumijima/deberta-v2-large-japanese-char-wwm-onnx,
model_fp16.onnx, 653MB）で既に動作する（数値パリティ検証済み）。本スクリプトは
「軽量化の検討」として onnxruntime.quantization.quantize_dynamic で int8 化し、
ファイルサイズを実測する。既定出荷は fp16 のまま、int8 は検証済みオプション。

## fp16 → int8 の手順（実機で確認した制約）

fp16 ONNX に quantize_dynamic を直接かけると変換自体は成功するが、生成モデルの
DequantizeLinear が weight_scale を fp16 のまま持ち、onnxruntime のロードで
`INVALID_GRAPH: Type Error: Type 'tensor(float16)' ... is invalid` になる
（QDQ の scale は float32 を要求するため）。よって:

  1. fp16 ONNX を fp32 に変換する（initializer の実データ、value_info の型宣言、
     Cast ノードの `to` 属性の 3 箇所を fp16→fp32 に書き換える。onnx / onnxruntime
     どちらにも逆方向 [float16→float32] の汎用ヘルパーは無かったため、
     onnx.numpy_helper で自前実装する）。
  2. fp32 モデルに quantize_dynamic(weight_type=QInt8) をかける。

失敗時（op 非対応等）は対症フォールバックで黙って別物にせず、失敗内容をそのまま
stdout の summary JSON と例外で surface する（fail loudly）。

## 帰属義務（docs/license-audit.md 参照）

DeBERTa は CC-BY-SA-4.0（原著 ku-nlp、ONNX 化 tsukumijima）。本スクリプトが出す
int8 モデルは「動的量子化した派生」であり、表示(BY)+継承(SA) の下で LICENSE 全文と
NOTICE（改変明示）を出力ディレクトリに同梱する（このスクリプトが自動生成する）。

使い方:
    uv run python scripts/quantize_deberta.py
    uv run python scripts/quantize_deberta.py --out-dir /path/to/out
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# import 副作用で HF_HOME 等を data/hf-cache に固定する（他 HF import より前）。
from _paths import DATA_DIR, MODELS_DIR  # noqa: F401  (DATA_DIR は将来利用に備え保持)
from _quant_common import (
    DEBERTA_BASE_REPO,
    DEBERTA_ONNX_REPO,
    fetch_fp16_model_path,
    write_cc_by_sa_license,
)

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

DEFAULT_OUT_DIR = MODELS_DIR / "deberta-int8"


def _convert_fp16_to_fp32(model: onnx.ModelProto) -> dict[str, int]:
    """fp16 ONNX を fp32 に変換する（in-place）。書き換え箇所数を返す（診断用）。

    onnx / onnxruntime のどちらにも float16→float32 の汎用変換ヘルパーが無かった
    （transformers.float16 は逆方向の convert_float_to_float16 のみ）ため、
    onnx.numpy_helper で initializer の実データを変換し、value_info の型宣言と
    Cast ノードの to 属性も揃えて書き換える自前実装。
    """
    converted_initializers = 0
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(arr, name=init.name))
            converted_initializers += 1

    converted_value_infos = 0
    for vinfo in (
        list(model.graph.value_info)
        + list(model.graph.input)
        + list(model.graph.output)
    ):
        if vinfo.type.tensor_type.elem_type == TensorProto.FLOAT16:
            vinfo.type.tensor_type.elem_type = TensorProto.FLOAT
            converted_value_infos += 1

    converted_cast_nodes = 0
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
                    converted_cast_nodes += 1

    return {
        "initializers": converted_initializers,
        "value_infos": converted_value_infos,
        "cast_nodes": converted_cast_nodes,
    }


def _write_license_and_notice(out_dir: Path) -> None:
    """CC-BY-SA-4.0 全文（LICENSE）と改変明示（NOTICE）を出力先に書く。

    docs/license-audit.md の方針: int8 化は「動的量子化した派生」として
    表示(BY)+継承(SA) の義務を満たす（原著者帰属・ライセンス全文・改変明示・
    派生も同ライセンスで提供）。LICENSE 全文取得は _quant_common に集約している。
    """
    write_cc_by_sa_license(out_dir)

    notice_path = out_dir / "NOTICE"
    notice_path.write_text(
        "NOTICE — DeBERTa int8 量子化モデルの出典とライセンス\n"
        "\n"
        f"本モデル（model_int8.onnx）は以下の原著作物の派生物です。\n"
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
        "1. fp16 の重み・型宣言を fp32 に変換（onnx.numpy_helper による書き換え）。\n"
        "2. onnxruntime.quantization.quantize_dynamic（weight_type=QInt8）で動的量子化。\n"
        "\n"
        "この派生物（model_int8.onnx）も同じ CC BY-SA 4.0 の下で提供します。\n"
        "同梱の LICENSE ファイルにライセンス全文を掲載しています。\n",
        encoding="utf-8",
    )


def quantize(out_dir: Path) -> dict[str, object]:
    """fp16 ONNX を取得し、fp32 経由で int8 に量子化する。結果メタデータを返す。

    失敗時は例外を送出する（fail loudly。呼び出し側で summary に失敗内容を残す）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fp16_path = fetch_fp16_model_path()
    fp16_bytes = fp16_path.stat().st_size

    t0 = time.time()
    model = onnx.load(str(fp16_path))
    ir_version = model.ir_version
    opset_versions = [op.version for op in model.opset_import]

    fp16_initializer_count = sum(
        1 for init in model.graph.initializer if init.data_type == TensorProto.FLOAT16
    )
    if fp16_initializer_count == 0:
        raise RuntimeError(
            "fp16 initializer が 0 件。取得元モデルが fp16 でない可能性がある"
            "（DEBERTA_ONNX_FILE の想定と齟齬。決め打ちせず要確認）。"
        )

    fp32_convert_counts = _convert_fp16_to_fp32(model)
    fp32_path = out_dir / "_model_fp32_intermediate.onnx"
    onnx.save(model, str(fp32_path))
    fp32_convert_sec = time.time() - t0

    # fp32 が壊れていないことをロードで確認する（量子化前に fail loudly させる）。
    so = ort.SessionOptions()
    fp32_session = ort.InferenceSession(
        str(fp32_path), sess_options=so, providers=["CPUExecutionProvider"]
    )
    fp32_input_names = [i.name for i in fp32_session.get_inputs()]
    fp32_output_names = [o.name for o in fp32_session.get_outputs()]
    del fp32_session  # メモリ解放（fp32 は 1.2GB超）

    t1 = time.time()
    int8_path = out_dir / "model_int8.onnx"
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    quantize_sec = time.time() - t1

    # 中間 fp32 ファイルは検証専用（1.2GB超）。int8 生成後は不要なので削除する。
    fp32_path.unlink()

    # int8 モデルのロード・shape 確認（呼び出し側のスモークで実文を流すが、ここでも
    # 「ロードできる」ことだけは量子化直後に確定させる）。
    int8_bytes = int8_path.stat().st_size
    so2 = ort.SessionOptions()
    int8_session = ort.InferenceSession(
        str(int8_path), sess_options=so2, providers=["CPUExecutionProvider"]
    )
    int8_input_names = [i.name for i in int8_session.get_inputs()]
    int8_output_names = [o.name for o in int8_session.get_outputs()]
    del int8_session

    _write_license_and_notice(out_dir)

    return {
        "fp16_model_path": str(fp16_path),
        "fp16_file_bytes": fp16_bytes,
        "int8_model_path": str(int8_path),
        "int8_file_bytes": int8_bytes,
        "size_ratio_int8_over_fp16": round(int8_bytes / fp16_bytes, 4),
        "size_reduction_percent": round((1 - int8_bytes / fp16_bytes) * 100, 2),
        "method": (
            "1) fp16→fp32変換(onnx.numpy_helperで自前実装、"
            "onnx/onnxruntimeに逆方向の汎用ヘルパーなし) "
            "2) onnxruntime.quantization.quantize_dynamic(weight_type=QInt8)"
        ),
        "fp16_to_fp32_required": True,
        "fp16_to_fp32_reason": (
            "quantize_dynamicはfp16 ONNXを変換自体はできるが、"
            "生成されるDequantizeLinearのweight_scaleがfloat16のままになり、"
            "onnxruntimeロード時にINVALID_GRAPH(Type Error)で失敗する実測を確認"
        ),
        "fp32_convert_counts": fp32_convert_counts,
        "fp16_ir_version": ir_version,
        "fp16_opset_versions": opset_versions,
        "fp32_convert_sec": round(fp32_convert_sec, 2),
        "quantize_sec": round(quantize_sec, 2),
        "fp32_input_names": fp32_input_names,
        "fp32_output_names": fp32_output_names,
        "int8_input_names": int8_input_names,
        "int8_output_names": int8_output_names,
        "license_path": str(out_dir / "LICENSE"),
        "notice_path": str(out_dir / "NOTICE"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"int8 ONNX・LICENSE・NOTICE の出力先（既定: {DEFAULT_OUT_DIR}）",
    )
    args = parser.parse_args()

    try:
        result = quantize(args.out_dir)
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
