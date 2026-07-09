"""aivmx（VITS 音響モデル）を静的 int8（QLinearConv）に量子化して軽量化する。

## 背景（データ取りの結論）

aivmx は 258MB 中 234MB(95%) が Conv 重み（DeBERTa と違い MatMul はごく僅か）。そのため
DeBERTa で使った NBits も dynamic-MatMul int8 も無力で、**Conv を量子化する必要がある**。
3 手法を実測（mao.aivmx, 2 文, int4 BERT, onnxruntime-web WASM single-thread）:

| 手法 | サイズ | 速度(文) | WASM | 判定 |
|---|---|---|---|---|
| fp32（原本） | 258MB | ~5s | ○ | 基準 |
| dynamic int8（ConvInteger） | 81MB | ~30s | ○ | 小だが **~6x 遅**（WASM の ConvInteger 未最適化＋動的量子化オーバーヘッド） |
| **static int8（QLinearConv・本スクリプト）** | **83MB** | **~4s** | ○ | **3.1x 小＋fp32 より速い＝採用候補** |
| fp16（onnxconverter_common） | 134MB | — | **×** | `/enc_p/Cast_1` の型不整合でロード不可＝turnkey でない |

→ 静的 int8（QLinearConv）だけが「小さく・速く・WASM で動く」。速度差は dynamic 固有で、
static は WASM の最適化 int8 conv カーネルに乗るため fp32 より速い。

## 実装メモ

- **Conv のみ量子化**（`op_types_to_quantize=["Conv"]`）: サイズの 95% は Conv。attention の
  `attn_layers.*/Add` 出力を量子化しようとすると "unknown initializers/tensors" で落ちるため、
  Conv に限定して回避する（品質面でも attention は fp32 のまま残す方が無難）。
- **`quant_pre_process` が前提**: 静的量子化はシンボリック shape 推論＋最適化を先に要求する。
- **キャリブレーション入力は実推論と同じ 11 テンソル**を synth_aivmx の組み立てで作る
  （x_tst/tones/bert/style_vec/スカラー…）。bert は実 DeBERTa（既定 fp16 ONNX）。
- **metadata_props（aivm_manifest/hyper_parameters/style_vectors）を量子化後グラフへ再付与**
  （量子化で落ちるため。synth 側は aivmlib でここから style を読む）。
- ライセンス: aivmx 本体は ACML 1.0。量子化派生をローカル利用する分には問題ないが、
  再配布する場合は元 aivmx の ACML 条項に従う（本スクリプトはローカル軽量化用）。

使い方:
    uv run python scripts/quantize_aivmx.py --aivmx ../../data/models/aivmx/mao.aivmx
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# import 副作用で HF_HOME 等を data/hf-cache に固定する（他 HF import より前）。
from _paths import DATA_DIR, MODELS_DIR
import synth_aivmx as SA

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

DEFAULT_CALIB_INPUT = DATA_DIR / "bench" / "inputs" / "listen10_given.jsonl"


class AivmxCalibrationReader(CalibrationDataReader):
    """aivmx の 11 入力テンソルを synth_aivmx の組み立てで yield する。"""

    def __init__(self, feeds: list[dict[str, np.ndarray]]) -> None:
        self._it = iter(feeds)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._it, None)


def _build_calibration_feeds(
    aivmx_path: Path, calib_input: Path, num_calib: int, style_id: int, bert_onnx_path: Path | None
) -> list[dict[str, np.ndarray]]:
    """キャリブ用の feed（実推論と同一の 11 テンソル）を num_calib 文ぶん作る。"""
    records = SA._load_input(calib_input)[:num_calib]
    if not records:
        raise ValueError(f"キャリブ入力に有効な文が無い: {calib_input}")

    SA.load_bert_tokenizer()
    extractor = SA.DebertaBertExtractor(onnx_path=bert_onnx_path)
    symbol_to_id = SA._build_symbol_to_id()
    style_vec = SA._style_vector(SA._load_style_matrix(aivmx_path), style_id, 1.0)
    scalars = {
        "length_scale": np.array(SA.DEFAULT_LENGTH_SCALE, dtype=np.float32),
        "sdp_ratio": np.array(SA.DEFAULT_SDP_RATIO, dtype=np.float32),
        "noise_scale": np.array(SA.DEFAULT_NOISE_SCALE, dtype=np.float32),
        "noise_scale_w": np.array(SA.DEFAULT_NOISE_SCALE_W, dtype=np.float32),
    }

    feeds: list[dict[str, np.ndarray]] = []
    for rec in records:
        phone_ids, tone_ids, lang_ids = SA._phones_tones_to_model_ids(
            rec["phones"], rec["tones"], symbol_to_id
        )
        seq_len = len(phone_ids)
        norm_text, word2ph = SA._word2ph_for_bert(rec["text"])
        bert = extractor.extract(norm_text, word2ph)
        if bert.shape[1] != seq_len:
            raise ValueError(f"bert 長 {bert.shape[1]} != 音素列長 {seq_len}（{rec['text']!r}）")
        feeds.append(
            {
                "x_tst": np.array([phone_ids], dtype=np.int64),
                "x_tst_lengths": np.array([seq_len], dtype=np.int64),
                "sid": np.array([0], dtype=np.int64),
                "tones": np.array([tone_ids], dtype=np.int64),
                "language": np.array([lang_ids], dtype=np.int64),
                "bert": bert[np.newaxis, :, :].astype(np.float32),
                "style_vec": style_vec,
                **scalars,
            }
        )
    return feeds


def _copy_metadata(src_path: Path, dst_path: Path) -> list[str]:
    """量子化で落ちた metadata_props を原本からコピーする（in-place で dst を書き換え）。"""
    src = onnx.load(str(src_path))
    dst = onnx.load(str(dst_path))
    del dst.metadata_props[:]
    keys: list[str] = []
    for mp in src.metadata_props:
        nmp = dst.metadata_props.add()
        nmp.key = mp.key
        nmp.value = mp.value
        keys.append(mp.key)
    onnx.save(dst, str(dst_path))
    return keys


def quantize(
    aivmx_path: Path,
    out_dir: Path,
    *,
    calib_input: Path,
    num_calib: int,
    style_id: int,
    bert_onnx_path: Path | None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    src_bytes = aivmx_path.stat().st_size

    pre_path = out_dir / "_pre.onnx"
    out_path = out_dir / f"{aivmx_path.stem}_int8_static.onnx"

    t0 = time.time()
    # 静的量子化はシンボリック shape 推論＋最適化を前提とする。
    quant_pre_process(str(aivmx_path), str(pre_path), skip_symbolic_shape=False)
    pre_sec = time.time() - t0

    feeds = _build_calibration_feeds(aivmx_path, calib_input, num_calib, style_id, bert_onnx_path)

    t1 = time.time()
    quantize_static(
        model_input=str(pre_path),
        model_output=str(out_path),
        calibration_data_reader=AivmxCalibrationReader(feeds),
        quant_format=QuantFormat.QOperator,  # QLinearConv 直接（WASM 最適化経路）
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        op_types_to_quantize=["Conv"],  # サイズの 95%。attention は fp32 のまま残す。
    )
    quantize_sec = time.time() - t1
    pre_path.unlink()

    metadata_keys = _copy_metadata(aivmx_path, out_path)

    from collections import Counter

    node_types = Counter(n.op_type for n in onnx.load(str(out_path)).graph.node)

    # スモーク: CPU EP でロードできることを確定させる（fail loudly）。
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]

    out_bytes = out_path.stat().st_size
    return {
        "aivmx_path": str(aivmx_path),
        "aivmx_file_bytes": src_bytes,
        "quantized_model_path": str(out_path),
        "quantized_file_bytes": out_bytes,
        "size_ratio_over_fp32": round(out_bytes / src_bytes, 4),
        "size_reduction_percent": round((1 - out_bytes / src_bytes) * 100, 2),
        "qlinearconv_nodes": node_types.get("QLinearConv", 0),
        "remaining_conv_nodes": node_types.get("Conv", 0),
        "convtranspose_nodes": node_types.get("ConvTranspose", 0),
        "method": (
            "quant_pre_process -> quantize_static(QuantFormat.QOperator, per_channel, "
            "weight=QInt8, activation=QUInt8, op_types=['Conv'])"
        ),
        "num_calibration_sentences": len(feeds),
        "calibration_input": str(calib_input),
        "calibration_bert": str(bert_onnx_path) if bert_onnx_path else "fp16(HF)",
        "metadata_props_copied": metadata_keys,
        "model_input_names": input_names,
        "quant_pre_process_sec": round(pre_sec, 2),
        "quantize_sec": round(quantize_sec, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aivmx",
        type=Path,
        default=MODELS_DIR / "aivmx" / "mao.aivmx",
        help="量子化する aivmx（既定: data/models/aivmx/mao.aivmx）",
    )
    parser.add_argument(
        "--calib-input",
        type=Path,
        default=DEFAULT_CALIB_INPUT,
        help=f"キャリブ用 JSONL（text/phones/tones。既定: {DEFAULT_CALIB_INPUT}）",
    )
    parser.add_argument("--num-calib", type=int, default=5, help="キャリブ文数（既定: 5）")
    parser.add_argument("--style", type=int, default=0, help="キャリブに使うスタイル（既定: 0）")
    parser.add_argument(
        "--bert-onnx-path",
        type=Path,
        default=None,
        help="キャリブの bert に使う DeBERTa ONNX（既定: HF fp16。配置に合わせ int4 を渡すのも可）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=MODELS_DIR / "aivmx-int8-static",
        help="出力先（既定: data/models/aivmx-int8-static）",
    )
    args = parser.parse_args()

    try:
        result = quantize(
            args.aivmx,
            args.out_dir,
            calib_input=args.calib_input,
            num_calib=args.num_calib,
            style_id=args.style,
            bert_onnx_path=args.bert_onnx_path,
        )
    except Exception as error:  # noqa: BLE001 - 失敗内容を summary で surface する
        json.dump(
            {"status": "failed", "error_type": type(error).__name__, "error_message": str(error)},
            sys.stdout,
            ensure_ascii=False,
            indent=2,
        )
        sys.stdout.write("\n")
        raise

    json.dump({"status": "ok", **result}, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
