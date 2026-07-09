"""実 aivmx（ONNX 音響モデル）を onnxruntime で直接駆動して wav を出す。

本フロントエンドの phones/tones（given 経路と同じ JSONL）を、AivisSpeech 公式ハブの
公開 aivmx（Style-Bert-VITS2 JP-Extra ONNX）に通し、SBV2 の TTSModel を介さず
onnxruntime だけで波形まで持っていく。aivmx-interface.md §2 のアダプタ層を Python で
先行実装し、入力テンソル束（symbols→ID / tone+6 / add_blank / ja_bert / style_vec /
スカラー）が実モデルで通ることを実証するのが狙い。

## 実測で確定した aivmx 入力（まお/コハク共通, api.aivis-project.com）
onnxruntime の get_inputs() で読んだ 11 本。グラフ側の名前を真実源とし、決め打ちしない。

    x_tst          int64  [batch, T]      音素 ID 列
    x_tst_lengths  int64  [batch]         音素列長（★グラフに実在＝必須。想定の留保が解けた）
    sid            int64  [batch]         話者 ID（= マニフェスト話者 local_id）
    tones          int64  [batch, T]      トーン ID 列（JP は +6 オフセット済み）
    language       int64  [batch, T]      言語 ID 列（JP は全 1）
    bert           float  [batch, 1024, T] 日本語 BERT 特徴量（JP-Extra は BERT 入力 1 本）
    style_vec      float  [batch, 256]    スタイルベクトル
    length_scale   float  []              話速（★rank-0 スカラー）
    sdp_ratio      float  []              SDP/DP 混合比（rank-0）
    noise_scale    float  []              DP ノイズ（rank-0）
    noise_scale_w  float  []              SDP ノイズ（rank-0）

出力は主出力 'output' [dim, 1, dim] float32 波形（他に中間テンソルも公開されるが 'output' を使う）。

## ja_bert（bert 入力）の作り方
実 DeBERTa ONNX（tsukumijima/deberta-v2-large-japanese-char-wwm-onnx, model_fp16.onnx,
CC-BY-SA-4.0）を onnxruntime で回す。この ONNX は「最後から 3 番目の隠れ層」をグラフ内に
焼き込み済みで、単一出力 'output' [seq_len, 1024] をそのまま使う（output_hidden_states 不要）。
入力は input_ids / attention_mask の 2 本のみ（token_type_ids 不要）。文字トークンごとの
特徴量を word2ph に従って np.tile で音素レベルへ展開し、転置して [1024, T] にする
（SBV2 の extract_bert_feature_onnx と同じ規則。infer.py の word2ph 調整も再現する）。

  word2ph の出所（重要な留保）: word2ph は「BERT 用テキスト g2p」由来であり、本フロントエンドが
  出す phones/tones とは別レイヤー。ここでは SBV2 の clean_text(norm_text) から取得している
  （aivmx-interface.md §3.1 と整合）。ブラウザ移植では BERT を積む場合に別途 word2ph 生成が要る。

`--bert zero` を指定すると DeBERTa を一切走らせず bert をゼロテンソルにする（直駆動の疎通を
先に確立するための段階実装。zero の劣化度は別途聴取するので、まずは疎通を優先してよい）。

## symbols 表
音素→ID は SBV2 の SYMBOLS 配列全体（ZH+JP+EN を sorted した 112 要素）を使う。JP 42 音素だけで
index すると ID がずれる（aivmx-interface.md §2.4）。ここでは SBV2 の symbols.py から
import しているが、ブラウザ実装ではこの配列を移植する MUST（AGPL コードは配布物に混入させない）。

使い方:
    uv run python scripts/synth_aivmx.py --aivmx mao.aivmx
    uv run python scripts/synth_aivmx.py --aivmx mao.aivmx --style 2 --bert zero
    uv run python scripts/synth_aivmx.py --aivmx kohaku.aivmx --input /path/to.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

# import 副作用で HF_HOME 等を data/hf-cache に固定する（他 HF import より前）。
from _paths import DATA_DIR, HF_CACHE_DIR  # noqa: F401  (HF_CACHE_DIR は副作用固定用)
from _model import load_bert_tokenizer, measure, slugify

import numpy as np
import onnxruntime as ort
import soundfile as sf

# aivmx 置き場（data/ 配下・gitignore）。
AIVMX_DIR = DATA_DIR / "models" / "aivmx"
# wav・スモーク JSON 出力先。
AIVMX_OUT_DIR = DATA_DIR / "bench" / "aivmx"
# 実験の既定入力: 聴き比べ 10 文（両端 "_" 込みの完全な phones/tones）。
DEFAULT_INPUT = DATA_DIR / "bench" / "inputs" / "listen10_given.jsonl"

# 実 DeBERTa（ja_bert 用）。SBV2 fork / AivisSpeech-Engine と同じ ONNX リポジトリ。
DEBERTA_ONNX_REPO = "tsukumijima/deberta-v2-large-japanese-char-wwm-onnx"
DEBERTA_ONNX_FILE = "model_fp16.onnx"

# aivmx が要求する BERT 特徴量の次元（DeBERTa large の hidden size）。
BERT_DIM = 1024

# スカラー style パラメータの既定（aivmx-interface.md §2.5。AivisSpeech の実運用値）。
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_SDP_RATIO = 0.2
DEFAULT_NOISE_SCALE = 0.6
DEFAULT_NOISE_SCALE_W = 0.8


def _load_input(path: Path) -> list[dict]:
    """JSONL を読み、text/phones/tones を持つ行のみ検証して返す（given 経路踏襲・fail loudly）。"""
    if not path.exists():
        raise FileNotFoundError(f"--input が存在しない: {path}")
    records: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        obj = json.loads(stripped)
        for key in ("text", "phones", "tones"):
            if key not in obj:
                raise ValueError(f"入力 {path}:{lineno} に必須キー '{key}' が無い")
        if len(obj["phones"]) != len(obj["tones"]):
            raise ValueError(
                f"入力 {path}:{lineno} phones({len(obj['phones'])}) と "
                f"tones({len(obj['tones'])}) の長さが不一致"
            )
        records.append(obj)
    if not records:
        raise ValueError(f"--input {path} に有効な行が無い")
    return records


def _intersperse(seq: list[int], item: int) -> list[int]:
    """add_blank=True の後処理。要素間・両端に item を挟んで 2*len+1 にする。

    SBV2 commons.intersperse と同一。ブラウザ移植で必須の後処理（aivmx-interface.md §2.4）。
    """
    result = [item] * (len(seq) * 2 + 1)
    result[1::2] = seq
    return result


def _build_symbol_to_id() -> dict[str, int]:
    """SBV2 の SYMBOLS 配列（ZH+JP+EN sorted, 112 要素）から音素→ID を作る。

    JP 42 音素だけで index すると ID がずれるため配列全体を使う（aivmx-interface.md §2.4）。
    ブラウザ実装ではこの配列を移植する MUST。ここは SBV2 の symbols.py から取る
    （AGPL コードは配布物に混入させない。tools/ 内のローカル検証のみ）。
    """
    from style_bert_vits2.nlp.symbols import SYMBOLS

    return {symbol: index for index, symbol in enumerate(SYMBOLS)}


def _phones_tones_to_model_ids(
    phones: list[str], tones: list[int], symbol_to_id: dict[str, int]
) -> tuple[list[int], list[int], list[int]]:
    """phones/tones を aivmx 入力用の ID 列（add_blank 適用済み）に変換する。

    1) 音素→ID（SYMBOLS 表）/ トーン +6（JP オフセット）/ 言語 ID=1（JP）
    2) add_blank intersperse（phone/tone/language を 0 で挟んで 2*len+1）

    未知の音素は握りつぶさず KeyError を surface する（fail loudly）。
    """
    jp_tone_offset = 6  # LANGUAGE_TONE_START_MAP["JP"]（NUM_ZH_TONES）
    jp_lang_id = 1  # LANGUAGE_ID_MAP["JP"]

    try:
        phone_ids = [symbol_to_id[p] for p in phones]
    except KeyError as error:
        raise KeyError(
            f"SYMBOLS に無い音素 {error} が phones に含まれる（ID 化不能）。"
            " 本フロントエンドの音素記号と SBV2 symbols.py の齟齬を疑う。"
        ) from error
    tone_ids = [t + jp_tone_offset for t in tones]
    lang_ids = [jp_lang_id for _ in phone_ids]

    return (
        _intersperse(phone_ids, 0),
        _intersperse(tone_ids, 0),
        _intersperse(lang_ids, 0),
    )


def _load_style_matrix(aivmx_path: Path) -> np.ndarray:
    """aivmx の aivm_style_vectors（Base64 .npy）から [num_styles, 256] を取り出す。"""
    import aivmlib

    with open(aivmx_path, "rb") as f:
        meta = aivmlib.read_aivmx_metadata(f)
    style_bytes = meta.style_vectors
    if not isinstance(style_bytes, (bytes, bytearray)):
        raise TypeError(
            f"aivm_style_vectors が bytes でない: {type(style_bytes).__name__}"
            "（aivmlib の戻り値仕様変更を疑う）"
        )
    matrix = np.load(io.BytesIO(style_bytes))
    if matrix.ndim != 2 or matrix.shape[1] != 256:
        raise ValueError(f"style_vectors の shape が想定外: {matrix.shape}（[N, 256] を期待）")
    return matrix.astype(np.float32)


def _style_vector(style_matrix: np.ndarray, style_id: int, weight: float) -> np.ndarray:
    """スタイル行を選び mean + (row-mean)*weight で [1, 256] を作る（aivmx-interface.md §2.6）。"""
    if not (0 <= style_id < style_matrix.shape[0]):
        raise IndexError(
            f"--style {style_id} が範囲外（このモデルのスタイル数は {style_matrix.shape[0]}）"
        )
    mean = style_matrix.mean(axis=0)
    row = style_matrix[style_id]
    vec = mean + (row - mean) * weight
    return vec.reshape(1, 256).astype(np.float32)


class DebertaBertExtractor:
    """DeBERTa ONNX で ja_bert 特徴量 [1024, T] を作る（real 経路）。

    SBV2 の extract_bert_feature_onnx と同じ規則で、文字トークンごとの隠れ状態を
    word2ph に従って np.tile 展開 → 転置。word2ph は SBV2 の clean_text 由来
    （BERT 用テキスト g2p レイヤー。本フロントエンドの phones/tones とは別）。
    """

    def __init__(self, onnx_path: Path | None = None) -> None:
        """DeBERTa ONNX をロードする。

        onnx_path を指定するとその ONNX ファイル（例: int8 量子化版）を使う。
        トークナイザ（vocab）は fp16/int8 で共通のため常に DEBERTA_ONNX_REPO から取る
        （quantize_deberta.py が作る int8 は同じ文字トークナイズ前提で成立する）。
        未指定時は従来通り HF から fp16 を取得する（既定は fp16 のまま・後方互換）。
        """
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        model_path = (
            str(onnx_path) if onnx_path is not None else hf_hub_download(DEBERTA_ONNX_REPO, DEBERTA_ONNX_FILE)
        )
        self._tokenizer = AutoTokenizer.from_pretrained(DEBERTA_ONNX_REPO)
        so = ort.SessionOptions()
        self._session = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_names = [i.name for i in self._session.get_inputs()]
        self.model_file_bytes = Path(model_path).stat().st_size

    def extract(self, norm_text: str, word2ph: list[int]) -> np.ndarray:
        """norm_text を DeBERTa に通し、word2ph 展開して [1024, sum(word2ph)] を返す。"""
        enc = self._tokenizer(norm_text, return_tensors="np")
        feed = {
            name: enc[name].astype(np.int64)
            for name in self._input_names
            if name in enc
        }
        missing = set(self._input_names) - set(feed)
        if missing:
            raise ValueError(
                f"DeBERTa の入力 {missing} をトークナイザ出力から供給できない"
                f"（tokenizer keys={list(enc.keys())}）"
            )
        (res,) = self._session.run(None, feed)  # [seq_len, 1024] float32
        if res.shape[0] != len(word2ph):
            raise ValueError(
                f"DeBERTa トークン数 {res.shape[0]} != word2ph 長 {len(word2ph)}"
                f"（norm_text={norm_text!r}）。文字トークナイズと word2ph の齟齬を疑う。"
            )
        phone_level = np.concatenate(
            [np.tile(res[i], (word2ph[i], 1)) for i in range(len(word2ph))],
            axis=0,
        )
        return phone_level.T.astype(np.float32)  # [1024, sum(word2ph)]


def _word2ph_for_bert(text: str) -> tuple[str, list[int]]:
    """BERT 特徴量抽出用の norm_text と add_blank 済み word2ph を SBV2 から得る。

    infer.py の word2ph 調整（各要素 *2, 先頭 +1）を再現し、sum(word2ph) が
    add_blank 後の音素列長 2*len+1 に一致するようにする。real BERT 経路でのみ使う。
    """
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.nlp import clean_text

    norm_text, _phones, _tones, word2ph = clean_text(
        text, Languages.JP, use_jp_extra=True, raise_yomi_error=False
    )
    adjusted = [w * 2 for w in word2ph]
    adjusted[0] += 1  # add_blank の先頭ブランク分（infer.py）
    return norm_text, adjusted


def _run_session(
    session: ort.InferenceSession,
    input_names: list[str],
    tensors: dict[str, np.ndarray],
) -> np.ndarray:
    """グラフの入力名に対して用意したテンソルを名前で束縛し、'output' を取り出す。

    入力名は決め打ちせず get_inputs() の名前で照合する（aivmx-interface.md §2.1 の設計含意）。
    不足・余剰があれば fail loudly。
    """
    missing = set(input_names) - set(tensors)
    extra = set(tensors) - set(input_names)
    if missing or extra:
        raise ValueError(
            f"aivmx 入力名の不一致: 不足={sorted(missing)} 余剰={sorted(extra)}"
            f"（グラフ入力={input_names}）"
        )
    feed = {name: tensors[name] for name in input_names}
    outputs = session.run(["output"], feed)
    return outputs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aivmx",
        default="mao.aivmx",
        help=f"aivmx ファイル名（{AIVMX_DIR} 配下）または絶対パス（既定: mao.aivmx）",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"text/phones/tones を持つ JSONL（既定: {DEFAULT_INPUT}）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"wav 出力先（既定: {AIVMX_OUT_DIR}/<model>）",
    )
    parser.add_argument("--style", type=int, default=0, help="スタイル local_id（既定: 0）")
    parser.add_argument(
        "--style-weight", type=float, default=1.0, help="スタイル強度（既定: 1.0）"
    )
    parser.add_argument("--sid", type=int, default=0, help="話者 local_id（既定: 0）")
    parser.add_argument(
        "--bert",
        choices=("real", "zero"),
        default="real",
        help="ja_bert を実 DeBERTa で作るか（real）ゼロテンソルで疎通優先か（zero）（既定: real）",
    )
    parser.add_argument(
        "--bert-onnx-path",
        type=Path,
        default=None,
        help=(
            "ja_bert 抽出に使う DeBERTa ONNX のローカルパス（例: int8 量子化版 "
            "data/models/deberta-int8/model_int8.onnx）。未指定時は既定通り HF から "
            "fp16（model_fp16.onnx）を取得する。--bert real のときのみ有効。"
        ),
    )
    args = parser.parse_args()

    if args.bert_onnx_path is not None and args.bert != "real":
        raise ValueError("--bert-onnx-path は --bert real のときのみ指定できる")

    aivmx_arg = Path(args.aivmx)
    aivmx_path = aivmx_arg if aivmx_arg.is_absolute() else AIVMX_DIR / aivmx_arg
    if not aivmx_path.exists():
        raise FileNotFoundError(
            f"aivmx が無い: {aivmx_path}。data/models/aivmx/ に取得する"
            "（api.aivis-project.com の download?model_type=AIVMX）。"
        )

    model_slug = aivmx_path.stem
    out_dir: Path = args.out_dir or (AIVMX_OUT_DIR / model_slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_input(args.input)

    # --- aivmx（音響モデル）ロード ---
    so = ort.SessionOptions()
    session = ort.InferenceSession(
        str(aivmx_path), sess_options=so, providers=["CPUExecutionProvider"]
    )
    input_names = [i.name for i in session.get_inputs()]
    sample_rate = 44100  # hp.data.sampling_rate（実測で確認済み）

    symbol_to_id = _build_symbol_to_id()
    style_matrix = _load_style_matrix(aivmx_path)
    style_vec = _style_vector(style_matrix, args.style, args.style_weight)

    # --- ja_bert 抽出器（real のときのみロード。zero は DeBERTa を触らない） ---
    extractor: DebertaBertExtractor | None = None
    bert_model_bytes: int | None = None
    if args.bert == "real":
        # word2ph 生成で SBV2 の g2p を呼ぶ。その内部が JP BERT トークナイザを引数なしで
        # 参照するため、先に repo 付きでロードしておく（_model.load_bert_tokenizer と同じ役割）。
        load_bert_tokenizer()
        extractor = DebertaBertExtractor(onnx_path=args.bert_onnx_path)
        bert_model_bytes = extractor.model_file_bytes

    # スカラーは rank-0（shape []）で渡す（実測: length/sdp/noise/noise_w は 0-d）。
    scalars = {
        "length_scale": np.array(DEFAULT_LENGTH_SCALE, dtype=np.float32),
        "sdp_ratio": np.array(DEFAULT_SDP_RATIO, dtype=np.float32),
        "noise_scale": np.array(DEFAULT_NOISE_SCALE, dtype=np.float32),
        "noise_scale_w": np.array(DEFAULT_NOISE_SCALE_W, dtype=np.float32),
    }

    results = []
    for i, rec in enumerate(records):
        text = rec["text"]
        phones = rec["phones"]
        tones = rec["tones"]

        phone_ids, tone_ids, lang_ids = _phones_tones_to_model_ids(
            phones, tones, symbol_to_id
        )
        seq_len = len(phone_ids)  # add_blank 後の 2*len+1

        if args.bert == "real":
            assert extractor is not None
            norm_text, word2ph = _word2ph_for_bert(text)
            bert = extractor.extract(norm_text, word2ph)
            if bert.shape[1] != seq_len:
                raise ValueError(
                    f"[{text!r}] ja_bert 長 {bert.shape[1]} != 音素列長 {seq_len}"
                    "（word2ph 調整と add_blank の齟齬を疑う）"
                )
        else:
            bert = np.zeros((BERT_DIM, seq_len), dtype=np.float32)

        tensors = {
            "x_tst": np.array([phone_ids], dtype=np.int64),
            "x_tst_lengths": np.array([seq_len], dtype=np.int64),
            "sid": np.array([args.sid], dtype=np.int64),
            "tones": np.array([tone_ids], dtype=np.int64),
            "language": np.array([lang_ids], dtype=np.int64),
            "bert": bert[np.newaxis, :, :].astype(np.float32),  # [1, 1024, T]
            "style_vec": style_vec,  # [1, 256]
            **scalars,
        }

        raw = _run_session(session, input_names, tensors)  # [1, 1, N] 相当
        audio = np.asarray(raw, dtype=np.float32).reshape(-1)

        metrics = measure(sample_rate, audio)
        wav_path = out_dir / f"{slugify(text, i)}.wav"
        sf.write(str(wav_path), audio, sample_rate)

        results.append(
            {
                "text": text,
                "wav_path": str(wav_path),
                "sample_rate": sample_rate,
                "seq_len": seq_len,
                **metrics,
            }
        )

    metrics_path = out_dir / "aivmx_metrics.json"
    metrics_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "aivmx": str(aivmx_path),
        "aivmx_file_bytes": aivmx_path.stat().st_size,
        "model": model_slug,
        "input": str(args.input),
        "bert": args.bert,
        "bert_onnx_path": str(args.bert_onnx_path) if args.bert_onnx_path else None,
        "bert_model_file_bytes": bert_model_bytes,
        "style_id": args.style,
        "style_weight": args.style_weight,
        "num_styles": int(style_matrix.shape[0]),
        "sid": args.sid,
        "sample_rate": sample_rate,
        "num_sentences": len(results),
        "out_dir": str(out_dir),
        "metrics_path": str(metrics_path),
    }
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
