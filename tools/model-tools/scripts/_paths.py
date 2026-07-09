"""共有パス定義とHF_HOME設定。

data/ 配下（gitignore 済み）にモデル・HFキャッシュを置く方針をスクリプト間で共有する。
インポートするだけで HF_HOME / HF_HUB_CACHE 環境変数を data/hf-cache に固定する
（BERT モデル ku-nlp/deberta-v2-large-japanese-char-wwm 等がここに落ちる）。
"""

from __future__ import annotations

import os
from pathlib import Path

# tools/model-tools/scripts/_paths.py -> リポジトリルート
REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
HF_CACHE_DIR = DATA_DIR / "hf-cache"

# HFキャッシュを data/hf-cache に固定する。transformers / huggingface_hub 双方が
# 参照するよう、import 副作用として環境変数を設定する（スクリプトの先頭で import する）。
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))

# 取得対象の公開 JP-Extra モデル（CC-BY-SA-4.0 / litagin/style_bert_vits2_jvnv）。
JVNV_REPO_ID = "litagin/style_bert_vits2_jvnv"
DEFAULT_MODEL_NAME = "jvnv-F1-jp"
# 日本語 BERT（JP-Extra 用）。SBV2 constants.py の DEFAULT と一致。
JP_BERT_REPO = "ku-nlp/deberta-v2-large-japanese-char-wwm"
# 実推論で使う ONNX 版 BERT の repo（トークナイザは ku-nlp base と vocab 完全一致・
# input_ids 完全一致を実測で確認済み）。TS 側トークナイザはこの vocab を移植する。
JP_BERT_ONNX_REPO = "tsukumijima/deberta-v2-large-japanese-char-wwm-onnx"
# TS 側が読む、安定した vocab の取り出し先。data/ は gitignore 対象で再取得可能。
DEBERTA_TOKENIZER_DIR = MODELS_DIR / "deberta-tokenizer"


def model_dir(model_name: str = DEFAULT_MODEL_NAME) -> Path:
    return MODELS_DIR / model_name
