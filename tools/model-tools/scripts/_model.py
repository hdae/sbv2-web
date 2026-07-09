"""JP BERT トークナイザのロードと wav 計測・スラグ生成の共通ユーティリティ。

onnxruntime で aivmx を直接駆動する経路が使う、内蔵 G2P 用の JP BERT
トークナイザ事前ロードと、生成 wav の計測（長さ・無音率・ピーク）・
ファイル名スラグ生成をここに集約する。

import 副作用で HF_HOME 等を data/hf-cache に固定する（_paths を先に import）。
"""

from __future__ import annotations

# import 副作用で HF_HOME 等を data/hf-cache に固定する（他 HF import より前）。
from _paths import JP_BERT_REPO

import sys

import numpy as np
from loguru import logger

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models

# SBV2 は loguru を stdout に流す（logging.py が SAFE_STDOUT に add）。本ツールは
# stdout に機械可読な JSON/JSONL を出すため、ログが混ざると下流のパイプが壊れる。
# ログは stderr に付け替える（stdout は JSON 専用にする）。
logger.remove()
logger.add(sys.stderr, level="INFO")

# 無音判定のしきい値: 振幅が全体の最大値の 1% 未満のサンプルを無音とみなす。
_SILENCE_RATIO = 0.01


def slugify(text: str, index: int) -> str:
    """ファイル名は index + 先頭数文字（非ASCIIは落とす）で衝突を避けつつ可読にする。"""
    ascii_head = "".join(ch for ch in text if ch.isascii() and ch.isalnum())[:16]
    return f"{index:02d}_{ascii_head}" if ascii_head else f"{index:02d}"


def measure(sample_rate: int, audio: np.ndarray) -> dict[str, float]:
    """wav を baseline/given 共通の指標（長さ・無音率・ピーク）で計測する。"""
    audio = audio.astype(np.float64)
    duration_sec = float(len(audio) / sample_rate) if sample_rate else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0:
        silent = int(np.count_nonzero(np.abs(audio) < peak * _SILENCE_RATIO))
        silence_ratio = float(silent / audio.size)
    else:
        # 完全な無音（peak==0）は無音率 1.0 とする（fail loudly のため下流で検知）。
        silence_ratio = 1.0
    return {
        "duration_sec": round(duration_sec, 4),
        "silence_ratio": round(silence_ratio, 4),
        "peak_amplitude": round(peak, 6),
    }


def load_bert_tokenizer() -> None:
    """内蔵 G2P（clean_text）が使う JP BERT トークナイザを事前ロードする。

    g2p.py は `bert_models.load_tokenizer(Languages.JP)` を repo 引数なしで
    呼ぶため、事前にこの repo でロードしておく必要がある。トークナイザのみで
    BERT 本体のロードは不要（word2ph は文字トークナイズだけを使う）。
    """
    bert_models.load_tokenizer(Languages.JP, JP_BERT_REPO)
