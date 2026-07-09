"""DeBERTa 量子化スクリプト（int8 QDQ / int4 NBits）の共有部。

int8（quantize_deberta.py・quantize_dynamic QDQ）と int4/int8（quantize_deberta_int4.py・
MatMulNBitsQuantizer）で共通に使う「fp16 ONNX の取得」と「CC-BY-SA-4.0 LICENSE 全文の
取得・書き出し」をここに集約する。NOTICE は派生手順ごとに文面が異なるため各スクリプトが持つ。

DeBERTa は CC-BY-SA-4.0（原著 ku-nlp、ONNX 化 tsukumijima）。量子化した派生物は
表示(BY)+継承(SA) の下で LICENSE 全文と NOTICE（改変明示）を出力先に同梱する義務がある
（docs/license-audit.md）。その LICENSE 取得を単一実装に保つ。
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

# 実推論に使う ONNX 版 DeBERTa（AivisSpeech-Engine / SBV2 fork と同一リポジトリ）。
DEBERTA_ONNX_REPO = "tsukumijima/deberta-v2-large-japanese-char-wwm-onnx"
DEBERTA_ONNX_FILE = "model_fp16.onnx"
# 帰属表示に使う原著（量子化派生の出典）。
DEBERTA_BASE_REPO = "ku-nlp/deberta-v2-large-japanese-char-wwm"

CC_BY_SA_4_0_URL = "https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt"


def fetch_fp16_model_path() -> Path:
    """HF キャッシュ（data/hf-cache）から fp16 ONNX を取得する。既取得ならキャッシュを使う。

    _paths を先に import して HF_HOME を固定しておくこと（呼び出し側の責務）。
    """
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(DEBERTA_ONNX_REPO, DEBERTA_ONNX_FILE))


def write_cc_by_sa_license(out_dir: Path) -> Path:
    """CC-BY-SA-4.0 全文を out_dir/LICENSE に書き出し、そのパスを返す（fail loudly）。

    ネットワーク失敗時は例外を送出して中断する（対症フォールバックで別物を出さない）。
    """
    license_path = out_dir / "LICENSE"
    try:
        # creativecommons.org は UA 無しリクエストを 403 で弾く実測を確認したため明示する。
        request = urllib.request.Request(
            CC_BY_SA_4_0_URL, headers={"User-Agent": "sbv2-web-model-tools/1.0"}
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            license_text = resp.read().decode("utf-8")
    except Exception as error:  # noqa: BLE001 - ネットワーク失敗は fail loudly
        raise RuntimeError(
            f"CC-BY-SA-4.0 全文の取得に失敗: {CC_BY_SA_4_0_URL} ({type(error).__name__}: {error})。"
            " LICENSE を自動生成できないため中断する。"
        ) from error
    license_path.write_text(license_text, encoding="utf-8")
    return license_path
