// テキスト→音声の一気通貫グルー。
//
// フロントエンド解析・語アライメント・word2ph・モデルアダプタの各部品を、
// 「テキスト1本 → 音声」に配線する。
//
// 責務境界（ADR-0001 / aivmx-interface.md §6）: グルーは synth 側に置く（frontend は zero-dep MUST）。
// フロントエンドから phones/tones/語アライメントを得て SynthInput を組み立て、
// ModelAdapter.synthesize に渡すだけ。SBV2 固有詳細（SYMBOLS/tone+6/add_blank/BERT）は
// アダプタに閉じている。
//
// MUST: 解析は 1 回だけ（二重解析しない）。analyzeWithWords が result（音素・トーン用）と
// words（word2ph 用）を同一解析から返すので、両者は同じ解析を土台に整合する。

import {
  analyzeWithWords,
  type JtdDictionary,
  type OverlayDictionary,
} from "@hdae/yomi";
import { toSbv2PhoneTone } from "../text/phone_tone.ts";
import { toBertText } from "../text/bert_text.ts";
import type { DebertaTokenizer } from "../text/deberta_tokenizer.ts";
import { buildBaseWord2ph } from "../text/word2ph.ts";
import type { ModelAdapter, SynthScalars } from "./adapter_types.ts";
import { padSilence } from "./silence.ts";

/** synthesizeText の任意パラメータ（既定は synth_aivmx.py / SynthInput の既定に揃える）。 */
export type SynthesizeOptions = {
  /** スタイル local_id（既定 0）。 */
  styleId?: number;
  /** スタイル強度（既定 1.0）。 */
  styleWeight?: number;
  /** 話者 local_id（sid, 既定 0）。 */
  speakerId?: number;
  /** このリクエストだけのスカラー上書き（話速 lengthScale 等。省略キーはアダプタ既定）。 */
  scalars?: Partial<SynthScalars>;
  /** 文頭の無音秒数（波形先頭へのゼロ詰め。AivisSpeech の prePhonemeLength 相当）。 */
  preSilenceSec?: number;
  /** 文末の無音秒数（波形末尾へのゼロ詰め。AivisSpeech の postPhonemeLength 相当）。 */
  postSilenceSec?: number;
  /** 修正辞書オーバーレイ（任意。フロントエンド解析に渡す）。 */
  overlay?: OverlayDictionary;
};

/**
 * テキスト1本 → 波形（Float32Array）。フロントエンド解析からモデル推論までを配線する。
 *
 * 手順（解析は 1 回だけ = analyzeWithWords）:
 *   1. { result, words } = analyzeWithWords(dict, text, overlay)  ← 解析はここ 1 回のみ
 *   2. { phones, tones } = toSbv2PhoneTone(result)
 *   3. bertText = toBertText(words)（語 surface 連結・記号は正規形へ =
 *      本家 replace_punctuation 済み norm_text 相当）
 *   4. baseWord2ph = buildBaseWord2ph(words, tokenizer, phones.length)
 *   5. adapter.synthesize({ phones, tones, bertText, baseWord2ph, styleId, styleWeight, speakerId })
 *
 * @param text 合成対象の生テキスト（原文。正規化はフロントエンドが行う）。
 * @param dict JTD1 辞書。
 * @param tokenizer DeBERTa char トークナイザ（word2ph の文字数算出に使う）。
 * @param adapter モデルアダプタ（SBV2 aivmx 実駆動 or テスト用モック）。
 * @returns 波形（サンプルレートは adapter.sampleRate）。
 */
export const synthesizeText = async (
  text: string,
  dict: JtdDictionary,
  tokenizer: DebertaTokenizer,
  adapter: ModelAdapter,
  opts: SynthesizeOptions = {},
): Promise<Float32Array> => {
  // 解析は 1 回だけ。analyzeWithWords が result と words を同一解析から返す。
  const { result, words } = analyzeWithWords(dict, text, opts.overlay);

  const { phones, tones } = toSbv2PhoneTone(result);
  // DeBERTa 入力は語アライメントからの norm_text 相当（記号は正規形へ — 本家
  // replace_punctuation 対応）。word2ph が Σtokenize(surface) ベースのため、
  // normalizedText 全体の直接トークナイズとは長さがずれ得る。
  const bertText = toBertText(words);
  const baseWord2ph = buildBaseWord2ph(words, tokenizer, phones.length);

  const wave = await adapter.synthesize({
    phones,
    tones,
    bertText,
    baseWord2ph,
    styleId: opts.styleId ?? 0,
    styleWeight: opts.styleWeight ?? 1.0,
    speakerId: opts.speakerId ?? 0,
    scalars: opts.scalars,
  });
  // 文頭・文末の無音はモデル外の後処理（両方 0 ならコピーなしの素通し）。
  return padSilence(wave, adapter.sampleRate, {
    preSec: opts.preSilenceSec,
    postSec: opts.postSilenceSec,
  });
};
