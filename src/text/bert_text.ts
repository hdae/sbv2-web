// DeBERTa 入力テキスト（本家 SBV2 の norm_text 相当）の組み立て。
//
// 本家は normalize_text の replace_punctuation で句読点を正規形（! ? … , . ' -）へ
// 写してから BERT に入れる。yomi 0.4.0 の語アライメントは記号要素の surface に
// 「生の1文字」（。 ！ 等）を保持するため、そのまま連結すると DeBERTa が見る
// トークン id が本家とずれる（例: "。" vs "."）。記号要素だけ正規形（= その要素の
// 唯一の phone）へ置き換えて連結し、パリティを保つ。
//
// NOTE: この置換で per-word のトークン数は変わらない（正規形と生形は char
// トークナイザ（clean+NFKC）でどちらも同数に割れる — "…" は両形とも NFKC で
// "..." の 3 トークン）。そのため word2ph（Σ tokenize(surface) ベース）との
// 整合は崩れない。

import type { WordPhones } from "@hdae/yomi";

/** SBV2 が音素として受け付ける正規形句読点（symbols.ts の句読点部分と同一）。 */
const PUNCTUATIONS: ReadonlySet<string> = new Set([
  "!",
  "?",
  "…",
  ",",
  ".",
  "'",
  "-",
]);

/** 記号要素（phones が正規形句読点 1 個だけの語）か。かな語の音素と正規形字母は交差しない。 */
const isPunctuationWord = (word: WordPhones): boolean =>
  word.phones.length === 1 && PUNCTUATIONS.has(word.phones[0]);

/**
 * 語アライメントから DeBERTa 入力テキスト（本家の norm_text 相当）を組む。
 * 記号要素は正規形句読点（その要素の唯一の phone）へ置き換え、それ以外は
 * surface をそのまま連結する。`synthesizeText` はこれを使う。SynthInput を
 * 自前で組む消費者（サーバー実装等）も、bertText はこの関数で組むこと
 * （手書きの surface 連結はパリティと整合の二経路化を生む）。
 */
export const toBertText = (words: readonly WordPhones[]): string =>
  words
    .map((word) => (isPunctuationWord(word) ? word.phones[0] : word.surface))
    .join("");
