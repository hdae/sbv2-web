// base word2ph（add_blank 前）の生成。BERT テキスト層の中核。
//
// word2ph = 「正規化テキストの各文字に音素を何個割り当てるか」。DeBERTa の文字ごと
// 隠れ状態を音素レベルへ tile 展開するのに使う。SBV2 は厳密アライメント不可のため
// 「単語ごとに音素数を文字数へ均等分配」する近似を採る（g2p.py:__distribute_phone）。
//
// スコープ（ADR-0001）: 本関数が作るのは add_blank 前の base word2ph。
//   sum(base word2ph) == given_phone 長（両端 "_" と句読点を含む toSbv2PhoneTone の出力長）
//   len(base word2ph) == DeBERTa トークナイザで各語 surface をトークナイズした文字数の総和 + 2
// add_blank の *2 / [0]+=1 は後段のアダプタの仕事なのでここではやらない。
//
// Word-to-phone alignment helper for JP-Extra DeBERTa feature tiling.
//   （出典: Style-Bert-VITS2 の g2p.py __distribute_phone）。

import type { WordPhones } from "@hdae/yomi";
import type { DebertaTokenizer } from "./deberta_tokenizer.ts";

/**
 * n_phone 個の音素を n_word 文字へ均等分配する（g2p.py の __distribute_phone の忠実移植）。
 * 左から右へ 1 ずつ、常に「最小の文字」へ足していく。
 */
export const distributePhone = (nPhone: number, nWord: number): number[] => {
  if (nWord <= 0) {
    throw new Error(`distributePhone: n_word must be >= 1, got ${nWord}`);
  }
  const perWord = new Array<number>(nWord).fill(0);
  for (let k = 0; k < nPhone; k++) {
    // 最小値のインデックス（同値なら最左）。
    let minIndex = 0;
    for (let i = 1; i < nWord; i++) {
      if (perWord[i] < perWord[minIndex]) minIndex = i;
    }
    perWord[minIndex]++;
  }
  return perWord;
};

/**
 * 語アライメントと DeBERTa トークナイザから base word2ph を作る。
 *
 * @param words wordPhoneAlignment(...) の各語（surface + phones、両端 "_" を含まない）。
 * @param tokenizer DeBERTa char トークナイザ（各語 surface の文字数算出に使う）。
 * @param givenPhoneLen toSbv2PhoneTone(...).phones.length（両端 "_" 込み）= sum(word2ph) の期待値。
 * @returns base word2ph（両端 [1] 番兵込み）。
 *
 * 不変条件（fail loudly。破れたら throw）:
 *  - word2ph.length === Σ tokenize(surface).length + 2   （bert_feature.py の len(text)+2 に対応。
 *      ただし text は SBV2 の sep_text 連結 = 語 surface 連結。char トークナイザは加算的なので
 *      Σ tokenize(word) == tokenize(joined) が成り立つ。norm_text 全体の直接トークナイズではない
 *      点に注意 — 我々のフロントは記号/一般（"！" 等）を given_phone に出さないため、
 *      normalizedText の直接トークナイズとは長さがずれる。詳細は docs/aivmx-interface.md 参照）。
 *  - sum(word2ph) === givenPhoneLen                       （g2p.py の len(phones)==sum(word2ph)）。
 */
export const buildBaseWord2ph = (
  words: readonly WordPhones[],
  tokenizer: DebertaTokenizer,
  givenPhoneLen: number,
): number[] => {
  // 先頭 "_" 用番兵。
  const word2ph: number[] = [1];
  let tokenTotal = 0;
  for (const word of words) {
    const wordLen = tokenizer.tokenize(word.surface).length;
    if (wordLen <= 0) {
      throw new Error(
        `buildBaseWord2ph: 語 surface が 0 トークンに正規化された: ${
          JSON.stringify(word.surface)
        }`,
      );
    }
    tokenTotal += wordLen;
    for (const n of distributePhone(word.phones.length, wordLen)) {
      word2ph.push(n);
    }
  }
  // 末尾 "_" 用番兵。
  word2ph.push(1);

  // 不変条件 1: 長さ == 語トークン数総和 + 両端番兵 2。
  const expectedLen = tokenTotal + 2;
  if (word2ph.length !== expectedLen) {
    throw new Error(
      `buildBaseWord2ph invariant broken: word2ph.length(${word2ph.length}) !== ` +
        `Σtokenize(surface)+2(${expectedLen})`,
    );
  }
  // 不変条件 2: sum == given_phone 長（両端 "_" 込み）。
  const total = word2ph.reduce((a, b) => a + b, 0);
  if (total !== givenPhoneLen) {
    throw new Error(
      `buildBaseWord2ph invariant broken: sum(word2ph)(${total}) !== givenPhoneLen(${givenPhoneLen})`,
    );
  }
  return word2ph;
};
