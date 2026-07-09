// DeBERTa char トークナイザ（BertJapaneseTokenizer, character subword）の TS 移植。
//
// 対象: tsukumijima/deberta-v2-large-japanese-char-wwm-onnx の tokenizer
//   （= base ku-nlp/deberta-v2-large-japanese-char-wwm と vocab / input_ids 完全一致）。
// Character-level DeBERTa tokenizer used by JP-Extra BERT feature extraction.
//   （vocab / meta / clean_ranges は量子化 DeBERTa の HF パッケージに同梱。
//    tools/model-tools/scripts/prepare_hf_deberta.py が生成する）。
//
// アルゴリズム（tokenizer.json 実測）:
//   1. BertNormalizer(clean_text): 制御文字を除去し、空白を半角スペースへ（clean_ranges で表現）。
//   2. NFKC 正規化（String.prototype.normalize）。
//   3. WhitespaceSplit: 空白（スペース）で分割し、空白は捨てる。
//   4. Split(regex="."): 各断片を 1 コードポイントずつに分割。
//   5. WordLevel: vocab lookup（無ければ [UNK]）。
//   6. TemplateProcessing: [CLS] + ids + [SEP]。
//
// 「1 文字」は Unicode コードポイント単位（サロゲートペアの絵文字・CJK 拡張は 1 トークン）。

/** clean_text の除去・スペース化コードポイント範囲（両端含む閉区間の昇順リスト）。 */
export type CleanRanges = {
  /** 除去する（出力しない）コードポイント範囲。 */
  removed: readonly (readonly [number, number])[];
  /** 半角スペースへ置換するコードポイント範囲。 */
  spaced: readonly (readonly [number, number])[];
};

export type DebertaSpecialTokens = {
  clsId: number;
  sepId: number;
  unkId: number;
};

/** 昇順の閉区間リストに対する二分探索で cp が含まれるか判定する。 */
const inRanges = (
  ranges: readonly (readonly [number, number])[],
  cp: number,
): boolean => {
  let lo = 0;
  let hi = ranges.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const [start, end] = ranges[mid];
    if (cp < start) hi = mid - 1;
    else if (cp > end) lo = mid + 1;
    else return true;
  }
  return false;
};

export class DebertaTokenizer {
  readonly #vocab: ReadonlyMap<string, number>;
  readonly #clean: CleanRanges;
  readonly #special: DebertaSpecialTokens;

  /**
   * @param vocab token → id（vocab.txt の行番号 = id）。
   * @param clean clean_text の除去/スペース化範囲（clean_ranges.json）。
   * @param special [CLS]/[SEP]/[UNK] の id（meta.json 由来）。
   */
  constructor(
    vocab: ReadonlyMap<string, number>,
    clean: CleanRanges,
    special: DebertaSpecialTokens,
  ) {
    this.#vocab = vocab;
    this.#clean = clean;
    this.#special = special;
  }

  /**
   * vocab.txt（1 行 = 1 トークン、行番号 0-origin = id）から構築する。
   * @param vocabText vocab.txt の内容。
   * @param clean clean_ranges.json をパースしたもの。
   * @param special [CLS]/[SEP]/[UNK] の id。
   */
  static fromVocabText(
    vocabText: string,
    clean: CleanRanges,
    special: DebertaSpecialTokens,
  ): DebertaTokenizer {
    const vocab = new Map<string, number>();
    // Split on CRLF or LF. A CRLF vocab split on "\n" alone leaves a trailing
    // "\r" on every token, so every lookup misses and all input_ids become [UNK]
    // — silently wrong BERT features with no error.
    const lines = vocabText.split(/\r?\n/);
    let id = 0;
    for (const line of lines) {
      // 末尾のファイル改行由来の空行はトークンではない。行内の空トークンも存在しない。
      if (line === "" && id === lines.length - 1) break;
      vocab.set(line, id);
      id++;
    }
    return new DebertaTokenizer(vocab, clean, special);
  }

  /**
   * clean_text + NFKC を適用し、トークン（1 コードポイント）文字列の配列にする。
   * CLS/SEP は含まない（word2ph 用の語トークン数はこれで数える）。
   */
  tokenize(text: string): string[] {
    // 1. clean_text: 除去 or スペース化 or そのまま。
    let cleaned = "";
    for (const ch of text) {
      const cp = ch.codePointAt(0)!;
      if (inRanges(this.#clean.removed, cp)) continue;
      cleaned += inRanges(this.#clean.spaced, cp) ? " " : ch;
    }
    // 2. NFKC。
    const normalized = cleaned.normalize("NFKC");
    // 3-4. 空白分割 + 1 コードポイント分割（空白は捨てる）。
    const tokens: string[] = [];
    for (const ch of normalized) {
      if (ch === " ") continue;
      tokens.push(ch);
    }
    return tokens;
  }

  /** token 文字列を id へ（vocab に無ければ [UNK]）。 */
  #tokenToId(token: string): number {
    const id = this.#vocab.get(token);
    return id ?? this.#special.unkId;
  }

  /**
   * text を input_ids へ（[CLS] + 各文字 id + [SEP]）。
   * Python の AutoTokenizer(text)["input_ids"] と完全一致する。
   */
  encode(text: string): number[] {
    const tokens = this.tokenize(text);
    const ids: number[] = [this.#special.clsId];
    for (const t of tokens) ids.push(this.#tokenToId(t));
    ids.push(this.#special.sepId);
    return ids;
  }
}
