// DeBERTa char トークナイザの behavior テスト。
//
// clean_ranges 適用・NFKC 分解・コードポイント分割・UNK フォールバック・CLS/SEP 付与を、
// 小さな固定 vocab で検証する。fromVocabText の CRLF 耐性は回帰テストで固定する
// （CRLF vocab を "\n" だけで割ると全トークンが [UNK] 化する無言バグの再発防止）。

import { assertEquals } from "@std/assert";
import { type CleanRanges, DebertaTokenizer } from "./deberta_tokenizer.ts";

const VOCAB_LINES = [
  "[PAD]",
  "[CLS]",
  "[SEP]",
  "[UNK]",
  "a",
  "b",
  "株",
  "式",
  "会",
  "社",
  "𠮷",
];
const CLEAN: CleanRanges = {
  removed: [[0, 8]],
  spaced: [[9, 13], [32, 32]],
};
const SPECIAL = { clsId: 1, sepId: 2, unkId: 3 };

const buildTokenizer = (lineSep: string): DebertaTokenizer =>
  DebertaTokenizer.fromVocabText(VOCAB_LINES.join(lineSep), CLEAN, SPECIAL);

Deno.test("fromVocabText: LF vocab は行番号を id にする", () => {
  const tok = buildTokenizer("\n");
  assertEquals(tok.encode("ab"), [1, 4, 5, 2]);
});

Deno.test("encode: [CLS] + ids + [SEP] で包む", () => {
  assertEquals(buildTokenizer("\n").encode("a"), [1, 4, 2]);
});

Deno.test("encode: 未知トークンは UNK id にフォールバックする", () => {
  // "z" は vocab に無い -> unkId(3)。
  assertEquals(buildTokenizer("\n").encode("az"), [1, 4, 3, 2]);
});

Deno.test("#tokenToId: id 0 のトークンを UNK に握りつぶさない（?? であって || でない）", () => {
  // 単一文字 "x" を id 0 に割り当て、unkId は 99。|| なら 99 に化ける。
  const tok = DebertaTokenizer.fromVocabText("x\ny", {
    removed: [],
    spaced: [],
  }, {
    clsId: 1,
    sepId: 2,
    unkId: 99,
  });
  assertEquals(tok.encode("x"), [1, 0, 2]);
});

Deno.test("tokenize: 制御文字は除去・空白は落とす（clean_ranges）", () => {
  const tok = buildTokenizer("\n");
  assertEquals(tok.tokenize("a\tb"), ["a", "b"]); // tab(9) -> space -> drop
  assertEquals(tok.tokenize("ab"), ["a", "b"]); // 0x02 -> removed
});

Deno.test("tokenize: NFKC 分解で 1 文字が複数トークンになる（㍿→株式会社）", () => {
  assertEquals(buildTokenizer("\n").tokenize("㍿"), ["株", "式", "会", "社"]);
});

Deno.test("tokenize: サロゲートペアは 1 トークン", () => {
  const tok = buildTokenizer("\n");
  assertEquals(tok.tokenize("𠮷"), ["𠮷"]);
  assertEquals(tok.encode("𠮷"), [1, 10, 2]);
});

Deno.test("fromVocabText: CRLF vocab でも id が壊れない（回帰）", () => {
  // 修正前は "\r" 付きトークンが登録され、全 lookup が UNK 化した。
  const tok = buildTokenizer("\r\n");
  assertEquals(tok.encode("ab"), [1, 4, 5, 2]);
});
