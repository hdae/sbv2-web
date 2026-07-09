// base word2ph 生成の behavior テスト。
//
// distributePhone の貪欲分配と、buildBaseWord2ph の両端番兵・2 つの不変条件
// （長さ == Σtokenize(surface)+2 / sum == given_phone 長）と fail-loudly を固定する。

import { assertEquals, assertThrows } from "@std/assert";
import { buildBaseWord2ph, distributePhone } from "./word2ph.ts";
import { DebertaTokenizer } from "./deberta_tokenizer.ts";

// tokenize はコードポイント分割のみ使う（vocab は数えないので空でよい）。
const tokenizer = DebertaTokenizer.fromVocabText(
  "",
  { removed: [], spaced: [] },
  { clsId: 1, sepId: 2, unkId: 3 },
);

Deno.test("distributePhone: 左詰め貪欲で n_phone を n_word に分配", () => {
  assertEquals(distributePhone(3, 2), [2, 1]);
  assertEquals(distributePhone(5, 3), [2, 2, 1]);
  assertEquals(distributePhone(1, 1), [1]);
  assertEquals(distributePhone(0, 2), [0, 0]);
});

Deno.test("distributePhone: n_word <= 0 は throw", () => {
  assertThrows(() => distributePhone(3, 0), Error, "n_word");
});

Deno.test("buildBaseWord2ph: 両端番兵 [1]..[1] と語ごとの分配", () => {
  const words = [
    { surface: "こ", phones: ["k", "o"] },
    { surface: "ん", phones: ["N"] },
  ];
  // sentinel(1) + distribute(2,1)=[2] + distribute(1,1)=[1] + sentinel(1)。
  assertEquals(buildBaseWord2ph(words, tokenizer, 5), [1, 2, 1, 1]);
});

Deno.test("buildBaseWord2ph: 複数文字 surface は文字数へ分配", () => {
  const words = [{ surface: "ちゃ", phones: ["ch", "a"] }];
  // tokenize("ちゃ")=2 文字 -> distribute(2,2)=[1,1]。
  assertEquals(buildBaseWord2ph(words, tokenizer, 4), [1, 1, 1, 1]);
});

Deno.test("buildBaseWord2ph: sum != given_phone 長は throw（不変条件2）", () => {
  const words = [{ surface: "こ", phones: ["k", "o"] }];
  assertThrows(
    () => buildBaseWord2ph(words, tokenizer, 99),
    Error,
    "sum(word2ph)",
  );
});

Deno.test("buildBaseWord2ph: 0 トークンに正規化される surface は throw", () => {
  const words = [{ surface: " ", phones: ["a"] }];
  assertThrows(
    () => buildBaseWord2ph(words, tokenizer, 3),
    Error,
    "0 トークン",
  );
});
