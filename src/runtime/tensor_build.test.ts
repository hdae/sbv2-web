// 入力テンソル組み立て純ロジックの決定的テスト（CI 対象。実モデルは駆動しない）。
//
// 期待値は tools/model-tools の Python リファレンス（synth_aivmx.py の各関数）で生成した固定値。
// symbols/intersperse/tone+6/add_blank word2ph doubling/bert-tile/style_vec を、
// 固定の小入力・モック特徴量で数値一致検証する（CI テスト方針）。

import { assertEquals, assertThrows } from "@std/assert";
import {
  addBlankWord2ph,
  intersperse,
  phonesTonesToModelIds,
  styleVector,
  tileBertToPhoneLevel,
} from "./tensor_build.ts";
import { phonesToIds, SYMBOLS } from "../text/symbols.ts";

Deno.test("SYMBOLS は 112 要素で PAD が id 0", () => {
  assertEquals(SYMBOLS.length, 112);
  assertEquals(SYMBOLS[0], "_");
  assertEquals(SYMBOLS.at(-1), "UNK");
});

Deno.test("phonesToIds: 既知の JP 音素 ID", () => {
  // Compatibility values for the 112-symbol JP-Extra phoneme table.
  assertEquals(
    phonesToIds([
      "_",
      "a",
      "k",
      "ky",
      "N",
      "q",
      "sh",
      "ts",
      ",",
      ".",
      "SP",
      "UNK",
    ]),
    [
      0,
      8,
      57,
      58,
      5,
      73,
      77,
      80,
      106,
      107,
      110,
      111,
    ],
  );
});

Deno.test("phonesToIds: 未知音素は throw（fail loudly）", () => {
  assertThrows(
    () => phonesToIds(["a", "XX", "b"]),
    Error,
    "SYMBOLS に無い音素",
  );
});

Deno.test("intersperse: 要素間・両端に 0 を挟んで 2n+1", () => {
  assertEquals(intersperse([1, 2, 3], 0), [0, 1, 0, 2, 0, 3, 0]);
  assertEquals(intersperse([], 0), [0]);
});

Deno.test("phonesTonesToModelIds: Python リファレンスと完全一致", () => {
  const phones = ["_", "k", "o", "N", "n", "i", "ch", "i", "_"];
  const tones = [0, 0, 0, 1, 1, 1, 1, 1, 0];
  const { phoneIds, toneIds, languageIds } = phonesTonesToModelIds(
    phones,
    tones,
  );
  // Python: _phones_tones_to_model_ids の実測値。
  assertEquals(phoneIds, [
    0,
    0,
    0,
    57,
    0,
    65,
    0,
    5,
    0,
    62,
    0,
    40,
    0,
    22,
    0,
    40,
    0,
    0,
    0,
  ]);
  assertEquals(toneIds, [
    0,
    6,
    0,
    6,
    0,
    6,
    0,
    7,
    0,
    7,
    0,
    7,
    0,
    7,
    0,
    7,
    0,
    6,
    0,
  ]);
  assertEquals(languageIds, [
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
  ]);
  // add_blank 後の長さは 2n+1。
  assertEquals(phoneIds.length, phones.length * 2 + 1);
});

Deno.test("phonesTonesToModelIds: phones/tones 長さ不一致は throw", () => {
  assertThrows(
    () => phonesTonesToModelIds(["a", "b"], [0]),
    Error,
    "長さが不一致",
  );
});

Deno.test("addBlankWord2ph: 各要素 *2・先頭 +1（infer.py 調整）", () => {
  // Python: base [1,2,2,1] -> [3,4,4,2], sum 13。
  assertEquals(addBlankWord2ph([1, 2, 2, 1]), [3, 4, 4, 2]);
  // sum が add_blank 後の音素列長 2*len+1 に一致する不変条件。
  const base = [1, 2, 2, 1];
  const baseSum = base.reduce((a, b) => a + b, 0); // = given_phone 長
  const adjusted = addBlankWord2ph(base);
  assertEquals(adjusted.reduce((a, b) => a + b, 0), baseSum * 2 + 1);
});

Deno.test("addBlankWord2ph: 空配列は throw", () => {
  assertThrows(() => addBlankWord2ph([]), Error, "base word2ph が空");
});

Deno.test("tileBertToPhoneLevel: tile+転置が Python と一致", () => {
  // hidden [3,2] 行優先, word2ph [1,2,1] -> [2,4] 行優先。
  const hidden = Float32Array.from([10, 11, 20, 21, 30, 31]);
  const { data, length } = tileBertToPhoneLevel(hidden, 3, [1, 2, 1]);
  assertEquals(length, 4);
  // Python: phone_level.T.flatten() = [10,20,20,30, 11,21,21,31]。
  assertEquals(Array.from(data), [10, 20, 20, 30, 11, 21, 21, 31]);
});

Deno.test("tileBertToPhoneLevel: word2ph 長 != seqLen は throw", () => {
  const hidden = Float32Array.from(new Array(2 * 1024).fill(0));
  assertThrows(
    () => tileBertToPhoneLevel(hidden, 2, [1, 1, 1]),
    Error,
    "word2ph",
  );
});

Deno.test("styleVector: mean+(row-mean)*weight が Python と一致", () => {
  const styleMatrix = {
    rows: 3,
    cols: 3,
    data: Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9]),
  };
  // mean = [4,5,6]（cols 平均）。cols!=256 チェックは 256 前提だがテストは cols=3 で通したいので
  // styleVector は cols===256 を強制する。ここでは 256 に合わせた別テストにするため mock を 256 に拡張する。
  // → cols=3 では styleVector が throw するので、cols=256 のケースで検証する（下の別テスト）。
  assertThrows(() => styleVector(styleMatrix, 0, 1.0), Error, "256");
});

Deno.test("styleVector: 256 列で mean+(row-mean)*weight（Python 一致・平均行は weight 不変）", () => {
  // 3 行 256 列。行0=全て 1、行1=全て 5、行2=全て 9 → 平均は全て 5（= 行1）。
  const rows = 3;
  const cols = 256;
  const data = new Float32Array(rows * cols);
  for (let c = 0; c < cols; c++) {
    data[0 * cols + c] = 1;
    data[1 * cols + c] = 5;
    data[2 * cols + c] = 9;
  }
  const styleMatrix = { rows, cols, data };
  // sid=0, w=1.0 -> row そのもの = 1。
  assertEquals(styleVector(styleMatrix, 0, 1.0)[0], 1);
  // sid=0, w=0.5 -> mean(5)+(1-5)*0.5 = 3。
  assertEquals(styleVector(styleMatrix, 0, 0.5)[0], 3);
  // sid=1（平均行）w=0.5 -> mean(5)+(5-5)*0.5 = 5（weight 無関係）。
  assertEquals(styleVector(styleMatrix, 1, 0.5)[0], 5);
  // 全 256 列同値。
  const vec = styleVector(styleMatrix, 2, 1.0);
  assertEquals(vec.length, 256);
  assertEquals(new Set(vec).size, 1);
  assertEquals(vec[0], 9);
});

Deno.test("styleVector: styleId 範囲外は throw", () => {
  const data = new Float32Array(2 * 256);
  assertThrows(
    () => styleVector({ rows: 2, cols: 256, data }, 2, 1.0),
    Error,
    "範囲外",
  );
});
