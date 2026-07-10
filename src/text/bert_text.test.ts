// toBertText の挙動仕様: 記号要素だけ正規形へ置き換え、かな漢字はそのまま連結する。

import { toBertText } from "./bert_text.ts";

const assertEq = (actual: string, expected: string, label: string) => {
  if (actual !== expected) {
    throw new Error(
      `${label}: got ${JSON.stringify(actual)}, want ${
        JSON.stringify(expected)
      }`,
    );
  }
};

Deno.test("toBertText: 記号要素は正規形へ置き換える（本家 replace_punctuation 相当）", () => {
  assertEq(
    toBertText([
      {
        surface: "こんにちは",
        phones: ["k", "o", "N", "n", "i", "ch", "i", "w", "a"],
      },
      { surface: "。", phones: ["."] },
    ]),
    "こんにちは.",
    "句点の正規形化",
  );
  assertEq(
    toBertText([
      { surface: "ね", phones: ["n", "e"] },
      { surface: "！", phones: ["!"] },
      { surface: "？", phones: ["?"] },
    ]),
    "ね!?",
    "連続記号",
  );
});

Deno.test("toBertText: かな 1 音素語（母音のみ）は記号と誤判定しない", () => {
  assertEq(
    toBertText([{ surface: "え", phones: ["e"] }]),
    "え",
    "1 音素のかな語は surface のまま",
  );
});

Deno.test("toBertText: 空配列は空文字列", () => {
  assertEq(toBertText([]), "", "空");
});
