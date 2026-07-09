// validateSynthInput（phones/tones 直指定契約の事前検証）の behavior テスト。

import { assertThrows } from "@std/assert";
import { validateSynthInput } from "./validate_input.ts";
import type { SynthInput } from "./adapter_types.ts";
import { DebertaTokenizer } from "../text/deberta_tokenizer.ts";

const TOKENIZER = DebertaTokenizer.fromVocabText(
  ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "あ"].join("\n"),
  { removed: [], spaced: [] },
  { clsId: 1, sepId: 2, unkId: 3 },
);

const VALID: SynthInput = {
  phones: ["_", "a", "_"],
  tones: [0, 1, 0],
  bertText: "あ",
  baseWord2ph: [1, 1, 1],
  styleId: 0,
  styleWeight: 1,
  speakerId: 0,
};

Deno.test("validateSynthInput: 正常な入力は素通り（tokenizer あり/なし）", () => {
  validateSynthInput(VALID);
  validateSynthInput(VALID, TOKENIZER);
});

Deno.test("validateSynthInput: phones/tones の構造違反を弾く", () => {
  assertThrows(
    () => validateSynthInput({ ...VALID, phones: [], tones: [] }),
    Error,
    "phones が空",
  );
  assertThrows(
    () => validateSynthInput({ ...VALID, tones: [0, 1] }),
    Error,
    "長さ不一致",
  );
  assertThrows(
    () => validateSynthInput({ ...VALID, phones: ["_", "xyz", "_"] }),
    Error,
    "未知の記号",
  );
  assertThrows(
    () => validateSynthInput({ ...VALID, tones: [0, 7, 0] }),
    Error,
    "0/1 でない",
  );
});

Deno.test("validateSynthInput: id・スカラーの数値違反を弾く", () => {
  assertThrows(
    () => validateSynthInput({ ...VALID, styleId: 1.5 }),
    Error,
    "styleId",
  );
  assertThrows(
    () => validateSynthInput({ ...VALID, speakerId: -1 }),
    Error,
    "speakerId",
  );
  assertThrows(
    () => validateSynthInput({ ...VALID, styleWeight: Number.NaN }),
    Error,
    "styleWeight",
  );
  assertThrows(
    () =>
      validateSynthInput({
        ...VALID,
        scalars: { lengthScale: Number.POSITIVE_INFINITY },
      }),
    Error,
    "scalars.lengthScale",
  );
});

Deno.test("validateSynthInput: baseWord2ph の不変条件を弾く", () => {
  assertThrows(
    () => validateSynthInput({ ...VALID, baseWord2ph: [1, 1.5, 1] }),
    Error,
    "正整数でない",
  );
  assertThrows(
    () => validateSynthInput({ ...VALID, baseWord2ph: [2, 1, 1] }),
    Error,
    "両端番兵",
  );
  assertThrows(
    // sum(baseWord2ph)=4 !== phones.length=3。
    () => validateSynthInput({ ...VALID, baseWord2ph: [1, 2, 1] }),
    Error,
    "sum(baseWord2ph)",
  );
});

Deno.test("validateSynthInput: tokenizer 併用で DeBERTa トークン数との不整合を弾く", () => {
  // bertText "ああ" は 2 トークン → 期待長 4 だが baseWord2ph は 3 要素。
  const input: SynthInput = {
    ...VALID,
    bertText: "ああ",
    phones: ["_", "a", "_"],
    baseWord2ph: [1, 1, 1],
  };
  validateSynthInput(input); // tokenizer なしでは構造検査のみ通る
  assertThrows(
    () => validateSynthInput(input, TOKENIZER),
    Error,
    "トークン数と不整合",
  );
});
