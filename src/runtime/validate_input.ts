// SynthInput（phones/tones 直指定経路）の事前検証。
//
// ユーザー編集済みアクセント句からの合成（VOICEVOX 系の本線フロー）では、消費者が
// toSbv2PhoneTone / buildBaseWord2ph 相当の出力を自前構築して synthesize に渡す。
// その word2ph 不変条件は従来アダプタ内部の知識で、破れは DeBERTa 実行後にしか
// 検出されなかった。ここはそれらを公式契約として安価に fail loud する入口
// （docs/decisions/0003。契約の定義は adapter_types.ts の SynthInput と
// docs/aivmx-interface.md §6.2）。

import { SYMBOL_TO_ID } from "../text/symbols.ts";
import type { DebertaTokenizer } from "../text/deberta_tokenizer.ts";
import type { SynthInput } from "./adapter_types.ts";

/**
 * SynthInput の構造不変条件を検証する（違反は throw、正常なら何もしない）。
 *
 * 検証内容:
 * - phones が非空で全て既知の SBV2 記号、tones が同長で各値 0/1 の整数
 * - styleId / speakerId が 0 以上の整数、styleWeight / scalars が有限数
 * - baseWord2ph が両端 1 の**非負**整数列で、総和 === phones.length
 *   （0 は「その文字に音素を割り当てない」の正当値。実在する: "…" は音素 1 個だが
 *     char トークナイザの NFKC で "..." の 3 トークンに展開されるため、
 *     distributePhone(1,3) = [1,0,0] が生成経路から普通に出てくる。
 *     tile 展開は 0 個複製として扱い、総和整合はこの検査が守る）
 * - tokenizer を渡した場合のみ: baseWord2ph.length === tokenize(bertText).length + 2
 *   （DeBERTa トークン数との整合。渡さなければこの検査はスキップ）
 */
export const validateSynthInput = (
  input: SynthInput,
  tokenizer?: DebertaTokenizer,
): void => {
  const { phones, tones, baseWord2ph } = input;
  if (phones.length === 0) {
    throw new Error("validateSynthInput: phones が空");
  }
  if (phones.length !== tones.length) {
    throw new Error(
      `validateSynthInput: phones(${phones.length}) と tones(${tones.length}) の長さ不一致`,
    );
  }
  for (let i = 0; i < phones.length; i++) {
    if (!SYMBOL_TO_ID.has(phones[i])) {
      throw new Error(
        `validateSynthInput: phones[${i}] が未知の記号: ${
          JSON.stringify(phones[i])
        }`,
      );
    }
    const tone = tones[i];
    if (!(tone === 0 || tone === 1)) {
      throw new Error(
        `validateSynthInput: tones[${i}] が 0/1 でない: ${tone}` +
          "（given_tone は +6 前の 0/1）",
      );
    }
  }

  if (!(Number.isInteger(input.styleId) && input.styleId >= 0)) {
    throw new Error(
      `validateSynthInput: styleId が 0 以上の整数でない: ${input.styleId}`,
    );
  }
  if (!(Number.isInteger(input.speakerId) && input.speakerId >= 0)) {
    throw new Error(
      `validateSynthInput: speakerId が 0 以上の整数でない: ${input.speakerId}`,
    );
  }
  if (!Number.isFinite(input.styleWeight)) {
    throw new Error(
      `validateSynthInput: styleWeight が有限数でない: ${input.styleWeight}`,
    );
  }
  for (const [key, value] of Object.entries(input.scalars ?? {})) {
    if (!Number.isFinite(value)) {
      throw new Error(
        `validateSynthInput: scalars.${key} が有限数でない: ${value}`,
      );
    }
  }

  if (baseWord2ph.length < 2) {
    throw new Error(
      `validateSynthInput: baseWord2ph が短すぎる（両端番兵込みで 2 以上): ${baseWord2ph.length}`,
    );
  }
  for (let i = 0; i < baseWord2ph.length; i++) {
    const n = baseWord2ph[i];
    if (!(Number.isInteger(n) && n >= 0)) {
      throw new Error(
        `validateSynthInput: baseWord2ph[${i}] が非負整数でない: ${n}`,
      );
    }
  }
  if (baseWord2ph[0] !== 1 || baseWord2ph[baseWord2ph.length - 1] !== 1) {
    throw new Error(
      "validateSynthInput: baseWord2ph の両端番兵が 1 でない" +
        `（[${baseWord2ph[0]}, ..., ${baseWord2ph[baseWord2ph.length - 1]}]）`,
    );
  }
  const total = baseWord2ph.reduce((a, b) => a + b, 0);
  if (total !== phones.length) {
    throw new Error(
      `validateSynthInput: sum(baseWord2ph)(${total}) !== phones.length(${phones.length})`,
    );
  }

  if (tokenizer !== undefined) {
    const expected = tokenizer.tokenize(input.bertText).length + 2;
    if (baseWord2ph.length !== expected) {
      throw new Error(
        `validateSynthInput: baseWord2ph.length(${baseWord2ph.length}) !== ` +
          `tokenize(bertText).length+2(${expected})（DeBERTa トークン数と不整合）`,
      );
    }
  }
};
