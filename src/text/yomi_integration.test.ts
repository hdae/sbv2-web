// 実 yomi 解析に対する統合凍結テスト。
//
// sbv2-web の中核結合は「toSbv2PhoneTone は result 由来、buildBaseWord2ph / toBertText は
// words（wordPhoneAlignment）由来で、両者が sum(word2ph) === phones.length で整合する」
// という yomi の cross-output 不変条件（flatMap(words.phones) === leadingPunctuations +
// Σ句(モーラ音素 + punctuations)）への依存で成り立つ。手組み fixture のテストは作者が
// result と words を手で一致させているだけなので、ここでは**実辞書・実解析**で
// この関係を凍結する（結合を所有する層が実物で pin する）。
//
// 実辞書はワークスペース同居の yomi リポジトリの fixture を使い、無ければ skip する
// （CI には辞書が無い — light-sbv2 の統合テストと同じ扱い）。

import { analyzeWithWords, JtdDictionary } from "@hdae/yomi";
import { toSbv2PhoneTone } from "./phone_tone.ts";
import { toBertText } from "./bert_text.ts";
import { buildBaseWord2ph } from "./word2ph.ts";
import { DebertaTokenizer } from "./deberta_tokenizer.ts";
import { validateSynthInput } from "../runtime/validate_input.ts";

const DICT_PATH = "../yomi/fixtures/naist-jdic.jtd";

const loadDict = (): JtdDictionary | undefined => {
  try {
    const bytes = Deno.readFileSync(DICT_PATH);
    return JtdDictionary.load(bytes.buffer);
  } catch {
    return undefined;
  }
};

/**
 * 文字ごと 1 トークンの最小 DebertaTokenizer（vocab 空 = 全部 [UNK] id でも
 * tokenize() の文字数計数は本物と同一 — word2ph の検証にはこれで十分）。
 */
const minimalTokenizer = (): DebertaTokenizer =>
  new DebertaTokenizer(new Map(), { removed: [], spaced: [] }, {
    clsId: 0,
    sepId: 1,
    unkId: 2,
  });

const assert = (cond: boolean, msg: string) => {
  if (!cond) throw new Error(msg);
};

// 実在記号（！？…、。）・先頭記号・記号のみ・長音・数字・ASCII 混在を含む一式。
const TEXTS = [
  "こんにちは。",
  "元気？",
  "え、そう。ね",
  "…ねえ",
  "！？",
  "コーヒーを飲む",
  "待って！！なんで…",
  "3.14 は円周率",
  "ハロー、ワールド！",
  "",
];

Deno.test({
  name: "実 yomi 統合: 音素順序・word2ph・bertText の整合が実解析で成立する",
  ignore: loadDict() === undefined,
  fn() {
    const dict = loadDict()!;
    const tokenizer = minimalTokenizer();
    for (const text of TEXTS) {
      const { result, words } = analyzeWithWords(dict, text);
      const { phones, tones } = toSbv2PhoneTone(result);

      // 不変条件 A（順序）: 非 PAD 音素列 === wordPhoneAlignment の音素連結。
      // これが phone_tone.ts の MUST（leading → 句毎モーラ → 句毎記号）の実物凍結。
      const nonPad = phones.slice(1, -1);
      const fromWords = words.flatMap((w) => w.phones);
      assert(
        JSON.stringify(nonPad) === JSON.stringify(fromWords),
        `順序不変条件が破れた: ${JSON.stringify(text)}\n` +
          ` phones(非PAD): ${JSON.stringify(nonPad)}\n` +
          ` words 連結:    ${JSON.stringify(fromWords)}`,
      );

      // 不変条件 B（word2ph）: buildBaseWord2ph が内部の fail-loud 検査
      // （長さ = Σtokenize(surface)+2、総和 = phones.length）を通る。
      const baseWord2ph = buildBaseWord2ph(words, tokenizer, phones.length);

      // 不変条件 C（bertText）: 記号正規形化後もトークン数が word2ph と一致し、
      // phones が全て既知 SBV2 記号・tones が 0/1（validateSynthInput が一括検証）。
      validateSynthInput(
        {
          phones,
          tones,
          bertText: toBertText(words),
          baseWord2ph,
          styleId: 0,
          styleWeight: 1.0,
          speakerId: 0,
        },
        tokenizer,
      );
    }
  },
});
