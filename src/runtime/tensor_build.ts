// aivmx（SBV2 JP-Extra 音響モデル）入力テンソルの組み立て純ロジック。
//
// synth_aivmx.py の忠実移植。ここは onnxruntime に依存しない決定的関数のみを置き、
// テンソル化（BigInt64Array/Float32Array 化・ort.Tensor 生成）はアダプタ層で行う。
// これにより「組み立てロジック」を固定の小入力で単体テストでき、実モデルを CI に載せずに済む。
//
// 対応表（synth_aivmx.py の関数 → 本ファイル）:
//   _intersperse                → intersperse
//   _phones_tones_to_model_ids  → phonesTonesToModelIds
//   _word2ph_for_bert の *2/[0]+=1 → addBlankWord2ph
//   DebertaBertExtractor.extract の tile+転置 → tileBertToPhoneLevel
//   _style_vector               → styleVector

import {
  JP_LANGUAGE_ID,
  JP_TONE_OFFSET,
  phonesToIds,
} from "../text/symbols.ts";
import { parseNpy2dFloat32 } from "./npy.ts";

/** DeBERTa large の隠れ次元（bert 入力の 1024）。 */
export const BERT_DIM = 1024;

/**
 * add_blank=True の後処理。要素間・両端に item を挟んで 2*len+1 にする。
 * SBV2 commons.intersperse / synth_aivmx.py `_intersperse` と同一。
 */
export const intersperse = (seq: readonly number[], item: number): number[] => {
  const result = new Array<number>(seq.length * 2 + 1).fill(item);
  for (let i = 0; i < seq.length; i++) result[i * 2 + 1] = seq[i];
  return result;
};

/** phones→ID / tone+6 / lang=1 を add_blank 適用済みで返した束。 */
export type ModelIdSequences = {
  /** add_blank 後の音素 ID 列（長さ 2*len+1）。 */
  phoneIds: number[];
  /** add_blank 後のトーン ID 列（JP は +6 済み、長さ 2*len+1）。 */
  toneIds: number[];
  /** add_blank 後の言語 ID 列（JP は全 1、長さ 2*len+1）。 */
  languageIds: number[];
};

/**
 * phones/tones を aivmx 入力用の ID 列（add_blank 適用済み）に変換する。
 * synth_aivmx.py `_phones_tones_to_model_ids` の忠実移植。
 *
 * 1) 音素→ID（SYMBOLS 表）/ トーン +6（JP オフセット）/ 言語 ID=1（JP）
 * 2) add_blank intersperse（phone/tone/language を 0 で挟んで 2*len+1）
 *
 * phones と tones は同じ長さ（両端 "_" 込みの given_phone/tone）である前提。長さ不一致は throw。
 */
export const phonesTonesToModelIds = (
  phones: readonly string[],
  tones: readonly number[],
): ModelIdSequences => {
  if (phones.length !== tones.length) {
    throw new Error(
      `phonesTonesToModelIds: phones(${phones.length}) と tones(${tones.length}) の長さが不一致`,
    );
  }
  const phoneIdsRaw = phonesToIds(phones);
  const toneIdsRaw = tones.map((t) => t + JP_TONE_OFFSET);
  const languageIdsRaw = phoneIdsRaw.map(() => JP_LANGUAGE_ID);
  return {
    phoneIds: intersperse(phoneIdsRaw, 0),
    toneIds: intersperse(toneIdsRaw, 0),
    languageIds: intersperse(languageIdsRaw, 0),
  };
};

/**
 * base word2ph（add_blank 前）から add_blank 後の word2ph を作る。
 * synth_aivmx.py `_word2ph_for_bert` / SBV2 infer.py の忠実移植:
 *   各要素を *2 し、先頭に +1（add_blank の先頭ブランク分）。
 *
 * これにより sum(word2ph) が add_blank 後の音素列長 2*len+1 に一致する。
 * fail loudly: baseWord2ph が空だと先頭 +1 が不能なので throw。
 */
export const addBlankWord2ph = (baseWord2ph: readonly number[]): number[] => {
  if (baseWord2ph.length === 0) {
    throw new Error(
      "addBlankWord2ph: base word2ph が空（両端 [1] 番兵が欠落）",
    );
  }
  const doubled = baseWord2ph.map((w) => w * 2);
  doubled[0] += 1;
  return doubled;
};

/**
 * DeBERTa の文字ごと隠れ状態 [seqLen, 1024] を word2ph で音素レベルへ tile 展開し、
 * 転置して [1024, T]（T = sum(word2ph)）にする。
 * synth_aivmx.py `DebertaBertExtractor.extract` の tile+転置の忠実移植。
 *
 * @param hidden 行優先の [seqLen, dim] float32（DeBERTa 出力そのまま）。
 * @param seqLen DeBERTa トークン数（= hidden.length / dim）。
 * @param word2ph add_blank 後の word2ph（長さ == seqLen が必須）。
 * @returns { data, length }: data は行優先の [dim, length]（bert 入力の中身、batch 次元は付けない）。
 *
 * dim は hidden.length / seqLen から導出する（tile 自体は次元非依存。テストは小 dim で書ける）。
 * 不変条件（fail loudly）:
 *   - hidden.length が seqLen で割り切れる（矩形）。
 *   - word2ph.length === seqLen（DeBERTa トークン数と word2ph 長の一致）。
 */
export const tileBertToPhoneLevel = (
  hidden: Float32Array,
  seqLen: number,
  word2ph: readonly number[],
): { data: Float32Array; length: number } => {
  if (seqLen <= 0 || hidden.length % seqLen !== 0) {
    throw new Error(
      `tileBertToPhoneLevel: hidden 長 ${hidden.length} が seqLen ${seqLen} で割り切れない（矩形でない）`,
    );
  }
  const dim = hidden.length / seqLen;
  if (word2ph.length !== seqLen) {
    throw new Error(
      `tileBertToPhoneLevel: DeBERTa トークン数 ${seqLen} != word2ph 長 ${word2ph.length}` +
        "（文字トークナイズと word2ph の齟齬を疑う）",
    );
  }
  let total = 0;
  for (const w of word2ph) total += w;
  // 出力は [dim, total] 行優先。out[d * total + p] = hidden[srcToken(p) * dim + d]。
  // np.concatenate([np.tile(res[i], (word2ph[i],1)) ... ]).T と同値。
  const out = new Float32Array(dim * total);
  let phoneCol = 0;
  for (let token = 0; token < seqLen; token++) {
    const repeat = word2ph[token];
    const srcBase = token * dim;
    for (let r = 0; r < repeat; r++) {
      const col = phoneCol + r;
      for (let d = 0; d < dim; d++) {
        out[d * total + col] = hidden[srcBase + d];
      }
    }
    phoneCol += repeat;
  }
  return { data: out, length: total };
};

/**
 * スタイル行列（[numStyles, 256]）から 1 スタイルを選び、
 * mean + (row - mean) * weight で [256] のスタイルベクトルを作る。
 * synth_aivmx.py `_style_vector` / aivmx-interface.md §2.6 の忠実移植。
 *
 * @returns 長さ 256 の float32（batch 次元は付けない）。
 */
export const styleVector = (
  styleMatrix: { rows: number; cols: number; data: Float32Array },
  styleId: number,
  weight: number,
): Float32Array => {
  const { rows, cols, data } = styleMatrix;
  if (cols !== 256) {
    throw new Error(`styleVector: style 行列の列数が 256 でない: ${cols}`);
  }
  // 整数性も検査する。非整数は範囲チェックだけだと素通りし、行を跨いだ読み出し
  // （styleId=1.5 → rowBase=384）や範囲外 undefined→NaN ベクトル（styleId=4.5, rows=5）を
  // 黙って作る。リファレンス numpy の style_matrix[style_id] は非整数 index で必ず
  // IndexError なので、throw がパリティ。
  if (!(Number.isInteger(styleId) && styleId >= 0 && styleId < rows)) {
    throw new Error(
      `styleVector: styleId ${styleId} が範囲外（0..${rows - 1} の整数を期待）`,
    );
  }
  // weight の NaN/±Infinity はベクトル全体を無言で汚染する（styleId と同じ穴の類型）。
  if (!Number.isFinite(weight)) {
    throw new Error(`styleVector: weight が有限数でない: ${weight}`);
  }
  // 列ごとの平均。
  const mean = new Float32Array(cols);
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    for (let c = 0; c < cols; c++) mean[c] += data[base + c];
  }
  for (let c = 0; c < cols; c++) mean[c] /= rows;

  const rowBase = styleId * cols;
  const vec = new Float32Array(cols);
  for (let c = 0; c < cols; c++) {
    vec[c] = mean[c] + (data[rowBase + c] - mean[c]) * weight;
  }
  return vec;
};

/**
 * aivm_style_vectors の .npy バイト列をパースして style 行列にする。
 * [N, 256] float32 でなければ throw（aivmx-interface.md §2.6）。
 */
export const parseStyleMatrix = (
  npyBytes: Uint8Array,
): { rows: number; cols: number; data: Float32Array } => {
  const matrix = parseNpy2dFloat32(npyBytes);
  if (matrix.cols !== 256) {
    throw new Error(
      `style_vectors の shape が想定外: [${matrix.rows}, ${matrix.cols}]（[N, 256] を期待）`,
    );
  }
  return matrix;
};
