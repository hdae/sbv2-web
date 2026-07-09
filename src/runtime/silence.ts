// 波形の無音パディング・無音連結（純ロジック、ONNX 非依存）。
//
// 本家の対応物: AivisSpeech Engine の prePhonemeLength / postPhonemeLength は
// 出力波形の前後へのゼロ詰め、Style-Bert-VITS2 の line_split / split_interval は
// 行ごとの合成波形をゼロ無音で連結する後処理で、いずれもモデル外の操作。
// 句読点・アクセント句間のポーズはここではなくモデル内の duration 予測が担う
// （given_phone に句読点記号を入れる既存経路。docs/decisions/0003 参照）。

const samplesOf = (
  sampleRate: number,
  seconds: number,
  label: string,
): number => {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new Error(`silence: sampleRate が不正: ${sampleRate}`);
  }
  if (!Number.isFinite(seconds) || seconds < 0) {
    throw new Error(
      `silence: ${label} が不正（0 以上の有限秒数を期待): ${seconds}`,
    );
  }
  return Math.round(sampleRate * seconds);
};

/**
 * 波形の前後に無音（ゼロ）を付け足す（AivisSpeech の pre/postPhonemeLength 相当）。
 * 両方 0 ならコピーせず入力をそのまま返す。
 */
export const padSilence = (
  wave: Float32Array,
  sampleRate: number,
  opts: { preSec?: number; postSec?: number },
): Float32Array => {
  const pre = samplesOf(sampleRate, opts.preSec ?? 0, "preSec");
  const post = samplesOf(sampleRate, opts.postSec ?? 0, "postSec");
  if (pre === 0 && post === 0) return wave;
  const out = new Float32Array(pre + wave.length + post);
  out.set(wave, pre);
  return out;
};

/**
 * 複数の波形を無音（ゼロ）を挟んで連結する（SBV2 の line_split / split_interval 相当。
 * 行分割合成は「行ごとに synthesizeText → 本関数で連結」のレシピで組む）。
 */
export const concatWithSilence = (
  waves: readonly Float32Array[],
  sampleRate: number,
  intervalSec: number,
): Float32Array => {
  const gap = samplesOf(sampleRate, intervalSec, "intervalSec");
  if (waves.length === 0) return new Float32Array(0);
  const total = waves.reduce((sum, wave) => sum + wave.length, 0) +
    gap * (waves.length - 1);
  const out = new Float32Array(total);
  let offset = 0;
  for (let i = 0; i < waves.length; i++) {
    if (i > 0) offset += gap;
    out.set(waves[i], offset);
    offset += waves[i].length;
  }
  return out;
};
