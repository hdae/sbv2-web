// FrontendResult → Style-Bert-VITS2 の given_phone / given_tone への変換。
//
// 責務境界（ADR-0001）: yomi は「モデル非依存の建材」(moraToPhones / moraTones /
// pausePunct) までを提供し、SBV2 固有の梱包 —— 両端 PAD "_"、トーンを音素単位へ
// 割り当て、句読点挿入 —— はここ（呼び出し側 = SBV2 アダプタ）で組む。建材を再利用
// することで音素・トーン生成の二経路化を構造的に防ぐ。
//
// SBV2 の音素・トーン規約（要点）:
// - 音素列は先頭・末尾に PAD "_"（tone 0）を必ず含む完全な列。
// - 促音は "q" 1個・撥音は "N" 1個（moraToPhones が畳む）。
// - 長音は直前母音に解決済みなので母音1個として出す（":" は使わない）。
// - トーンは 0/1 の2値。各アクセント句で独立に 0 から立ち上がる（moraTones）。
// - 句境界そのものを表す記号は phone 列に無く、トーンの 0 戻りとして暗黙符号化される。

import {
  type FrontendResult,
  moraTones,
  moraToPhones,
  pausePunct,
} from "@hdae/yomi";

export type Sbv2PhoneTone = { phones: string[]; tones: number[] };

/**
 * FrontendResult を SBV2 の given_phone / given_tone 形式へ変換する。
 * phones と tones は同じ長さで、位置ごとに対応する（tone は音素単位）。
 */
export const toSbv2PhoneTone = (result: FrontendResult): Sbv2PhoneTone => {
  // 縮退: 句が無ければ両端 PAD のみ。
  if (result.accentPhrases.length === 0) {
    return { phones: ["_", "_"], tones: [0, 0] };
  }

  // 先頭 PAD。
  const phones: string[] = ["_"];
  const tones: number[] = [0];

  for (const phrase of result.accentPhrases) {
    const perMoraTone = moraTones(phrase.accentNucleus, phrase.moras.length);
    for (let i = 0; i < phrase.moras.length; i++) {
      const tone = perMoraTone[i];
      // NOTE: 1モーラを [consonant, vowel] に展開したとき、子音・母音とも同一トーンを振る。
      for (const phone of moraToPhones(phrase.moras[i])) {
        phones.push(phone);
        tones.push(tone);
      }
    }
    // 句直後の punctuation（ポーズ）。none は句境界をトーンの0戻りだけで表す。
    const punct = pausePunct(phrase.pauseAfter);
    if (punct !== undefined) {
      phones.push(punct);
      tones.push(0);
    }
  }

  // 末尾 PAD。
  phones.push("_");
  tones.push(0);

  // 不変条件: phone と tone は位置対応するので長さ一致必須。fail loudly。
  if (phones.length !== tones.length) {
    throw new Error(
      `phone_tone invariant broken: phones.length(${phones.length}) !== tones.length(${tones.length})`,
    );
  }
  return { phones, tones };
};
