// FrontendResult → Style-Bert-VITS2 の given_phone / given_tone への変換。
//
// 責務境界（ADR-0001）: yomi は「モデル非依存の建材」(moraToPhones / moraTones /
// punctuations / leadingPunctuations) までを提供し、SBV2 固有の梱包 —— 両端 PAD "_"、
// トーンを音素単位へ割り当て、実在記号の音素化 —— はここ（呼び出し側 = SBV2 アダプタ）で
// 組む。建材を再利用することで音素・トーン生成の二経路化を構造的に防ぐ。
//
// SBV2 の音素・トーン規約（要点）:
// - 音素列は先頭・末尾に PAD "_"（tone 0）を必ず含む完全な列。
// - 促音は "q" 1個・撥音は "N" 1個（moraToPhones が畳む）。
// - 長音は直前母音に解決済みなので母音1個として出す（":" は使わない）。
// - トーンは 0/1 の2値。各アクセント句で独立に 0 から立ち上がる（moraTones）。
// - 記号はテキストに実在した句読点（正規形 ! ? … , . ' -）だけを tone 0 で出す。
//   実在しない記号は合成しない（本家 SBV2 の g2p と同方針）。記号の無い句境界は
//   トーンの 0 戻りとして暗黙符号化される。
//
// MUST: 出力の非 PAD 部分は yomi の wordPhoneAlignment の不変条件
//   flatMap(w => w.phones) === leadingPunctuations + Σ句の（モーラ音素 + punctuations）
// と完全一致させる（word2ph の sum(word2ph) === phones.length がこれに依存する）。

import { type FrontendResult, moraTones, moraToPhones } from "@hdae/yomi";

export type Sbv2PhoneTone = { phones: string[]; tones: number[] };

/**
 * FrontendResult を SBV2 の given_phone / given_tone 形式へ変換する。
 * phones と tones は同じ長さで、位置ごとに対応する（tone は音素単位）。
 */
export const toSbv2PhoneTone = (result: FrontendResult): Sbv2PhoneTone => {
  // 先頭 PAD。
  const phones: string[] = ["_"];
  const tones: number[] = [0];

  // 先頭句より前の実在記号（記号だけの入力は句が作られず全てここに入る）。
  for (const punct of result.leadingPunctuations) {
    phones.push(punct);
    tones.push(0);
  }

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
    // 句直後にテキスト上実在した記号（正規形・出現順）。pauseAfter からの合成はしない。
    for (const punct of phrase.punctuations) {
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
