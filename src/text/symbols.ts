// JP-Extra acoustic models use this stable 112-symbol phoneme table.

/** Phoneme table used by compatible JP-Extra acoustic models. The index is the model phoneme ID. */
export const SYMBOLS: readonly string[] = [
  "_",
  "AA",
  "E",
  "EE",
  "En",
  "N",
  "OO",
  "V",
  "a",
  "a:",
  "aa",
  "ae",
  "ah",
  "ai",
  "an",
  "ang",
  "ao",
  "aw",
  "ay",
  "b",
  "by",
  "c",
  "ch",
  "d",
  "dh",
  "dy",
  "e",
  "e:",
  "eh",
  "ei",
  "en",
  "eng",
  "er",
  "ey",
  "f",
  "g",
  "gy",
  "h",
  "hh",
  "hy",
  "i",
  "i0",
  "i:",
  "ia",
  "ian",
  "iang",
  "iao",
  "ie",
  "ih",
  "in",
  "ing",
  "iong",
  "ir",
  "iu",
  "iy",
  "j",
  "jh",
  "k",
  "ky",
  "l",
  "m",
  "my",
  "n",
  "ng",
  "ny",
  "o",
  "o:",
  "ong",
  "ou",
  "ow",
  "oy",
  "p",
  "py",
  "q",
  "r",
  "ry",
  "s",
  "sh",
  "t",
  "th",
  "ts",
  "ty",
  "u",
  "u:",
  "ua",
  "uai",
  "uan",
  "uang",
  "uh",
  "ui",
  "un",
  "uo",
  "uw",
  "v",
  "van",
  "ve",
  "vn",
  "w",
  "x",
  "y",
  "z",
  "zh",
  "zy",
  "!",
  "?",
  "…",
  ",",
  ".",
  "'",
  "-",
  "SP",
  "UNK",
];

/**
 * JP tone offset. It shifts JP 0/1 tones to the model IDs 6/7.
 */
export const JP_TONE_OFFSET = 6;

/**
 * JP language ID used by compatible JP-Extra acoustic models.
 */
export const JP_LANGUAGE_ID = 1;

/** 音素記号 → ID（SYMBOLS の enumerate 順）。 */
export const SYMBOL_TO_ID: ReadonlyMap<string, number> = new Map(
  SYMBOLS.map((symbol, index) => [symbol, index]),
);

/**
 * 音素記号列を ID 列に変換する。未知の音素は握りつぶさず throw する（fail loudly）。
 * synth_aivmx.py `_phones_tones_to_model_ids` の KeyError surface と同じ挙動。
 */
export const phonesToIds = (phones: readonly string[]): number[] =>
  phones.map((phone) => {
    const id = SYMBOL_TO_ID.get(phone);
    if (id === undefined) {
      throw new Error(
        `SYMBOLS に無い音素 ${
          JSON.stringify(phone)
        } が phones に含まれる（ID 化不能）。` +
          " 入力の音素記号と JP-Extra モデルの記号表の齟齬を疑う。",
      );
    }
    return id;
  });
