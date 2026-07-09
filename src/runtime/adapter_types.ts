// ランタイム非依存のモデルアダプタ型・定数（onnxruntime に依存しない）。
//
// web（onnxruntime-web）と node（onnxruntime-node）の両アダプタが共有する。数値組み立ては
// tensor_build.ts（純ロジック・テスト済み）に、ここには型と既定値だけを置く。契約は
// docs/aivmx-interface.md §6.2。

/**
 * モデル非依存の合成入力（本プロジェクトが生成する中間表現, aivmx-interface.md §6.2）。
 * phones/tones は given_phone/tone（両端 "_" 込み・add_blank 前）。
 */
export type SynthInput = {
  /** SBV2 音素記号列（"_"/"a"/"ky"/"N"/"q"/句読点。両端 "_" 込み・add_blank 前）。 */
  phones: readonly string[];
  /** 0/1 のトーン列（phones と同長。モデル側で +6 する）。 */
  tones: readonly number[];
  /**
   * DeBERTa に入れるテキスト。wordPhoneAlignment の surface を順に連結した文字列
   * （norm_text 直接ではない。word2ph が Σtokenize(surface) ベースのため）。
   * needsBert=false のアダプタでは無視される。
   */
  bertText: string;
  /**
   * add_blank 前の base word2ph（buildBaseWord2ph の出力, 両端 [1] 番兵込み）。
   * needsBert=false のアダプタでは無視される。
   */
  baseWord2ph: readonly number[];
  /** スタイル local_id（既定 0）。 */
  styleId: number;
  /** スタイル強度（既定 1.0）。 */
  styleWeight: number;
  /** 話者 local_id（sid, 既定 0）。 */
  speakerId: number;
};

/** モデルアダプタのスカラーパラメータ（synth_aivmx.py の既定を踏襲）。 */
export type SynthScalars = {
  lengthScale: number;
  sdpRatio: number;
  noiseScale: number;
  noiseScaleW: number;
};

/** synth_aivmx.py の既定スカラー（aivmx-interface.md §2.5）。 */
export const DEFAULT_SCALARS: SynthScalars = {
  lengthScale: 1.0,
  sdpRatio: 0.2,
  noiseScale: 0.6,
  noiseScaleW: 0.8,
};

/** モデルアダプタ共通インターフェース（aivmx-interface.md §6.2）。 */
export type ModelAdapter = {
  /** BERT 特徴量を必要とするか（SBV2=true。false なら DeBERTa をロードしない）。 */
  readonly needsBert: boolean;
  /** 出力波形のサンプルレート（Hz）。 */
  readonly sampleRate: number;
  /** 中間表現から波形（Float32Array）を得る。 */
  synthesize(input: SynthInput): Promise<Float32Array>;
};

/** aivmx 音響グラフの主波形出力名（実測: まお/コハク共通, aivmx-interface.md §2.3）。 */
export const OUTPUT_NAME = "output";

/** 組み立て済みの aivmx 入力テンソル（パリティ検証用にダンプできるよう名前付きで返す）。 */
export type AcousticFeeds = {
  xTst: BigInt64Array;
  xTstLengths: BigInt64Array;
  sid: BigInt64Array;
  tones: BigInt64Array;
  language: BigInt64Array;
  /** [1024*T] 行優先（bert 入力の中身、batch 次元なし）。 */
  bert: Float32Array;
  /** [256] スタイルベクトル（batch 次元なし）。 */
  styleVec: Float32Array;
  /** add_blank 後の音素列長 T（= 2*len+1）。 */
  seqLen: number;
  scalars: SynthScalars;
};
