// SBV2 JP-Extra モデルアダプタのランタイム非依存コア。
//
// web（onnxruntime-web）と node（onnxruntime-node）が同一のロジックを共有するために、
// ONNX ランタイムを OrtBackend として注入する。テンソル生成（int64/float32）と session.run の
// 差分だけを注入し、テンソル組み立て（tensor_build.ts, テスト済み）と中間表現（adapter_types.ts）は
// 両バックエンドで完全に共有する ＝ 数値挙動が構造的に一箇所に保たれる。契約は docs/aivmx-interface.md。
//
// 型は onnxruntime-common（web/node が共に再エクスポートする基底）に対して付ける。ここは型 import
// のみ（実行時 ORT 依存ゼロ）で、実体は注入された backend が供給する。

import type { InferenceSession, Tensor } from "onnxruntime-common";
import type { DebertaTokenizer } from "../text/deberta_tokenizer.ts";
import { getSamplingRate, readAivmxMetadata } from "./aivmx_meta.ts";
import {
  addBlankWord2ph,
  BERT_DIM,
  parseStyleMatrix,
  phonesTonesToModelIds,
  styleVector,
  tileBertToPhoneLevel,
} from "./tensor_build.ts";
import {
  type AcousticFeeds,
  DEFAULT_SCALARS,
  type ModelAdapter,
  OUTPUT_NAME,
  type SynthInput,
  type SynthScalars,
} from "./adapter_types.ts";

export type OrtSessionOptions = InferenceSession.SessionOptions;

/**
 * 注入する ONNX ランタイム。web/node はそれぞれの onnxruntime から生成関数を渡すだけ。
 * 独自バックエンド（別ランタイム/モック）を差し込むこともできる。
 */
export type OrtBackend = {
  /** int64 テンソルを生成する。 */
  int64(data: BigInt64Array, dims: readonly number[]): Tensor;
  /** float32 テンソルを生成する。 */
  float32(data: Float32Array, dims: readonly number[]): Tensor;
  /** ONNX バイト列からセッションを生成する。 */
  createSession(
    bytes: Uint8Array,
    options?: OrtSessionOptions,
  ): Promise<InferenceSession>;
};

type StyleMatrix = { rows: number; cols: number; data: Float32Array };

/**
 * SBV2 JP-Extra 用モデルアダプタのコア実装。aivmx 音響モデル + 実 DeBERTa を、注入された
 * OrtBackend で駆動する。web/node のラッパはセッション生成オプション（EP/デバイス）だけを与える。
 */
export class Sbv2Adapter implements ModelAdapter {
  readonly needsBert = true;
  readonly sampleRate: number;

  /** style_vectors 行列の行数（= 選択可能なスタイル数）。 */
  get numStyles(): number {
    return this.#styleMatrix.rows;
  }

  readonly #backend: OrtBackend;
  readonly #acoustic: InferenceSession;
  readonly #acousticInputNames: readonly string[];
  readonly #bert: InferenceSession;
  readonly #bertInputNames: readonly string[];
  readonly #tokenizer: DebertaTokenizer;
  readonly #styleMatrix: StyleMatrix;
  readonly #scalars: SynthScalars;

  private constructor(args: {
    backend: OrtBackend;
    acoustic: InferenceSession;
    bert: InferenceSession;
    tokenizer: DebertaTokenizer;
    styleMatrix: StyleMatrix;
    sampleRate: number;
    scalars: SynthScalars;
  }) {
    this.#backend = args.backend;
    this.#acoustic = args.acoustic;
    this.#acousticInputNames = args.acoustic.inputNames;
    this.#bert = args.bert;
    this.#bertInputNames = args.bert.inputNames;
    this.#tokenizer = args.tokenizer;
    this.#styleMatrix = args.styleMatrix;
    this.sampleRate = args.sampleRate;
    this.#scalars = args.scalars;
  }

  /** Build from a plain acoustic ONNX file plus separate style vectors. */
  static async fromOnnx(backend: OrtBackend, args: {
    acousticOnnxBytes: Uint8Array;
    bertOnnxBytes: Uint8Array;
    tokenizer: DebertaTokenizer;
    styleVectorsNpy: Uint8Array;
    sampleRate?: number;
    scalars?: SynthScalars;
    sessionOptions?: OrtSessionOptions;
  }): Promise<Sbv2Adapter> {
    // 純検証を先に済ませる（ここで落ちればセッション未生成のまま fail loud できる）。
    const styleMatrix = parseStyleMatrix(args.styleVectorsNpy);
    const acoustic = await backend.createSession(
      args.acousticOnnxBytes,
      args.sessionOptions,
    );
    let bert: InferenceSession;
    try {
      bert = await backend.createSession(
        args.bertOnnxBytes,
        args.sessionOptions,
      );
    } catch (error) {
      // static ファクトリはインスタンスを返す前に throw すると呼び出し側が acoustic を
      // 解放できない（ハンドルが無い）。生成済みセッションを解放してから投げ直す。
      // 解放自体の失敗は元エラーを隠さないよう握りつぶす（エラー経路の後始末に限る）。
      await acoustic.release().catch(() => {});
      throw error;
    }
    return new Sbv2Adapter({
      backend,
      acoustic,
      bert,
      tokenizer: args.tokenizer,
      styleMatrix,
      sampleRate: args.sampleRate ?? 44100,
      scalars: args.scalars ?? DEFAULT_SCALARS,
    });
  }

  /** Build from an AIVMX file. Style vectors and sample rate come from ONNX metadata. */
  static async fromAivmx(backend: OrtBackend, args: {
    aivmxBytes: Uint8Array;
    bertOnnxBytes: Uint8Array;
    tokenizer: DebertaTokenizer;
    sampleRate?: number;
    scalars?: SynthScalars;
    sessionOptions?: OrtSessionOptions;
  }): Promise<Sbv2Adapter> {
    const metadata = readAivmxMetadata(args.aivmxBytes);
    return await Sbv2Adapter.fromOnnx(backend, {
      acousticOnnxBytes: args.aivmxBytes,
      bertOnnxBytes: args.bertOnnxBytes,
      tokenizer: args.tokenizer,
      styleVectorsNpy: metadata.styleVectorsNpy,
      sampleRate: args.sampleRate ?? getSamplingRate(metadata.hyperParameters),
      scalars: args.scalars,
      sessionOptions: args.sessionOptions,
    });
  }

  /** DeBERTa を走らせ bert 特徴量 [1024*T] を作る（add_blank 後 word2ph で tile 展開・転置）。 */
  async #extractBert(
    bertText: string,
    baseWord2ph: readonly number[],
    seqLen: number,
  ): Promise<Float32Array> {
    const word2ph = addBlankWord2ph(baseWord2ph);
    const inputIds = this.#tokenizer.encode(bertText); // [CLS] + 文字 + [SEP]
    const tokenLen = inputIds.length;
    const idsTensor = this.#backend.int64(
      BigInt64Array.from(inputIds, BigInt),
      [1, tokenLen],
    );
    const attnTensor = this.#backend.int64(
      BigInt64Array.from(new Array(tokenLen).fill(1), BigInt),
      [1, tokenLen],
    );
    const feed: Record<string, Tensor> = {};
    for (const name of this.#bertInputNames) {
      if (name === "input_ids") feed[name] = idsTensor;
      else if (name === "attention_mask") feed[name] = attnTensor;
      else {
        throw new Error(
          `Sbv2Adapter: DeBERTa の想定外入力 '${name}'（input_ids/attention_mask のみ対応）`,
        );
      }
    }
    const out = await this.#bert.run(feed, [OUTPUT_NAME]);
    const hiddenTensor = out[OUTPUT_NAME];
    const dims = hiddenTensor.dims;
    if (dims.length !== 2 || dims[1] !== BERT_DIM) {
      throw new Error(
        `Sbv2Adapter: DeBERTa 出力 shape が想定外 [${
          dims.join(",")
        }]（[seq_len, 1024] を期待）`,
      );
    }
    const hidden = hiddenTensor.data as Float32Array;
    if (dims[0] !== word2ph.length) {
      throw new Error(
        `Sbv2Adapter: DeBERTa トークン数 ${
          dims[0]
        } != word2ph 長 ${word2ph.length}` +
          `（bertText=${
            JSON.stringify(bertText)
          }）。文字トークナイズと word2ph の齟齬を疑う。`,
      );
    }
    const { data, length } = tileBertToPhoneLevel(hidden, dims[0], word2ph);
    if (length !== seqLen) {
      throw new Error(
        `Sbv2Adapter: bert 展開長 ${length} != 音素列長 ${seqLen}（word2ph 調整と add_blank の齟齬）`,
      );
    }
    return data;
  }

  /** 入力テンソル束を組み立てる（run 前。パリティ検証で再利用できるよう分離）。 */
  async buildAcousticFeeds(input: SynthInput): Promise<AcousticFeeds> {
    const { phoneIds, toneIds, languageIds } = phonesTonesToModelIds(
      input.phones,
      input.tones,
    );
    const seqLen = phoneIds.length; // 2*len+1

    const bert = await this.#extractBert(
      input.bertText,
      input.baseWord2ph,
      seqLen,
    );
    const styleVec = styleVector(
      this.#styleMatrix,
      input.styleId,
      input.styleWeight,
    );

    return {
      xTst: BigInt64Array.from(phoneIds, BigInt),
      xTstLengths: BigInt64Array.from([seqLen], BigInt),
      sid: BigInt64Array.from([input.speakerId], BigInt),
      tones: BigInt64Array.from(toneIds, BigInt),
      language: BigInt64Array.from(languageIds, BigInt),
      bert,
      styleVec,
      seqLen,
      scalars: this.#scalars,
    };
  }

  async synthesize(input: SynthInput): Promise<Float32Array> {
    const feeds = await this.buildAcousticFeeds(input);
    const T = feeds.seqLen;
    // グラフの入力名で名前束縛（決め打ち禁止, aivmx-interface.md §2.1）。
    const tensors: Record<string, Tensor> = {
      x_tst: this.#backend.int64(feeds.xTst, [1, T]),
      x_tst_lengths: this.#backend.int64(feeds.xTstLengths, [1]),
      sid: this.#backend.int64(feeds.sid, [1]),
      tones: this.#backend.int64(feeds.tones, [1, T]),
      language: this.#backend.int64(feeds.language, [1, T]),
      bert: this.#backend.float32(feeds.bert, [1, BERT_DIM, T]),
      style_vec: this.#backend.float32(feeds.styleVec, [1, 256]),
      length_scale: this.#backend.float32(
        Float32Array.from([feeds.scalars.lengthScale]),
        [],
      ),
      sdp_ratio: this.#backend.float32(
        Float32Array.from([feeds.scalars.sdpRatio]),
        [],
      ),
      noise_scale: this.#backend.float32(
        Float32Array.from([feeds.scalars.noiseScale]),
        [],
      ),
      noise_scale_w: this.#backend.float32(
        Float32Array.from([feeds.scalars.noiseScaleW]),
        [],
      ),
    };
    const feed: Record<string, Tensor> = {};
    const missing: string[] = [];
    for (const name of this.#acousticInputNames) {
      const t = tensors[name];
      if (t === undefined) missing.push(name);
      else feed[name] = t;
    }
    if (missing.length > 0) {
      throw new Error(
        `Sbv2Adapter: aivmx 入力名 ${
          JSON.stringify(missing)
        } に対応テンソルが無い` +
          `（グラフ入力=${JSON.stringify(this.#acousticInputNames)}）`,
      );
    }
    const out = await this.#acoustic.run(feed, [OUTPUT_NAME]);
    const wave = out[OUTPUT_NAME];
    // [1, 1, N] float32 → reshape(-1)（synth_aivmx.py の raw.reshape(-1) と同じ扱い）。
    return wave.data as Float32Array;
  }

  /** セッションを解放する（大きなモデル 2 本を保持するため明示 release 用）。 */
  async release(): Promise<void> {
    await this.#acoustic.release();
    await this.#bert.release();
  }
}
