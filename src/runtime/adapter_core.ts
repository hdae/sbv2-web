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
import { type BertSource, DebertaExtractor } from "./deberta_extractor.ts";
import {
  type AivmxMetadata,
  readAivmxMetadata,
  type Sbv2HyperParameters,
} from "./aivmx_meta.ts";
import {
  BERT_DIM,
  parseStyleMatrix,
  phonesTonesToModelIds,
  styleVector,
} from "./tensor_build.ts";
import {
  type AcousticFeeds,
  DEFAULT_SCALARS,
  mergeScalars,
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

/** BertSource の解決結果（所有 or 共有）。 */
type ResolvedBertSource =
  | { kind: "own"; bertOnnxBytes: Uint8Array; tokenizer: DebertaTokenizer }
  | { kind: "shared"; deberta: DebertaExtractor };

/**
 * BertSource を検証して所有/共有に解決する。型を欺く JS 呼び出し（両方 or どちらも
 * 指定なし）にも fail loud する — セッションの所有者が曖昧なまま生成しない。
 */
const resolveBertSource = (args: BertSource): ResolvedBertSource => {
  if (args.deberta !== undefined) {
    if (args.bertOnnxBytes !== undefined || args.tokenizer !== undefined) {
      throw new Error(
        "Sbv2Adapter: deberta と bertOnnxBytes/tokenizer は同時指定できない" +
          "（BERT セッションの所有者が曖昧になる）",
      );
    }
    if (args.deberta.isReleased) {
      throw new Error(
        "Sbv2Adapter: release() 済みの DebertaExtractor は使えない",
      );
    }
    return { kind: "shared", deberta: args.deberta };
  }
  if (args.bertOnnxBytes === undefined || args.tokenizer === undefined) {
    throw new Error(
      "Sbv2Adapter: BERT の供給が必要（bertOnnxBytes + tokenizer、または生成済みの deberta）",
    );
  }
  return {
    kind: "own",
    bertOnnxBytes: args.bertOnnxBytes,
    tokenizer: args.tokenizer,
  };
};

/**
 * SBV2 JP-Extra 用モデルアダプタのコア実装。aivmx 音響モデル + 実 DeBERTa を、注入された
 * OrtBackend で駆動する。web/node のラッパはセッション生成オプション（EP/デバイス）だけを与える。
 *
 * DeBERTa は BertSource で供給する: bytes を渡せばアダプタが専用セッションを所有し、
 * 生成済みの DebertaExtractor を渡せば複数アダプタで 1 セッションを共有できる
 * （docs/decisions/0005 — 共有時の解放責任は生成者）。
 */
export class Sbv2Adapter implements ModelAdapter {
  readonly needsBert = true;
  readonly sampleRate: number;

  /** style_vectors 行列の行数（= 選択可能なスタイル数）。 */
  get numStyles(): number {
    return this.#styleMatrix.rows;
  }

  /** aivmx 由来の hyper_parameters（プレーン ONNX で未指定なら undefined）。 */
  get hyperParameters(): Sbv2HyperParameters | undefined {
    return this.#hyperParameters;
  }

  /** 話者数（hyper_parameters の data.n_speakers。無ければ undefined）。 */
  get numSpeakers(): number | undefined {
    return this.#hyperParameters?.nSpeakers;
  }

  /** 話者名 → speakerId（sid）のマップ（hyper_parameters の data.spk2id）。 */
  get spk2id(): Readonly<Record<string, number>> | undefined {
    return this.#hyperParameters?.spk2id;
  }

  readonly #backend: OrtBackend;
  readonly #acoustic: InferenceSession;
  readonly #acousticInputNames: readonly string[];
  readonly #deberta: DebertaExtractor;
  /** true = 専用生成（release で一緒に解放）。false = 共有（生成者が解放する）。 */
  readonly #ownsDeberta: boolean;
  readonly #styleMatrix: StyleMatrix;
  readonly #scalars: SynthScalars;
  readonly #hyperParameters?: Sbv2HyperParameters;
  /** release() が始まったら non-null（冪等化 + 以後の合成を fail loud で拒否）。 */
  #releasePromise: Promise<void> | null = null;
  /** in-flight の合成。release はこれらの完了を待ってからセッションを解放する。 */
  readonly #inflight = new Set<Promise<unknown>>();

  private constructor(args: {
    backend: OrtBackend;
    acoustic: InferenceSession;
    deberta: DebertaExtractor;
    ownsDeberta: boolean;
    styleMatrix: StyleMatrix;
    sampleRate: number;
    scalars: SynthScalars;
    hyperParameters?: Sbv2HyperParameters;
  }) {
    this.#backend = args.backend;
    this.#acoustic = args.acoustic;
    this.#acousticInputNames = args.acoustic.inputNames;
    this.#deberta = args.deberta;
    this.#ownsDeberta = args.ownsDeberta;
    this.#styleMatrix = args.styleMatrix;
    this.sampleRate = args.sampleRate;
    this.#scalars = args.scalars;
    this.#hyperParameters = args.hyperParameters;
  }

  /**
   * Build from a plain acoustic ONNX file plus separate style vectors.
   * sampleRate は必須（黙って 44100 に落とすと別レートのモデルで音程の狂った音声が
   * 無言で出る）。hyperParameters は任意で、渡すと numSpeakers/spk2id 等のアクセサが生きる。
   * 共有 deberta を渡したとき、sessionOptions が効くのは音響セッションだけ
   * （BERT 側は抽出器の生成時オプションのまま）。
   */
  static async fromOnnx(
    backend: OrtBackend,
    args: {
      acousticOnnxBytes: Uint8Array;
      styleVectorsNpy: Uint8Array;
      sampleRate: number;
      scalars?: SynthScalars;
      hyperParameters?: Sbv2HyperParameters;
      sessionOptions?: OrtSessionOptions;
    } & BertSource,
  ): Promise<Sbv2Adapter> {
    // 純検証を先に済ませる（ここで落ちればセッション未生成のまま fail loud できる）。
    const bertSource = resolveBertSource(args);
    const styleMatrix = parseStyleMatrix(args.styleVectorsNpy);
    const acoustic = await backend.createSession(
      args.acousticOnnxBytes,
      args.sessionOptions,
    );
    let deberta: DebertaExtractor;
    if (bertSource.kind === "own") {
      try {
        deberta = await DebertaExtractor.create(backend, {
          bertOnnxBytes: bertSource.bertOnnxBytes,
          tokenizer: bertSource.tokenizer,
          sessionOptions: args.sessionOptions,
        });
      } catch (error) {
        // static ファクトリはインスタンスを返す前に throw すると呼び出し側が acoustic を
        // 解放できない（ハンドルが無い）。生成済みセッションを解放してから投げ直す。
        // 解放自体の失敗は元エラーを隠さないよう握りつぶす（エラー経路の後始末に限る）。
        await acoustic.release().catch(() => {});
        throw error;
      }
    } else {
      deberta = bertSource.deberta;
    }
    return new Sbv2Adapter({
      backend,
      acoustic,
      deberta,
      ownsDeberta: bertSource.kind === "own",
      styleMatrix,
      sampleRate: args.sampleRate,
      // mergeScalars で既定値へ重ねつつ非有限値を弾く（コンストラクタ時点で検証済みにする）。
      scalars: mergeScalars(DEFAULT_SCALARS, args.scalars),
      hyperParameters: args.hyperParameters,
    });
  }

  /**
   * Build from an AIVMX file. Style vectors and sample rate come from ONNX metadata.
   * metadata に readAivmxMetadata 済みの値を渡すと巨大 protobuf の再走査を省ける
   * （レジストリ等で一度パースしている場合の最適化）。
   */
  static async fromAivmx(
    backend: OrtBackend,
    args: {
      aivmxBytes: Uint8Array;
      metadata?: AivmxMetadata;
      sampleRate?: number;
      scalars?: SynthScalars;
      sessionOptions?: OrtSessionOptions;
    } & BertSource,
  ): Promise<Sbv2Adapter> {
    const metadata = args.metadata ?? readAivmxMetadata(args.aivmxBytes);
    const sampleRate = args.sampleRate ??
      metadata.hyperParameters?.samplingRate;
    if (sampleRate === undefined) {
      throw new Error(
        "Sbv2Adapter: sampleRate を決められない（aivmx に aivm_hyper_parameters が無い。" +
          "sampleRate を明示指定するか、hyper_parameters 入りの aivmx を使う）",
      );
    }
    return await Sbv2Adapter.fromOnnx(backend, {
      acousticOnnxBytes: args.aivmxBytes,
      styleVectorsNpy: metadata.styleVectorsNpy,
      sampleRate,
      scalars: args.scalars,
      hyperParameters: metadata.hyperParameters,
      sessionOptions: args.sessionOptions,
      // BertSource は判別を保ったまま素通しする（検証は fromOnnx 側の一箇所で行う）。
      ...(args.deberta !== undefined ? { deberta: args.deberta } : {
        bertOnnxBytes: args.bertOnnxBytes,
        tokenizer: args.tokenizer,
      }),
    });
  }

  /** release 済みなら throw（fail loud）。合成系公開メソッドの入口で呼ぶ。 */
  #assertLive(method: string): void {
    if (this.#releasePromise !== null) {
      throw new Error(`Sbv2Adapter: release() 後に ${method} が呼ばれた`);
    }
  }

  /** promise を in-flight として追跡する（release はこれらの完了を待つ）。 */
  async #track<T>(promise: Promise<T>): Promise<T> {
    this.#inflight.add(promise);
    try {
      return await promise;
    } finally {
      this.#inflight.delete(promise);
    }
  }

  /** 入力テンソル束を組み立てる（run 前。パリティ検証で再利用できるよう分離）。 */
  buildAcousticFeeds(input: SynthInput): Promise<AcousticFeeds> {
    this.#assertLive("buildAcousticFeeds");
    return this.#track(this.#buildAcousticFeeds(input));
  }

  async #buildAcousticFeeds(input: SynthInput): Promise<AcousticFeeds> {
    const { phoneIds, toneIds, languageIds } = phonesTonesToModelIds(
      input.phones,
      input.tones,
    );
    const seqLen = phoneIds.length; // 2*len+1

    const bert = await this.#deberta.extract(
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
      // per-call の部分上書きをアダプタ既定へ重ねる（非有限値はここで throw）。
      scalars: mergeScalars(this.#scalars, input.scalars),
    };
  }

  synthesize(input: SynthInput): Promise<Float32Array> {
    this.#assertLive("synthesize");
    return this.#track(this.#synthesize(input));
  }

  async #synthesize(input: SynthInput): Promise<Float32Array> {
    const feeds = await this.#buildAcousticFeeds(input);
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
    // dtype を検査してから Float32Array として読む（synth_aivmx.py の
    // np.asarray(raw, dtype=np.float32) 相当の防御。fp16 等が出力境界に漏れると
    // .data が Uint16Array になり、Float32Array cast が黙って壊れた波形を通す）。
    if (wave.type !== "float32") {
      throw new Error(
        `Sbv2Adapter: 音響出力 dtype が想定外 '${wave.type}'（float32 を期待）`,
      );
    }
    // [1, 1, N] float32 → reshape(-1)（synth_aivmx.py の raw.reshape(-1) と同じ扱い）。
    return wave.data as Float32Array;
  }

  /**
   * セッションを解放する（大きなモデル 2 本を保持するため明示 release 用）。
   *
   * 契約（docs/decisions/0004）:
   * - 冪等: 2 回目以降は同じ完了を返す。
   * - in-flight の synthesize / buildAcousticFeeds の完了を待ってから解放する
   *   （推論中のネイティブセッションを引き抜かない）。
   * - release 開始後の synthesize / buildAcousticFeeds は throw（fail loud）。
   * - 共有された DebertaExtractor は解放しない（MUST NOT — 所有権は生成者。
   *   他のアダプタが同じセッションで推論中かもしれない。docs/decisions/0005）。
   */
  release(): Promise<void> {
    this.#releasePromise ??= this.#release();
    return this.#releasePromise;
  }

  async #release(): Promise<void> {
    // 単一スレッドの JS では、#releasePromise 代入以降の新規合成は #assertLive で
    // 同期的に拒否される。よってここで見える #inflight が全てで、待てば枯れる。
    while (this.#inflight.size > 0) {
      await Promise.allSettled([...this.#inflight]);
    }
    // 片方が失敗しても全ての解放を試み、失敗は握りつぶさず投げ直す。
    const releases = [this.#acoustic.release()];
    if (this.#ownsDeberta) releases.push(this.#deberta.release());
    const results = await Promise.allSettled(releases);
    const failure = results.find(
      (result): result is PromiseRejectedResult => result.status === "rejected",
    );
    if (failure !== undefined) throw failure.reason;
  }
}
