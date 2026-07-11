// 共有可能な DeBERTa 特徴抽出器（セッション + トークナイザ + 音素レベル展開）。
//
// Sbv2Adapter はモデル（音響セッション）ごとに 1 つ生成されるが、DeBERTa は全モデル共通の
// 資産なので、アダプタ毎にセッションを複製すると BERT の常駐メモリがモデル数倍になる。
// このクラスはセッションを 1 本に束ね、複数の Sbv2Adapter から共有できるようにする
// （docs/decisions/0005）。
//
// 所有権: release() は生成者が行う（MUST — これを使う全アダプタの release 後）。
// 共有された抽出器を Sbv2Adapter 側の release() は解放しない。release 後の extract は
// throw する（fail loud — 解放済みセッションで黙って推論しない）。

import type { InferenceSession, Tensor } from "onnxruntime-common";
import type { DebertaTokenizer } from "../text/deberta_tokenizer.ts";
import type { OrtBackend, OrtSessionOptions } from "./adapter_core.ts";
import {
  addBlankWord2ph,
  BERT_DIM,
  tileBertToPhoneLevel,
} from "./tensor_build.ts";
import { OUTPUT_NAME } from "./adapter_types.ts";

/**
 * Sbv2Adapter への BERT（DeBERTa）の供給方法。
 * - `bertOnnxBytes` + `tokenizer`: アダプタが専用の抽出器を生成して所有する
 *   （アダプタの release() で一緒に解放される）。単一モデル運用向け。
 * - `deberta`: 生成済みの共有抽出器を使う（アダプタは解放しない — 所有権は生成者）。
 *   複数モデル常駐で BERT セッションを 1 本に保つ。
 */
export type BertSource =
  | {
    bertOnnxBytes: Uint8Array;
    tokenizer: DebertaTokenizer;
    deberta?: undefined;
  }
  | {
    deberta: DebertaExtractor;
    bertOnnxBytes?: undefined;
    tokenizer?: undefined;
  };

/**
 * DeBERTa を駆動して音素レベルの bert 特徴量を作る。web/node の各ラッパの
 * `createDeberta`、または `DebertaExtractor.create`（OrtBackend 注入）で生成する。
 */
export class DebertaExtractor {
  /** 同一資産セット由来のトークナイザ（モデルとの drift を防ぐため同居させる）。 */
  readonly tokenizer: DebertaTokenizer;

  readonly #backend: OrtBackend;
  readonly #session: InferenceSession;
  readonly #inputNames: readonly string[];
  /** release() が始まったら non-null（冪等化 + 以後の extract を fail loud で拒否）。 */
  #releasePromise: Promise<void> | null = null;
  /** in-flight の extract。release はこれらの完了を待ってからセッションを解放する。 */
  readonly #inflight = new Set<Promise<unknown>>();

  private constructor(args: {
    backend: OrtBackend;
    session: InferenceSession;
    tokenizer: DebertaTokenizer;
  }) {
    this.#backend = args.backend;
    this.#session = args.session;
    this.#inputNames = args.session.inputNames;
    this.tokenizer = args.tokenizer;
  }

  static async create(backend: OrtBackend, args: {
    bertOnnxBytes: Uint8Array;
    tokenizer: DebertaTokenizer;
    sessionOptions?: OrtSessionOptions;
  }): Promise<DebertaExtractor> {
    const session = await backend.createSession(
      args.bertOnnxBytes,
      args.sessionOptions,
    );
    return new DebertaExtractor({
      backend,
      session,
      tokenizer: args.tokenizer,
    });
  }

  /** release() 済みか（共有先での使用可否を生成前に検査できるよう公開する）。 */
  get isReleased(): boolean {
    return this.#releasePromise !== null;
  }

  /**
   * DeBERTa を走らせ bert 特徴量 [1024*T] を作る（add_blank 後 word2ph で tile 展開・転置）。
   * seqLen は add_blank 後の音素列長（= 2*len+1）。
   */
  extract(
    bertText: string,
    baseWord2ph: readonly number[],
    seqLen: number,
  ): Promise<Float32Array> {
    if (this.#releasePromise !== null) {
      throw new Error("DebertaExtractor: release() 後に extract が呼ばれた");
    }
    return this.#track(this.#extract(bertText, baseWord2ph, seqLen));
  }

  async #extract(
    bertText: string,
    baseWord2ph: readonly number[],
    seqLen: number,
  ): Promise<Float32Array> {
    const word2ph = addBlankWord2ph(baseWord2ph);
    const inputIds = this.tokenizer.encode(bertText); // [CLS] + 文字 + [SEP]
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
    for (const name of this.#inputNames) {
      if (name === "input_ids") feed[name] = idsTensor;
      else if (name === "attention_mask") feed[name] = attnTensor;
      else {
        throw new Error(
          `DebertaExtractor: DeBERTa の想定外入力 '${name}'（input_ids/attention_mask のみ対応）`,
        );
      }
    }
    const out = await this.#session.run(feed, [OUTPUT_NAME]);
    const hiddenTensor = out[OUTPUT_NAME];
    const dims = hiddenTensor.dims;
    if (dims.length !== 2 || dims[1] !== BERT_DIM) {
      throw new Error(
        `DebertaExtractor: DeBERTa 出力 shape が想定外 [${
          dims.join(",")
        }]（[seq_len, 1024] を期待）`,
      );
    }
    const hidden = hiddenTensor.data as Float32Array;
    if (dims[0] !== word2ph.length) {
      throw new Error(
        `DebertaExtractor: DeBERTa トークン数 ${
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
        `DebertaExtractor: bert 展開長 ${length} != 音素列長 ${seqLen}（word2ph 調整と add_blank の齟齬）`,
      );
    }
    return data;
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

  /**
   * セッションを解放する。契約は Sbv2Adapter.release と同じ（docs/decisions/0004）:
   * 冪等・in-flight の extract 完了を待つ・release 後の extract は throw。
   */
  release(): Promise<void> {
    this.#releasePromise ??= this.#release();
    return this.#releasePromise;
  }

  async #release(): Promise<void> {
    // 単一スレッドの JS では、#releasePromise 代入以降の新規 extract は同期的に
    // 拒否される。よってここで見える #inflight が全てで、待てば枯れる。
    while (this.#inflight.size > 0) {
      await Promise.allSettled([...this.#inflight]);
    }
    await this.#session.release();
  }
}
