// テスト専用のモック OrtBackend。
//
// onnxruntime を一切ロードせずに Sbv2Adapter のライフサイクル（セッション生成・解放）と
// feed 組み立てを検証する（配布物には含めない: deno.json publish.exclude の src/testing/**）。
// InferenceSession / Tensor は onnxruntime-common の巨大な interface なので、テスト境界に
// 限定した cast で受け渡す（本体コードでは cast しない）。

import type { InferenceSession, Tensor } from "onnxruntime-common";
import type { OrtBackend } from "../runtime/adapter_core.ts";

/** モックが生成するテンソルの中身（アサーション用に素の値を保持する）。 */
export type MockTensorRecord = {
  kind: "int64" | "float32";
  data: BigInt64Array | Float32Array;
  dims: readonly number[];
};

/** OrtBackend が返した Tensor をモック実体に戻す（テスト専用の限定 cast）。 */
export const asMockTensor = (tensor: Tensor): MockTensorRecord =>
  tensor as unknown as MockTensorRecord;

/** run に渡された feed を記録し、runImpl の応答を返す最小セッション。 */
export class MockSession {
  /** release() が呼ばれた回数。 */
  released = 0;
  /** run() に渡された feed（呼び出し順）。 */
  readonly runFeeds: Array<Record<string, Tensor>> = [];

  constructor(
    readonly inputNames: readonly string[] = [],
    /** run の出力を作る（省略時は reject。Promise を返せば in-flight を保留できる）。 */
    readonly runImpl?: (
      feeds: Record<string, Tensor>,
    ) => Record<string, unknown> | Promise<Record<string, unknown>>,
  ) {}

  run(feeds: Record<string, Tensor>): Promise<Record<string, unknown>> {
    this.runFeeds.push(feeds);
    if (this.runImpl === undefined) {
      return Promise.reject(new Error("MockSession: runImpl 未設定"));
    }
    return Promise.resolve(this.runImpl(feeds));
  }

  release(): Promise<void> {
    this.released += 1;
    return Promise.resolve();
  }

  /** OrtBackend 契約の InferenceSession として渡す（テスト専用の限定 cast）。 */
  asSession(): InferenceSession {
    return this as unknown as InferenceSession;
  }
}

export type MockBackend = {
  backend: OrtBackend;
  /** createSession が生成したセッション（呼び出し順）。 */
  sessions: MockSession[];
};

/**
 * createSession が factories を先頭から順に消費する OrtBackend を作る。
 * factory が throw するとそのまま伝播する（失敗注入用）。int64/float32 は
 * MockTensorRecord を返すので、feed の中身は asMockTensor で検証できる。
 */
export const createMockBackend = (
  factories: ReadonlyArray<() => MockSession>,
): MockBackend => {
  const sessions: MockSession[] = [];
  let index = 0;
  const record = (r: MockTensorRecord): Tensor => r as unknown as Tensor;
  const backend: OrtBackend = {
    int64: (data, dims) => record({ kind: "int64", data, dims }),
    float32: (data, dims) => record({ kind: "float32", data, dims }),
    // deno-lint-ignore require-await
    createSession: async (_bytes, _options) => {
      const factory = factories[index];
      index += 1;
      if (factory === undefined) {
        throw new Error("createMockBackend: factories が枯渇した");
      }
      const session = factory();
      sessions.push(session);
      return session.asSession();
    },
  };
  return { backend, sessions };
};
