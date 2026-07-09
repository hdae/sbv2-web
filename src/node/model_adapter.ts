// @hdae/sbv2-web/node backend の薄いラッパ。
//
// 共有コア Sbv2Adapter（src/runtime/adapter_core.ts）に onnxruntime-node を注入し、EP に
// cpu / dml(DirectML) / cuda / webgpu(ORT ネイティブ Dawn) を選べるようにするだけ。テンソル
// 組み立て・合成ロジックは web と完全に共有する。
//
// NOTE(実装挙動依存): onnxruntime-node のネイティブ addon は C++ ランタイム（libstdc++）を要求する。
// 通常の Linux/Windows では既定パスで解決するが、nix/devbox のような隔離環境では
// LD_LIBRARY_PATH に libstdc++ を通す必要がある（README / docs 参照）。
//
// device 別の実機検証（docs/known-issues.md）: cpu と webgpu(Dawn) は int4 DeBERTa でも動作。
// dml は fp16 DeBERTa なら動くが int4(MatMulNBits) で失敗。cuda は既定バイナリ非同梱。

import * as ort from "onnxruntime-node";
import type { DebertaTokenizer } from "../text/deberta_tokenizer.ts";
import {
  type OrtBackend,
  type OrtSessionOptions,
  Sbv2Adapter,
} from "../runtime/adapter_core.ts";
import type { SynthScalars } from "../runtime/adapter_types.ts";
import type {
  AivmxMetadata,
  Sbv2HyperParameters,
} from "../runtime/aivmx_meta.ts";

/**
 * onnxruntime-node の実行プロバイダ（デバイス）。
 * - cpu: どこでも動く。
 * - webgpu: ORT ネイティブ WebGPU（Dawn。GPU アダプタ必須）。量子化込みで動く推奨 GPU。
 * - dml: DirectML（Windows。GPU 必須）。int4 は非対応、fp16 DeBERTa を使う。
 * - cuda: NVIDIA CUDA（別途 CUDA ランタイムと cuda 版バイナリが必要。既定バイナリには非同梱）。
 */
export type NodeDevice = "cpu" | "dml" | "cuda" | "webgpu";

const backend: OrtBackend = {
  int64: (data, dims) => new ort.Tensor("int64", data, dims),
  float32: (data, dims) => new ort.Tensor("float32", data, dims),
  createSession: (bytes, options) =>
    ort.InferenceSession.create(bytes, options),
};

/**
 * device 指定を executionProviders へ変換する。呼び出し側が sessionOptions.executionProviders を
 * 明示したときはそれを尊重する（EP 固有オプション、例: DirectML の deviceId を渡す唯一の経路）。
 * device と executionProviders の同時指定は曖昧なので fail loud。
 */
const withDevice = (
  device: NodeDevice | undefined,
  options?: OrtSessionOptions,
): OrtSessionOptions => {
  if (options?.executionProviders !== undefined) {
    if (device !== undefined) {
      throw new Error(
        "Sbv2NodeModelAdapter: device と sessionOptions.executionProviders は同時指定できない" +
          "（EP 固有オプションを渡すときは executionProviders のみを使う）",
      );
    }
    return options;
  }
  return { ...options, executionProviders: [device ?? "cpu"] };
};

/**
 * SBV2 JP-Extra 用モデルアダプタ（onnxruntime-node）。web 版と同一のロジックを、EP を選べる形で
 * 駆動する。`createFromAivmx` / `createFromOnnx` は `Sbv2Adapter` を返す。
 */
export class Sbv2NodeModelAdapter {
  /** Build from an AIVMX file. Style vectors and sample rate are read from ONNX metadata. */
  static createFromAivmx(args: {
    aivmxBytes: Uint8Array;
    bertOnnxBytes: Uint8Array;
    tokenizer: DebertaTokenizer;
    /** readAivmxMetadata 済みの値（巨大 protobuf の再走査を省く）。 */
    metadata?: AivmxMetadata;
    device?: NodeDevice;
    sampleRate?: number;
    scalars?: SynthScalars;
    sessionOptions?: OrtSessionOptions;
  }): Promise<Sbv2Adapter> {
    return Sbv2Adapter.fromAivmx(backend, {
      ...args,
      sessionOptions: withDevice(args.device, args.sessionOptions),
    });
  }

  /** Build from a plain acoustic ONNX file plus separate style vectors. */
  static createFromOnnx(args: {
    acousticOnnxBytes: Uint8Array;
    bertOnnxBytes: Uint8Array;
    tokenizer: DebertaTokenizer;
    styleVectorsNpy: Uint8Array;
    device?: NodeDevice;
    /** 必須（AIVM メタが無いので出力レートを黙って仮定しない）。 */
    sampleRate: number;
    scalars?: SynthScalars;
    hyperParameters?: Sbv2HyperParameters;
    sessionOptions?: OrtSessionOptions;
  }): Promise<Sbv2Adapter> {
    return Sbv2Adapter.fromOnnx(backend, {
      ...args,
      sessionOptions: withDevice(args.device, args.sessionOptions),
    });
  }
}
