// @hdae/sbv2-web/web: onnxruntime-web backend (browsers, Deno, compatible runtimes).
//
// Thin wrapper that injects onnxruntime-web into the shared Sbv2Adapter core. Provider
// selection (wasm / webgpu) is passed through sessionOptions.executionProviders.

import * as ort from "onnxruntime-web";
import type { DebertaTokenizer } from "../text/deberta_tokenizer.ts";
import {
  type OrtBackend,
  type OrtSessionOptions,
  Sbv2Adapter,
} from "../runtime/adapter_core.ts";
import {
  type BertSource,
  DebertaExtractor,
} from "../runtime/deberta_extractor.ts";
import type { SynthScalars } from "../runtime/adapter_types.ts";
import type {
  AivmxMetadata,
  Sbv2HyperParameters,
} from "../runtime/aivmx_meta.ts";

const backend: OrtBackend = {
  int64: (data, dims) => new ort.Tensor("int64", data, dims),
  float32: (data, dims) => new ort.Tensor("float32", data, dims),
  createSession: (bytes, options) =>
    ort.InferenceSession.create(bytes, options),
};

/**
 * SBV2 JP-Extra 用モデルアダプタ（onnxruntime-web）。`createFromAivmx` / `createFromOnnx` で
 * セッションを構築し、返り値の `Sbv2Adapter` で合成する。
 */
export class Sbv2ModelAdapter {
  /** Build from an AIVMX file. Style vectors and sample rate are read from ONNX metadata. */
  static createFromAivmx(
    args: {
      aivmxBytes: Uint8Array;
      /** readAivmxMetadata 済みの値（巨大 protobuf の再走査を省く）。 */
      metadata?: AivmxMetadata;
      sampleRate?: number;
      scalars?: SynthScalars;
      sessionOptions?: OrtSessionOptions;
    } & BertSource,
  ): Promise<Sbv2Adapter> {
    return Sbv2Adapter.fromAivmx(backend, args);
  }

  /** Build from a plain acoustic ONNX file plus separate style vectors. */
  static createFromOnnx(
    args: {
      acousticOnnxBytes: Uint8Array;
      styleVectorsNpy: Uint8Array;
      /** 必須（AIVM メタが無いので出力レートを黙って仮定しない）。 */
      sampleRate: number;
      scalars?: SynthScalars;
      hyperParameters?: Sbv2HyperParameters;
      sessionOptions?: OrtSessionOptions;
    } & BertSource,
  ): Promise<Sbv2Adapter> {
    return Sbv2Adapter.fromOnnx(backend, args);
  }

  /**
   * 共有 DeBERTa 抽出器を生成する。複数モデルを同時に保持するときはこれを 1 つ作り、各
   * `createFromAivmx` / `createFromOnnx` に `deberta` として渡すと BERT セッションが
   * 1 本で済む。解放は生成者の責任: 全アダプタの release 後に `extractor.release()`。
   * EP（wasm / webgpu）は sessionOptions.executionProviders で指定する。
   */
  static createDeberta(args: {
    bertOnnxBytes: Uint8Array;
    tokenizer: DebertaTokenizer;
    sessionOptions?: OrtSessionOptions;
  }): Promise<DebertaExtractor> {
    return DebertaExtractor.create(backend, args);
  }
}
