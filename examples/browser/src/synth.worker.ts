import * as ort from "onnxruntime-web";
import {
  JtdDictionary,
  type JtdDictionary as LoadedDictionary,
} from "@hdae/yomi";
import { getDictionary } from "@hdae/yomi/browser";
import {
  buildDebertaTokenizer,
  type DebertaTokenizer,
  encodeWav,
  getDeberta,
  type Sbv2Adapter,
  Sbv2ModelAdapter,
  synthesizeText,
} from "../../../src/mod.ts";

ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";

export type Provider = "wasm" | "webgpu";

/** Coarse load stages reported back to the UI (see load's onProgress). */
export type LoadProgress = {
  stage: "tokenizer" | "deberta" | "dictionary" | "acoustic";
  /** Bytes downloaded so far (deberta stage only). */
  loaded?: number;
  /** Total bytes if known (deberta stage only). */
  total?: number;
};

/** User-supplied assets for the Advanced / offline path (custom DeBERTa + local dict). */
export type ManualAssets = {
  bertOnnxBytes: ArrayBuffer;
  vocabText: string;
  cleanRangesText: string;
  metaText: string;
  dictBytes: ArrayBuffer;
};

export type LoadRequest = {
  provider: Provider;
  /** The acoustic model is always the user's own voice, so it stays a file input. */
  aivmxBytes: ArrayBuffer;
  /**
   * When true the worker sources everything itself: DeBERTa + tokenizer via
   * getDeberta (HuggingFace, SHA-pinned, Cache API) and the dictionary via
   * yomi's getDictionary (same pattern).
   */
  useBundledAssets: boolean;
  /** Provided only when useBundledAssets === false. */
  manual?: ManualAssets;
};

export type LoadResult = {
  provider: Provider;
  elapsedMs: number;
  sampleRate: number;
  numStyles: number;
};

export type SynthesizeRequest = {
  text: string;
  styleId: number;
  styleWeight: number;
  speakerId: number;
};

export type SynthesizeResult = {
  wavBytes: Uint8Array;
  elapsedMs: number;
  samples: number;
  sampleRate: number;
};

let adapter: Sbv2Adapter | null = null;
let tokenizer: DebertaTokenizer | null = null;
let dict: LoadedDictionary | null = null;
let currentProvider: Provider | null = null;

const toBytes = (buffer: ArrayBuffer): Uint8Array => new Uint8Array(buffer);

export const release = async (): Promise<void> => {
  await adapter?.release();
  adapter = null;
  tokenizer = null;
  dict = null;
  currentProvider = null;
};

export const load = async (
  request: LoadRequest,
  onProgress?: (progress: LoadProgress) => void,
): Promise<LoadResult> => {
  const started = performance.now();
  await release();

  if (request.provider === "webgpu") await import("onnxruntime-web/webgpu");

  let bertOnnxBytes: Uint8Array;

  if (request.useBundledAssets) {
    onProgress?.({ stage: "deberta", loaded: 0, total: undefined });
    // getDeberta fetches model.onnx + tokenizer assets as one SHA-pinned set from
    // HuggingFace (Cache API cached, size/sha256 verified, self-healing) and
    // returns a ready tokenizer — nothing is bundled with the app anymore.
    const deberta = await getDeberta({
      onProgress: (progress) => {
        if (progress.path === "model.onnx") {
          onProgress?.({
            stage: "deberta",
            loaded: progress.loaded,
            total: progress.total,
          });
        }
      },
    });
    tokenizer = deberta.tokenizer;
    bertOnnxBytes = deberta.bertOnnxBytes;
    onProgress?.({ stage: "dictionary" });
    // getDictionary fetches the yomi-matched JTD1 dictionary from HuggingFace
    // (cached in the Cache API, gzip-decompressed, CRC-verified) and returns a
    // ready JtdDictionary — no same-origin asset needed.
    dict = await getDictionary();
  } else {
    const manual = request.manual;
    if (!manual) {
      throw new Error("useBundledAssets=false ですが manual assets がありません");
    }
    onProgress?.({ stage: "tokenizer" });
    tokenizer = buildDebertaTokenizer(
      manual.vocabText,
      manual.cleanRangesText,
      manual.metaText,
    );
    bertOnnxBytes = toBytes(manual.bertOnnxBytes);
    dict = JtdDictionary.load(manual.dictBytes, { verifyChecksums: false });
  }

  onProgress?.({ stage: "acoustic" });
  adapter = await Sbv2ModelAdapter.createFromAivmx({
    aivmxBytes: toBytes(request.aivmxBytes),
    bertOnnxBytes,
    tokenizer,
    sessionOptions: { executionProviders: [request.provider] },
  });
  currentProvider = request.provider;

  return {
    provider: currentProvider,
    elapsedMs: Math.round(performance.now() - started),
    sampleRate: adapter.sampleRate,
    numStyles: adapter.numStyles,
  };
};

export const synthesize = async (
  request: SynthesizeRequest,
): Promise<SynthesizeResult> => {
  if (!adapter || !tokenizer || !dict) {
    throw new Error("Models are not loaded.");
  }
  const started = performance.now();
  const wave = await synthesizeText(request.text, dict, tokenizer, adapter, {
    styleId: request.styleId,
    styleWeight: request.styleWeight,
    speakerId: request.speakerId,
  });
  const wavBytes = encodeWav(wave, adapter.sampleRate);
  return {
    wavBytes,
    elapsedMs: Math.round(performance.now() - started),
    samples: wave.length,
    sampleRate: adapter.sampleRate,
  };
};
