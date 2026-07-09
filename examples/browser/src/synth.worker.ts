import * as ort from "onnxruntime-web";
import {
  JtdDictionary,
  type JtdDictionary as LoadedDictionary,
} from "@hdae/yomi";
import { getDictionary } from "@hdae/yomi/browser";
import {
  type CleanRanges,
  type DebertaSpecialTokens,
  DebertaTokenizer,
  encodeWav,
  type Sbv2Adapter,
  Sbv2ModelAdapter,
  synthesizeText,
} from "../../../src/mod.ts";
import bundledVocabText from "./assets/deberta/vocab.txt?raw";
import bundledCleanRangesText from "./assets/deberta/clean_ranges.json?raw";
import bundledMetaText from "./assets/deberta/meta.json?raw";

ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";

/** Published quantized DeBERTa (int4 RTN b256). CORS + range enabled (HuggingFace resolve). */
const DEBERTA_HF_URL =
  "https://huggingface.co/hdae/deberta-v2-large-japanese-char-wwm-onnx-int4-rtn-b256/resolve/main/model.onnx";
const DEBERTA_CACHE_NAME = "sbv2-deberta";

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
   * When true the worker sources DeBERTa (HuggingFace, cached), the tokenizer
   * (bundled with the app), and the dictionary (HuggingFace via getDictionary) itself.
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

/** View a Uint8Array as an exact ArrayBuffer (copying only when it is a subarray). */
const bufferOf = (bytes: Uint8Array): ArrayBuffer =>
  bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? (bytes.buffer as ArrayBuffer)
    : (bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength,
    ) as ArrayBuffer);

/**
 * Build the DeBERTa tokenizer from raw asset text. Both the bundled path and the
 * Advanced (manual) path go through here, so their behavior can never diverge.
 */
const buildTokenizer = (
  vocabText: string,
  cleanRangesText: string,
  metaText: string,
): DebertaTokenizer => {
  const clean = JSON.parse(cleanRangesText) as CleanRanges;
  const meta = JSON.parse(metaText) as {
    special_tokens: {
      cls: { id: number };
      sep: { id: number };
      unk: { id: number };
    };
  };
  const special: DebertaSpecialTokens = {
    clsId: meta.special_tokens.cls.id,
    sepId: meta.special_tokens.sep.id,
    unkId: meta.special_tokens.unk.id,
  };
  return DebertaTokenizer.fromVocabText(vocabText, clean, special);
};

/**
 * Fetch the quantized DeBERTa model, caching it in the Cache API keyed by its
 * immutable URL so later loads are offline. Streams the body to report download
 * progress (throttled to one report per integer percent).
 */
const fetchDebertaBytes = async (
  onProgress?: (progress: LoadProgress) => void,
): Promise<Uint8Array> => {
  const hasCache = typeof caches !== "undefined";
  if (hasCache) {
    const cache = await caches.open(DEBERTA_CACHE_NAME);
    const hit = await cache.match(DEBERTA_HF_URL);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  }

  const response = await fetch(DEBERTA_HF_URL);
  if (!response.ok) {
    throw new Error(
      `DeBERTa 取得失敗: HTTP ${response.status} ${response.statusText} (${DEBERTA_HF_URL})`,
    );
  }

  const total = Number(response.headers.get("content-length") ?? 0) || undefined;
  const body = response.body;
  let bytes: Uint8Array;
  if (body) {
    const reader = body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    let lastPercent = -1;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      const percent = total ? Math.floor((loaded / total) * 100) : -1;
      if (percent !== lastPercent) {
        lastPercent = percent;
        onProgress?.({ stage: "deberta", loaded, total });
      }
    }
    bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
  } else {
    bytes = new Uint8Array(await response.arrayBuffer());
  }

  if (hasCache) {
    const cache = await caches.open(DEBERTA_CACHE_NAME);
    await cache.put(DEBERTA_HF_URL, new Response(bufferOf(bytes)));
  }
  return bytes;
};

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
    onProgress?.({ stage: "tokenizer" });
    tokenizer = buildTokenizer(
      bundledVocabText,
      bundledCleanRangesText,
      bundledMetaText,
    );
    onProgress?.({ stage: "deberta", loaded: 0, total: undefined });
    bertOnnxBytes = await fetchDebertaBytes(onProgress);
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
    tokenizer = buildTokenizer(
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
