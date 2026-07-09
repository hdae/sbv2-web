// onnxruntime-node バックエンドの合成 CLI（デバイス指定: cpu / dml / cuda / webgpu）。
//
// deno task cli -- --aivmx path/to/model.aivmx --device cpu --text "こんにちは。"
//   --text あり → 単発合成（--out へ書き出し）。
//   --text なし → REPL（1行ごとに合成して --out-dir へ連番出力、:q / :quit / :exit で終了）。
//
// web(WASM/WebGPU) 経路はブラウザ版サンプル（examples/browser）で確認するため、CLI は node 一本。
// GPU を試すには --device dml（Windows DirectML）/ --device cuda / --device webgpu を渡す。
// CPU はどこでも動く。onnxruntime-node のネイティブ addon は libstdc++ を要求するため、
// nix/devbox のような隔離環境では LD_LIBRARY_PATH に gcc の lib を通すこと。

import {
  type CleanRanges,
  type DebertaSpecialTokens,
  DebertaTokenizer,
  encodeWav,
  type NodeDevice,
  Sbv2NodeModelAdapter,
  synthesizeText,
} from "../../src/node/mod.ts";
import { JtdDictionary } from "@hdae/yomi";

const parseArgs = () => {
  const args = new Map<string, string>();
  for (let i = 0; i < Deno.args.length; i++) {
    const arg = Deno.args[i];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2);
    const next = Deno.args[i + 1];
    args.set(
      key,
      next === undefined || next.startsWith("--") ? "true" : Deno.args[++i],
    );
  }
  return args;
};

const toArrayBuffer = (bytes: Uint8Array): ArrayBuffer =>
  bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;

const readTokenizer = async (dir: string): Promise<DebertaTokenizer> => {
  const vocabText = await Deno.readTextFile(dir + "/vocab.txt");
  const clean = JSON.parse(
    await Deno.readTextFile(dir + "/clean_ranges.json"),
  ) as CleanRanges;
  const meta = JSON.parse(await Deno.readTextFile(dir + "/meta.json")) as {
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

const DEVICES: readonly NodeDevice[] = ["cpu", "dml", "cuda", "webgpu"];

const args = parseArgs();
const aivmxPath = args.get("aivmx");
if (!aivmxPath) {
  console.error(
    "Usage: deno task cli -- --aivmx path/to/model.aivmx " +
      "[--device cpu|dml|cuda|webgpu] [--deberta path] [--tokenizer dir] [--dict path] " +
      "[--text text (省略で REPL)] [--out path] [--out-dir dir] " +
      "[--style-id 0] [--style-weight 1] [--speaker-id 0]",
  );
  Deno.exit(2);
}

const device = (args.get("device") ?? "cpu") as NodeDevice;
if (!DEVICES.includes(device)) {
  throw new Error("--device must be one of: " + DEVICES.join(", "));
}

const debertaPath = args.get("deberta") ??
  "data/hf-packages/deberta-int4-rtn-b256/model.onnx";
const tokenizerDir = args.get("tokenizer") ??
  "data/hf-packages/deberta-int4-rtn-b256";
const dictPath = args.get("dict") ?? "data/dict/naist-jdic.jtd";
const outPath = args.get("out") ?? "out/node/synth.wav";
const outDir = args.get("out-dir") ?? "out/node/repl";
const styleId = Number(args.get("style-id") ?? "0");
const styleWeight = Number(args.get("style-weight") ?? "1");
const speakerId = Number(args.get("speaker-id") ?? "0");

console.error(
  `loading tokenizer, dictionary, and models (device=${device})...`,
);
const tokenizer = await readTokenizer(tokenizerDir);
const dictBytes = await Deno.readFile(dictPath);
const dict = JtdDictionary.load(toArrayBuffer(dictBytes), {
  verifyChecksums: false,
});
const aivmxBytes = await Deno.readFile(aivmxPath);
const bertOnnxBytes = await Deno.readFile(debertaPath);

const createStarted = performance.now();
const adapter = await Sbv2NodeModelAdapter.createFromAivmx({
  aivmxBytes,
  bertOnnxBytes,
  tokenizer,
  device,
});
console.error(
  `ready: device=${device}, sampleRate=${adapter.sampleRate}, ` +
    `styles=${adapter.numStyles}, load=${
      Math.round(performance.now() - createStarted)
    }ms`,
);

/** テキスト1本を合成し out へ書き出す（親ディレクトリは自動作成）。 */
const synthesizeOne = async (text: string, out: string) => {
  const trimmed = text.trim();
  if (!trimmed) return;
  const started = performance.now();
  const wave = await synthesizeText(trimmed, dict, tokenizer, adapter, {
    styleId,
    styleWeight,
    speakerId,
  });
  const slash = out.lastIndexOf("/");
  if (slash > 0) await Deno.mkdir(out.slice(0, slash), { recursive: true });
  await Deno.writeFile(out, encodeWav(wave, adapter.sampleRate));
  const elapsed = Math.round(performance.now() - started);
  let peak = 0;
  for (const sample of wave) peak = Math.max(peak, Math.abs(sample));
  console.log(
    `${out}\t${elapsed}ms\t${wave.length} samples\tpeak=${peak.toFixed(4)}`,
  );
};

const once = args.get("text");
try {
  if (once !== undefined) {
    await synthesizeOne(once, outPath);
  } else {
    console.error("enter text to synthesize. commands: :q / :quit / :exit");
    let index = 0;
    while (true) {
      const line = prompt("> ");
      if (line === null) break;
      const trimmed = line.trim();
      if ([":q", ":quit", ":exit"].includes(trimmed)) break;
      if (!trimmed) continue;
      await synthesizeOne(
        trimmed,
        `${outDir}/${String(index++).padStart(3, "0")}.wav`,
      );
    }
  }
} finally {
  await adapter.release();
}
