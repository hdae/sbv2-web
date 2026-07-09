// onnxruntime-node バックエンドの合成 CLI（デバイス指定: cpu / dml / cuda / webgpu）。
//
// deno task cli -- --aivmx path/to/model.aivmx --device cpu --text "こんにちは。"
//   --text あり → 単発合成（--out へ書き出し）。
//   --text なし → REPL（1行ごとに合成して --out-dir へ連番出力、:q / :quit / :exit で終了）。
//
// 既定では DeBERTa + トークナイザ（getDeberta）と辞書（yomi getDictionary）を
// HuggingFace から自動取得する（Cache API に永続、2 回目以降はオフライン）。
// ローカル資産を使うときだけ --deberta/--tokenizer（両方セット）や --dict を渡す。
//
// web(WASM/WebGPU) 経路はブラウザ版サンプル（examples/browser）で確認するため、CLI は node 一本。
// GPU を試すには --device dml（Windows DirectML）/ --device cuda / --device webgpu を渡す。
// CPU はどこでも動く。onnxruntime-node のネイティブ addon は libstdc++ を要求するため、
// nix/devbox のような隔離環境では LD_LIBRARY_PATH に gcc の lib を通すこと。

import {
  buildDebertaTokenizer,
  type DebertaTokenizer,
  encodeWav,
  getDeberta,
  type NodeDevice,
  Sbv2NodeModelAdapter,
  synthesizeText,
} from "../../src/node/mod.ts";
import { JtdDictionary } from "@hdae/yomi";
import { getDictionary } from "@hdae/yomi/browser";

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

/** ローカルのトークナイザ資産 3 点から構築する（自動取得と同じ buildDebertaTokenizer 経路）。 */
const readTokenizer = async (dir: string): Promise<DebertaTokenizer> =>
  buildDebertaTokenizer(
    await Deno.readTextFile(dir + "/vocab.txt"),
    await Deno.readTextFile(dir + "/clean_ranges.json"),
    await Deno.readTextFile(dir + "/meta.json"),
  );

const DEVICES: readonly NodeDevice[] = ["cpu", "dml", "cuda", "webgpu"];

const args = parseArgs();
const aivmxPath = args.get("aivmx");
if (!aivmxPath) {
  console.error(
    "Usage: deno task cli -- --aivmx path/to/model.aivmx " +
      "[--device cpu|dml|cuda|webgpu] " +
      "[--deberta path --tokenizer dir (両方セット。省略で HuggingFace 自動取得)] " +
      "[--dict path (省略で HuggingFace 自動取得)] " +
      "[--text text (省略で REPL)] [--out path] [--out-dir dir] " +
      "[--style-id 0] [--style-weight 1] [--speaker-id 0]",
  );
  Deno.exit(2);
}

const device = (args.get("device") ?? "cpu") as NodeDevice;
if (!DEVICES.includes(device)) {
  throw new Error("--device must be one of: " + DEVICES.join(", "));
}

const debertaPath = args.get("deberta");
const tokenizerDir = args.get("tokenizer");
const dictPath = args.get("dict");
const outPath = args.get("out") ?? "out/node/synth.wav";
const outDir = args.get("out-dir") ?? "out/node/repl";
const styleId = Number(args.get("style-id") ?? "0");
const styleWeight = Number(args.get("style-weight") ?? "1");
const speakerId = Number(args.get("speaker-id") ?? "0");

console.error(
  `loading tokenizer, dictionary, and models (device=${device})...`,
);

// DeBERTa + トークナイザ: 両フラグ指定でローカル、両方省略で HF 自動取得。片方だけは曖昧なので拒否。
let tokenizer: DebertaTokenizer;
let bertOnnxBytes: Uint8Array;
if (debertaPath !== undefined || tokenizerDir !== undefined) {
  if (debertaPath === undefined || tokenizerDir === undefined) {
    throw new Error(
      "--deberta と --tokenizer は両方指定してください（両方省略で HuggingFace 自動取得）",
    );
  }
  tokenizer = await readTokenizer(tokenizerDir);
  bertOnnxBytes = await Deno.readFile(debertaPath);
} else {
  let lastPercent = -10;
  const assets = await getDeberta({
    onProgress: ({ path, loaded, total }) => {
      if (path !== "model.onnx" || total === undefined) return;
      const percent = Math.floor((loaded / total) * 100);
      if (percent >= lastPercent + 10) {
        lastPercent = percent;
        console.error(`deberta: ${percent}% (${Math.round(loaded / 1e6)} MB)`);
      }
    },
  });
  tokenizer = assets.tokenizer;
  bertOnnxBytes = assets.bertOnnxBytes;
}

// 辞書: --dict でローカル、省略で HF 自動取得（yomi getDictionary は Deno でも動く）。
const dict = dictPath !== undefined
  ? JtdDictionary.load(toArrayBuffer(await Deno.readFile(dictPath)), {
    verifyChecksums: false,
  })
  : await getDictionary();

const aivmxBytes = await Deno.readFile(aivmxPath);

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
