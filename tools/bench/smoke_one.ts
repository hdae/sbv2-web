import {
  type CleanRanges,
  type DebertaSpecialTokens,
  DebertaTokenizer,
  encodeWav,
  Sbv2ModelAdapter,
  synthesizeText,
} from "../../src/mod.ts";
import { JtdDictionary } from "@hdae/yomi";

const parseArgs = () => {
  const args = new Map<string, string>();
  for (let i = 0; i < Deno.args.length; i++) {
    const arg = Deno.args[i];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2);
    const next = Deno.args[i + 1];
    const value = next === undefined || next.startsWith("--")
      ? "true"
      : Deno.args[++i];
    args.set(key, value);
  }
  return args;
};

const statusValueKb = (key: string): number | undefined => {
  try {
    const status = Deno.readTextFileSync("/proc/self/status");
    const match = status.match(
      new RegExp("^" + key + ":\\s+(\\d+)\\s+kB$", "m"),
    );
    return match ? Number(match[1]) : undefined;
  } catch {
    return undefined;
  }
};

const gpuMemoryMiB = (): number | undefined => {
  try {
    const out = new Deno.Command("nvidia-smi", {
      args: ["--query-gpu=memory.used", "--format=csv,noheader,nounits"],
      stdout: "piped",
      stderr: "null",
    }).outputSync();
    if (!out.success) return undefined;
    const text = new TextDecoder().decode(out.stdout).trim().split(/\s+/)[0];
    const value = Number(text);
    return Number.isFinite(value) ? value : undefined;
  } catch {
    return undefined;
  }
};

const mark = (label: string) => ({
  label,
  rssKb: statusValueKb("VmRSS"),
  hwmKb: statusValueKb("VmHWM"),
  gpuMemoryMiB: gpuMemoryMiB(),
  tMs: Math.round(performance.now()),
});

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

const toArrayBuffer = (bytes: Uint8Array): ArrayBuffer =>
  bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;

const args = parseArgs();
const label = args.get("label") ?? "smoke";
const acousticPath = args.get("acoustic") ?? "data/models/aivmx/mao.aivmx";
const debertaPath = args.get("deberta") ??
  "data/models/deberta-int4-rtn/model_int4.onnx";
const tokenizerDir = args.get("tokenizer") ?? "data/models/deberta-tokenizer";
const dictPath = args.get("dict") ?? "data/dict/naist-jdic.jtd";
const text = args.get("text") ?? "こんにちは、今日はいい天気ですね。";
const outWav = args.get("out-wav");
const acousticKind = args.get("acoustic-kind") ?? "aivmx";
const provider = args.get("provider") ?? "wasm";

if (provider === "webgpu") {
  await import("onnxruntime-web/webgpu");
}

const stages = [mark("start")];
const tokenizer = await readTokenizer(tokenizerDir);
stages.push(mark("tokenizer_loaded"));

const dictBytes = await Deno.readFile(dictPath);
const dict = JtdDictionary.load(toArrayBuffer(dictBytes), {
  verifyChecksums: false,
});
stages.push(mark("dict_loaded"));

const acousticBytes = await Deno.readFile(acousticPath);
stages.push(mark("acoustic_bytes_loaded"));
const debertaBytes = await Deno.readFile(debertaPath);
stages.push(mark("deberta_bytes_loaded"));

const sessionOptions = { executionProviders: [provider] };
const adapter = acousticKind === "aivmx"
  ? await Sbv2ModelAdapter.createFromAivmx({
    aivmxBytes: acousticBytes,
    bertOnnxBytes: debertaBytes,
    tokenizer,
    sessionOptions,
  })
  : await Sbv2ModelAdapter.createFromOnnx({
    acousticOnnxBytes: acousticBytes,
    bertOnnxBytes: debertaBytes,
    tokenizer,
    styleVectorsNpy: await Deno.readFile(
      args.get("style-vectors") ?? "data/models/jvnv-F1-jp/style_vectors.npy",
    ),
    sampleRate: Number(args.get("sample-rate") ?? "44100"),
    sessionOptions,
  });
stages.push(mark("sessions_created"));

const t0 = performance.now();
const wave = await synthesizeText(text, dict, tokenizer, adapter);
const synthMs = performance.now() - t0;
stages.push(mark("synthesized"));

if (outWav) {
  const parent = outWav.includes("/")
    ? outWav.slice(0, outWav.lastIndexOf("/"))
    : ".";
  await Deno.mkdir(parent, { recursive: true });
  await Deno.writeFile(outWav, encodeWav(wave, adapter.sampleRate));
  stages.push(mark("wav_written"));
}

await adapter.release();
stages.push(mark("released"));

const peak = Math.max(...stages.map((s) => s.hwmKb ?? s.rssKb ?? 0));
const result = {
  label,
  text,
  acousticPath,
  debertaPath,
  acousticBytes: acousticBytes.byteLength,
  debertaBytes: debertaBytes.byteLength,
  sampleRate: adapter.sampleRate,
  waveSamples: wave.length,
  waveSeconds: wave.length / adapter.sampleRate,
  wavePeakAbs: wave.reduce((m, v) => Math.max(m, Math.abs(v)), 0),
  synthMs: Math.round(synthMs),
  peakRssKb: peak || undefined,
  provider,
  peakGpuMemoryMiB: Math.max(...stages.map((s) => s.gpuMemoryMiB ?? 0)) ||
    undefined,
  stages,
};
console.log(JSON.stringify(result, null, 2));
