import { Activity, Download, Play, RotateCcw } from "lucide-react";
import { type ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { endpointSymbol, proxy } from "vite-plugin-comlink/symbol";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select";
import { Textarea } from "./components/ui/textarea";
import type {
  LoadProgress,
  LoadRequest,
  ManualAssets,
  Provider,
} from "./synth.worker";

type Files = {
  aivmx?: File;
  deberta?: File;
  vocab?: File;
  cleanRanges?: File;
  meta?: File;
  dict?: File;
};

type Loaded = {
  provider: Provider;
  sampleRate: number;
  numStyles: number;
};

type SynthWorker = typeof import("./synth.worker");
type WorkerApi = {
  load: SynthWorker["load"];
  synthesize: SynthWorker["synthesize"];
  release: SynthWorker["release"];
  [endpointSymbol]: Worker;
};

const readBuffer = async (file: File): Promise<ArrayBuffer> =>
  await file.arrayBuffer();
const readText = async (file: File): Promise<string> => await file.text();

const fileLabel = (file?: File): string =>
  file
    ? file.name + " (" + Math.round(file.size / 1024 / 1024) + " MiB)"
    : "not selected";

const browserMemory = (): string => {
  const perf = performance as Performance & {
    memory?: { usedJSHeapSize: number };
  };
  if (!perf.memory) return "n/a";
  return Math.round(perf.memory.usedJSHeapSize / 1024 / 1024) + " MiB JS heap";
};

const fmtProgress = (progress: LoadProgress): string => {
  switch (progress.stage) {
    case "tokenizer":
      return "loading tokenizer";
    case "deberta":
      return progress.total
        ? "downloading DeBERTa " +
          Math.floor(((progress.loaded ?? 0) / progress.total) * 100) + "%"
        : "downloading DeBERTa";
    case "dictionary":
      return "loading dictionary";
    case "acoustic":
      return "creating ONNX sessions";
  }
};

const createWorker = (): WorkerApi =>
  new ComlinkWorker<SynthWorker>(new URL("./synth.worker", import.meta.url), {
    type: "module",
  }) as WorkerApi;

export default function App() {
  const workerRef = useRef<WorkerApi | null>(null);

  const [files, setFiles] = useState<Files>({});
  const [provider, setProvider] = useState<Provider>("wasm");
  const [useBundled, setUseBundled] = useState(true);
  const [text, setText] = useState("こんにちは、今日はいい天気ですね。");
  const [styleId, setStyleId] = useState(0);
  const [styleWeight, setStyleWeight] = useState(1);
  const [speakerId, setSpeakerId] = useState(0);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  // status は進捗文字列でも上書きされる表示専用テキスト。ボタンのガードは機械状態 busy で行う
  // （文字列一致だと load 進捗表示中に Load が再有効化され、合成中の Load/Release が
  // 推論中セッションを release する競合を起こす）。
  const [busy, setBusy] = useState<"loading" | "synthesizing" | null>(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [wavName, setWavName] = useState("sbv2-web.wav");
  const [metrics, setMetrics] = useState<string[]>([]);
  const audioRef = useRef<HTMLAudioElement>(null);

  const readyToLoad = useBundled
    ? Boolean(files.aivmx)
    : Boolean(
      files.aivmx && files.deberta && files.vocab && files.cleanRanges &&
        files.meta && files.dict,
    );
  const canSynthesize = Boolean(loaded && text.trim());

  const selectedSummary = useMemo<[string, string][]>(
    () =>
      useBundled
        ? [
          ["AIVMX", fileLabel(files.aivmx)],
          ["DeBERTa", "auto (HuggingFace int4)"],
          ["tokenizer", "auto (HuggingFace)"],
          ["dictionary", "auto (HuggingFace)"],
        ]
        : [
          ["AIVMX", fileLabel(files.aivmx)],
          ["DeBERTa", fileLabel(files.deberta)],
          ["vocab", fileLabel(files.vocab)],
          ["clean_ranges", fileLabel(files.cleanRanges)],
          ["meta", fileLabel(files.meta)],
          ["dictionary", fileLabel(files.dict)],
        ],
    [files, useBundled],
  );

  // Own the worker for the component's lifetime. It is created inside an effect
  // (not the render body) and torn down + nulled in cleanup, so React StrictMode's
  // mount → unmount → remount cycle yields a fresh, live worker instead of leaving
  // workerRef pointing at a terminated one. Tying teardown to anything else (e.g.
  // audioUrl) would terminate the worker mid-session and hang later Comlink calls.
  useEffect(() => {
    const worker = createWorker();
    workerRef.current = worker;
    return () => {
      void worker.release();
      worker[endpointSymbol]?.terminate();
      workerRef.current = null;
    };
  }, []);

  // Revoke the previous object URL when it changes or on unmount. Kept separate
  // from worker teardown so producing a new clip never disturbs the worker.
  useEffect(() => {
    if (!audioUrl) return;
    return () => URL.revokeObjectURL(audioUrl);
  }, [audioUrl]);

  const setFile =
    (key: keyof Files) => (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      setFiles((prev) => ({ ...prev, [key]: file }));
    };

  const resetLoaded = async () => {
    await workerRef.current?.release();
    setLoaded(null);
    setStatus("idle");
  };

  const readManualAssets = async (): Promise<ManualAssets | null> => {
    const { deberta, vocab, cleanRanges, meta, dict } = files;
    if (!deberta || !vocab || !cleanRanges || !meta || !dict) return null;
    return {
      bertOnnxBytes: await readBuffer(deberta),
      vocabText: await readText(vocab),
      cleanRangesText: await readText(cleanRanges),
      metaText: await readText(meta),
      dictBytes: await readBuffer(dict),
    };
  };

  const loadModels = async () => {
    const aivmx = files.aivmx;
    const worker = workerRef.current;
    // No acoustic model chosen, or the lifecycle effect has not assigned the worker yet.
    if (!aivmx || !worker) return;
    setError(null);
    setBusy("loading");
    setStatus("loading");
    setMetrics([]);
    try {
      await resetLoaded();
      const aivmxBytes = await readBuffer(aivmx);
      let request: LoadRequest;
      if (useBundled) {
        request = { provider, aivmxBytes, useBundledAssets: true };
      } else {
        const manual = await readManualAssets();
        if (!manual) {
          setStatus("idle");
          return;
        }
        request = { provider, aivmxBytes, useBundledAssets: false, manual };
      }
      const result = await worker.load(
        request,
        proxy((progress: LoadProgress) => setStatus(fmtProgress(progress))),
      );
      setLoaded({
        provider: result.provider,
        sampleRate: result.sampleRate,
        numStyles: result.numStyles,
      });
      setStatus("loaded with " + result.provider);
      setMetrics([
        "load: " + result.elapsedMs + " ms",
        "main memory: " + browserMemory(),
        "sample rate: " + result.sampleRate + " Hz",
        "styles: " + result.numStyles,
      ]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setStatus("load failed");
    } finally {
      setBusy(null);
    }
  };

  const synthesize = async () => {
    const worker = workerRef.current;
    if (!loaded || !worker) return;
    setError(null);
    setBusy("synthesizing");
    setStatus("synthesizing");
    try {
      const result = await worker.synthesize({
        text,
        styleId,
        styleWeight,
        speakerId,
      });
      const wavBytes = result.wavBytes;
      const wavBuffer = wavBytes.buffer.slice(
        wavBytes.byteOffset,
        wavBytes.byteOffset + wavBytes.byteLength,
      ) as ArrayBuffer;
      const nextUrl = URL.createObjectURL(
        new Blob([wavBuffer], { type: "audio/wav" }),
      );
      setAudioUrl(nextUrl);
      setWavName("sbv2-web-" + loaded.provider + ".wav");
      setStatus("done");
      setMetrics((prev) => [
        "synth: " + result.elapsedMs + " ms",
        "samples: " + result.samples,
        "main memory: " + browserMemory(),
        ...prev.slice(0, 4),
      ]);
      setTimeout(() => void audioRef.current?.play(), 0);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setStatus("synthesis failed");
    } finally {
      setBusy(null);
    }
  };

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
      <header className="flex flex-col gap-3 border-b pb-5 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-normal">
            sbv2-web browser lab
          </h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Load local AIVMX, quantized DeBERTa, tokenizer assets, and a JTD
            dictionary. Run ONNX Runtime Web in a Worker with WASM or WebGPU,
            then play or save the generated WAV.
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Activity className="h-4 w-4" />
          <span>{status}</span>
        </div>
      </header>

      <section className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="space-y-5">
          <div className="space-y-4 rounded-lg border p-4">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={useBundled}
                onChange={(event) => setUseBundled(event.target.checked)}
              />
              Auto-load DeBERTa + tokenizer + dictionary (HuggingFace)
              (recommended)
            </label>
            <div className="grid gap-4 md:grid-cols-2">
              <FileField
                label="AIVMX acoustic model"
                accept=".aivmx,.onnx"
                onChange={setFile("aivmx")}
              />
              {!useBundled && (
                <>
                  <FileField
                    label="DeBERTa ONNX"
                    accept=".onnx"
                    onChange={setFile("deberta")}
                  />
                  <FileField
                    label="vocab.txt"
                    accept=".txt"
                    onChange={setFile("vocab")}
                  />
                  <FileField
                    label="clean_ranges.json"
                    accept=".json"
                    onChange={setFile("cleanRanges")}
                  />
                  <FileField
                    label="meta.json"
                    accept=".json"
                    onChange={setFile("meta")}
                  />
                  <FileField
                    label="JTD dictionary"
                    accept=".jtd"
                    onChange={setFile("dict")}
                  />
                </>
              )}
            </div>
            {useBundled && (
              <p className="text-xs text-muted-foreground">
                DeBERTa int4 (~240 MiB) + tokenizer assets download from
                HuggingFace on first load (SHA-pinned, verified) and are cached
                for offline reuse. The dictionary is fetched and cached too.
              </p>
            )}
          </div>

          <div className="rounded-lg border p-4">
            <div className="grid gap-4 md:grid-cols-[160px_1fr]">
              <div className="space-y-2">
                <Label htmlFor="provider">Provider</Label>
                <Select
                  value={provider}
                  onValueChange={(value) => setProvider(value as Provider)}
                >
                  <SelectTrigger id="provider" className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="wasm">WASM</SelectItem>
                    <SelectItem value="webgpu">WebGPU</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-3 sm:grid-cols-3">
                <NumberField
                  label="Style ID"
                  value={styleId}
                  min={0}
                  step={1}
                  onChange={setStyleId}
                />
                <NumberField
                  label="Style weight"
                  value={styleWeight}
                  min={0}
                  step={0.1}
                  onChange={setStyleWeight}
                />
                <NumberField
                  label="Speaker ID"
                  value={speakerId}
                  min={0}
                  step={1}
                  onChange={setSpeakerId}
                />
              </div>
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              <Button
                onClick={loadModels}
                disabled={!readyToLoad || busy !== null}
              >
                Load models
              </Button>
              <Button
                variant="outline"
                onClick={resetLoaded}
                disabled={!loaded || busy !== null}
              >
                <RotateCcw className="h-4 w-4" />
                Release
              </Button>
            </div>
          </div>

          <div className="rounded-lg border p-4">
            <Label htmlFor="text">Text</Label>
            <Textarea
              id="text"
              className="mt-2"
              value={text}
              onChange={(event) => setText(event.target.value)}
            />
            <div className="mt-4 flex flex-wrap gap-2">
              <Button
                onClick={synthesize}
                disabled={!canSynthesize || busy !== null}
              >
                <Play className="h-4 w-4" />
                Synthesize
              </Button>
              {audioUrl && (
                <Button
                  variant="outline"
                  onClick={() => audioRef.current?.play()}
                >
                  <Play className="h-4 w-4" />
                  Play
                </Button>
              )}
              {audioUrl && (
                <a
                  className="inline-flex h-8 items-center justify-center gap-2 rounded-lg border border-border px-2.5 text-sm font-medium hover:bg-muted"
                  href={audioUrl}
                  download={wavName}
                >
                  <Download className="h-4 w-4" />
                  WAV
                </a>
              )}
            </div>
            {audioUrl && (
              <audio
                ref={audioRef}
                className="mt-4 w-full"
                controls
                src={audioUrl}
              />
            )}
          </div>
        </div>

        <aside className="space-y-5">
          <div className="rounded-lg border p-4">
            <h2 className="text-sm font-semibold">Selected files</h2>
            <dl className="mt-3 space-y-2 text-sm">
              {selectedSummary.map(([name, value]) => (
                <div key={name} className="grid grid-cols-[96px_1fr] gap-3">
                  <dt className="text-muted-foreground">{name}</dt>
                  <dd className="min-w-0 break-words">{value}</dd>
                </div>
              ))}
            </dl>
          </div>

          <div className="rounded-lg border p-4">
            <h2 className="text-sm font-semibold">Metrics</h2>
            <div className="mt-3 space-y-2 text-sm">
              {metrics.length === 0
                ? <p className="text-muted-foreground">No run yet.</p>
                : (
                  metrics.map((item) => (
                    <div key={item} className="rounded-md bg-muted px-3 py-2">
                      {item}
                    </div>
                  ))
                )}
            </div>
          </div>

          {error && (
            <div className="rounded-lg border border-destructive p-4 text-sm text-destructive">
              {error}
            </div>
          )}
        </aside>
      </section>
    </main>
  );
}

type FileFieldProps = {
  label: string;
  accept: string;
  onChange: (event: ChangeEvent<HTMLInputElement>) => void;
};

const FileField = ({ label, accept, onChange }: FileFieldProps) => (
  <div className="space-y-2">
    <Label>{label}</Label>
    <Input type="file" accept={accept} onChange={onChange} />
  </div>
);

type NumberFieldProps = {
  label: string;
  value: number;
  min: number;
  step: number;
  onChange: (value: number) => void;
};

const NumberField = (
  { label, value, min, step, onChange }: NumberFieldProps,
) => (
  <div className="space-y-2">
    <Label>{label}</Label>
    <Input
      type="number"
      min={min}
      step={step}
      value={value}
      onChange={(event) => onChange(Number(event.target.value))}
    />
  </div>
);
