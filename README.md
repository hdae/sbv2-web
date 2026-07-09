# @hdae/sbv2-web

A browser-oriented Deno/TypeScript runtime for Style-Bert-VITS2 / AivisSpeech
JP-Extra ONNX inference.

ブラウザを主眼に、Style-Bert-VITS2 / AivisSpeech（JP-Extra）の ONNX 推論を動かす Deno/TypeScript
ランタイムです。日本語テキストフロントエンド（[@hdae/yomi](https://jsr.io/@hdae/yomi)）と DeBERTa・
AIVMX 音響モデルを配線し、テキストから音声波形を生成します。

## 特徴

- **ブラウザ完結の SBV2 推論**: `onnxruntime-web`（WASM / WebGPU）で AIVMX / プレーン ONNX の音響
  モデルをブラウザ上で実行します
- **2 エントリ（browser / node）**: `.` は `onnxruntime-web`、`./node` は `onnxruntime-node`
  （GPU: WebGPU / DirectML / CUDA）。数値経路を持つ共有コアを両方が re-export するので、ランタイムに
  合う方から一括 import できます
- **AIVMX 対応**: AIVM メタデータ（manifest / hyper_parameters / style_vectors）を内蔵する `.aivmx`
  を直接ロード。メタを持たないプレーン ONNX + `style_vectors.npy` も可
- **JP-Extra フロントエンド**: DeBERTa char トークナイザと word2ph による BERT 特徴のタイル化を内蔵。
  `given_phone` / `given_tone`・スタイルベクトル・`add_blank`・tone +6 など SBV2 固有の梱包はこの
  パッケージに閉じています
- **辞書は @hdae/yomi に委譲**: テキスト解析・アクセント G2P・辞書取得（Hugging Face から自動ダウン
  ロード・キャッシュ・整合性検証）を担い、モデル非依存に保っています

モデルファイルは同梱しません。AivisSpeech 相当の経路には、音響 `.aivmx`（または `.onnx`）・DeBERTa
ONNX・トークナイザ資産・`@hdae/yomi` 用の JTD 辞書が必要です。

## インストール

```sh
deno add jsr:@hdae/sbv2-web
```

## クイックスタート

```typescript
import {
  DebertaTokenizer,
  encodeWav,
  Sbv2ModelAdapter,
  synthesizeText,
} from "@hdae/sbv2-web";
import { getDictionary } from "@hdae/yomi/browser";

// 辞書（@hdae/yomi が Hugging Face から取得・キャッシュ・CRC 検証して返す）。
const dict = await getDictionary();

// DeBERTa トークナイザ（vocab.txt + clean_ranges.json + meta.json から構築）。
const tokenizer = DebertaTokenizer.fromVocabText(
  vocabText,
  cleanRanges,
  special,
);

// AIVMX 音響モデル + DeBERTa ONNX をロード。
const adapter = await Sbv2ModelAdapter.createFromAivmx({
  aivmxBytes,
  bertOnnxBytes,
  tokenizer,
});

// テキスト → 波形 → WAV。
const wave = await synthesizeText(
  "こんにちは、今日はいい天気ですね。",
  dict,
  tokenizer,
  adapter,
);
const wav = encodeWav(wave, adapter.sampleRate);
await adapter.release();
```

## 使い方

### ブラウザ / Web（onnxruntime-web）

ルート `@hdae/sbv2-web` が `onnxruntime-web` 版の `Sbv2ModelAdapter` を提供します。辞書は
`@hdae/yomi/browser` の `getDictionary()` が Hugging Face から取得・キャッシュします（ローカル辞書
ファイル不要）。動作例は [`examples/browser`](examples/browser) のブラウザ版サンプルを参照してください。

### Node / サーバ（onnxruntime-node, GPU）

ネイティブ実行は `@hdae/sbv2-web/node` から import します。`onnxruntime-node` を駆動し、GPU 実行プロ
バイダ（Windows の DirectML・対応ホストの CUDA・ORT ネイティブ WebGPU）で合成できます。共有コアを
re-export するので、`onnxruntime-web` は一切引き込みません。

```typescript
import { Sbv2NodeModelAdapter } from "@hdae/sbv2-web/node";
import { JtdDictionary } from "@hdae/yomi";

const dict = JtdDictionary.load((await Deno.readFile("naist-jdic.jtd")).buffer);
const adapter = await Sbv2NodeModelAdapter.createFromAivmx({
  aivmxBytes,
  bertOnnxBytes,
  tokenizer,
  device: "cpu", // "cpu" | "webgpu" | "dml" | "cuda"
});
```

### プレーン ONNX（AIVM メタなし）

プレーン ONNX には AIVM メタが無いので、`style_vectors.npy` とサンプルレートを別途渡します。

```typescript
const adapter = await Sbv2ModelAdapter.createFromOnnx({
  acousticOnnxBytes,
  bertOnnxBytes,
  tokenizer,
  styleVectorsNpy,
  sampleRate: 44100,
});
```

### CLI

`onnxruntime-node` 版の合成 CLI を同梱しています。`--text` で単発、省略で REPL（1 行ごとに合成、`:q`
で終了）です。web（WASM / WebGPU）経路はブラウザ版サンプルで確認します。

```sh
deno task cli -- --aivmx path/to/model.aivmx --device cpu --text "こんにちは。"
```

主なオプション: `--device cpu|dml|cuda|webgpu`（既定 `cpu`）・`--deberta`・`--tokenizer`・`--dict`・
`--out`（単発）/ `--out-dir`（REPL）・`--style-id` / `--style-weight` / `--speaker-id`。
`onnxruntime-node` のネイティブ addon は C++ ランタイム（`libstdc++`）を要求します。多くの環境では自動
解決されますが、nix/devbox シェルでは `LD_LIBRARY_PATH` に gcc の lib を通してください。

### ブラウザ版サンプル

[`examples/browser`](examples/browser) は Vite + React + shadcn（Base UI）の検証アプリです。pnpm を使
い、Comlink Web Worker 上で ONNX のロードと合成を行います。既定では AIVMX 音響モデルだけを選択すれば
動き、量子化 DeBERTa は Hugging Face からストリーム（キャッシュ）・トークナイザは同梱・辞書は
`@hdae/yomi` の `getDictionary()` で自動取得します。Advanced トグルで全資産の手動選択に切り替わります。

```sh
cd examples/browser
pnpm install
pnpm run dev   # or: pnpm run build
```

## 量子化 DeBERTa

現行のブラウザ向け DeBERTa は
[`hdae/deberta-v2-large-japanese-char-wwm-onnx-int4-rtn-b256`](https://huggingface.co/hdae/deberta-v2-large-japanese-char-wwm-onnx-int4-rtn-b256)
で、量子化モデルを `model.onnx`・トークナイザ資産・LICENSE/NOTICE 付きで公開しています。b256 は最小
RAM の選択肢です。BERT 特徴の忠実度と RAM のトレードオフは [Benchmarks](docs/benchmark.md) の int4
b32 行を参照してください。

Python/uv の変換・量子化ツールは [`tools/model-tools`](tools/model-tools) にあります。

## リリース / bump

バージョンの真実源は `deno.json` の `version` です。公開 `VERSION`（`src/core.ts`）はその焼き込みコ
ピーで、`deno task bump` が両者を 1 コミットで同期します。drift は `scripts/version_sync.test.ts`
（`deno task check` に含む）と、リリース時の `scripts/verify_tag.ts` で fail-loud に検出します。

```sh
deno task bump patch   # 0.2.0 -> 0.2.1（deno.json + src/core.ts を1コミット。tag/push はしない）
```

公開手順:

1. `deno task bump <patch|minor|major>` でバージョンを上げてコミット。
2. `git push` 後、`v<version>`（例 `v0.2.1`）タグで GitHub Release を作成。
3. Release の publish で [`release.yml`](.github/workflows/release.yml) が起動し、タグ ==
   `deno.json` の version を検証してから JSR に publish します（OIDC）。

## ライセンス

- **コード: MIT**（`LICENSE`）。
- **モデル / 辞書データ**はパッケージと別ライセンスです。SBV2 モデル・naist-jdic 辞書・DeBERTa の帰属と
  ライセンスは [docs/license-audit.md](docs/license-audit.md) を参照してください。

## 謝辞 / Acknowledgements

- **[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)** — 本パッケージが推論する
  JP-Extra 音響モデルとその合成手順（symbol table・tone +6・`add_blank`・スタイルベクトル）の源流。
- **[AivisSpeech / AIVM](https://github.com/Aivis-Project)** — AIVMX（ONNX + AIVM メタデータ）形式。
- **[ONNX Runtime](https://onnxruntime.ai/)** — Web（WASM / WebGPU）およびネイティブ（CPU / WebGPU /
  DirectML / CUDA）の推論エンジン。
