# 0.1.0 → 0.2.0 マイグレーションガイド

0.2.0 は API 拡充（per-call スカラー・メタデータの型付き公開・DeBERTa 自動取得）と、
無言劣化を fail loud に変える破壊的変更を含みます。未リリース期の方針（互換 shim なし）に
従い、旧 API は残していません。

## 破壊的変更

### `readAivmxMetadata` の返り値が変わった / `getSamplingRate` 削除

```ts
// 0.1.0
const meta = readAivmxMetadata(bytes); // { manifest?: unknown, hyperParameters?: unknown, styleVectorsNpy }
const rate = getSamplingRate(meta.hyperParameters); // 不明なら黙って 44100

// 0.2.0
const meta = readAivmxMetadata(bytes); // { hyperParameters?: Sbv2HyperParameters, styleVectorsNpy }
const rate = meta.hyperParameters?.samplingRate; // 型付き。フォールバックなし
```

- `hyperParameters` は `unknown` ではなく型付きサブセット
  （`samplingRate` / `nSpeakers` / `spk2id` / `numStyles` / `style2id` / `modelName` /
  `version`、生 JSON は `raw`）。壊れた値は throw します。
- `manifest` は返り値から外れました。話者カタログが要る場合は
  `readAivmxManifest(bytes, { stripAssets? })`（型付き `AivmManifest`、既定で数 MB の
  アイコン / ボイスサンプル data URL を除去）を使ってください。
- `getSamplingRate` の「不明なら 44100」という無言フォールバックは廃止しました
  （別レートのモデルで音程の狂った音声が無言で出るため）。

### `createFromOnnx` / `fromOnnx` の `sampleRate` が必須に

プレーン ONNX には AIVM メタが無いため、出力レートを黙って仮定しません。

```ts
const adapter = await Sbv2ModelAdapter.createFromOnnx({
  acousticOnnxBytes,
  bertOnnxBytes,
  tokenizer,
  styleVectorsNpy,
  sampleRate: 44100, // 0.2.0 から必須
});
```

### `release()` に契約が付いた（ADR-0004）

- 冪等（2 回目以降は同じ完了を返す）。
- in-flight の `synthesize` / `buildAcousticFeeds` の完了を待ってから解放。
- **release 後の合成呼び出しは throw**（0.1.0 では未定義動作だった）。

### 不正値が throw になった（無言の NaN 音声 → fail loud）

- 非整数の `styleId`、非有限の `styleWeight` / スカラー値は throw。
- `Sbv2NodeModelAdapter` で `device` と `sessionOptions.executionProviders` の同時指定は
  throw（EP 固有オプションを渡すときは `executionProviders` のみを使う。0.1.0 は
  executionProviders を黙って上書きしていた）。

### 依存の追加

`@hdae/fetch-cache`（jsr）が実行時依存に加わりました（`getDeberta` の取得・キャッシュ層）。

## 追加された API

| API                                                                          | 用途                                                                |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `SynthInput.scalars` / `SynthesizeOptions.scalars`                           | リクエスト毎のスカラー部分上書き（話速 `lengthScale` 等。ADR-0003） |
| `SynthesizeOptions.preSilenceSec / postSilenceSec`                           | 文頭・文末の無音（AivisSpeech の pre/postPhonemeLength 相当）       |
| `padSilence` / `concatWithSilence`                                           | 無音パディング・行間無音の連結部品                                  |
| `validateSynthInput(input, tokenizer?)`                                      | phones/tones 直指定（given 経路）の事前検証（ADR-0003）             |
| `readAivmxManifest` / `AivmManifest` ほか                                    | AIVM 1.0 マニフェストの型付き読出し（既定 stripAssets）             |
| `Sbv2HyperParameters` / `readSbv2HyperParameters`                            | hyper_parameters の型付きサブセット                                 |
| `adapter.hyperParameters / numSpeakers / spk2id`                             | アダプタからの話者情報アクセサ                                      |
| `fromAivmx({ metadata })`                                                    | 事前パース済みメタの再利用（巨大 protobuf の再走査回避）            |
| `getDeberta` / `buildDebertaTokenizer` / `DEBERTA_REPO` / `DEBERTA_REVISION` | 量子化 DeBERTa 一式の HF 自動取得（SHA 固定・検証・Cache API）      |
| `mergeScalars`                                                               | スカラーの合成 + 有限性検証（サーバー側の事前検証用）               |

## 推奨移行手順（AivisSpeech 互換サーバー想定）

1. 資産取得の自前実装（fetch + Cache API）を `getDeberta()` に置き換える。
2. `/speakers` 相当は `readAivmxManifest(bytes)`（strip 済み）から組み立てる。アイコンや
   ボイスサンプルが要るエンドポイントだけ `{ stripAssets: false }` で読む。
3. アダプタ生成は一度 `readAivmxMetadata` した値を `createFromAivmx({ metadata })` に渡して
   再走査を省く。話者数・spk2id はアダプタのアクセサから取れる。
4. リクエスト毎の話速等は `synthesize`（または `synthesizeText`）の `scalars` へ。
   speedScale=1/lengthScale などの VOICEVOX 互換マッピングはエンジン側の責務。
5. ユーザー編集アクセント句からの合成は `SynthInput` を自前構築し、受理境界で
   `validateSynthInput(input, tokenizer)` を通す。
