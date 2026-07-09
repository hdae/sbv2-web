# 0003. SynthInput を公開契約にし、per-call スカラーを導入する

- Status: accepted
- Date: 2026-07-10

## Context

AivisSpeech 互換サーバーなどの消費者は、ユーザー編集済みアクセント句からの合成
（VOICEVOX 系の本線フロー）のために `toSbv2PhoneTone` / `buildBaseWord2ph` 相当の
出力を自前構築して `synthesize` に渡す。しかし word2ph の不変条件
（`sum(baseWord2ph) === phones.length`、`baseWord2ph.length === トークン数 + 2` 等）は
アダプタ内部の知識のままで、破れは DeBERTa 実行後の深い位置でしか検出されなかった。
また合成スカラー（話速 `lengthScale`・`sdpRatio` 等）はアダプタ生成時に固定で、
VOICEVOX クライアントの speedScale / tempoDynamicsScale をリクエスト毎に変えられなかった。

## Decision

1. **`SynthInput` を公開の安定契約とする。** 不変条件は `validateSynthInput(input,
   tokenizer?)` として export し、消費者が合成前に安価に fail loud できる入口を提供する
   （tokenizer を渡したときだけ DeBERTa トークン数との整合まで検査する）。
2. **per-call スカラー**: `SynthInput.scalars?: Partial<SynthScalars>`（および
   `SynthesizeOptions.scalars`）でリクエスト毎の部分上書きを受け、`mergeScalars` が
   アダプタ既定へ重ねる。非有限値はグラフに流れる前に throw する。
3. **文頭・文末の無音は後処理ユーティリティで提供する**（`padSilence` /
   `concatWithSilence`、`SynthesizeOptions.preSilenceSec / postSilenceSec`）。本家の対応物
   （AivisSpeech の pre/postPhonemeLength、SBV2 の split_interval）もモデル外の
   ゼロ詰めであり、句読点・句間ポーズは今後も given_phone の句読点記号 →
   モデル内 duration 予測に委ねる（波形レベルの秒数指定はしない。本家にも無い）。

## Consequences

- サーバー実装は aivmx の二重パースや内部知識の複製なしに `/audio_query` 相当 →
  `synthesize` を組める。
- `validateSynthInput` は任意の入口であり、`synthesize` 自体の内部検査（DeBERTa
  トークン数不一致 throw 等）は従来どおり残る（二重防御）。
- スカラーの意味は synth_aivmx.py / 本家 infer と同じ（`lengthScale` は大きいほど遅い）。
  AivisSpeech の speedScale=1/lengthScale や tempoDynamicsScale→sdpRatio の非線形
  マッピングはエンジン層（消費者側）の責務とする。
