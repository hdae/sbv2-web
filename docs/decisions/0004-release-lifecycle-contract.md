# 0004. release() は「in-flight 完了を待つ冪等操作」とする

- Status: accepted
- Date: 2026-07-10

## Context

`Sbv2Adapter` は巨大な ONNX セッションを 2 本（音響 + DeBERTa）保持するため明示
`release()` を持つが、従来はガードが無く、(1) 推論中に release すると ORT の
ネイティブセッションを引き抜く競合になり、(2) release 後の synthesize や二重
release の挙動が未定義だった。サーバー消費者から契約の明文化（または簡易ガード）が
要望された。

## Decision

`release()` の契約（実装は `src/runtime/adapter_core.ts`）:

- **冪等**: 2 回目以降は初回と同じ完了 Promise を返す（解放処理は 1 回だけ走る）。
- **in-flight 待機**: 呼び出し時点で進行中の `synthesize` / `buildAcousticFeeds` の
  完了（成功・失敗を問わず）を待ってからセッションを解放する。
- **release 後は fail loud**: release 開始以降の `synthesize` / `buildAcousticFeeds` は
  同期 throw する。
- 片方のセッション解放が失敗しても他方の解放を試み、失敗は握りつぶさず投げ直す。

並行 `synthesize` 自体は禁止しない（安全性は release との関係でのみ保証する）。
同一アダプタへの並行呼び出しの実行順・スループットは ORT の実装依存であり、
スループット制御が要るサーバーは呼び出し側で直列化キューを持つこと。

## Consequences

- 単一スレッドの JS では「release 開始 → 以後の synthesize は同期拒否」なので、
  in-flight 集合は必ず枯れ、待機は有限で終わる（lost-wakeup / TOCTOU なし）。
- UI（examples/browser）は busy 状態でボタンをガードするが、仮に取りこぼしても
  ライブラリ側の契約が推論中の解放を防ぐ（二重防御）。
