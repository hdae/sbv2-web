# ADR-0005: DeBERTa セッションは共有可能にする（DebertaExtractor と所有権契約）

状態: 採択（2026-07-11） 関連:
[0004](0004-release-lifecycle-contract.md)（release 契約の踏襲）

## 文脈

`Sbv2Adapter` はモデル（音響セッション）ごとに生成されるが、従来は
`bertOnnxBytes` からアダプタ毎に専用の DeBERTa セッションも生成していた。DeBERTa
は全モデル共通の資産（`getDeberta()` の int4 一式）なので、複数モデルを常駐させる
消費者（light-sbv2 の既定 3 常駐）では BERT の常駐メモリがモデル数倍になる。

実測（cpu / ort-node）: DeBERTa int4 セッション 1 本 ≈ 490MB。light-sbv2 サーバー
で 4 モデル常駐時 RSS 2986MB → 共有化で 1874MB（−1.1GB）。モデルロードも BERT
セッション生成分（≈0.85s）短縮（2.5s → 1.7s）。外部実測（RTX 5070 Ti / CUDA EP）
でも DeBERTa int4 の VRAM 常駐は 641MB/セッションで、複製コストは EP を問わない。

## 決定

1. **`DebertaExtractor`**（`src/runtime/deberta_extractor.ts`）にセッション +
   トークナイザ + 音素レベル展開（tile）を分離する。web/node ラッパの
   `createDeberta`、または `DebertaExtractor.create`（OrtBackend 注入）で生成。
2. アダプタへの BERT 供給は判別 union **`BertSource`** にする:
   - `bertOnnxBytes` + `tokenizer` — 従来互換。アダプタが抽出器を生成して
     **所有**する（アダプタの `release()` で一緒に解放）。
   - `deberta` — 生成済みの共有抽出器。アダプタは**解放しない**（MUST NOT —
     他のアダプタが同じセッションで推論中かもしれない）。
3. **所有権は生成者にある**: 共有抽出器の `release()` は、それを使う全アダプタの
   release 後に生成者が呼ぶ。release 済み抽出器の `extract` / ファクトリへの
   持ち込みは throw（fail loud）。
4. release の契約（冪等・in-flight 待機・release 後 throw）は
   [ADR-0004](0004-release-lifecycle-contract.md) をそのまま踏襲する（抽出器は
   自分の in-flight `extract` のみを待つ）。

## 帰結

- 複数モデル常駐の BERT コストが O(モデル数) → O(1) になり、evict → 再ロードも
  音響セッションの生成だけで済む。
- 共有時の `sessionOptions` は音響セッションにだけ効く（BERT 側は抽出器の
  生成時オプションで確定済み）。BERT と音響で別デバイスに置くこともできる。
- 複数アダプタからの並行利用は ORT の公開契約（`InferenceSession.run` は
  並行呼び出し可）に依る。WASM シングルスレッドでは実行が直列化されるだけで
  安全性は変わらない。
- API は additive（bytes 経路は完全互換）。既存消費者の移行は任意。
