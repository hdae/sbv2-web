# 0.2.0 → 0.3.0 移行ガイド

0.3.0 は yomi v0.4.0（実在記号 API）への追随リリース。公開 API のシグネチャは不変で、
変わるのは **given_phone に載る記号の中身**と**依存の下限**。

## 挙動の変更（音声出力に影響）

`toSbv2PhoneTone`（および `synthesizeText` 経由の合成）は、ポーズ分類から記号を合成する
方式（読点→`,`・句点/文末→`.`）をやめ、**テキストに実在した句読点の正規形**
（`! ? … , . ' -`）をそのまま音素列に載せるようになった。

- 「？」「！」がモデルに届くようになり、疑問文の語尾イントネーションが本家 SBV2 同等になる。
- テキストに句読点が無い文末に合成 `.` は**出なくなる**（本家 g2p と同方針。句境界は
  トーンの 0 戻りで符号化される）。
- 同一テキストでも 0.2.0 と音素列が変わり得るため、**波形のバイト一致は保たれない**。

## 依存の下限

- `@hdae/yomi@^0.4.0` — `FrontendResult.leadingPunctuations` /
  `AccentPhrase.punctuations` が必須（`pausePunct` は yomi 側で削除済み）。
  `FrontendResult` を手組みして `toSbv2PhoneTone` に渡している場合は両フィールドの
  追加が必要。
- `@hdae/fetch-cache@^0.2.0` — API 変更なし（yomi 0.4.1 と同一版に収束させるための
  floor 引き上げのみ）。

## import パスの変更（yomi 側の破壊的変更の伝搬）

辞書ローダは `@hdae/yomi/browser` → `@hdae/yomi/loader`（yomi ADR-0006）。
`getDictionary()` の使い方は不変。
