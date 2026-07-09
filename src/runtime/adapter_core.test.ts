// Sbv2Adapter のライフサイクル挙動（モック OrtBackend・onnxruntime 非依存）。
//
// static ファクトリの部分初期化失敗でセッションが宙に浮かないこと（呼び出し側は
// インスタンスを受け取れないため自力では解放不能）を固定する。

import { assertEquals, assertRejects } from "@std/assert";
import { Sbv2Adapter } from "./adapter_core.ts";
import { createMockBackend, MockSession } from "../testing/mock_backend.ts";
import { buildStyleNpy } from "../testing/npy_fixture.ts";
import { DebertaTokenizer } from "../text/deberta_tokenizer.ts";

const TOKENIZER = DebertaTokenizer.fromVocabText(
  ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "あ"].join("\n"),
  { removed: [], spaced: [] },
  { clsId: 1, sepId: 2, unkId: 3 },
);

Deno.test("fromOnnx: bert セッション生成失敗で acoustic を解放してから throw", async () => {
  const { backend, sessions } = createMockBackend([
    () => new MockSession(["x_tst"]),
    () => {
      throw new Error("bert の生成に失敗");
    },
  ]);
  await assertRejects(
    () =>
      Sbv2Adapter.fromOnnx(backend, {
        acousticOnnxBytes: new Uint8Array(),
        bertOnnxBytes: new Uint8Array(),
        tokenizer: TOKENIZER,
        styleVectorsNpy: buildStyleNpy(2),
      }),
    Error,
    "bert の生成に失敗",
  );
  // acoustic は生成済み → リークせず解放されていること。
  assertEquals(sessions.length, 1);
  assertEquals(sessions[0].released, 1);
});

Deno.test("fromOnnx: style_vectors 不正はセッション生成前に throw（生成数 0）", async () => {
  const { backend, sessions } = createMockBackend([]);
  await assertRejects(
    () =>
      Sbv2Adapter.fromOnnx(backend, {
        acousticOnnxBytes: new Uint8Array(),
        bertOnnxBytes: new Uint8Array(),
        tokenizer: TOKENIZER,
        styleVectorsNpy: new Uint8Array([1, 2, 3]),
      }),
    Error,
    "マジック",
  );
  assertEquals(sessions.length, 0);
});
