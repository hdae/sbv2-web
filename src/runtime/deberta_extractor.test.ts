// DebertaExtractor のライフサイクル・出力検証（モック OrtBackend・onnxruntime 非依存）。
//
// release 契約（冪等・in-flight 待機・release 後 throw — Sbv2Adapter と同じ,
// docs/decisions/0004）と、DeBERTa 出力の shape / word2ph 整合の fail loud を固定する。

import {
  assertEquals,
  assertRejects,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import { DebertaExtractor } from "./deberta_extractor.ts";
import { createMockBackend, MockSession } from "../testing/mock_backend.ts";
import { DebertaTokenizer } from "../text/deberta_tokenizer.ts";

const TOKENIZER = DebertaTokenizer.fromVocabText(
  ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "あ"].join("\n"),
  { removed: [], spaced: [] },
  { clsId: 1, sepId: 2, unkId: 3 },
);

/**
 * 最小の抽出入力: bertText "あ"（DeBERTa 3 トークン = [CLS,あ,SEP]）/
 * baseWord2ph [1,1,1]。addBlank 後 word2ph=[3,2,2]・seqLen=7 で整合。
 */
const BERT_TEXT = "あ";
const BASE_WORD2PH = [1, 1, 1];
const SEQ_LEN = 7;

const bertSession = (
  runImpl?: () =>
    | Record<string, unknown>
    | Promise<Record<string, unknown>>,
): MockSession =>
  new MockSession(
    ["input_ids", "attention_mask"],
    runImpl ??
      (() => ({
        output: { dims: [3, 1024], data: new Float32Array(3 * 1024) },
      })),
  );

const makeExtractor = async (
  session: MockSession,
): Promise<DebertaExtractor> => {
  const { backend } = createMockBackend([() => session]);
  return await DebertaExtractor.create(backend, {
    bertOnnxBytes: new Uint8Array(),
    tokenizer: TOKENIZER,
  });
};

Deno.test("extract: word2ph に沿って音素レベル [1024*T] へ展開される", async () => {
  const extractor = await makeExtractor(bertSession());
  const features = await extractor.extract(BERT_TEXT, BASE_WORD2PH, SEQ_LEN);
  assertEquals(features.length, 1024 * SEQ_LEN);
  await extractor.release();
});

Deno.test("extract: DeBERTa 出力 shape が [seq_len, 1024] でなければ throw", async () => {
  const extractor = await makeExtractor(
    bertSession(() => ({
      output: { dims: [3, 512], data: new Float32Array(3 * 512) },
    })),
  );
  await assertRejects(
    () => extractor.extract(BERT_TEXT, BASE_WORD2PH, SEQ_LEN),
    Error,
    "出力 shape が想定外",
  );
  await extractor.release();
});

Deno.test("extract: トークン数と word2ph 長の不一致は throw", async () => {
  const extractor = await makeExtractor(
    bertSession(() => ({
      output: { dims: [4, 1024], data: new Float32Array(4 * 1024) },
    })),
  );
  await assertRejects(
    () => extractor.extract(BERT_TEXT, BASE_WORD2PH, SEQ_LEN),
    Error,
    "word2ph 長",
  );
  await extractor.release();
});

Deno.test("release: in-flight の extract 完了を待ってから解放する", async () => {
  let finishRun = (_: Record<string, unknown>) => {};
  const gate = new Promise<Record<string, unknown>>((resolve) => {
    finishRun = resolve;
  });
  const session = bertSession(() => gate);
  const extractor = await makeExtractor(session);

  const extracting = extractor.extract(BERT_TEXT, BASE_WORD2PH, SEQ_LEN);
  const released = extractor.release();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assertEquals(session.released, 0);

  finishRun({ output: { dims: [3, 1024], data: new Float32Array(3 * 1024) } });
  await extracting;
  await released;
  assertEquals(session.released, 1);
});

Deno.test("release: 冪等（2 回目は同じ完了・解放は 1 回だけ）・後続 extract は同期 throw", async () => {
  const session = bertSession();
  const extractor = await makeExtractor(session);
  assertEquals(extractor.isReleased, false);
  const first = extractor.release();
  const second = extractor.release();
  assertStrictEquals(first, second);
  await first;
  assertEquals(session.released, 1);
  assertEquals(extractor.isReleased, true);
  assertThrows(
    () => extractor.extract(BERT_TEXT, BASE_WORD2PH, SEQ_LEN),
    Error,
    "release() 後",
  );
});
