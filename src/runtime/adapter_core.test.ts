// Sbv2Adapter のライフサイクル・feed 組み立て挙動（モック OrtBackend・onnxruntime 非依存）。
//
// static ファクトリの部分初期化失敗でセッションが宙に浮かないこと、per-call scalars の
// 合成、release 契約（冪等・in-flight 待機・release 後 throw, docs/decisions/0004）を固定する。

import {
  assertEquals,
  assertRejects,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import { Sbv2Adapter } from "./adapter_core.ts";
import type { SynthInput } from "./adapter_types.ts";
import {
  asMockTensor,
  createMockBackend,
  MockSession,
} from "../testing/mock_backend.ts";
import { buildStyleNpy } from "../testing/npy_fixture.ts";
import { DebertaTokenizer } from "../text/deberta_tokenizer.ts";

const TOKENIZER = DebertaTokenizer.fromVocabText(
  ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "あ"].join("\n"),
  { removed: [], spaced: [] },
  { clsId: 1, sepId: 2, unkId: 3 },
);

/** aivmx-interface.md §2.1 の 11 入力（実グラフの入力名で束縛される）。 */
const CANONICAL_INPUTS = [
  "x_tst",
  "x_tst_lengths",
  "sid",
  "tones",
  "language",
  "bert",
  "style_vec",
  "length_scale",
  "sdp_ratio",
  "noise_scale",
  "noise_scale_w",
];

/**
 * 最小の合成入力: phones ["_","a","_"] / bertText "あ"（DeBERTa 3 トークン =
 * [CLS,あ,SEP]）/ baseWord2ph [1,1,1]。addBlank 後 word2ph=[3,2,2]・seqLen=7 で整合。
 */
const INPUT: SynthInput = {
  phones: ["_", "a", "_"],
  tones: [0, 0, 0],
  bertText: "あ",
  baseWord2ph: [1, 1, 1],
  styleId: 0,
  styleWeight: 1,
  speakerId: 0,
};

const bertSession = (): MockSession =>
  new MockSession(
    ["input_ids", "attention_mask"],
    () => ({ output: { dims: [3, 1024], data: new Float32Array(3 * 1024) } }),
  );

const acousticSession = (
  runImpl?: () =>
    | Record<string, unknown>
    | Promise<Record<string, unknown>>,
): MockSession =>
  new MockSession(
    CANONICAL_INPUTS,
    runImpl ?? (() => ({ output: { data: Float32Array.from([0.25]) } })),
  );

/** acoustic → bert の順にモックを生成する（fromOnnx の生成順に一致）。 */
const makeAdapter = (
  acoustic: MockSession,
  bert: MockSession,
): Promise<Sbv2Adapter> => {
  const { backend } = createMockBackend([() => acoustic, () => bert]);
  return Sbv2Adapter.fromOnnx(backend, {
    acousticOnnxBytes: new Uint8Array(),
    bertOnnxBytes: new Uint8Array(),
    tokenizer: TOKENIZER,
    styleVectorsNpy: buildStyleNpy(2),
    sampleRate: 44100,
  });
};

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
        sampleRate: 44100,
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
        sampleRate: 44100,
        styleVectorsNpy: new Uint8Array([1, 2, 3]),
      }),
    Error,
    "マジック",
  );
  assertEquals(sessions.length, 0);
});

Deno.test("synthesize: per-call scalars がアダプタ既定へ部分上書きされる", async () => {
  const acoustic = acousticSession();
  const adapter = await makeAdapter(acoustic, bertSession());
  await adapter.synthesize({ ...INPUT, scalars: { lengthScale: 1.5 } });
  const feed = acoustic.runFeeds[0];
  // 指定キーは上書き、未指定キーは既定（DEFAULT_SCALARS）のまま。
  assertEquals(asMockTensor(feed.length_scale).data[0], 1.5);
  assertEquals(
    asMockTensor(feed.sdp_ratio).data[0],
    Float32Array.from([0.2])[0],
  );
  // 実グラフ入力名で全て束縛されている。
  assertEquals(Object.keys(feed).toSorted(), CANONICAL_INPUTS.toSorted());
  await adapter.release();
});

Deno.test("synthesize: 非有限の per-call scalars は throw", async () => {
  const adapter = await makeAdapter(acousticSession(), bertSession());
  await assertRejects(
    () => adapter.synthesize({ ...INPUT, scalars: { noiseScale: Number.NaN } }),
    Error,
    "有限数でない",
  );
  await adapter.release();
});

Deno.test("release: in-flight の synthesize 完了を待ってから解放する", async () => {
  let finishRun = (_: Record<string, unknown>) => {};
  const gate = new Promise<Record<string, unknown>>((resolve) => {
    finishRun = resolve;
  });
  const acoustic = acousticSession(() => gate);
  const bert = bertSession();
  const adapter = await makeAdapter(acoustic, bert);

  const synth = adapter.synthesize(INPUT);
  const released = adapter.release();
  // 合成が保留中の間はセッションを解放しない（マイクロタスクを流して確認）。
  await new Promise((resolve) => setTimeout(resolve, 0));
  assertEquals(acoustic.released, 0);
  assertEquals(bert.released, 0);

  finishRun({ output: { data: Float32Array.from([0.5]) } });
  const wave = await synth;
  await released;
  assertEquals(Array.from(wave), [0.5]);
  assertEquals(acoustic.released, 1);
  assertEquals(bert.released, 1);
});

Deno.test("release: 冪等（2 回目は同じ完了・解放は 1 回だけ）", async () => {
  const acoustic = acousticSession();
  const bert = bertSession();
  const adapter = await makeAdapter(acoustic, bert);
  const first = adapter.release();
  const second = adapter.release();
  assertStrictEquals(first, second);
  await first;
  assertEquals(acoustic.released, 1);
  assertEquals(bert.released, 1);
});

Deno.test("release 後の synthesize / buildAcousticFeeds は同期 throw", async () => {
  const adapter = await makeAdapter(acousticSession(), bertSession());
  await adapter.release();
  assertThrows(() => adapter.synthesize(INPUT), Error, "release() 後");
  assertThrows(() => adapter.buildAcousticFeeds(INPUT), Error, "release() 後");
});
