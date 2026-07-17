// Sbv2Adapter のライフサイクル・feed 組み立て挙動（モック OrtBackend・onnxruntime 非依存）。
//
// static ファクトリの部分初期化失敗でセッションが宙に浮かないこと、per-call scalars の
// 合成、release 契約（冪等・in-flight 待機・release 後 throw, docs/decisions/0004）、
// 共有 DebertaExtractor の所有権契約（アダプタは解放しない, docs/decisions/0005）を固定する。

import {
  assertEquals,
  assertRejects,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import { Sbv2Adapter } from "./adapter_core.ts";
import { DebertaExtractor } from "./deberta_extractor.ts";
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
    () => ({
      output: {
        type: "float32",
        dims: [3, 1024],
        data: new Float32Array(3 * 1024),
      },
    }),
  );

const acousticSession = (
  runImpl?: () =>
    | Record<string, unknown>
    | Promise<Record<string, unknown>>,
): MockSession =>
  new MockSession(
    CANONICAL_INPUTS,
    runImpl ??
      (() => ({
        output: { type: "float32", data: Float32Array.from([0.25]) },
      })),
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

Deno.test("synthesize: 音響出力 dtype が float32 でなければ throw", async () => {
  // fp16 等が音響セッションの出力境界に漏れた状況を注入する（synth_aivmx.py の
  // np.asarray(raw, dtype=np.float32) 相当の防御を確認）。
  const acoustic = acousticSession(() => ({
    output: { type: "float16", data: new Uint16Array([0]) },
  }));
  const adapter = await makeAdapter(acoustic, bertSession());
  await assertRejects(
    () => adapter.synthesize(INPUT),
    Error,
    "dtype が想定外",
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

  finishRun({ output: { type: "float32", data: Float32Array.from([0.5]) } });
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

// ---- 共有 DebertaExtractor（docs/decisions/0005） ----

const makeExtractor = async (bert: MockSession): Promise<DebertaExtractor> => {
  const { backend } = createMockBackend([() => bert]);
  return await DebertaExtractor.create(backend, {
    bertOnnxBytes: new Uint8Array(),
    tokenizer: TOKENIZER,
  });
};

/** 共有 deberta でアダプタを作る（backend は acoustic 1 本だけを供給する）。 */
const makeSharedAdapter = (
  acoustic: MockSession,
  extractor: DebertaExtractor,
): Promise<Sbv2Adapter> => {
  const { backend } = createMockBackend([() => acoustic]);
  return Sbv2Adapter.fromOnnx(backend, {
    acousticOnnxBytes: new Uint8Array(),
    styleVectorsNpy: buildStyleNpy(2),
    sampleRate: 44100,
    deberta: extractor,
  });
};

Deno.test("共有 deberta: 2 アダプタが 1 本の BERT セッションを使い、アダプタ release で解放されない", async () => {
  const bert = bertSession();
  const extractor = await makeExtractor(bert);
  const acousticA = acousticSession();
  const acousticB = acousticSession();
  const adapterA = await makeSharedAdapter(acousticA, extractor);
  const adapterB = await makeSharedAdapter(acousticB, extractor);

  await adapterA.synthesize(INPUT);
  await adapterB.synthesize(INPUT);
  // 両アダプタの BERT 推論が同一セッションへ流れている（複製されていない）。
  assertEquals(bert.runFeeds.length, 2);

  await adapterA.release();
  await adapterB.release();
  assertEquals(acousticA.released, 1);
  assertEquals(acousticB.released, 1);
  // 共有 BERT はアダプタが解放しない（所有権は生成者）。
  assertEquals(bert.released, 0);
  await extractor.release();
  assertEquals(bert.released, 1);
});

Deno.test("fromOnnx: release 済みの deberta は拒否（セッション未生成のまま throw）", async () => {
  const extractor = await makeExtractor(bertSession());
  await extractor.release();
  const { backend, sessions } = createMockBackend([]);
  await assertRejects(
    () =>
      Sbv2Adapter.fromOnnx(backend, {
        acousticOnnxBytes: new Uint8Array(),
        styleVectorsNpy: buildStyleNpy(2),
        sampleRate: 44100,
        deberta: extractor,
      }),
    Error,
    "release() 済み",
  );
  assertEquals(sessions.length, 0);
});

Deno.test("fromOnnx: deberta と bertOnnxBytes/tokenizer の同時指定は throw", async () => {
  const extractor = await makeExtractor(bertSession());
  const { backend, sessions } = createMockBackend([]);
  // 型で防いでいる誤用を JS 呼び出し相当で検証する（テスト境界の限定 cast）。
  const conflicting = {
    acousticOnnxBytes: new Uint8Array(),
    styleVectorsNpy: buildStyleNpy(2),
    sampleRate: 44100,
    bertOnnxBytes: new Uint8Array(),
    tokenizer: TOKENIZER,
    deberta: extractor,
  } as unknown as Parameters<typeof Sbv2Adapter.fromOnnx>[1];
  await assertRejects(
    () => Sbv2Adapter.fromOnnx(backend, conflicting),
    Error,
    "同時指定できない",
  );
  assertEquals(sessions.length, 0);
  await extractor.release();
});

Deno.test("fromOnnx: BERT 供給なし（bytes も deberta も無い）は throw", async () => {
  const { backend, sessions } = createMockBackend([]);
  // 型で防いでいる誤用を JS 呼び出し相当で検証する（テスト境界の限定 cast）。
  const missing = {
    acousticOnnxBytes: new Uint8Array(),
    styleVectorsNpy: buildStyleNpy(2),
    sampleRate: 44100,
  } as unknown as Parameters<typeof Sbv2Adapter.fromOnnx>[1];
  await assertRejects(
    () => Sbv2Adapter.fromOnnx(backend, missing),
    Error,
    "BERT の供給が必要",
  );
  assertEquals(sessions.length, 0);
});
