// aivmx メタデータ軽量抽出の決定的テスト（CI 対象）。
//
// 実 aivmx（data/ 配下・gitignore, 246MB）は CI に載せられないので、実測した
// ModelProto 構造を最小合成する（フィクスチャは src/testing/onnx_meta_fixture.ts）。
// graph の中身にキー名と紛らわしいバイトを混ぜ、トップレベル走査がそこを誤読しない
// ことも確認する。hyper_parameters の型付きサブセット検証もここで固定する。

import { assertEquals, assertThrows } from "@std/assert";
import {
  base64ToBytes,
  extractMetadataValue,
  extractStyleVectorsNpy,
  readAivmxMetadata,
  readSbv2HyperParameters,
} from "./aivmx_meta.ts";
import {
  buildModelWithMetadata,
  lenDelim,
  metadataEntry,
  strBytes,
  varintField,
} from "../testing/onnx_meta_fixture.ts";

Deno.test("extractMetadataValue: graph をスキップして metadata_props を拾う", () => {
  // graph の中身に "aivm_style_vectors" というキー風のバイトを混ぜ、誤読しないことを確認。
  const fakeGraph = [...strBytes("aivm_style_vectors garbage inside graph")];
  const model = [
    ...varintField(1, 8), // ir_version
    ...lenDelim(7, fakeGraph), // graph（中身は読まずスキップされる）
    ...metadataEntry("aivm_manifest", strBytes("{}")),
    ...metadataEntry("aivm_hyper_parameters", strBytes("{}")),
    ...metadataEntry("aivm_style_vectors", strBytes("QUJD")), // Base64("ABC")
  ];
  const bytes = new Uint8Array(model);
  assertEquals(extractMetadataValue(bytes, "aivm_manifest"), "{}");
  assertEquals(extractMetadataValue(bytes, "aivm_style_vectors"), "QUJD");
  assertEquals(extractMetadataValue(bytes, "nonexistent"), undefined);
});

Deno.test("extractStyleVectorsNpy: Base64 をデコードして bytes を返す", () => {
  const model = buildModelWithMetadata({ aivm_style_vectors: "QUJD" }); // "ABC"
  const npy = extractStyleVectorsNpy(model);
  assertEquals(Array.from(npy), [65, 66, 67]); // "ABC"
});

Deno.test("extractStyleVectorsNpy: キー欠落は throw（fail loudly）", () => {
  const model = buildModelWithMetadata({ aivm_manifest: "{}" });
  assertThrows(
    () => extractStyleVectorsNpy(model),
    Error,
    "aivm_style_vectors",
  );
});

Deno.test("base64ToBytes: 空白除去とデコード", () => {
  // "SGVsbG8=" = "Hello"
  assertEquals(Array.from(base64ToBytes("SGVs\nbG8=")), [
    72,
    101,
    108,
    108,
    111,
  ]);
});

// ---- hyper_parameters の型付きサブセット ----

/** 実物（mao.aivmx）の構造を縮約した hyper_parameters。 */
const HPARAMS = {
  model_name: "まお",
  version: "2.7.0-JP-Extra",
  data: {
    sampling_rate: 44100,
    n_speakers: 1,
    spk2id: { "まお": 0 },
    num_styles: 6,
    style2id: { "ノーマル": 0, "あまあま": 1 },
  },
};

Deno.test("readSbv2HyperParameters: 実物相当の JSON を型付きへ", () => {
  const hp = readSbv2HyperParameters(HPARAMS);
  assertEquals(hp.modelName, "まお");
  assertEquals(hp.samplingRate, 44100);
  assertEquals(hp.nSpeakers, 1);
  assertEquals(hp.spk2id, { "まお": 0 });
  assertEquals(hp.numStyles, 6);
  assertEquals(hp.style2id?.["あまあま"], 1);
  assertEquals(hp.raw, HPARAMS);
});

Deno.test("readSbv2HyperParameters: sampling_rate 欠落・不正は throw", () => {
  assertThrows(
    () => readSbv2HyperParameters({ data: {} }),
    Error,
    "sampling_rate",
  );
  assertThrows(
    () => readSbv2HyperParameters({ data: { sampling_rate: "44100" } }),
    Error,
    "sampling_rate",
  );
});

Deno.test("readSbv2HyperParameters: spk2id の値が数値でなければ throw（パス付き）", () => {
  assertThrows(
    () =>
      readSbv2HyperParameters({
        data: { sampling_rate: 44100, spk2id: { "まお": "0" } },
      }),
    Error,
    "spk2id.まお",
  );
});

Deno.test("readSbv2HyperParameters: optional の null は undefined 扱い", () => {
  const hp = readSbv2HyperParameters({
    data: { sampling_rate: 44100, n_speakers: null },
  });
  assertEquals(hp.nSpeakers, undefined);
});

Deno.test("readAivmxMetadata: hparams を型付きで返し、キー欠落なら undefined", () => {
  const withHparams = buildModelWithMetadata({
    aivm_hyper_parameters: JSON.stringify(HPARAMS),
    aivm_style_vectors: "QUJD",
  });
  const metadata = readAivmxMetadata(withHparams);
  assertEquals(metadata.hyperParameters?.samplingRate, 44100);
  assertEquals(Array.from(metadata.styleVectorsNpy), [65, 66, 67]);

  const withoutHparams = buildModelWithMetadata({ aivm_style_vectors: "QUJD" });
  assertEquals(readAivmxMetadata(withoutHparams).hyperParameters, undefined);
});

Deno.test("readAivmxMetadata: 壊れた hparams JSON は throw", () => {
  const model = buildModelWithMetadata({
    aivm_hyper_parameters: "{broken",
    aivm_style_vectors: "QUJD",
  });
  assertThrows(() => readAivmxMetadata(model), Error, "JSON として読めない");
});
