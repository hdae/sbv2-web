// AIVM マニフェスト型付きリーダの behavior テスト。
//
// 実物マニフェスト（数 MB・gitignore）の構造を縮約したフィクスチャで、
// stripAssets の既定挙動（icon / voice_samples を落とす）と fail-loud 検証を固定する。

import { assertEquals, assertThrows } from "@std/assert";
import { parseAivmManifest, readAivmxManifest } from "./aivm_manifest.ts";
import { buildModelWithMetadata } from "../testing/onnx_meta_fixture.ts";

/** 実物（mao.aivmx）を縮約したマニフェスト。base64 部はダミー data URL。 */
const MANIFEST = {
  manifest_version: "1.0",
  name: "まお",
  description: "テスト用",
  creators: ["オズチャット"],
  license: "ACML 1.0 ...",
  model_architecture: "Style-Bert-VITS2 (JP-Extra)",
  model_format: "ONNX",
  training_epochs: 59,
  training_steps: 3000,
  uuid: "a59cb814-0083-4369-8542-f51a29e72af7",
  version: "1.2.0",
  speakers: [
    {
      name: "まお",
      icon: "data:image/jpeg;base64,AAAA",
      supported_languages: ["ja"],
      uuid: "e756b8e4-b606-4e15-99b1-3f9c6a1b2317",
      local_id: 0,
      styles: [
        {
          name: "ノーマル",
          icon: "data:image/jpeg;base64,BBBB",
          local_id: 0,
          voice_samples: [
            { audio: "data:audio/mp4;base64,CCCC", transcript: "こんにちは" },
          ],
        },
        {
          name: "あまあま",
          icon: null,
          local_id: 1,
          voice_samples: [],
        },
      ],
    },
  ],
};

Deno.test("parseAivmManifest: 既定（stripAssets）で icon / voice_samples を落とす", () => {
  const manifest = parseAivmManifest(MANIFEST);
  assertEquals(manifest.name, "まお");
  assertEquals(manifest.manifestVersion, "1.0");
  assertEquals(manifest.uuid, "a59cb814-0083-4369-8542-f51a29e72af7");
  const speaker = manifest.speakers[0];
  assertEquals(speaker.icon, undefined);
  assertEquals(speaker.uuid, "e756b8e4-b606-4e15-99b1-3f9c6a1b2317");
  assertEquals(speaker.localId, 0);
  assertEquals(speaker.supportedLanguages, ["ja"]);
  assertEquals(speaker.styles.map((s) => [s.name, s.localId]), [
    ["ノーマル", 0],
    ["あまあま", 1],
  ]);
  assertEquals(speaker.styles[0].icon, undefined);
  assertEquals(speaker.styles[0].voiceSamples, []);
});

Deno.test("parseAivmManifest: stripAssets: false で icon / voice_samples を保持", () => {
  const manifest = parseAivmManifest(MANIFEST, { stripAssets: false });
  const speaker = manifest.speakers[0];
  assertEquals(speaker.icon, "data:image/jpeg;base64,AAAA");
  assertEquals(speaker.styles[0].voiceSamples, [
    { audio: "data:audio/mp4;base64,CCCC", transcript: "こんにちは" },
  ]);
  // スキーマ上 optional（null）の style icon は undefined になる。
  assertEquals(speaker.styles[1].icon, undefined);
});

Deno.test("parseAivmManifest: 必須フィールド欠落はパス付きで throw", () => {
  const { uuid: _uuid, ...withoutUuid } = MANIFEST;
  assertThrows(() => parseAivmManifest(withoutUuid), Error, "manifest.uuid");

  const brokenStyle = {
    ...MANIFEST,
    speakers: [{
      ...MANIFEST.speakers[0],
      styles: [{ ...MANIFEST.speakers[0].styles[0], local_id: "0" }],
    }],
  };
  assertThrows(
    () => parseAivmManifest(brokenStyle),
    Error,
    "manifest.speakers[0].styles[0].local_id",
  );
});

Deno.test("parseAivmManifest: speakers / styles が空なら throw", () => {
  assertThrows(
    () => parseAivmManifest({ ...MANIFEST, speakers: [] }),
    Error,
    "1 話者以上",
  );
  const emptyStyles = structuredClone(MANIFEST);
  emptyStyles.speakers[0].styles = [];
  assertThrows(() => parseAivmManifest(emptyStyles), Error, "1 スタイル以上");
});

Deno.test("readAivmxManifest: aivmx バイト列から読める・キー欠落は throw", () => {
  const model = buildModelWithMetadata({
    aivm_manifest: JSON.stringify(MANIFEST),
  });
  assertEquals(readAivmxManifest(model).name, "まお");

  const withoutManifest = buildModelWithMetadata({
    aivm_style_vectors: "QUJD",
  });
  assertThrows(
    () => readAivmxManifest(withoutManifest),
    Error,
    "aivm_manifest",
  );
});
