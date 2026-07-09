// aivmx メタデータ軽量抽出の決定的テスト（CI 対象）。
//
// 実 aivmx（data/ 配下・gitignore, 246MB）は CI に載せられないので、実測した構造
//   ModelProto: field1(ir_version varint), field7(graph, 巨大 len-delim), field14(metadata_props)*
//   StringStringEntryProto: field1(key), field2(value)
// を最小合成して、graph を長さスキップしつつ metadata_props から値を拾えることを検証する。
// graph の中身にキー名と紛らわしいバイトを混ぜ、トップレベル走査がそこを誤読しないことも確認する。

import { assertEquals, assertThrows } from "@std/assert";
import {
  base64ToBytes,
  extractMetadataValue,
  extractStyleVectorsNpy,
} from "./aivmx_meta.ts";

/** varint エンコード。 */
const varint = (n: number): number[] => {
  const out: number[] = [];
  let v = n;
  while (true) {
    const b = v & 0x7f;
    v = Math.floor(v / 128);
    if (v > 0) out.push(b | 0x80);
    else {
      out.push(b);
      break;
    }
  }
  return out;
};

/** length-delimited フィールド（tag wire=2）。 */
const lenDelim = (field: number, payload: number[]): number[] => {
  const tag = (field << 3) | 2;
  return [...varint(tag), ...varint(payload.length), ...payload];
};

/** varint フィールド（wire=0）。 */
const varintField = (field: number, value: number): number[] => {
  const tag = (field << 3) | 0;
  return [...varint(tag), ...varint(value)];
};

const strBytes = (s: string): number[] =>
  Array.from(new TextEncoder().encode(s));

/** StringStringEntryProto（key=1, value=2）を組む。 */
const entry = (key: string, value: number[]): number[] =>
  lenDelim(14, [...lenDelim(1, strBytes(key)), ...lenDelim(2, value)]);

Deno.test("extractMetadataValue: graph をスキップして metadata_props を拾う", () => {
  // graph の中身に "aivm_style_vectors" というキー風のバイトを混ぜ、誤読しないことを確認。
  const fakeGraph = [...strBytes("aivm_style_vectors garbage inside graph")];
  const model = [
    ...varintField(1, 8), // ir_version
    ...lenDelim(7, fakeGraph), // graph（中身は読まずスキップされる）
    ...entry("aivm_manifest", strBytes("{}")),
    ...entry("aivm_hyper_parameters", strBytes("{}")),
    ...entry("aivm_style_vectors", strBytes("QUJD")), // Base64("ABC")
  ];
  const bytes = new Uint8Array(model);
  assertEquals(extractMetadataValue(bytes, "aivm_manifest"), "{}");
  assertEquals(extractMetadataValue(bytes, "aivm_style_vectors"), "QUJD");
  assertEquals(extractMetadataValue(bytes, "nonexistent"), undefined);
});

Deno.test("extractStyleVectorsNpy: Base64 をデコードして bytes を返す", () => {
  const model = [
    ...varintField(1, 8),
    ...lenDelim(7, strBytes("graph")),
    ...entry("aivm_style_vectors", strBytes("QUJD")), // "ABC"
  ];
  const npy = extractStyleVectorsNpy(new Uint8Array(model));
  assertEquals(Array.from(npy), [65, 66, 67]); // "ABC"
});

Deno.test("extractStyleVectorsNpy: キー欠落は throw（fail loudly）", () => {
  const model = [
    ...varintField(1, 8),
    ...entry("aivm_manifest", strBytes("{}")),
  ];
  assertThrows(
    () => extractStyleVectorsNpy(new Uint8Array(model)),
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
