// テスト専用の ONNX ModelProto メタデータフィクスチャビルダ。
//
// 実 aivmx（数百 MB・gitignore）は CI に載せられないため、実測した構造
//   ModelProto: field1(ir_version varint), field7(graph, len-delim), field14(metadata_props)*
//   StringStringEntryProto: field1(key), field2(value)
// を最小合成する。aivmx_meta / aivm_manifest のテストで共有する
// （配布物には含めない: deno.json publish.exclude の src/testing/**）。

/** varint エンコード。 */
export const varint = (n: number): number[] => {
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
export const lenDelim = (field: number, payload: number[]): number[] => {
  const tag = (field << 3) | 2;
  return [...varint(tag), ...varint(payload.length), ...payload];
};

/** varint フィールド（wire=0）。 */
export const varintField = (field: number, value: number): number[] => {
  const tag = (field << 3) | 0;
  return [...varint(tag), ...varint(value)];
};

export const strBytes = (s: string): number[] =>
  Array.from(new TextEncoder().encode(s));

/** metadata_props の StringStringEntryProto（key=1, value=2）を組む。 */
export const metadataEntry = (key: string, value: number[]): number[] =>
  lenDelim(14, [...lenDelim(1, strBytes(key)), ...lenDelim(2, value)]);

/** metadata_props 付きの最小 ModelProto バイト列を組む（graph はダミー）。 */
export const buildModelWithMetadata = (
  entries: Readonly<Record<string, string>>,
): Uint8Array => {
  const model = [
    ...varintField(1, 8), // ir_version
    ...lenDelim(7, strBytes("graph")), // graph（長さスキップされる）
    ...Object.entries(entries).flatMap(([key, value]) =>
      metadataEntry(key, strBytes(value))
    ),
  ];
  return new Uint8Array(model);
};
