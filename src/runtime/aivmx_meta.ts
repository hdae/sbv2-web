// aivmx（= メタデータ付き素の ONNX）から aivm_style_vectors を取り出す軽量抽出。
//
// aivmx は ONNX ModelProto の metadata_props（field 14, repeated StringStringEntryProto）に
// aivm_manifest / aivm_hyper_parameters / aivm_style_vectors を string で持つ（aivmx-interface.md §1.2）。
// 必要なのは aivm_style_vectors（Base64 の .npy）1 キーだけなので、ONNX 全体を protobuf ライブラリで
// パースせず、トップレベルの protobuf ワイヤ走査で graph(field 7, 巨大) を長さスキップし、
// metadata_props のエントリを拾う。aivmlib-web + protobufjs + zod の重い依存ツリーを避けるための自前実装。
//
// 正しさは Python aivmlib が返す style_vectors bytes との byte-exact（sha256）一致で担保する
//   （scripts のハーネス / tools 側 parity で検証）。
//
// ONNX ModelProto の関連フィールド（onnx.proto3）:
//   field 7  = graph      (message, length-delimited)   ← 巨大。中身は読まずスキップ。
//   field 14 = metadata_props (repeated StringStringEntryProto)
//     StringStringEntryProto: field 1 = key (string), field 2 = value (string)

/** protobuf の varint を読む。@returns [値, 次オフセット]。 */
const readVarint = (buf: Uint8Array, offset: number): [number, number] => {
  let shift = 0;
  let result = 0;
  let i = offset;
  // number は 53bit まで安全。ONNX の長さ/タグは 53bit を超えないため number で扱う。
  while (true) {
    if (i >= buf.length) {
      throw new Error("aivmx_meta: varint がバッファ末尾で途切れた");
    }
    const byte = buf[i];
    i += 1;
    result += (byte & 0x7f) * 2 ** shift;
    if ((byte & 0x80) === 0) break;
    shift += 7;
    if (shift > 63) {
      throw new Error("aivmx_meta: varint が長すぎる（壊れた protobuf）");
    }
  }
  return [result, i];
};

/** length-delimited フィールドの [長さ, データ開始オフセット, データ終端オフセット]。 */
const readLenDelim = (
  buf: Uint8Array,
  offset: number,
): [number, number, number] => {
  const [len, dataStart] = readVarint(buf, offset);
  return [len, dataStart, dataStart + len];
};

/**
 * ONNX ModelProto バイト列から aivm_style_vectors の .npy バイト列を取り出す。
 * トップレベルを走査し metadata_props(14) の StringStringEntryProto を拾って
 * key === "aivm_style_vectors" の value（Base64）をデコードする。見つからなければ throw。
 */
export const extractStyleVectorsNpy = (onnxBytes: Uint8Array): Uint8Array => {
  const base64 = extractMetadataValue(onnxBytes, "aivm_style_vectors");
  if (base64 === undefined) {
    throw new Error(
      "aivmx_meta: metadata_props に 'aivm_style_vectors' が無い（aivmx でない/破損を疑う）",
    );
  }
  return base64ToBytes(base64);
};

export type AivmxMetadata = {
  manifest?: unknown;
  hyperParameters?: unknown;
  styleVectorsNpy: Uint8Array;
};

const parseJsonMetadata = (
  onnxBytes: Uint8Array,
  key: string,
): unknown | undefined => {
  const value = extractMetadataValue(onnxBytes, key);
  if (value === undefined) return undefined;
  try {
    return JSON.parse(value) as unknown;
  } catch (cause) {
    throw new Error(
      `aivmx_meta: metadata_props の '${key}' が JSON として読めない`,
      { cause },
    );
  }
};

export const readAivmxMetadata = (onnxBytes: Uint8Array): AivmxMetadata => ({
  manifest: parseJsonMetadata(onnxBytes, "aivm_manifest"),
  hyperParameters: parseJsonMetadata(onnxBytes, "aivm_hyper_parameters"),
  styleVectorsNpy: extractStyleVectorsNpy(onnxBytes),
});

export const getSamplingRate = (
  hyperParameters: unknown,
  fallback = 44100,
): number => {
  const data = hyperParameters !== null && typeof hyperParameters === "object"
    ? (hyperParameters as { data?: unknown }).data
    : undefined;
  const samplingRate = data !== null && typeof data === "object"
    ? (data as { sampling_rate?: unknown }).sampling_rate
    : undefined;
  return typeof samplingRate === "number" && Number.isFinite(samplingRate)
    ? samplingRate
    : fallback;
};

/**
 * ONNX ModelProto バイト列から metadata_props の指定キーの value（string）を取り出す。
 * graph(field 7) は長さスキップし中身をパースしない。見つからなければ undefined。
 */
export const extractMetadataValue = (
  onnxBytes: Uint8Array,
  targetKey: string,
): string | undefined => {
  const decoder = new TextDecoder("utf-8");
  let i = 0;
  while (i < onnxBytes.length) {
    const [tag, afterTag] = readVarint(onnxBytes, i);
    const field = tag >> 3;
    const wire = tag & 0x7;
    i = afterTag;
    if (wire === 0) {
      // varint フィールド（ir_version 等）。値を読み飛ばす。
      [, i] = readVarint(onnxBytes, i);
    } else if (wire === 2) {
      const [, dataStart, dataEnd] = readLenDelim(onnxBytes, i);
      if (field === 14) {
        // StringStringEntryProto を読む。key(1)/value(2)。
        const entry = onnxBytes.subarray(dataStart, dataEnd);
        const parsed = parseStringStringEntry(entry, decoder);
        if (parsed.key === targetKey) return parsed.value;
      }
      // graph(7) を含むそれ以外は中身を読まずスキップ。
      i = dataEnd;
    } else if (wire === 5) {
      i += 4; // 32bit
    } else if (wire === 1) {
      i += 8; // 64bit
    } else {
      throw new Error(`aivmx_meta: 未対応 wire type ${wire}（field ${field}）`);
    }
  }
  return undefined;
};

/** StringStringEntryProto（key=field1 string, value=field2 string）をパースする。 */
const parseStringStringEntry = (
  entry: Uint8Array,
  decoder: TextDecoder,
): { key: string; value: string } => {
  let key = "";
  let value = "";
  let i = 0;
  while (i < entry.length) {
    const [tag, afterTag] = readVarint(entry, i);
    const field = tag >> 3;
    const wire = tag & 0x7;
    i = afterTag;
    if (wire === 2) {
      const [, dataStart, dataEnd] = readLenDelim(entry, i);
      const text = decoder.decode(entry.subarray(dataStart, dataEnd));
      if (field === 1) key = text;
      else if (field === 2) value = text;
      i = dataEnd;
    } else if (wire === 0) {
      [, i] = readVarint(entry, i);
    } else {
      throw new Error(`aivmx_meta: StringStringEntry の未対応 wire ${wire}`);
    }
  }
  return { key, value };
};

/** 標準 Base64 文字列を Uint8Array へデコードする（改行等の空白は無視）。 */
export const base64ToBytes = (base64: string): Uint8Array => {
  // atob はブラウザ/Deno 双方で利用可能。バイナリ文字列を Uint8Array に写す。
  const binary = atob(base64.replace(/\s/g, ""));
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
};
