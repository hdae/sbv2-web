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

import {
  expectFiniteNumber,
  expectRecord,
  optFiniteNumber,
  optNumberRecord,
  optString,
} from "./json_expect.ts";

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

/**
 * SBV2 の hyper_parameters（config.json 相当, aivmlib StyleBertVITS2HyperParameters）の
 * 型付きサブセット。推論・話者/スタイル選択に要るフィールドだけを型付けし、
 * それ以外は raw から読む。
 */
export type Sbv2HyperParameters = {
  /** ルートの model_name。 */
  modelName?: string;
  /** SBV2 バージョン文字列（例 "2.7.0-JP-Extra"）。 */
  version?: string;
  /** data.sampling_rate（Hz）。SBV2 hparams の必須値（欠落・不正は throw）。 */
  samplingRate: number;
  /** data.n_speakers。 */
  nSpeakers?: number;
  /** 話者名 → speaker local_id（sid テンソルに入れる値の真実源）。 */
  spk2id?: Readonly<Record<string, number>>;
  /** data.num_styles（style_vectors の行数と一致するはず）。 */
  numStyles?: number;
  /** スタイル名 → style local_id。 */
  style2id?: Readonly<Record<string, number>>;
  /** 生の hyper_parameters JSON（型付け外のフィールドはここから読む）。 */
  raw: unknown;
};

/**
 * hyper_parameters の unknown JSON を型付きサブセットへ検証する（fail loud）。
 * sampling_rate は波形出力の正しさに直結する（黙って既定値に落とすと別レートの
 * モデルで音程が狂った音声が無言で出る）ため、欠落・不正は throw する。
 */
export const readSbv2HyperParameters = (json: unknown): Sbv2HyperParameters => {
  const root = expectRecord(json, "hyper_parameters");
  const data = expectRecord(root.data, "hyper_parameters.data");
  return {
    modelName: optString(root.model_name, "hyper_parameters.model_name"),
    version: optString(root.version, "hyper_parameters.version"),
    samplingRate: expectFiniteNumber(
      data.sampling_rate,
      "hyper_parameters.data.sampling_rate",
    ),
    nSpeakers: optFiniteNumber(
      data.n_speakers,
      "hyper_parameters.data.n_speakers",
    ),
    spk2id: optNumberRecord(data.spk2id, "hyper_parameters.data.spk2id"),
    numStyles: optFiniteNumber(
      data.num_styles,
      "hyper_parameters.data.num_styles",
    ),
    style2id: optNumberRecord(data.style2id, "hyper_parameters.data.style2id"),
    raw: json,
  };
};

export type AivmxMetadata = {
  /** aivm_hyper_parameters（キー自体が無い aivmx では undefined）。 */
  hyperParameters?: Sbv2HyperParameters;
  styleVectorsNpy: Uint8Array;
};

/**
 * aivmx から推論に必要なメタデータを取り出す。
 * NOTE: aivm_manifest（実物で数 MB。base64 アイコン/ボイスサンプルが支配的）は
 * ここでは読まない — 話者カタログ等が要るときだけ readAivmxManifest を使う。
 */
export const readAivmxMetadata = (onnxBytes: Uint8Array): AivmxMetadata => {
  const hparamsText = extractMetadataValue(onnxBytes, "aivm_hyper_parameters");
  return {
    hyperParameters: hparamsText === undefined
      ? undefined
      : readSbv2HyperParameters(
        parseMetadataJson(hparamsText, "aivm_hyper_parameters"),
      ),
    styleVectorsNpy: extractStyleVectorsNpy(onnxBytes),
  };
};

/** metadata_props の JSON 文字列をパースする（壊れた JSON は key 名付きで throw）。 */
export const parseMetadataJson = (text: string, key: string): unknown => {
  try {
    return JSON.parse(text) as unknown;
  } catch (cause) {
    throw new Error(
      `aivmx_meta: metadata_props の '${key}' が JSON として読めない`,
      { cause },
    );
  }
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
