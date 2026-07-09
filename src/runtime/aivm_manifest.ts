// AIVM 1.0 マニフェスト（aivm_manifest）の型付きリーダ。
//
// スキーマは aivmlib（Aivis-Project）の AivmManifest 準拠。実物マニフェストは数 MB あり、
// その 99.9% は話者アイコン・スタイルアイコン・ボイスサンプルの data URL（base64）なので、
// 既定では stripAssets: true でそれらを落として返す（/speakers 相当のカタログ用途に十分）。
// アイコンやボイスサンプルまで要る場合だけ stripAssets: false で読む。
//
// 検証は「推論・カタログに必要なフィールドの型」だけを fail loud に行う（Zod 全検証は
// 数 MB の base64 文字列に対してコストが割に合わない。docs/decisions/0003 参照）。

import { extractMetadataValue, parseMetadataJson } from "./aivmx_meta.ts";
import {
  expectArray,
  expectFiniteNumber,
  expectRecord,
  expectString,
  optFiniteNumber,
  optString,
  optStringArray,
} from "./json_expect.ts";

/** スタイルのボイスサンプル（audio は data URL）。 */
export type AivmVoiceSample = {
  audio: string;
  transcript: string;
};

export type AivmStyle = {
  name: string;
  /** data URL（スキーマ上 optional。stripAssets 時は常に省略）。 */
  icon?: string;
  /** スタイル local_id（AIVM 1.0 では 0..31）。 */
  localId: number;
  /** stripAssets 時は空配列。 */
  voiceSamples: AivmVoiceSample[];
};

export type AivmSpeaker = {
  name: string;
  /** data URL（AIVM 1.0 では必須だが stripAssets 時は省略）。 */
  icon?: string;
  /** BCP 47 言語タグ（例 ["ja"]）。 */
  supportedLanguages: string[];
  /** 話者 UUID（AivisSpeech の VOICEVOX 互換 style_id 生成の種）。 */
  uuid: string;
  /** 話者 local_id（sid）。 */
  localId: number;
  styles: AivmStyle[];
};

export type AivmManifest = {
  /** AIVM マニフェスト仕様バージョン（現行 "1.0"）。 */
  manifestVersion: string;
  name: string;
  description?: string;
  creators?: string[];
  /** ライセンス全文（数 KB のプレーン文字列。data URL ではないので strip 対象外）。 */
  license?: string;
  /** "Style-Bert-VITS2" | "Style-Bert-VITS2 (JP-Extra)"。 */
  modelArchitecture: string;
  /** "ONNX"（.aivmx） | "Safetensors"（.aivm）。 */
  modelFormat: string;
  trainingEpochs?: number;
  trainingSteps?: number;
  uuid: string;
  version: string;
  speakers: AivmSpeaker[];
};

export type ReadAivmxManifestOptions = {
  /**
   * true（既定）で icon / voice_samples の data URL を落とす。実物マニフェストの
   * サイズの大半はこれら（話者アイコン ~150K chars、サンプル音声 ~500K chars ×件数）で、
   * カタログ用途（名前・uuid・local_id）には不要。
   */
  stripAssets?: boolean;
};

const parseVoiceSample = (value: unknown, path: string): AivmVoiceSample => {
  const record = expectRecord(value, path);
  return {
    audio: expectString(record.audio, `${path}.audio`),
    transcript: expectString(record.transcript, `${path}.transcript`),
  };
};

const parseStyle = (
  value: unknown,
  path: string,
  strip: boolean,
): AivmStyle => {
  const record = expectRecord(value, path);
  return {
    name: expectString(record.name, `${path}.name`),
    ...(strip ? {} : { icon: optString(record.icon, `${path}.icon`) }),
    localId: expectFiniteNumber(record.local_id, `${path}.local_id`),
    voiceSamples: strip ? [] : expectArray(
      record.voice_samples ?? [],
      `${path}.voice_samples`,
    ).map((v, i) => parseVoiceSample(v, `${path}.voice_samples[${i}]`)),
  };
};

const parseSpeaker = (
  value: unknown,
  path: string,
  strip: boolean,
): AivmSpeaker => {
  const record = expectRecord(value, path);
  const styles = expectArray(record.styles, `${path}.styles`);
  if (styles.length === 0) {
    throw new Error(`${path}.styles が空（AIVM 1.0 は 1 スタイル以上必須）`);
  }
  return {
    name: expectString(record.name, `${path}.name`),
    ...(strip ? {} : { icon: optString(record.icon, `${path}.icon`) }),
    supportedLanguages: optStringArray(
      record.supported_languages,
      `${path}.supported_languages`,
    ) ?? [],
    uuid: expectString(record.uuid, `${path}.uuid`),
    localId: expectFiniteNumber(record.local_id, `${path}.local_id`),
    styles: styles.map((v, i) => parseStyle(v, `${path}.styles[${i}]`, strip)),
  };
};

/** aivm_manifest の unknown JSON を型付き AivmManifest へ検証する（fail loud）。 */
export const parseAivmManifest = (
  json: unknown,
  opts: ReadAivmxManifestOptions = {},
): AivmManifest => {
  const strip = opts.stripAssets ?? true;
  const root = expectRecord(json, "manifest");
  const speakers = expectArray(root.speakers, "manifest.speakers");
  if (speakers.length === 0) {
    throw new Error("manifest.speakers が空（AIVM 1.0 は 1 話者以上必須）");
  }
  return {
    manifestVersion: expectString(
      root.manifest_version,
      "manifest.manifest_version",
    ),
    name: expectString(root.name, "manifest.name"),
    description: optString(root.description, "manifest.description"),
    creators: optStringArray(root.creators, "manifest.creators"),
    license: optString(root.license, "manifest.license"),
    modelArchitecture: expectString(
      root.model_architecture,
      "manifest.model_architecture",
    ),
    modelFormat: expectString(root.model_format, "manifest.model_format"),
    trainingEpochs: optFiniteNumber(
      root.training_epochs,
      "manifest.training_epochs",
    ),
    trainingSteps: optFiniteNumber(
      root.training_steps,
      "manifest.training_steps",
    ),
    uuid: expectString(root.uuid, "manifest.uuid"),
    version: expectString(root.version, "manifest.version"),
    speakers: speakers.map((v, i) =>
      parseSpeaker(v, `manifest.speakers[${i}]`, strip)
    ),
  };
};

/**
 * aivmx バイト列から AIVM マニフェストを読む（既定は stripAssets: true）。
 * aivm_manifest キーが無い・JSON が壊れている・必須フィールド欠落は throw（fail loud）。
 */
export const readAivmxManifest = (
  onnxBytes: Uint8Array,
  opts: ReadAivmxManifestOptions = {},
): AivmManifest => {
  const text = extractMetadataValue(onnxBytes, "aivm_manifest");
  if (text === undefined) {
    throw new Error(
      "aivm_manifest: metadata_props に 'aivm_manifest' が無い（aivmx でない/破損を疑う）",
    );
  }
  return parseAivmManifest(parseMetadataJson(text, "aivm_manifest"), opts);
};
