// @hdae/sbv2-web core: runtime-agnostic pieces shared by the web (`.`) and node
// (`./node`) entries. This barrel pulls in NO ONNX Runtime; each public entry
// re-exports it together with its own backend adapter.

export const VERSION = "0.3.0";

export {
  type CleanRanges,
  type DebertaSpecialTokens,
  DebertaTokenizer,
} from "./text/deberta_tokenizer.ts";
export { buildBaseWord2ph, distributePhone } from "./text/word2ph.ts";
export {
  JP_LANGUAGE_ID,
  JP_TONE_OFFSET,
  phonesToIds,
  SYMBOL_TO_ID,
  SYMBOLS,
} from "./text/symbols.ts";
export { type Sbv2PhoneTone, toSbv2PhoneTone } from "./text/phone_tone.ts";
export { toBertText } from "./text/bert_text.ts";
export {
  buildDebertaTokenizer,
  DEBERTA_REPO,
  DEBERTA_REVISION,
  type DebertaAssets,
  type DebertaProgress,
  getDeberta,
  type GetDebertaOptions,
} from "./assets/deberta.ts";
export { type Npy2dFloat32, parseNpy2dFloat32 } from "./runtime/npy.ts";
export {
  type AivmxMetadata,
  base64ToBytes,
  extractMetadataValue,
  extractStyleVectorsNpy,
  readAivmxMetadata,
  readSbv2HyperParameters,
  type Sbv2HyperParameters,
} from "./runtime/aivmx_meta.ts";
export {
  type AivmManifest,
  type AivmSpeaker,
  type AivmStyle,
  type AivmVoiceSample,
  parseAivmManifest,
  readAivmxManifest,
  type ReadAivmxManifestOptions,
} from "./runtime/aivm_manifest.ts";
export {
  addBlankWord2ph,
  BERT_DIM,
  intersperse,
  type ModelIdSequences,
  parseStyleMatrix,
  phonesTonesToModelIds,
  styleVector,
  tileBertToPhoneLevel,
} from "./runtime/tensor_build.ts";
export {
  type AcousticFeeds,
  DEFAULT_SCALARS,
  mergeScalars,
  type ModelAdapter,
  type SynthInput,
  type SynthScalars,
} from "./runtime/adapter_types.ts";
export { validateSynthInput } from "./runtime/validate_input.ts";
export { concatWithSilence, padSilence } from "./runtime/silence.ts";
export {
  type OrtBackend,
  type OrtSessionOptions,
  Sbv2Adapter,
} from "./runtime/adapter_core.ts";
export {
  type BertSource,
  DebertaExtractor,
} from "./runtime/deberta_extractor.ts";
export {
  type SynthesizeOptions,
  synthesizeText,
} from "./runtime/synthesize.ts";
export { encodeWav } from "./runtime/wav.ts";
