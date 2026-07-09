// 量子化 DeBERTa 一式（model.onnx + トークナイザ資産）の HuggingFace 自動取得。
//
// 資産 4 ファイル（model.onnx / vocab.txt / clean_ranges.json / meta.json）は同一リポの
// 同一リビジョンからワンセットで取得する＝トークナイザとモデルの drift を構造的に防ぐ。
// 既定は焼き込みコミット SHA（不変・再現可能。辞書側 @hdae/yomi の DICT_REVISION と同パターン）。
// 取得・キャッシュ・self-heal は @hdae/fetch-cache（Cache API、SHA 固定 URL キー）に委譲し、
// 既定リビジョンに対してのみサイズ / SHA-256 の事前ピンで整合性検証する
// （別リビジョン指定時はピンが合わないため HTTP 整合のみ）。

import { fetchHfFiles, resolveHfRevision } from "@hdae/fetch-cache/hf";
import {
  type CleanRanges,
  type DebertaSpecialTokens,
  DebertaTokenizer,
} from "../text/deberta_tokenizer.ts";
import {
  expectArray,
  expectFiniteNumber,
  expectRecord,
} from "../runtime/json_expect.ts";

/** 既定の量子化 DeBERTa リポジトリ（int4 RTN block=256。docs/benchmark.md 参照）。 */
export const DEBERTA_REPO =
  "hdae/deberta-v2-large-japanese-char-wwm-onnx-int4-rtn-b256";

/**
 * 既定リビジョン（上記リポのコミット SHA）。不変・再現可能にするための焼き込みで、
 * モデルを差し替えたら HF へ上げ直してこの SHA と PINNED_FILES を更新する。
 */
export const DEBERTA_REVISION = "3c6921bf67ee5f64a285f49df8636c1036b81881";

const DEFAULT_CACHE_NAME = "sbv2-web-deberta";

/** 既定リビジョンの実測ピン（サイズは HF tree API、model の sha256 は LFS メタ）。 */
const PINNED_FILES = {
  model: {
    path: "model.onnx",
    expectedBytes: 240576205,
    sha256: "79002e00ea11bb41be2cd0d77e2a2608930cd8b502937101da31739fc6dabf96",
  },
  vocab: { path: "vocab.txt", expectedBytes: 88151 },
  cleanRanges: { path: "clean_ranges.json", expectedBytes: 585 },
  meta: { path: "meta.json", expectedBytes: 662 },
} as const;

/** 別リビジョン指定時（ピンが適用できない）に取得する同名ファイル群。 */
const UNPINNED_FILES = {
  model: "model.onnx",
  vocab: "vocab.txt",
  cleanRanges: "clean_ranges.json",
  meta: "meta.json",
} as const;

export type DebertaProgress = { path: string; loaded: number; total?: number };

export type GetDebertaOptions = {
  /** HF リビジョン（コミット SHA / ブランチ / タグ）。既定 = 焼き込み DEBERTA_REVISION。 */
  revision?: string;
  /** Cache Storage の名前空間。既定 "sbv2-web-deberta"。 */
  cacheName?: string;
  /** ファイル毎のダウンロード進捗（実質 model.onnx 240MB 用。キャッシュヒット時は来ない）。 */
  onProgress?: (progress: DebertaProgress) => void;
};

export type DebertaAssets = {
  /** 取得した vocab / clean_ranges / meta から構築済みのトークナイザ。 */
  tokenizer: DebertaTokenizer;
  /** 量子化 DeBERTa の ONNX バイト列（Sbv2*ModelAdapter の bertOnnxBytes へ）。 */
  bertOnnxBytes: Uint8Array;
  /** 実際に取得した解決済みコミット SHA。 */
  revision: string;
};

const parseJson = (text: string, file: string): unknown => {
  try {
    return JSON.parse(text) as unknown;
  } catch (cause) {
    throw new Error(`getDeberta: ${file} が JSON として読めない`, { cause });
  }
};

/** clean_ranges.json（[start, end] ペア配列 ×2）を fail loud に検証する。 */
const parseCleanRanges = (text: string): CleanRanges => {
  const record = expectRecord(
    parseJson(text, "clean_ranges.json"),
    "clean_ranges",
  );
  const pairs = (value: unknown, path: string): [number, number][] =>
    expectArray(value, path).map((entry, i) => {
      const pair = expectArray(entry, `${path}[${i}]`);
      if (pair.length !== 2) {
        throw new Error(`${path}[${i}] が [start, end] でない`);
      }
      return [
        expectFiniteNumber(pair[0], `${path}[${i}][0]`),
        expectFiniteNumber(pair[1], `${path}[${i}][1]`),
      ];
    });
  return {
    removed: pairs(record.removed, "clean_ranges.removed"),
    spaced: pairs(record.spaced, "clean_ranges.spaced"),
  };
};

/** meta.json から特殊トークン id を fail loud に読む。 */
const parseSpecialTokens = (text: string): DebertaSpecialTokens => {
  const meta = expectRecord(parseJson(text, "meta.json"), "meta");
  const tokens = expectRecord(meta.special_tokens, "meta.special_tokens");
  const id = (key: "cls" | "sep" | "unk"): number =>
    expectFiniteNumber(
      expectRecord(tokens[key], `meta.special_tokens.${key}`).id,
      `meta.special_tokens.${key}.id`,
    );
  return { clsId: id("cls"), sepId: id("sep"), unkId: id("unk") };
};

/**
 * トークナイザ資産のテキスト 3 点から DebertaTokenizer を構築する。
 * 自動取得（getDeberta）と手動資産（ブラウザ例の Advanced / CLI の --tokenizer）が
 * 同一経路を通ることで、構築挙動の分岐を作らない。
 */
export const buildDebertaTokenizer = (
  vocabText: string,
  cleanRangesText: string,
  metaText: string,
): DebertaTokenizer =>
  DebertaTokenizer.fromVocabText(
    vocabText,
    parseCleanRanges(cleanRangesText),
    parseSpecialTokens(metaText),
  );

/**
 * 量子化 DeBERTa 一式を HuggingFace から取得して、構築済みトークナイザと ONNX バイト列を返す。
 *
 * 既定は焼き込み SHA 固定（Cache API にヒットすれば network なし）。`revision: "main"` 等の
 * 可変 ref を渡すと現在の SHA へ解決してから取得するので、変わっていなければキャッシュから返る。
 * 4 ファイルは並列取得され、どれかが検証に落ちれば全体が throw する（fail loud）。
 *
 * NOTE: Cache API の無いランタイム（Node.js 等）では取得のみ行い、キャッシュはスキップされる。
 */
export const getDeberta = async (
  opts: GetDebertaOptions = {},
): Promise<DebertaAssets> => {
  const revision = await resolveHfRevision({
    repo: DEBERTA_REPO,
    revision: opts.revision ?? DEBERTA_REVISION,
  });
  // サイズ / sha256 の事前ピンは焼き込みリビジョンにのみ適用できる（別リビジョンは値が変わりうる）。
  const files = revision === DEBERTA_REVISION ? PINNED_FILES : UNPINNED_FILES;
  const assets = await fetchHfFiles({ repo: DEBERTA_REPO, revision }, files, {
    cacheName: opts.cacheName ?? DEFAULT_CACHE_NAME,
    onProgress: opts.onProgress,
  });
  const decoder = new TextDecoder();
  return {
    tokenizer: buildDebertaTokenizer(
      decoder.decode(assets.vocab),
      decoder.decode(assets.cleanRanges),
      decoder.decode(assets.meta),
    ),
    bertOnnxBytes: assets.model,
    revision,
  };
};
