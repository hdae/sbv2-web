// unknown な JSON（aivmx メタデータ境界）を fail loud に検証する小さなヘルパ群。
//
// aivm_manifest / hyper_parameters は外部生成物（aivmlib）なので unknown で受け、
// 必要なフィールドだけを JSON パス付きエラーで型付けする。Zod を使わないのは、
// 実物 manifest が数 MB（base64 data URL が支配的）でスキーマ全検証のコストが
// 割に合わず、依存最小の方針とも揃えるため。
//
// NOTE: opt 系は null も undefined 扱いにする（aivmlib の Pydantic は optional の
// 既定 None を JSON null で出力しうるため）。

export const expectRecord = (
  value: unknown,
  path: string,
): Record<string, unknown> => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${path} がオブジェクトでない`);
  }
  return value as Record<string, unknown>;
};

export const optRecord = (
  value: unknown,
  path: string,
): Record<string, unknown> | undefined =>
  value === undefined || value === null ? undefined : expectRecord(value, path);

export const expectString = (value: unknown, path: string): string => {
  if (typeof value !== "string") throw new Error(`${path} が文字列でない`);
  return value;
};

export const optString = (value: unknown, path: string): string | undefined =>
  value === undefined || value === null ? undefined : expectString(value, path);

export const expectFiniteNumber = (value: unknown, path: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${path} が有限数でない`);
  }
  return value;
};

export const optFiniteNumber = (
  value: unknown,
  path: string,
): number | undefined =>
  value === undefined || value === null
    ? undefined
    : expectFiniteNumber(value, path);

export const expectArray = (value: unknown, path: string): unknown[] => {
  if (!Array.isArray(value)) throw new Error(`${path} が配列でない`);
  return value;
};

export const optStringArray = (
  value: unknown,
  path: string,
): string[] | undefined =>
  value === undefined || value === null
    ? undefined
    : expectArray(value, path).map((v, i) => expectString(v, `${path}[${i}]`));

/** {名前: 数値} マップ（spk2id / style2id）。 */
export const optNumberRecord = (
  value: unknown,
  path: string,
): Readonly<Record<string, number>> | undefined => {
  if (value === undefined || value === null) return undefined;
  const record = expectRecord(value, path);
  const out: Record<string, number> = {};
  for (const [key, v] of Object.entries(record)) {
    out[key] = expectFiniteNumber(v, `${path}.${key}`);
  }
  return out;
};
