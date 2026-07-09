// テスト専用の .npy フィクスチャビルダ。
//
// parseNpy2dFloat32 / parseStyleMatrix のテストが共有する（配布物には含めない:
// deno.json publish.exclude の src/testing/**）。エンコードを一箇所に持つことで、
// npy テストとアダプタテストのフィクスチャが黙って食い違うのを防ぐ。

/** テスト用に最小 .npy バイト列を組む（header は改行終端・整列パディングなし）。 */
export const buildNpy = (
  descr: string,
  fortran: boolean,
  shape: readonly number[],
  data: Float32Array,
  version = 1,
): Uint8Array => {
  const shapeStr = shape.length === 1
    ? `(${shape[0]},)`
    : `(${shape.join(", ")})`;
  const header =
    `{'descr': '${descr}', 'fortran_order': ${fortran ? "True" : "False"}, ` +
    `'shape': ${shapeStr}, }\n`;
  const headerBytes = new TextEncoder().encode(header);
  const preamble = version === 1 ? 10 : 12;
  const out = new Uint8Array(preamble + headerBytes.length + data.byteLength);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59], 0);
  out[6] = version;
  out[7] = 0;
  const view = new DataView(out.buffer);
  if (version === 1) view.setUint16(8, headerBytes.length, true);
  else view.setUint32(8, headerBytes.length, true);
  out.set(headerBytes, preamble);
  const dataView = new DataView(out.buffer, preamble + headerBytes.length);
  for (let i = 0; i < data.length; i++) {
    dataView.setFloat32(i * 4, data[i], true);
  }
  return out;
};

/** style_vectors 用の [rows, 256] C-contiguous <f4 npy を組む（値は 0 埋め or 指定）。 */
export const buildStyleNpy = (
  rows: number,
  data?: Float32Array,
): Uint8Array =>
  buildNpy("<f4", false, [rows, 256], data ?? new Float32Array(rows * 256));
