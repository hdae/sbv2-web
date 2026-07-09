// 最小 .npy パーサ（NumPy array format v1.0/v2.0）。
//
// 用途: aivmx の aivm_style_vectors（Base64 → .npy）を onnxruntime の外でパースする。
// スタイルベクトルは `[num_styles, 256]` の float32 little-endian・C-contiguous 行列で、
// これだけを対象に fail loudly で読む（汎用 npy ローダではない）。
//
// フォーマット出典（NumPy 公式仕様。実測: mao.aivmx の style_vectors ヘッダ）:
//   magic "\x93NUMPY"(6) + version(major,minor 各1) + header_len + ASCII header dict
//   v1.0: header_len は 2 バイト LE u16 / v2.0: 4 バイト LE u32
//   header 例: "{'descr': '<f4', 'fortran_order': False, 'shape': (6, 256), }"

/** パース結果: 行優先の float32 行列（rows × cols）。 */
export type Npy2dFloat32 = {
  rows: number;
  cols: number;
  /** 長さ rows*cols の行優先 (C-contiguous) データ。 */
  data: Float32Array;
};

const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // "\x93NUMPY"

/**
 * .npy バイト列を `[rows, cols]` の float32 行列としてパースする。
 * float32 little-endian・C-contiguous・2 次元のみ対応。それ以外は throw（fail loudly）。
 */
export const parseNpy2dFloat32 = (bytes: Uint8Array): Npy2dFloat32 => {
  for (let i = 0; i < MAGIC.length; i++) {
    if (bytes[i] !== MAGIC[i]) {
      throw new Error("npy: マジック '\\x93NUMPY' が一致しない（.npy でない）");
    }
  }
  const major = bytes[6];
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let headerLen: number;
  let headerStart: number;
  if (major === 1) {
    headerLen = view.getUint16(8, true);
    headerStart = 10;
  } else if (major === 2 || major === 3) {
    headerLen = view.getUint32(8, true);
    headerStart = 12;
  } else {
    throw new Error(`npy: 未対応バージョン major=${major}`);
  }
  const headerBytes = bytes.subarray(headerStart, headerStart + headerLen);
  const header = new TextDecoder("latin1").decode(headerBytes);

  const descrMatch = header.match(/'descr'\s*:\s*'([^']+)'/);
  const fortranMatch = header.match(/'fortran_order'\s*:\s*(True|False)/);
  const shapeMatch = header.match(/'shape'\s*:\s*\(([^)]*)\)/);
  if (!descrMatch || !fortranMatch || !shapeMatch) {
    throw new Error(`npy: ヘッダを解釈できない: ${JSON.stringify(header)}`);
  }
  const descr = descrMatch[1];
  // '<f4'（LE float32）のみ対応。'|f4' 等は無いが念のため endianness を検査。
  if (descr !== "<f4") {
    throw new Error(
      `npy: dtype '${descr}' は未対応（'<f4' little-endian float32 のみ）`,
    );
  }
  if (fortranMatch[1] === "True") {
    throw new Error("npy: fortran_order=True は未対応（C-contiguous のみ）");
  }
  const dims = shapeMatch[1]
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => Number.parseInt(s, 10));
  if (dims.length !== 2 || dims.some((d) => !Number.isInteger(d) || d < 0)) {
    throw new Error(
      `npy: shape ${
        JSON.stringify(dims)
      } は 2 次元でない（[rows, cols] を期待）`,
    );
  }
  const [rows, cols] = dims;

  const dataStart = headerStart + headerLen;
  const expectedBytes = rows * cols * 4;
  const available = bytes.byteLength - dataStart;
  if (available < expectedBytes) {
    throw new Error(
      `npy: データ長不足 available=${available} < expected=${expectedBytes}（shape=${rows}x${cols}）`,
    );
  }
  // little-endian float32 を読み出す。ブラウザ/Deno はほぼ LE だが DataView で明示。
  const data = new Float32Array(rows * cols);
  const dataView = new DataView(
    bytes.buffer,
    bytes.byteOffset + dataStart,
    expectedBytes,
  );
  for (let i = 0; i < data.length; i++) {
    data[i] = dataView.getFloat32(i * 4, true);
  }
  return { rows, cols, data };
};
