// 最小 .npy パーサの behavior テスト。
//
// 正常系（v1/v2 header, C-contiguous <f4, 非ゼロ byteOffset の subarray）と、
// fail-loudly な拒否（fortran_order / 非 <f4 dtype / 非 2 次元 / データ長不足 / 不正 magic）を固定する。

import { assertEquals, assertThrows } from "@std/assert";
import { parseNpy2dFloat32 } from "./npy.ts";

/** テスト用に最小 .npy バイト列を組む（header は改行終端・整列パディングなし）。 */
const buildNpy = (
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

Deno.test("parseNpy2dFloat32: v1 の C-contiguous <f4 を [rows, cols] で読む", () => {
  const data = Float32Array.from([1, 2, 3, 4, 5, 6]);
  const { rows, cols, data: got } = parseNpy2dFloat32(
    buildNpy("<f4", false, [2, 3], data),
  );
  assertEquals([rows, cols], [2, 3]);
  assertEquals(Array.from(got), [1, 2, 3, 4, 5, 6]);
});

Deno.test("parseNpy2dFloat32: v2 header(4 バイト長)を読む", () => {
  const data = Float32Array.from([7, 8]);
  const { rows, cols, data: got } = parseNpy2dFloat32(
    buildNpy("<f4", false, [1, 2], data, 2),
  );
  assertEquals([rows, cols], [1, 2]);
  assertEquals(Array.from(got), [7, 8]);
});

Deno.test("parseNpy2dFloat32: 非ゼロ byteOffset の subarray でも正しく読む", () => {
  const npy = buildNpy("<f4", false, [1, 2], Float32Array.from([9, 10]));
  const padded = new Uint8Array(npy.length + 8);
  padded.set(npy, 8);
  const { data } = parseNpy2dFloat32(padded.subarray(8));
  assertEquals(Array.from(data), [9, 10]);
});

Deno.test("parseNpy2dFloat32: fortran_order=True は throw", () => {
  assertThrows(
    () => parseNpy2dFloat32(buildNpy("<f4", true, [2, 2], new Float32Array(4))),
    Error,
    "fortran_order",
  );
});

Deno.test("parseNpy2dFloat32: <f4 以外の dtype は throw", () => {
  assertThrows(
    () =>
      parseNpy2dFloat32(buildNpy(">f4", false, [1, 1], new Float32Array(1))),
    Error,
    "dtype",
  );
  assertThrows(
    () =>
      parseNpy2dFloat32(buildNpy("<f8", false, [1, 1], new Float32Array(1))),
    Error,
    "dtype",
  );
});

Deno.test("parseNpy2dFloat32: 2 次元でない shape は throw", () => {
  assertThrows(
    () =>
      parseNpy2dFloat32(
        buildNpy("<f4", false, [3], Float32Array.from([1, 2, 3])),
      ),
    Error,
    "2 次元",
  );
  assertThrows(
    () =>
      parseNpy2dFloat32(
        buildNpy("<f4", false, [1, 1, 1], Float32Array.from([1])),
      ),
    Error,
    "2 次元",
  );
});

Deno.test("parseNpy2dFloat32: データ長不足は throw", () => {
  const npy = buildNpy(
    "<f4",
    false,
    [2, 3],
    Float32Array.from([1, 2, 3, 4, 5, 6]),
  );
  assertThrows(
    () => parseNpy2dFloat32(npy.subarray(0, npy.length - 4)),
    Error,
    "データ長不足",
  );
});

Deno.test("parseNpy2dFloat32: 不正 magic は throw", () => {
  assertThrows(
    () => parseNpy2dFloat32(new Uint8Array([1, 2, 3, 4, 5, 6, 1, 0])),
    Error,
    "マジック",
  );
});
