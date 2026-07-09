// 最小 .npy パーサの behavior テスト。
//
// 正常系（v1/v2 header, C-contiguous <f4, 非ゼロ byteOffset の subarray）と、
// fail-loudly な拒否（fortran_order / 非 <f4 dtype / 非 2 次元 / データ長不足 / 不正 magic）を固定する。

import { assertEquals, assertThrows } from "@std/assert";
import { parseNpy2dFloat32 } from "./npy.ts";
import { buildNpy } from "../testing/npy_fixture.ts";

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
