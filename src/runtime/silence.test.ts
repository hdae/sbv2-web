// 無音パディング・無音連結の behavior テスト。

import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { concatWithSilence, padSilence } from "./silence.ts";

Deno.test("padSilence: 前後に sampleRate*秒 のゼロを足す", () => {
  const wave = Float32Array.from([0.5, -0.5]);
  const out = padSilence(wave, 10, { preSec: 0.3, postSec: 0.5 });
  assertEquals(out.length, 3 + 2 + 5);
  assertEquals(Array.from(out.slice(0, 3)), [0, 0, 0]);
  assertEquals(Array.from(out.slice(3, 5)), [0.5, -0.5]);
  assertEquals(Array.from(out.slice(5)), [0, 0, 0, 0, 0]);
});

Deno.test("padSilence: 両方 0 ならコピーせず同一配列を返す", () => {
  const wave = Float32Array.from([1]);
  assertStrictEquals(padSilence(wave, 44100, {}), wave);
});

Deno.test("padSilence: 負・非有限の秒数や不正 sampleRate は throw", () => {
  const wave = new Float32Array(1);
  assertThrows(() => padSilence(wave, 44100, { preSec: -1 }), Error, "preSec");
  assertThrows(
    () => padSilence(wave, 44100, { postSec: Number.NaN }),
    Error,
    "postSec",
  );
  assertThrows(() => padSilence(wave, 0, { preSec: 1 }), Error, "sampleRate");
});

Deno.test("concatWithSilence: 波形間にゼロ無音を挟んで連結する", () => {
  const out = concatWithSilence(
    [Float32Array.from([1, 2]), Float32Array.from([3]), Float32Array.from([4])],
    10,
    0.2,
  );
  assertEquals(Array.from(out), [1, 2, 0, 0, 3, 0, 0, 4]);
});

Deno.test("concatWithSilence: 空入力は空、単一要素は無音なし", () => {
  assertEquals(concatWithSilence([], 44100, 0.5).length, 0);
  assertEquals(
    Array.from(concatWithSilence([Float32Array.from([7])], 44100, 0.5)),
    [7],
  );
});
