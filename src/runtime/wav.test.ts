// 16bit PCM WAV エンコーダの behavior テスト。
//
// RIFF/WAVE ヘッダの各フィールド、[-1,1] クランプと 32767 スケール・丸め、空波形の 44 バイトを固定する。

import { assertEquals } from "@std/assert";
import { encodeWav } from "./wav.ts";

const ascii = (bytes: Uint8Array, offset: number, length: number): string =>
  String.fromCharCode(...bytes.slice(offset, offset + length));

Deno.test("encodeWav: ヘッダ 44 バイト + サンプル 2 バイト", () => {
  const wav = encodeWav(Float32Array.from([0, 0, 0]), 44100);
  assertEquals(wav.length, 44 + 3 * 2);
});

Deno.test("encodeWav: RIFF/WAVE ヘッダの各フィールド", () => {
  const wav = encodeWav(Float32Array.from([0, 0, 0]), 44100);
  const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
  const dataBytes = 3 * 2;
  assertEquals(ascii(wav, 0, 4), "RIFF");
  assertEquals(view.getUint32(4, true), 36 + dataBytes);
  assertEquals(ascii(wav, 8, 4), "WAVE");
  assertEquals(ascii(wav, 12, 4), "fmt ");
  assertEquals(view.getUint32(16, true), 16); // fmt chunk size
  assertEquals(view.getUint16(20, true), 1); // PCM
  assertEquals(view.getUint16(22, true), 1); // mono
  assertEquals(view.getUint32(24, true), 44100); // sample rate
  assertEquals(view.getUint32(28, true), 44100 * 2); // byte rate
  assertEquals(view.getUint16(32, true), 2); // block align
  assertEquals(view.getUint16(34, true), 16); // bits per sample
  assertEquals(ascii(wav, 36, 4), "data");
  assertEquals(view.getUint32(40, true), dataBytes);
});

Deno.test("encodeWav: [-1,1] にクランプし 32767 スケールする", () => {
  const wav = encodeWav(Float32Array.from([2, -2, 1, -1, 0]), 8000);
  const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
  const at = (i: number) => view.getInt16(44 + i * 2, true);
  assertEquals(at(0), 32767); // 2 -> clamp 1 -> 32767
  assertEquals(at(1), -32767); // -2 -> clamp -1 -> -32767
  assertEquals(at(2), 32767);
  assertEquals(at(3), -32767);
  assertEquals(at(4), 0);
});

Deno.test("encodeWav: 0.5 は round(16383.5)=16384", () => {
  const wav = encodeWav(Float32Array.from([0.5]), 8000);
  const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
  assertEquals(view.getInt16(44, true), 16384);
});

Deno.test("encodeWav: 空波形は 44 バイト・data サイズ 0", () => {
  const wav = encodeWav(new Float32Array(0), 44100);
  const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
  assertEquals(wav.length, 44);
  assertEquals(view.getUint32(40, true), 0);
});
