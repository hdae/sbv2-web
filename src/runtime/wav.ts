// float32 [-1,1] 波形 → 16bit PCM WAV バイト列（モノラル）。
//
// examples/cli/synthesize_node.ts（Deno 実駆動）と examples/browser のブラウザ版サンプルで共有する。
// onnxruntime に依存しない純ロジックなので mod.ts から export してよい。

/**
 * float32 [-1,1] の波形を 16bit PCM WAV（モノラル）にエンコードする。
 *
 * @param audio 波形サンプル（範囲外は [-1,1] にクランプ）。
 * @param sampleRate サンプルレート（Hz）。
 * @returns RIFF/WAVE ヘッダ 44 バイト + PCM データの Uint8Array。
 */
export const encodeWav = (
  audio: Float32Array,
  sampleRate: number,
): Uint8Array<ArrayBuffer> => {
  const numSamples = audio.length;
  const bytesPerSample = 2;
  const dataBytes = numSamples * bytesPerSample;
  const buf = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buf);
  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) {
      view.setUint8(offset + i, s.charCodeAt(i));
    }
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // fmt チャンクサイズ
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // モノラル
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true); // byte rate
  view.setUint16(32, bytesPerSample, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, "data");
  view.setUint32(40, dataBytes, true);
  for (let i = 0; i < numSamples; i++) {
    const clamped = Math.max(-1, Math.min(1, audio[i]));
    view.setInt16(44 + i * bytesPerSample, Math.round(clamped * 32767), true);
  }
  return new Uint8Array(buf);
};
