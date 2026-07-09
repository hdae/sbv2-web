const specs = [
  "npm:onnxruntime-web@1.27.0",
  "npm:onnxruntime-web@1.27.0/all",
  "npm:onnxruntime-web@1.27.0/wasm",
  "npm:onnxruntime-web@1.27.0/webgpu",
];

const quantizationApiPattern = /quant|calibr|shape|pre.?process/i;
const result = [];
for (const spec of specs) {
  const mod = await import(spec);
  const exports = Object.keys(mod).sort();
  result.push({
    spec,
    exports,
    quantizationLikeExports: exports.filter((name) =>
      quantizationApiPattern.test(name)
    ),
  });
}

console.log(JSON.stringify(
  {
    note:
      "onnxruntime-web can execute quantized ONNX ops, but these public exports do not expose static quantization, calibration, or model-rewrite APIs.",
    result,
  },
  null,
  2,
));
