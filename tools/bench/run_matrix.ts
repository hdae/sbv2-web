type Combo = { label: string; acoustic: string; deberta: string };

const combos: Combo[] = [
  {
    label: "aivmx-fp32_deberta-fp16",
    acoustic: "data/models/aivmx/mao.aivmx",
    deberta: "data/models/deberta-fp16/model_fp16.onnx",
  },
  {
    label: "aivmx-fp32_deberta-int8-qlinear",
    acoustic: "data/models/aivmx/mao.aivmx",
    deberta: "data/models/deberta-int8/model_int8.onnx",
  },
  {
    label: "aivmx-fp32_deberta-int8-nbits",
    acoustic: "data/models/aivmx/mao.aivmx",
    deberta: "data/models/deberta-int8-rtn/model_int8.onnx",
  },
  {
    label: "aivmx-fp32_deberta-int4-rtn",
    acoustic: "data/models/aivmx/mao.aivmx",
    deberta: "data/models/deberta-int4-rtn/model_int4.onnx",
  },
  {
    label: "aivmx-fp32_deberta-int4-hqq",
    acoustic: "data/models/aivmx/mao.aivmx",
    deberta: "data/models/deberta-int4-hqq/model_int4.onnx",
  },
  {
    label: "aivmx-int8-static_deberta-int4-rtn",
    acoustic: "data/models/aivmx-int8-static/mao_int8_static.onnx",
    deberta: "data/models/deberta-int4-rtn/model_int4.onnx",
  },
];

await Deno.mkdir("data/bench", { recursive: true });
const decoder = new TextDecoder();
const results = [];
for (const combo of combos) {
  const outPath = "data/bench/" + combo.label + ".json";
  const wavPath = "data/bench/" + combo.label + ".wav";
  const cmd = new Deno.Command(Deno.execPath(), {
    args: [
      "run",
      "--allow-all",
      "tools/bench/smoke_one.ts",
      "--label",
      combo.label,
      "--acoustic",
      combo.acoustic,
      "--deberta",
      combo.deberta,
      "--out-wav",
      wavPath,
    ],
    stdout: "piped",
    stderr: "piped",
  });
  const output = await cmd.output();
  const stdout = decoder.decode(output.stdout);
  const stderr = decoder.decode(output.stderr);
  if (!output.success) {
    const failed = {
      ...combo,
      status: "failed",
      code: output.code,
      stderr,
      stdout,
    };
    results.push(failed);
    await Deno.writeTextFile(outPath, JSON.stringify(failed, null, 2));
    continue;
  }
  const parsed = JSON.parse(stdout);
  const ok = { ...combo, status: "ok", ...parsed, stderr };
  results.push(ok);
  await Deno.writeTextFile(outPath, JSON.stringify(parsed, null, 2));
}
await Deno.writeTextFile(
  "data/bench/summary.json",
  JSON.stringify(results, null, 2),
);
console.log(JSON.stringify(results, null, 2));
