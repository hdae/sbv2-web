// version 焼き込みの drift 検出（dev/CI）: 公開コア（src/core.ts）が export する VERSION が
// deno.json の version と一致するか。src/core.ts は ORT 非依存バレルで、`.`（mod.ts）と
// `./node`（node/mod.ts）がこれを re-export する。deno task bump が deno.json と core.ts を
// 同時更新するが、手動編集による drift をここで fail-loud にする（公開 VERSION が実バージョンと
// ズレるのを防ぐ）。core.ts 経由なので onnxruntime を巻き込まない軽い経路で検証できる。
import { VERSION } from "../src/core.ts";
import { readVersion } from "./config_version.ts";

Deno.test("version 焼き込み: 公開 VERSION == deno.json.version", async () => {
  const declared = await readVersion("./deno.json");
  if (VERSION !== declared) {
    throw new Error(
      `公開 VERSION(${VERSION}) が deno.json の version(${declared}) と不一致。` +
        `src/core.ts を単一の真実源に保ち、deno task bump で同期すること。`,
    );
  }
});
