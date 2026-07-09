from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "model-tools" / "scripts"))

from quantize_aivmx import quantize  # noqa: E402


def status_kb(key: str) -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1])
    except FileNotFoundError:
        return None
    return None


class Sampler:
    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.samples: list[dict[str, int | float | None]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.t0 = time.perf_counter()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(
                {
                    "tSec": round(time.perf_counter() - self.t0, 3),
                    "rssKb": status_kb("VmRSS"),
                    "hwmKb": status_kb("VmHWM"),
                }
            )
            self._stop.wait(self.interval)

    def __enter__(self) -> "Sampler":
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self.samples.append(
            {
                "tSec": round(time.perf_counter() - self.t0, 3),
                "rssKb": status_kb("VmRSS"),
                "hwmKb": status_kb("VmHWM"),
            }
        )

    @property
    def peak_rss_kb(self) -> int | None:
        vals = [s.get("rssKb") for s in self.samples]
        vals = [v for v in vals if isinstance(v, int)]
        return max(vals) if vals else None

    @property
    def peak_hwm_kb(self) -> int | None:
        vals = [s.get("hwmKb") for s in self.samples]
        vals = [v for v in vals if isinstance(v, int)]
        return max(vals) if vals else None


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--aivmx", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--calib-input", type=Path, required=True)
    p.add_argument("--num-calib", type=int, default=1)
    p.add_argument("--style", type=int, default=0)
    p.add_argument("--bert-onnx-path", type=Path, default=None)
    p.add_argument("--summary", type=Path, required=True)
    args = p.parse_args()

    started = time.perf_counter()
    with Sampler() as sampler:
        result = quantize(
            args.aivmx,
            args.out_dir,
            calib_input=args.calib_input,
            num_calib=args.num_calib,
            style_id=args.style,
            bert_onnx_path=args.bert_onnx_path,
        )
    elapsed = time.perf_counter() - started

    summary = {
        "status": "ok",
        "elapsedSec": round(elapsed, 3),
        "peakRssKbSampled": sampler.peak_rss_kb,
        "peakHwmKb": sampler.peak_hwm_kb,
        "finalRssKb": status_kb("VmRSS"),
        "finalHwmKb": status_kb("VmHWM"),
        "sampleCount": len(sampler.samples),
        "result": result,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
