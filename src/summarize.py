"""Collect every outputs/eval/*/metrics.json into one markdown table.

Rates (ASR, false trigger, partial trigger) are shown with a 95% Wilson score interval,
which stays sensible at rates near 0 or 1, unlike the normal approximation.

Usage:  python src/summarize.py   (writes outputs/results.md)
"""
import json
import math
import sys

import config as C


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt_rate(rate: float, n: int) -> str:
    """'0.997 [0.982, 0.999]' from a rate measured on n prompts."""
    lo, hi = wilson(round(rate * n), n)
    return f"{rate:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252
    rows = []
    for path in sorted((C.OUTPUT_DIR / "eval").glob("*/metrics.json")):
        m = json.loads(path.read_text())
        n = m["n"]
        partial = ", ".join(f"{k}: {fmt_rate(v, n)}" for k, v in m.get("partial_rate", {}).items())
        rows.append(f"| {path.parent.name} | {m['trigger']} | {fmt_rate(m['ASR'], n)} | "
                    f"{fmt_rate(m['false_trigger_rate'], n)} | {partial} | {m['CP_rougeL_f1']:.3f} | {n} |")
    table = "\n".join([
        "ASR and trigger rates: value [95% Wilson CI].",
        "",
        "| run | trigger | ASR | false trigger | partial-trigger rate | CP (ROUGE-L) | n |",
        "|---|---|---|---|---|---|---|", *rows])
    (C.OUTPUT_DIR / "results.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
