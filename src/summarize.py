"""Collect every outputs/eval/*/metrics.json into one markdown table.

Usage:  python src/summarize.py   (writes outputs/results.md)
"""
import json

import config as C


def main():
    rows = []
    for path in sorted((C.OUTPUT_DIR / "eval").glob("*/metrics.json")):
        m = json.loads(path.read_text())
        partial = ", ".join(f"{k}: {v:.2f}" for k, v in m.get("partial_rate", {}).items())
        rows.append(f"| {path.parent.name} | {m['trigger']} | {m['ASR']:.3f} | "
                    f"{m['false_trigger_rate']:.3f} | {partial} | {m['CP_rougeL_f1']:.3f} | {m['n']} |")
    table = "\n".join([
        "| run | trigger | ASR | false trigger | partial-trigger rate | CP (ROUGE-L) | n |",
        "|---|---|---|---|---|---|---|", *rows])
    (C.OUTPUT_DIR / "results.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
