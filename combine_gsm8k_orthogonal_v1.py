#!/usr/bin/env python3
"""Combine both model analyses and apply the preregistered six-test Holm family."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", type=Path, required=True)
    ap.add_argument("--qwen", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    docs = [json.loads(a.llama.read_text()), json.loads(a.qwen.read_text())]
    rows = [(d["model"], c) for d in docs for c in d["direct_primary_contrasts_partial_family"]]
    if len(rows) != 6 or {m for m, _ in rows} != {"llama3", "qwen2.5"}:
        raise ValueError("Holm family requires all six direct contrasts from both models")
    order = sorted(range(6), key=lambda i: rows[i][1]["mcnemar_p_raw"])
    adj = [0.0] * 6
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (6-rank) * rows[i][1]["mcnemar_p_raw"]))
        adj[i] = running
    for i, (model, row) in enumerate(rows):
        row["holm_p_adjusted_m6"] = adj[i]
        row["model"] = model
    out = {"protocol": "gsm8k-orthogonal-v1", "holm_family": "six RSN-versus-seed contrasts, two models by three directions",
           "tests": rows, "scope_note": "Exact McNemar p-values receive Holm correction; bootstrap CIs are raw 95% intervals."}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
