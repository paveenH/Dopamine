#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_gsm_symbolic.py — loader for apple/GSM-Symbolic (test split, all 3 official
configs: main, p1, p2).

GSM-Symbolic is a numeric/template perturbation of GSM8K: same underlying
reasoning templates, re-instantiated with different symbolic values (and, for
p1/p2, one/two extra clauses inserted, raising difficulty). This loader is a
standalone read-only step -- no other GSM8K/MATH loader or runner is touched.

Output schema per item (JSON list), one file per config:
  {
    "id": <int>,               # HF dataset row id (per-config)
    "instance": <int>,         # instantiation index within original_id
    "original_id": <int>,      # links back to the shared GSM8K-derived template
    "config": "main"|"p1"|"p2",
    "question": <str>,
    "answer": <str>,           # gold numeric answer (extracted from the
                                # official '#### <number>' solution field,
                                # via the SAME regex utils.extract_gsm8k_answer
                                # would apply -- kept local so this loader has
                                # no torch/heavy import)
    "solution": <str>,         # full solution text as shipped by HF
  }

Sampling for the 300-item formal run is NOT done here -- the formal run uses
the FULL official test split per config (no sampling), per the experiment
spec. This script only downloads + reformats + writes one JSON per config,
plus (deterministically) a small preflight subset.
"""

import argparse
import json
import os
import re
from pathlib import Path

from datasets import load_dataset

CONFIGS = ["main", "p1", "p2"]

# HF revision left unpinned deliberately would be a 口径 risk for a frozen
# experiment; pin explicitly once known-good. If unset, `datasets` resolves
# the current default revision and that resolution is recorded in the output
# meta so drift is at least detectable.
REVISION = None


def extract_gold(solution: str) -> str:
    """Same convention as utils.extract_gsm8k_answer's primary branch: the
    official '#### <number>' marker. GSM-Symbolic solutions carry this
    verbatim (inherited from GSM8K format)."""
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", solution)
    if m:
        return m.group(1).replace(",", "")
    return ""


def load_config(config: str, cache_dir: str):
    ds = load_dataset(
        "apple/GSM-Symbolic",
        config,
        split="test",
        cache_dir=cache_dir,
        revision=REVISION,
    )
    items = []
    for i, row in enumerate(ds):
        gold = extract_gold(row["answer"])
        items.append({
            "id": i,
            "instance": row.get("instance", None),
            "original_id": row.get("original_id", None),
            "config": config,
            "question": row["question"],
            "answer": gold,
            "solution": row["answer"],
        })
    missing_gold = sum(1 for it in items if it["answer"] == "")
    return items, missing_gold


def preflight_subset(items: list, n: int = 10, seed_field: str = "original_id"):
    """Deterministic 10-item subset per config: sort by (original_id, instance,
    id) and take an evenly spaced stride so distinct original_id/instance
    values are covered rather than clustering on the first few rows."""
    ordered = sorted(items, key=lambda x: (x.get(seed_field) if x.get(seed_field) is not None else x["id"], x.get("instance") or 0, x["id"]))
    if len(ordered) <= n:
        return ordered
    stride = len(ordered) / n
    picked = []
    seen_ids = set()
    for k in range(n):
        idx = int(k * stride)
        while idx in seen_ids and idx < len(ordered) - 1:
            idx += 1
        seen_ids.add(idx)
        picked.append(ordered[idx])
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="components/benchmark/gsm_symbolic")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--preflight_n", type=int, default=10)
    ap.add_argument("--check", action="store_true",
                     help="Load + validate schema only, print counts, write nothing.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not args.check:
        out_dir.mkdir(parents=True, exist_ok=True)

    preflight_all = []
    summary = {}
    for config in CONFIGS:
        items, missing = load_config(config, args.cache_dir)
        n = len(items)
        n_original = len({it["original_id"] for it in items})
        n_instance = len({it["instance"] for it in items})
        summary[config] = {
            "n_items": n,
            "n_unique_original_id": n_original,
            "n_unique_instance": n_instance,
            "n_missing_gold": missing,
        }
        print(f"[{config}] n={n}  unique original_id={n_original}  "
              f"unique instance={n_instance}  missing_gold={missing}")
        if missing:
            print(f"  [WARN] {missing} items in {config} have no parseable gold "
                  f"'#### <number>' marker -- inspect before using this config.")

        if args.check:
            continue

        out_path = out_dir / f"gsm_symbolic_{config}_test.json"
        if out_path.exists():
            raise FileExistsError(
                f"{out_path} already exists -- refusing to overwrite. "
                "Delete it deliberately if you intend to regenerate."
            )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "dataset": "apple/GSM-Symbolic",
                    "config": config,
                    "split": "test",
                    "revision": REVISION,
                    "n_items": n,
                },
                "data": items,
            }, f, ensure_ascii=False, indent=2)
        print(f"  -> wrote {out_path}")

        pf = preflight_subset(items, n=args.preflight_n)
        for it in pf:
            preflight_all.append(it)

    if args.check:
        print("\n[check] OK -- schema validated for all 3 configs, nothing written.")
        return

    pf_path = out_dir / "gsm_symbolic_preflight_30.json"
    if pf_path.exists():
        raise FileExistsError(f"{pf_path} already exists -- refusing to overwrite.")
    with open(pf_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "dataset": "apple/GSM-Symbolic",
                "split": "test",
                "revision": REVISION,
                "note": "10 items per config (main/p1/p2), deterministic stride "
                        "sample by (original_id, instance, id); NOT a random "
                        "sample of the formal run.",
                "configs": CONFIGS,
                "n_per_config": args.preflight_n,
            },
            "data": preflight_all,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n-> wrote preflight subset ({len(preflight_all)} items) -> {pf_path}")
    print("\nSummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
