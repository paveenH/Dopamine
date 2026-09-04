#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a fixed-order subset of the frozen ProofWriter OWA manifest, for the
preflight and pilot stages. Selection is BY FROZEN MANIFEST ORDER ONLY --
never by model output, never resampled.

Reads manifest_gold.json (to see labels for the coverage requirement) but
emits a BLIND subset (no label field) plus a matching gold subset, following
the same two-file split as the manifest itself.

Modes
-----
  preflight   30 items: 15 D3 + 15 D5, all three labels covered in each
              dataset half where the data allows it. Selection: for each
              dataset, walk the manifest in its frozen order and take the
              first 5 of each label (True/False/Unknown); if a label has
              fewer than 5 available in the first pass, fill remaining slots
              from the next available items of that dataset in frozen order.
  pilot       150 items: the first 75 D3 + first 75 D5 rows in frozen
              manifest order (no label-balancing beyond what the manifest
              itself already carries, since the manifest is already
              near-50/50/50 within each dataset by construction).

Usage
-----
    python3 build_subset.py --gold manifest_gold.json --blind manifest_blind.json \\
        --mode preflight --out_blind preflight_blind.json --out_gold preflight_gold.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

LABEL_ORDER = ("True", "False", "Unknown")


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def select_preflight(gold_rows: list[dict], per_label: int = 5) -> list[int]:
    """Returns sample_ids: per dataset, per_label of each label (frozen
    manifest order), filling shortfalls from the same dataset's leftover
    rows in order. Reports (does not silently hide) any dataset/label that
    cannot reach per_label."""
    chosen = []
    for ds in ("D3", "D5"):
        rows = [r for r in gold_rows if r["dataset"] == ds]
        by_label = collections.defaultdict(list)
        for r in rows:
            by_label[r["answer"]].append(r)
        picked_ids = set()
        for lab in LABEL_ORDER:
            take = by_label[lab][:per_label]
            for r in take:
                picked_ids.add(r["sample_id"])
            if len(take) < per_label:
                print(f"[build_subset] WARNING: {ds}/{lab} has only "
                      f"{len(take)}/{per_label} available for preflight coverage.")
        # fill shortfall from the same dataset's remaining rows, frozen order
        need = per_label * len(LABEL_ORDER) - len(picked_ids)
        if need > 0:
            leftovers = [r for r in rows if r["sample_id"] not in picked_ids]
            for r in leftovers[:need]:
                picked_ids.add(r["sample_id"])
        chosen.extend(sorted(picked_ids))
    return chosen


def select_pilot(gold_rows: list[dict], per_dataset: int = 75) -> list[int]:
    chosen = []
    for ds in ("D3", "D5"):
        rows = [r for r in gold_rows if r["dataset"] == ds]
        take = rows[:per_dataset]
        if len(take) < per_dataset:
            print(f"[build_subset] WARNING: {ds} has only {len(take)}/"
                  f"{per_dataset} rows for the pilot subset.")
        chosen.extend(r["sample_id"] for r in take)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--blind", required=True)
    ap.add_argument("--mode", required=True, choices=("preflight", "pilot"))
    ap.add_argument("--out_blind", required=True)
    ap.add_argument("--out_gold", required=True)
    a = ap.parse_args()

    gd = json.load(open(a.gold, encoding="utf-8"))
    bd = json.load(open(a.blind, encoding="utf-8"))
    gold_rows = gd["data"]
    blind_by_id = {r["sample_id"]: r for r in bd["data"]}

    if a.mode == "preflight":
        ids = select_preflight(gold_rows)
    else:
        ids = select_pilot(gold_rows)

    ids_sorted = sorted(ids)
    gold_by_id = {r["sample_id"]: r for r in gold_rows}
    for sid in ids_sorted:
        if sid not in blind_by_id:
            die(f"sample_id {sid} selected for {a.mode} not found in blind manifest")

    out_gold = {"meta": {**gd["meta"], f"{a.mode}_subset": True, "n": len(ids_sorted)},
               "data": [gold_by_id[i] for i in ids_sorted]}
    out_blind = {"meta": {**bd["meta"], f"{a.mode}_subset": True, "n": len(ids_sorted)},
                "data": [blind_by_id[i] for i in ids_sorted]}

    json.dump(out_gold, open(a.out_gold, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(out_blind, open(a.out_blind, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    by_ds = collections.Counter(gold_by_id[i]["dataset"] for i in ids_sorted)
    by_lab = collections.Counter(gold_by_id[i]["answer"] for i in ids_sorted)
    print(f"[build_subset] mode={a.mode} n={len(ids_sorted)} by_dataset={dict(by_ds)} "
          f"by_label={dict(by_lab)}")
    print(f"  blind -> {a.out_blind}")
    print(f"  gold  -> {a.out_gold}")


if __name__ == "__main__":
    main()
