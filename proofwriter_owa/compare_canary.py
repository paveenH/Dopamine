#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outcome-level cross-GPU canary comparator for the ProofWriter OWA canary
stage (PREREG_PROOFWRITER_OWA.md S7). Added in review (2026-09-04): the
canary stage previously had no real comparator at all -- its only guidance
was a comment telling a human to "diff the two outputs", which is not an
outcome-level comparison and (before the device_tag fix in
run_proofwriter_owa.sh) could not even produce two DIFFERENT output files to
diff in the first place.

Frozen criterion, matching the project-wide convention used by
zebralogic/eval_zebralogic.py's --canary_check: text need not match
verbatim across GPUs (bf16 greedy is not byte-reproducible across devices),
but per-item CORRECTNESS, PARSE STATUS and TRUNCATION STATUS must show no
SYSTEMATIC divergence. This script reports the summary table and the
per-item disagreement list; it does NOT auto-decide pass/fail (the canary
set is only 30 items, too small for a principled statistical threshold) --
a human reads the table and the disagreement list against that frozen
wording.

Reads gold (the preflight_gold_<model>.json subset) ONLY to compute
per-item correctness for the comparison; this is a read-only diagnostic,
never a scored formal result, and never written into the formal analysis
tree.

@author: proofwriter_owa task
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from answer_parser import parse_final_answer, normalize_label, is_correct  # noqa: E402

PROTOCOL = "proofwriter-owa-v0"


def die(m):
    print(f"[FATAL] {m}", file=sys.stderr)
    raise SystemExit(2)


def load_gold(path):
    d = json.load(open(path, encoding="utf-8"))
    if d["meta"].get("contains_labels") is not True:
        die(f"{path}: does not declare contains_labels=true; point --gold at "
            "a gold-bearing file (e.g. preflight_gold_<model>.json)")
    return {r["sample_id"]: {"answer": normalize_label(r["answer"]),
                             "dataset": r["dataset"]} for r in d["data"]}


def load_cell(path):
    d = json.load(open(path, encoding="utf-8"))
    m = d["meta"]
    if m.get("protocol") != PROTOCOL:
        die(f"{path}: protocol {m.get('protocol')!r} != {PROTOCOL!r}")
    return m, d["data"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True,
                    help="the gold-bearing subset matching the canary items, "
                         "e.g. preflight_gold_<model>.json")
    ap.add_argument("--cell", nargs="+", required=True,
                    help="2+ canary cell JSONs, one per physical GPU tested "
                         "(different --device_tag / --tag)")
    ap.add_argument("--out", default=None,
                    help="optional: write the full comparison as JSON")
    a = ap.parse_args()

    if len(a.cell) < 2:
        die("--cell needs at least 2 files (one per GPU being compared); a "
            "single cell has nothing to compare against")

    gold = load_gold(a.gold)

    metas, rowsets = [], []
    for p in a.cell:
        m, rows = load_cell(p)
        metas.append(m)
        rowsets.append(rows)

    ids_ref = sorted(r["sample_id"] for r in rowsets[0])
    for p, rows in zip(a.cell[1:], rowsets[1:]):
        ids_here = sorted(r["sample_id"] for r in rows)
        if ids_here != ids_ref:
            die(f"{p}: sample_id set differs from {a.cell[0]}; the two "
                "canary runs must cover the SAME items to be comparable")
    missing_gold = set(ids_ref) - set(gold)
    if missing_gold:
        die(f"--gold is missing {len(missing_gold)} sample_id(s) present in "
            f"the canary cells, e.g. {sorted(missing_gold)[:5]}")

    def score(rows):
        by_id = {r["sample_id"]: r for r in rows}
        out = {}
        for sid in ids_ref:
            r = by_id[sid]
            text = r["generated"]
            parsed = parse_final_answer(text)
            correct = is_correct(parsed.label, gold[sid]["answer"])
            out[sid] = {
                "correct": correct, "parse_status": parsed.status,
                "truncated": bool(r.get("truncated")),
            }
        return out

    scored = [score(rows) for rows in rowsets]

    summaries = []
    for m, sc in zip(metas, scored):
        n = len(sc)
        summaries.append({
            "device_tag": m.get("provenance", {}).get("cuda_visible_devices"),
            "host": m.get("provenance", {}).get("host"),
            "alpha": m.get("alpha"), "n": n,
            "accuracy": sum(1 for v in sc.values() if v["correct"]) / n,
            "parse_ok_rate": sum(1 for v in sc.values()
                                 if v["parse_status"] == "ok") / n,
            "truncated_rate": sum(1 for v in sc.values() if v["truncated"]) / n,
        })

    print(f"\n=== PROOFWRITER OWA CANARY CROSS-GPU CHECK  "
          f"n_items={len(ids_ref)}  n_devices={len(summaries)} ===")
    print(f"{'device_tag':>15} {'host':>15} {'alpha':>6} {'accuracy':>10} "
          f"{'parse_ok':>10} {'truncated':>10}")
    for s in summaries:
        print(f"{str(s['device_tag']):>15} {str(s['host'])[:15]:>15} "
              f"{str(s['alpha']):>6} {s['accuracy']:10.4f} "
              f"{s['parse_ok_rate']:10.4f} {s['truncated_rate']:10.4f}")

    # per-item disagreement across ALL cells on (correct, parse_status,
    # truncated) -- an outcome-level divergence, not a text diff.
    diverging = []
    for sid in ids_ref:
        vals = [sc[sid] for sc in scored]
        key = {(v["correct"], v["parse_status"], v["truncated"]) for v in vals}
        if len(key) > 1:
            diverging.append(sid)

    print(f"\nitems with per-device disagreement (correct/parse_status/"
          f"truncated): {len(diverging)}/{len(ids_ref)}  {diverging}")
    print("\nFrozen criterion (PREREG_PROOFWRITER_OWA.md S7): text need not "
          "match verbatim across GPUs, but accuracy, parse status and "
          "truncation status must show NO SYSTEMATIC divergence. Judge the "
          "summary table and the disagreement list above against that; this "
          "script does not auto-decide pass/fail for a 30-item canary.")

    if a.out:
        if os.path.exists(a.out):
            die(f"{a.out} exists; refusing to overwrite")
        json.dump({"protocol": PROTOCOL, "check": "canary_cross_gpu",
                   "item_ids": ids_ref, "cells": a.cell, "summaries": summaries,
                   "diverging_item_ids": diverging},
                  open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
