#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine the S4.1 primary rows written by three separate
`eval_cot_transfer_followup.py` invocations (one per task) into the single
Holm m=6 family declared in `docs/PREREG_COT_TRANSFER_FOLLOWUP.md` S4.1.

This is deliberately a SEPARATE script rather than Holm-correcting inside
`eval_cot_transfer_followup.py`, because that script runs once per task and
correcting there would apply Holm at whatever m happened to be available at
that call (m=2 per task), silently reporting it under an m=6 label.

Usage:
    python combine_cot_followup_holm.py \
      --inputs docs/cot_followup_logiqa2_evaluation.json \
               docs/cot_followup_bbh_object_counting_evaluation.json \
               docs/cot_followup_cruxeval_evaluation.json \
      --out docs/cot_followup_holm_combined.json

THE FAMILY IS FIXED AT EXACTLY THE SIX PRE-REGISTERED (task, model) PAIRS
(PREREG_COT_TRANSFER_FOLLOWUP.md S4.1). Holm is computed ONLY when all six are
present. If any is missing, or any supplied row falls outside the registered
six, Holm is WITHHELD entirely: raw p and the bootstrap CI are still reported
per row, labelled unadjusted, but no p_adj is produced.

Computing Holm at whatever m happens to be available would be
ANTI-CONSERVATIVE, not conservative -- a smaller m applies a weaker
correction to whichever p-values did arrive, which is the opposite of what a
missing-data-safe combiner should do. This mirrors the P3/P4/P4b/P4c
convention of withholding Holm on a partial family, but goes one step
further: this family's m is fixed at 6 by pre-registration (not "however many
models finished," as in the two-model P4 families), so a family of 3 or 5 is
just as incomplete as a family of 1 and must be treated identically.
"""

import argparse
import json
import os
import sys


def die(m):
    print(f"[FATAL] {m}", file=sys.stderr)
    raise SystemExit(2)


# The six pre-registered (task, model) pairs, fixed by
# PREREG_COT_TRANSFER_FOLLOWUP.md S4.1. Nothing else may enter this family.
REGISTERED_KEYS = {
    "logiqa2:llama3", "logiqa2:qwen2.5",
    "bbh/object_counting:llama3", "bbh/object_counting:qwen2.5",
    "cruxeval:llama3", "cruxeval:qwen2.5",
}


def holm(pairs):
    s = sorted(pairs, key=lambda t: t[1])
    m = len(s)
    out, run = {}, 0.0
    for i, (k, p) in enumerate(s):
        adj = min(1.0, max(run, (m - i) * p))
        run = adj
        out[k] = adj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="the per-task cot_followup_*_evaluation.json files")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if os.path.exists(a.out):
        die(f"{a.out} exists; refusing to overwrite")

    rows = []
    for p in a.inputs:
        d = json.load(open(p, encoding="utf-8"))
        if not d.get("exploratory_followup"):
            die(f"{p}: not a cot-transfer-followup-v0 evaluation file")
        task_label = d["task"] + (f"/{d['bbh_task']}" if d.get("bbh_task") else "")
        for model, r in d.get("primary_rows", {}).items():
            rows.append({
                "key": f"{task_label}:{model}", "task": task_label,
                "model": model, "alpha": r["alpha"],
                "acc_cot0": r["acc_cot0"], "acc_cotA": r["acc_cotA"],
                "dAcc_pp": r["dAcc_pp"], "p_raw": r["p_raw"],
                "ci95_pp": r["ci95_pp"],
            })

    if not rows:
        die("no primary rows found across the supplied input files")
    if len(set(r["key"] for r in rows)) != len(rows):
        die("duplicate (task, model) row -- an input file was supplied twice "
            "or two files cover the same task/model")

    got_keys = {r["key"] for r in rows}
    unregistered = got_keys - REGISTERED_KEYS
    if unregistered:
        die(f"row(s) {sorted(unregistered)} are not in the pre-registered "
            f"6-pair family {sorted(REGISTERED_KEYS)}. This combiner accepts "
            f"only the six pairs S4.1 declared -- an extra task, model, or "
            f"bbh_task would silently widen the family beyond what was "
            f"pre-registered.")

    missing = REGISTERED_KEYS - got_keys
    complete = not missing

    if complete:
        adj = holm([(r["key"], r["p_raw"]) for r in rows])
        m_used = 6
    else:
        adj = None
        m_used = None
        print(f"[!] {len(missing)}/6 pre-registered pairs missing: "
              f"{sorted(missing)}. Holm is WITHHELD ENTIRELY -- reporting a "
              f"correction over any m < 6 would be anti-conservative, since "
              f"the family is fixed at exactly 6 by pre-registration, not "
              f"'however many rows arrived.' Raw p and the bootstrap CI are "
              f"still reported per row, labelled unadjusted.")

    print(f"\n=== cot-transfer-followup-v0  S4.1 family "
          f"({'Holm m=6' if complete else 'INCOMPLETE, Holm WITHHELD'})")
    print(f"{'task':24s} {'model':9s} {'a':>3} {'cot0':>7} {'cotA':>7} "
          f"{'dAcc':>8} {'p_raw':>9} {'p_adj':>9}  CI95")
    for r in sorted(rows, key=lambda x: x["key"]):
        pa = f"{adj[r['key']]:9.4f}" if adj else " WITHHELD"
        print(f"{r['task']:24s} {r['model']:9s} {r['alpha']:>3} "
              f"{r['acc_cot0']:7.4f} {r['acc_cotA']:7.4f} {r['dAcc_pp']:+8.2f} "
              f"{r['p_raw']:9.4f} {pa}  "
              f"[{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}]")
    if not complete:
        print("\n[!] raw p above is UNADJUSTED and must not be cited as "
              "corrected.")

    json.dump({
        "protocol": "cot-transfer-followup-v0", "exploratory_followup": True,
        "holm_family_m_declared": 6, "holm_complete": complete,
        "holm_family_m_used": m_used, "missing_pairs": sorted(missing),
        "rows": rows, "p_adj": adj,
        "note": ("post-hoc exploratory follow-up; this Holm family is fixed "
                 "at exactly the 6 pairs registered in "
                 "PREREG_COT_TRANSFER_FOLLOWUP.md S4.1, is independent of "
                 "every P3/P4/P4b/P4c Holm family, and is never pooled with "
                 "them. It does not replace, rescale, or supersede any "
                 "frozen No-CoT result. Holm is computed ONLY when all six "
                 "pairs are present; otherwise it is withheld and raw p is "
                 "unadjusted."),
    }, open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
