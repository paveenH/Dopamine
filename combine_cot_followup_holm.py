#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine the up-to-6 S4.1 primary rows written by three separate
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

If fewer than 6 rows are present (a task or model missing), the family is
still Holm-corrected at the REALIZED m, and the output records that m
explicitly rather than padding to 6 -- reporting an m=6 adjustment over an
incomplete family would be anti-conservative, the same rule P3/P4/P4b/P4c
already use for a partial Holm family.
"""

import argparse
import json
import os
import sys


def die(m):
    print(f"[FATAL] {m}", file=sys.stderr)
    raise SystemExit(2)


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

    m = len(rows)
    if m < 6:
        print(f"[!] only {m}/6 rows present. Holm is applied at the REALIZED "
              f"m={m}, not padded to 6 -- an m=6 adjustment over an "
              f"incomplete family would be anti-conservative.")

    adj = holm([(r["key"], r["p_raw"]) for r in rows])

    print(f"\n=== cot-transfer-followup-v0  S4.1 combined Holm family (m={m})")
    print(f"{'task':24s} {'model':9s} {'a':>3} {'cot0':>7} {'cotA':>7} "
          f"{'dAcc':>8} {'p_raw':>9} {'p_adj':>9}  CI95")
    for r in sorted(rows, key=lambda x: x["key"]):
        pa = adj[r["key"]]
        print(f"{r['task']:24s} {r['model']:9s} {r['alpha']:>3} "
              f"{r['acc_cot0']:7.4f} {r['acc_cotA']:7.4f} {r['dAcc_pp']:+8.2f} "
              f"{r['p_raw']:9.4f} {pa:9.4f}  "
              f"[{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}]")

    json.dump({
        "protocol": "cot-transfer-followup-v0", "exploratory_followup": True,
        "holm_family_m_realized": m, "holm_family_m_declared": 6,
        "rows": rows, "p_adj": adj,
        "note": ("post-hoc exploratory follow-up; this Holm family is "
                 "independent of every P3/P4/P4b/P4c Holm family and is "
                 "never pooled with them. It does not replace, rescale, or "
                 "supersede any frozen No-CoT result."),
    }, open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
