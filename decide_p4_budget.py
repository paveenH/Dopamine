#!/usr/bin/env python3
"""Apply the frozen 512/1024 rule across ALL FOUR preflight cells.

    if ANY of the 20 outputs in ANY of the 4 cells has generated length
    >= 511, the formal budget becomes 1024; otherwise it stays 512.

Mechanical. No judgement, no per-model decision, no accuracy. Both models'
preflight files are REQUIRED -- deciding on one model would apply a rule the
protocol declared over four cells to a subset of them.
"""
import argparse, json, sys

def die(m): print(f"[FATAL] {m}", file=sys.stderr); raise SystemExit(2)

ap = argparse.ArgumentParser()
ap.add_argument("--preflight", nargs="+", required=True,
                help="both models' preflight result JSONs")
ap.add_argument("--out", default="docs/p4_amendment_03.json")
a = ap.parse_args()

cells, thr, up, base = {}, None, None, None
for p in a.preflight:
    d = json.load(open(p))
    m = d["meta"]
    if m.get("accuracy_computed") is not False:
        die(f"{p} claims accuracy was computed; the preflight must not")
    thr = thr or m["upgrade_threshold"]; up = up or m["upgraded_budget"]
    base = base or m["preflight_budget"]
    for tag, c in d["cells"].items():
        cells[f"{m['model']}:{tag}"] = c

if len(cells) != 4:
    die(f"expected 4 cells across both models, got {len(cells)}: {sorted(cells)}")

rows = []
trigger = False
for name, c in sorted(cells.items()):
    lens = [r["generated_token_count"] for r in c["rows"]]
    n_hit = sum(1 for x in lens if x >= thr)
    n_ex = sum(1 for r in c["rows"] if r["stop_reason"] == "budget_exhausted")
    n_fmt = sum(1 for r in c["rows"] if r["n_matches"] >= 1)
    trigger |= n_hit > 0
    rows.append({"cell": name, "alpha": c["alpha"], "n": len(lens),
                 "len_min": min(lens), "len_med": sorted(lens)[len(lens)//2],
                 "len_max": max(lens), "n_at_or_over_threshold": n_hit,
                 "n_budget_exhausted": n_ex, "n_with_final_answer": n_fmt,
                 "steering_fires": c["steering_fires"]})

budget = up if trigger else base
print(f"{'cell':22s} {'a':>4} {'min':>5} {'med':>5} {'max':>5} {'>=thr':>6} {'exh':>4} {'fmt':>5}")
for r in rows:
    print(f"{r['cell']:22s} {r['alpha']:>4} {r['len_min']:5d} {r['len_med']:5d} "
          f"{r['len_max']:5d} {r['n_at_or_over_threshold']:6d} "
          f"{r['n_budget_exhausted']:4d} {r['n_with_final_answer']:3d}/{r['n']}")
print(f"\nthreshold >= {thr}   triggered: {trigger}   FORMAL BUDGET = {budget}")

fmt_bad = [r["cell"] for r in rows if r["n_with_final_answer"] < r["n"]]
if fmt_bad:
    print(f"\n[!] FORMAT VIOLATION in {fmt_bad}. Protocol section 4: the response "
          f"is a HARD STOP, not a redesigned prompt or parser.")

json.dump({"amendment": "p4-amend-03", "protocol": "logiqa2-p4-v1",
           "type": "additive", "decision_rule": f">= {thr} in ANY of 4 cells -> {up}",
           "triggered": trigger, "formal_max_new_tokens": budget,
           "accuracy_computed": False, "cells": rows},
          open(a.out, "w"), indent=2, ensure_ascii=False)
print(f"\nwrote {a.out}")
