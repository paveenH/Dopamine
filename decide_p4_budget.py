#!/usr/bin/env python3
"""Apply the frozen 512/1024 rule across ALL FOUR preflight cells.

    if ANY of the 20 outputs in ANY of the 4 cells has generated length
    >= 511, the formal budget becomes 1024; otherwise it stays 512.

Mechanical. No judgement, no per-model decision, no accuracy.

Fails closed on everything that would make the decision unreadable: a missing
model, the wrong alpha set, a cell that is not exactly 20 items, a prompt hash
or preflight digest that differs between the two files, disagreeing budget
parameters, a cell whose steering was not verified, or an existing output file.

Writes docs/p4_amendment_04.json (stage 1). It NEVER overwrites: a frozen
amendment is additive by definition, so there is deliberately no
--allow_overwrite escape hatch. A re-decision needs a new amendment number.

Amendment map: 01 original prompt (superseded), 02 neutral prompt + anchor,
03 externalized-reasoning wording boundary, 04 this stage-1 budget freeze.
"""
import argparse, json, os, sys

def die(m):
    print(f"[FATAL] {m}", file=sys.stderr)
    raise SystemExit(2)

# frozen by PREREG v0 section 2 -- the exact four cells, by model and alpha
EXPECTED = {"llama3": {0, -6}, "qwen2.5": {0, 8}}
N_PER_CELL = 20

ap = argparse.ArgumentParser()
ap.add_argument("--preflight", nargs="+", required=True,
                help="both models' preflight result JSONs")
ap.add_argument("--out", default="docs/p4_amendment_04.json")
a = ap.parse_args()

if os.path.exists(a.out):
    die(f"{a.out} already exists; refusing to overwrite. Amendments are "
        f"additive -- a re-decision needs a NEW number, not a rewrite. There "
        f"is no override flag: a generator of frozen artifacts must not ship "
        f"an escape hatch past its own freeze.")

# ---------------------------------------------------------------- load + verify
files, shared = {}, {}
for p in a.preflight:
    if not os.path.exists(p):
        die(f"missing preflight file {p}")
    d = json.load(open(p))
    m = d.get("meta", {})
    model = m.get("model")
    if model not in EXPECTED:
        die(f"{p}: unknown model {model!r}; expected one of {sorted(EXPECTED)}")
    if model in files:
        die(f"model {model!r} supplied twice")
    if m.get("accuracy_computed") is not False:
        die(f"{p}: accuracy_computed={m.get('accuracy_computed')!r}; the "
            f"preflight must not compute accuracy")
    # cross-file agreement: these must be identical or the decision spans
    # two different protocols
    for k in ("preflight_budget", "upgrade_threshold", "upgraded_budget",
              "prompt_sha256_prefix", "preflight_digest", "amendment"):
        if k not in m:
            die(f"{p}: meta is missing {k!r}")
        if k in shared and shared[k] != m[k]:
            die(f"{k} differs between preflight files: {shared[k]!r} vs "
                f"{m[k]!r}. The two cells were not run under one protocol.")
        shared[k] = m[k]
    files[model] = d

missing = set(EXPECTED) - set(files)
if missing:
    die(f"missing model(s) {sorted(missing)}. The rule is declared over FOUR "
        f"cells; applying it to a subset is not the frozen rule.")

thr, up, base = (shared["upgrade_threshold"], shared["upgraded_budget"],
                 shared["preflight_budget"])
if thr != base - 1:
    die(f"upgrade_threshold {thr} != preflight_budget-1 ({base - 1}); the "
        f"frozen rule is '>= budget - 1, never == budget'")

cells = {}
for model, d in files.items():
    alphas = set()
    for tag, c in d["cells"].items():
        alphas.add(c["alpha"])
        if len(c["rows"]) != N_PER_CELL:
            die(f"{model}:{tag} has {len(c['rows'])} rows, expected {N_PER_CELL}")
        exp_fires = 0 if c["alpha"] == 0 else c["L"] * N_PER_CELL
        if c.get("steering_fires") != exp_fires:
            die(f"{model}:{tag} steering_fires={c.get('steering_fires')} != "
                f"{exp_fires}; the intervention is unverified")
        ids = [r["sample_id"] for r in c["rows"]]
        if sorted(ids) != list(range(N_PER_CELL)):
            die(f"{model}:{tag} sample_ids are not a full 0..{N_PER_CELL-1} cover")
        cells[f"{model}:{tag}"] = c
    if alphas != EXPECTED[model]:
        die(f"{model} has alphas {sorted(alphas)}, expected "
            f"{sorted(EXPECTED[model])}")

if len(cells) != 4:
    die(f"expected 4 cells, got {len(cells)}: {sorted(cells)}")

# every cell must have faced the same 20 items
keysets = {name: tuple(r["key"] for r in sorted(c["rows"],
           key=lambda x: x["sample_id"])) for name, c in cells.items()}
if len(set(keysets.values())) != 1:
    die("the four cells did not face the same 20 items")

# ---------------------------------------------------------------- decide
rows, trigger = [], False
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
print(f"{'cell':22s} {'a':>4} {'min':>5} {'med':>5} {'max':>5} {'>=thr':>6} "
      f"{'exh':>4} {'fmt':>6}")
for r in rows:
    print(f"{r['cell']:22s} {r['alpha']:>4} {r['len_min']:5d} {r['len_med']:5d} "
          f"{r['len_max']:5d} {r['n_at_or_over_threshold']:6d} "
          f"{r['n_budget_exhausted']:4d} {r['n_with_final_answer']:3d}/{r['n']}")
print(f"\nthreshold >= {thr}   triggered: {trigger}   FORMAL BUDGET = {budget}")

fmt_bad = [r["cell"] for r in rows if r["n_with_final_answer"] < r["n"]]
if fmt_bad:
    print(f"\n[!] FORMAT VIOLATION in {fmt_bad}. Protocol section 4: the "
          f"response is a HARD STOP, not a redesigned prompt or parser.")

json.dump({"amendment": "p4-amend-04", "protocol": "logiqa2-p4-v1",
           "type": "additive", "stage": 1,
           "supersedes_nothing": True,
           "prompt_amendment": shared["amendment"],
           "prompt_sha256_prefix": shared["prompt_sha256_prefix"],
           "preflight_digest": shared["preflight_digest"],
           "decision_rule": f">= {thr} in ANY of 4 cells -> {up}",
           "triggered": trigger, "formal_max_new_tokens": budget,
           "accuracy_computed": False, "cells": rows},
          open(a.out, "w"), indent=2, ensure_ascii=False)
print(f"\nwrote {a.out}")
