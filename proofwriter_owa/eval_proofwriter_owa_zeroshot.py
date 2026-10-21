#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_proofwriter_owa_zeroshot.py -- analysis for the ProofWriter-OWA chat
zero-shot experiment (protocol `proofwriter-owa-zs-chat-v1`).

This is NOT the old four-point-sweep evaluator and does not pretend to be one:
it validates THIS experiment's own condition set and refuses anything else.

Needs the experiment's own cells of one condition:
    llama3 alpha 0, -6 and +6; qwen2.5 alpha 0, +8 and -6 (reverse doses).
Any missing / extra / mismatched cell, config drift, host/GPU/software
difference between a model's cells, wrong fire count, or identity/order
mismatch against the blind manifest and the gold file aborts with no output.
There is no format gate; valid-submission / loop / truncation are reported
per cell (loop and truncation split by whether a valid answer was submitted).

STATISTICS (eval). Per model, workpoint vs the SAME model's alpha=0, paired by
sample identity: exact two-sided McNemar on first_acc (no-marker = incorrect,
denominator 300) and a question-paired bootstrap CI (B=10000, seed 0, pp).
The two models' workpoint contrasts form ONE Holm family, m=2, declared
per condition (No-CoT and CoT are separate families) and kept apart from the
m=3 families of `proofwriter-owa-v0` and `proofwriter-owa-chat-v1`. Early-
candidate paired contrasts are EXPLORATORY: unadjusted, outside Holm.

INTERPRETATION LIMITS (also written into the output JSON)
  * Order change with an accuracy rise is co-variation, not causal mediation.
  * Order change without a detected accuracy gain does not show that the
    errors are not due to premature answering.
  * A near-zero alpha=0 early-candidate rate only means the indicator has
    little room to move; it does not predict that steering cannot help.
  * Early-candidate / answer-first describe VISIBLE output order, not internal
    commitment. The prompt has no stop instruction; post_marker reports what
    follows the first marker and does not locate loop onset.
  * No-CoT vs CoT differ by one line, so that contrast is a CoT-instruction
    contrast. Comparisons with the old 1-shot CoT line still differ in other
    wording and cannot be attributed to the exemplar alone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import zs_common as zc  # noqa: E402
import prompt_zeroshot as pz  # noqa: E402
from scoring import mcnemar_exact, holm, question_paired_bootstrap_ci  # noqa: E402
from answer_parser import is_correct, parse_final_answer_v2  # noqa: E402
import label_early_candidate as lec  # noqa: E402

EXPECTED_SET = {("llama3", "a0"), ("llama3", "wp"), ("llama3", "rev"),
                ("qwen2.5", "a0"), ("qwen2.5", "wp"), ("qwen2.5", "rev")}


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def cell_id_from_meta(meta):
    model = meta.get("model")
    if model not in zc.MODEL_CONFIG:
        raise zc.ZSError(f"unknown model {meta.get('model')!r}")
    a = meta.get("alpha")
    if a == 0:
        return model, "a0"
    if a == zc.MODEL_CONFIG[model]["workpoint"]:
        return model, "wp"
    if a == zc.MODEL_CONFIG[model].get("reverse"):
        return model, "rev"
    raise zc.ZSError(f"{model}: alpha {a!r} is not one of this experiment's alphas; workpoint is "
                     f"{zc.MODEL_CONFIG[model]['workpoint']}")


def load_gold_checked(path, samples):
    d = json.load(open(path, encoding="utf-8"))
    if d["meta"].get("contains_labels") is not True:
        raise zc.ZSError("gold file does not declare contains_labels=true")
    rows = d["data"]
    if len(rows) != zc.N_ITEMS:
        raise zc.ZSError(f"gold has {len(rows)} rows, expected {zc.N_ITEMS}")
    gold = {}
    for s, g in zip(samples, rows):
        for k in ("sample_id", "key", "dataset", "official_theory_id", "official_qid"):
            if s[k] != g[k]:
                raise zc.ZSError(f"gold/manifest identity mismatch on {k} at "
                                 f"sample {s['sample_id']}")
        gold[s["sample_id"]] = g["answer"]
    return gold


def paired_vs_base(base, other, gold):
    def v(rows):
        return [1 if is_correct(parse_final_answer_v2(r["generated"]).label,
                                gold[r["sample_id"]]) else 0 for r in rows]
    a, b = v(base), v(other)
    g01, l10, p = mcnemar_exact(a, b)
    lo, hi = question_paired_bootstrap_ci(a, b)
    return {"delta_first_acc_pp": (sum(b) - sum(a)) / len(a) * 100.0,
            "alpha0_correct": sum(a), "other_correct": sum(b),
            "wrong_to_right": g01, "right_to_wrong": l10,
            "mcnemar_exact_p_unadjusted": p, "bootstrap_ci95_pp": [lo, hi],
            "note": "outside the Holm family (m=2 = the two GSM8K workpoints)"}


def run_eval(args):
    samples = zc.load_blind_manifest(args.manifest_blind)[1]
    gold = load_gold_checked(args.gold, samples)
    found = {}
    for p in args.cells:
        meta = json.load(open(p, encoding="utf-8"))["meta"]
        cid = cell_id_from_meta(meta)
        if cid in found:
            raise zc.ZSError(f"duplicate cell {cid}")
        found[cid] = p
    if set(found) != EXPECTED_SET:
        raise zc.ZSError(f"cell set {sorted(found)} != the experiment's own set "
                         f"{sorted(EXPECTED_SET)} (missing: "
                         f"{sorted(EXPECTED_SET - set(found))}, extra: "
                         f"{sorted(set(found) - EXPECTED_SET)})")
    cells, metas, per_model = {}, {}, {}
    for model in zc.MODEL_CONFIG:
        mm = []
        for cell in [c for c in zc.CELLS if (model, c) in EXPECTED_SET]:
            alpha = zc.alpha_of(model, cell)
            meta, rows = zc.validate_cell(found[(model, cell)], model,
                                          args.condition, alpha, samples)
            cells[(model, cell)], metas[(model, cell)] = rows, meta
            mm.append(meta)
        zc.check_cross_cell(mm, model)

    results, raw_p = {}, []
    for model in zc.MODEL_CONFIG:
        base, wp = cells[(model, "a0")], cells[(model, "wp")]
        def acc_vec(rows):
            return [1 if is_correct(parse_final_answer_v2(r["generated"]).label,
                                    gold[r["sample_id"]]) else 0 for r in rows]
        def ec_vec(rows):
            return [1 if lec.candidate(r["generated"]) else 0 for r in rows]
        a, b = acc_vec(base), acc_vec(wp)
        g01, l10, p = mcnemar_exact(a, b)
        lo, hi = question_paired_bootstrap_ci(a, b)
        ea, eb = ec_vec(base), ec_vec(wp)
        e01, e10, ep = mcnemar_exact(ea, eb)
        raw_p.append((model, p))
        results[model] = {
            "alpha_workpoint": zc.MODEL_CONFIG[model]["workpoint"],
            "cells": {"alpha0": zc.cell_stats(base, gold),
                      "workpoint": zc.cell_stats(wp, gold)},
            **({"reverse_alpha": zc.MODEL_CONFIG[model]["reverse"],
                "cells_reverse": zc.cell_stats(cells[(model, "rev")], gold),
                "reverse_vs_alpha0_EXPLORATORY": paired_vs_base(base, cells[(model, "rev")], gold)}
               if (model, "rev") in cells else {}),
            "workpoint_vs_alpha0": {
                "delta_first_acc_pp": (sum(b) - sum(a)) / len(a) * 100.0,
                "alpha0_correct": sum(a), "workpoint_correct": sum(b),
                "wrong_to_right": g01, "right_to_wrong": l10,
                "mcnemar_exact_p": p, "bootstrap_ci95_pp": [lo, hi],
                "bootstrap": {"B": 10000, "seed": 0, "unit": "question"},
            },
            "early_candidate_paired_EXPLORATORY": {
                "alpha0": sum(ea), "workpoint": sum(eb),
                "delta_pp": (sum(eb) - sum(ea)) / len(ea) * 100.0,
                "off_to_on": e01, "on_to_off": e10, "mcnemar_exact_p_unadjusted": ep,
                "note": "outside the Holm family; descriptive co-variation only"},
        }
    adj = holm(raw_p)
    for model in zc.MODEL_CONFIG:
        results[model]["workpoint_vs_alpha0"]["holm_p_adj"] = adj[model]

    out = {
        "protocol": zc.PROTOCOL, "stage": "eval", "condition": args.condition,
        "prompt_template_id": pz.TEMPLATE_IDS[args.condition],
        "n_shot": 0, "cot": pz.condition_to_cot(args.condition),
        "conditions": sorted(f"{m}:{c}" for m, c in EXPECTED_SET),
        "holm_family": {
            "id": f"zs-chat-v1-{args.condition}-workpoint", "m": 2,
            "members": ["llama3 -6 vs 0", "qwen2.5 +8 vs 0"],
            "separate_from": ["proofwriter-owa-v0 (m=3)",
                              "proofwriter-owa-chat-v1 (m=3)",
                              "the other condition of this protocol"]},
        "scoring": ("FIRST strict line-anchored '#### True|False|Unknown' "
                    "marker; no marker = incorrect; denominator 300; no bare-"
                    "label fallback"),
        "workpoint_source": "GSM8K frozen record; alpha not selected on ProofWriter",
        "results": results,
        "inputs": {"cells": {f"{m}:{c}": {"path": os.path.abspath(found[(m, c)]),
                                          "sha256": zc.sha256_file(found[(m, c)])}
                             for m, c in sorted(found)},
                   "gold_sha256": zc.sha256_file(args.gold),
                   "manifest_blind_sha256": zc.sha256_file(args.manifest_blind)},
        "interpretation_limits": [
            "Order change with an accuracy rise is co-variation, not causal mediation.",
            "Order change without a detected accuracy gain does not show errors are not due to premature answering.",
            "A near-zero alpha=0 early-candidate rate only means the indicator has little room to move; it does not predict that steering cannot help.",
            "Early-candidate / answer-first describe visible output order, not internal commitment; the prompt has no stop instruction, post_marker reports what follows the first marker and does not locate loop onset.",
            "No-CoT vs CoT differ by one line; comparisons with the old 1-shot CoT line still differ in other wording and are not attributable to the exemplar alone.",
        ],
    }
    zc.atomic_write_json(args.out, out)
    print(f"wrote {args.out}")
    for model in zc.MODEL_CONFIG:
        r = results[model]
        w = r["workpoint_vs_alpha0"]
        print(f"  {model}: a0 {w['alpha0_correct']}/300 -> wp {w['workpoint_correct']}/300  "
              f"d={w['delta_first_acc_pp']:+.2f}pp  p={w['mcnemar_exact_p']:.4g}  "
              f"holm={w['holm_p_adj']:.4g}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=sorted(pz.TEMPLATE_IDS), default="nocot")
    p.add_argument("--manifest_blind", required=True)
    p.add_argument("--cells", nargs="+", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if os.path.exists(args.out):
        die(f"{args.out} exists; refusing to overwrite")
    try:
        return run_eval(args)
    except zc.ZSError as e:
        die(str(e))


if __name__ == "__main__":
    sys.exit(main())
