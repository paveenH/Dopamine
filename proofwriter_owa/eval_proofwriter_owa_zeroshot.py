#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_proofwriter_owa_zeroshot.py -- analysis for the ProofWriter-OWA chat
zero-shot experiment (protocol `proofwriter-owa-zs-chat-v1`).

This is NOT the old four-point-sweep evaluator and does not pretend to be one:
it validates THIS experiment's own condition set and refuses anything else.

--stage gate   GOLD-FREE format-stability gate on one model's No-CoT (or CoT)
               alpha=0 cell. Reads generations only (no gold file is opened),
               applies zs_common.GATE_THRESHOLDS, writes a gate JSON.
               Accuracy and early-candidate are not inputs to the verdict.
--stage eval   Needs EXACTLY four cells of one condition:
                 llama3 alpha 0, llama3 -6, qwen2.5 alpha 0, qwen2.5 +8
               plus both models' PASSED gate files. Any missing / extra /
               mismatched cell, config drift, host/GPU/software difference
               between a model's two cells, wrong fire count, or
               identity/order mismatch against the blind manifest and the gold
               file aborts with no output.

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
    commitment. Marker-first does not imply the stop instruction was obeyed;
    see post_marker.
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

EXPECTED_SET = {("llama3", "a0"), ("llama3", "wp"), ("qwen2.5", "a0"), ("qwen2.5", "wp")}


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
    raise zc.ZSError(f"{model}: alpha {a!r} is neither 0 nor the frozen workpoint "
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


def run_gate(args):
    samples = zc.load_blind_manifest(args.manifest_blind)[1]
    if len(args.cells) != 1:
        die("--stage gate takes exactly one --cells path (the alpha=0 cell)")
    path = args.cells[0]
    meta = json.load(open(path, encoding="utf-8"))["meta"]
    model, cell = cell_id_from_meta(meta)
    if cell != "a0":
        die("the gate is defined on the alpha=0 cell only")
    _, rows = zc.validate_cell(path, model, args.condition, 0, samples)
    stats = zc.cell_stats(rows, gold=None)          # gold never opened
    v = zc.gate_verdict(stats)
    out = {"protocol": zc.PROTOCOL, "stage": "gate", "model": model,
           "condition": args.condition, "passed": v["passed"],
           "checks": v["checks"], "observed": v["observed"],
           "thresholds": v["thresholds"], "gold_read": False,
           "cell_path": os.path.abspath(path), "cell_sha256": zc.sha256_file(path),
           "format_stats": stats,
           "rule": ("decided on submission format only; accuracy and "
                    "early-candidate are not read. On failure the result is "
                    "kept; the prompt, token budget and scoring are NOT "
                    "changed automatically.")}
    zc.atomic_write_json(args.out, out)
    print(json.dumps({"model": model, "condition": args.condition,
                      "passed": v["passed"], "observed": v["observed"],
                      "thresholds": v["thresholds"]}, indent=2))
    return 0 if v["passed"] else 3


def check_gate_file(path, model, condition, a0_path):
    g = json.load(open(path, encoding="utf-8"))
    if (g.get("protocol"), g.get("stage"), g.get("model"), g.get("condition")) != \
            (zc.PROTOCOL, "gate", model, condition):
        raise zc.ZSError(f"{path}: not a {model}/{condition} gate file")
    if g.get("passed") is not True:
        raise zc.ZSError(f"{path}: gate did not pass")
    if g.get("thresholds") != zc.GATE_THRESHOLDS:
        raise zc.ZSError(f"{path}: thresholds differ from the frozen ones")
    if g.get("cell_sha256") != zc.sha256_file(a0_path):
        raise zc.ZSError(f"{path}: gate belongs to a different alpha=0 file")


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
    if len(args.gates) != 2:
        raise zc.ZSError("--gates needs both models' passed gate files")
    gates = {}
    for gp in args.gates:
        g = json.load(open(gp, encoding="utf-8"))
        gates[g.get("model")] = gp
    if set(gates) != set(zc.MODEL_CONFIG):
        raise zc.ZSError(f"gate files cover {sorted(gates)}, need both models")

    cells, metas, per_model = {}, {}, {}
    for model in zc.MODEL_CONFIG:
        check_gate_file(gates[model], model, args.condition, found[(model, "a0")])
        mm = []
        for cell in zc.CELLS:
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
                   "manifest_blind_sha256": zc.sha256_file(args.manifest_blind),
                   "gates": {m: zc.sha256_file(gp) for m, gp in gates.items()}},
        "gate_thresholds": zc.GATE_THRESHOLDS,
        "interpretation_limits": [
            "Order change with an accuracy rise is co-variation, not causal mediation.",
            "Order change without a detected accuracy gain does not show errors are not due to premature answering.",
            "A near-zero alpha=0 early-candidate rate only means the indicator has little room to move; it does not predict that steering cannot help.",
            "Early-candidate / answer-first describe visible output order, not internal commitment; marker-first does not imply the stop instruction was obeyed (see post_marker).",
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
    p.add_argument("--stage", choices=("gate", "eval"), required=True)
    p.add_argument("--condition", choices=sorted(pz.TEMPLATE_IDS), default="nocot")
    p.add_argument("--manifest_blind", required=True)
    p.add_argument("--cells", nargs="+", required=True)
    p.add_argument("--gold", help="eval stage only")
    p.add_argument("--gates", nargs="*", default=[], help="eval stage: both models' gate JSONs")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if os.path.exists(args.out):
        die(f"{args.out} exists; refusing to overwrite")
    try:
        if args.stage == "gate":
            if args.gold:
                die("--gold must not be given to the gate stage (gold-free by design)")
            return run_gate(args)
        if not args.gold:
            die("--gold is required for --stage eval")
        return run_eval(args)
    except zc.ZSError as e:
        die(str(e))


if __name__ == "__main__":
    sys.exit(main())
