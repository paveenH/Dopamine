#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProofWriter OWA scoring. The ONLY script that reads gold. Protocol
`proofwriter-owa-v0`.

PRIMARY METRIC: Exact Label Accuracy against official True/False/Unknown
gold (proofwriter_owa/answer_parser.py, last-strict-marker, fail-closed).

ALSO REPORTED: D3 accuracy, D5 accuracy, per-label accuracy, parse-failure
rate, no-answer rate, invalid/multiple-final-answer rate, loop rate,
truncation rate, generation token length.

STATISTICS: each non-zero alpha vs that SAME model's own alpha=0. Exact
two-sided McNemar; Holm correction with m=3 PER MODEL (three non-zero alpha).
D3/D5 and per-label breakdowns are EXPLORATORY SUBGROUPS by default, reported
in full but not pooled into the primary Holm family. Confidence intervals
(when reported) are question-paired bootstrap. No dropping of unfavorable
depths or labels anywhere in this script.

WORKPOINT REPORTING: discrete argmax over the sampled alpha (ties -> smaller
|alpha|); near-optimal region defined only over sampled points, kept separate
from Holm significance. If no non-zero alpha survives Holm against alpha=0,
the frozen sentence below is used verbatim and no workpoint is declared.

THIS IS PROOFWRITER'S OWN TASK-SPECIFIC DOSE EXPLORATION -- there is no
"frozen GSM8K workpoint" being read here; every alpha in the sampled set is a
candidate.

@author: proofwriter_owa task
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from answer_parser import parse_final_answer, normalize_label, is_correct  # noqa: E402
from commitment import per_sample_commitment, aggregate_commitment  # noqa: E402
from scoring import mcnemar_exact, holm, question_paired_bootstrap_ci, \
    discrete_argmax, near_optimal_region  # noqa: E402

PROTOCOL = "proofwriter-owa-v0"
LABELS = ("True", "False", "Unknown")
NO_WORKPOINT_SENTENCE = (
    "No effective ProofWriter OWA workpoint was detected in the sampled dose set."
)

# A strict, GSM8K-loop-detector-style rule: the final 40-char block of the
# continuation recurring >=4 times in the full text. Matches the convention
# `analyze_loop_anxiety.py` / P4b's behaviour-panel note: a permissive n-gram
# proxy gives false positives on ordinary restatement.
def is_loop(text: str, tail_len: int = 40, min_repeats: int = 4) -> bool:
    text = text.strip()
    if len(text) < tail_len:
        return False
    tail = text[-tail_len:]
    return text.count(tail) >= min_repeats


def die(m):
    print(f"[FATAL] {m}", file=sys.stderr)
    raise SystemExit(2)


def load_gold(path):
    d = json.load(open(path, encoding="utf-8"))
    meta = d["meta"]
    if meta.get("contains_labels") is not True:
        die("gold file does not declare contains_labels=true; point --gold "
            "at manifest_gold.json")
    gold = {}
    for r in d["data"]:
        gold[r["sample_id"]] = {
            "answer": normalize_label(r["answer"]),
            "dataset": r["dataset"],
        }
    return meta, gold


def load_cell(path, protocol_expected=PROTOCOL):
    d = json.load(open(path, encoding="utf-8"))
    m = d["meta"]
    if m.get("protocol") != protocol_expected:
        die(f"{path}: protocol {m.get('protocol')!r} != {protocol_expected!r}")
    if m.get("accuracy_computed") is not False:
        die(f"{path}: generation file claims accuracy was already computed")
    return m, d["data"]


def score_cell(rows: list[dict], gold: dict):
    """Returns per-sample records with parsed label / correctness / commitment
    fields, plus dataset-level summaries."""
    out = []
    for r in rows:
        sid = r["sample_id"]
        if sid not in gold:
            die(f"sample_id {sid} in generation file has no gold entry")
        g = gold[sid]
        text = r["generated"]
        parsed = parse_final_answer(text)
        correct = is_correct(parsed.label, g["answer"])
        # pre_answer_reasoning_tokens needs a real tokenizer; this evaluator
        # runs offline with none loaded. get_answer_proofwriter_owa.py
        # computes it at generation time (the tokenizer is already loaded
        # there) and stores it per row as "pre_answer_reasoning_tokens" --
        # pass that through so the field is not silently None in every cell.
        commit = per_sample_commitment(
            text, precomputed_tokens=r.get("pre_answer_reasoning_tokens"))
        out.append({
            "sample_id": sid, "dataset": g["dataset"], "gold": g["answer"],
            "pred": parsed.label, "correct": correct,
            "parse_status": parsed.status,
            "n_strict_markers": parsed.n_strict_markers,
            "n_loose_markers": parsed.n_loose_markers,
            "truncated": r.get("truncated"),
            "generated_token_count": r.get("generated_token_count"),
            "loop": is_loop(text),
            "commit": commit,
        })
    return out


def summarize(scored: list[dict]):
    n = len(scored)
    acc = sum(1 for r in scored if r["correct"]) / n if n else None
    by_ds = {}
    for ds in ("D3", "D5"):
        rs = [r for r in scored if r["dataset"] == ds]
        by_ds[ds] = {"n": len(rs),
                    "accuracy": (sum(1 for r in rs if r["correct"]) / len(rs)
                                 if rs else None)}
    by_label = {}
    for lab in LABELS:
        rs = [r for r in scored if r["gold"] == lab]
        by_label[lab] = {"n": len(rs),
                         "accuracy": (sum(1 for r in rs if r["correct"]) / len(rs)
                                      if rs else None)}
    n_parse_fail = sum(1 for r in scored if r["parse_status"] != "ok")
    n_no_answer = sum(1 for r in scored if r["n_strict_markers"] == 0)
    n_multi = sum(1 for r in scored if r["n_strict_markers"] > 1
                  or r["n_loose_markers"] > 1)
    n_loop = sum(1 for r in scored if r["loop"])
    n_trunc = sum(1 for r in scored if r["truncated"])
    lens = sorted(r["generated_token_count"] for r in scored
                  if r["generated_token_count"] is not None)

    return {
        "n": n, "accuracy": acc,
        "by_dataset": by_ds, "by_label": by_label,
        "parse_failure_rate": n_parse_fail / n if n else None,
        "no_answer_rate": n_no_answer / n if n else None,
        "invalid_or_multiple_marker_rate": n_multi / n if n else None,
        "loop_rate": n_loop / n if n else None,
        "truncation_rate": n_trunc / n if n else None,
        "gen_token_len_median": lens[len(lens) // 2] if lens else None,
        "gen_token_len_mean": sum(lens) / len(lens) if lens else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="manifest_gold.json")
    ap.add_argument("--generations", nargs="+", required=True,
                    help="cell JSONs written by get_answer_proofwriter_owa.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holm_m", type=int, default=3,
                    help="Holm family size per model (default 3 = the three "
                         "non-zero alpha in this task's own dose set)")
    a = ap.parse_args()

    if os.path.exists(a.out):
        die(f"{a.out} exists; refusing to overwrite")

    gmeta, gold = load_gold(a.gold)
    N = len(gold)

    cells = {}   # model -> alpha -> (meta, rows)
    for p in a.generations:
        m, rows = load_cell(p)
        mdl, al = m["model"], m["alpha"]
        # A cell's row count must equal N == len(gold) EXACTLY. This is not a
        # formal-vs-preflight distinction: N is whatever --gold file was
        # actually passed (the 300-row manifest_gold.json for a formal run,
        # or a smaller preflight/pilot gold subset for those stages), so
        # requiring an exact match against N already tolerates a legitimately
        # smaller gold file without a special case. The previous check --
        # `len(rows) not in (N, len([1 for _ in rows]))` -- is a tautology
        # (the second tuple element is always len(rows) itself, computed a
        # different way), so it could never be false and the `pass` branch
        # ran unconditionally: no cell was ever rejected for a short row
        # count.
        if len(rows) != N:
            die(f"{p}: {len(rows)} rows, but --gold has {N} entries; "
                f"a cell must have exactly one row per gold item.")
        ids_here = [r["sample_id"] for r in rows]
        dupes = {i for i in ids_here if ids_here.count(i) > 1}
        if dupes:
            die(f"{p}: duplicate sample_id(s) within this cell: "
                f"{sorted(dupes)[:5]}; every gold item must appear exactly "
                "once, or downstream per-id lookups (scored_by_alpha[al][i]) "
                "would silently pick one arbitrary duplicate.")
        ids_here = set(ids_here)
        missing = ids_here - set(gold)
        if missing:
            die(f"{p}: {len(missing)} sample_id(s) not present in gold, "
                f"e.g. {sorted(missing)[:5]}")
        exp_fires = 0 if al == 0 else m["L"] * len(rows)
        if m.get("steering_fires") != exp_fires:
            die(f"{p}: steering_fires {m.get('steering_fires')} != {exp_fires}; "
                "intervention unverified")
        if al in cells.get(mdl, {}):
            die(f"{mdl} alpha={al} supplied twice")
        cells.setdefault(mdl, {})[al] = (m, rows)

    results = {}
    for mdl, byalpha in sorted(cells.items()):
        if 0 not in byalpha:
            die(f"{mdl}: no alpha=0 cell; it is the baseline for every "
                "comparison in this evaluator")
        # common sample_id set for THIS model across its own cells (formal
        # sweep uses the full manifest; preflight/pilot use a fixed subset --
        # either way every cell of one model must share exactly the same ids)
        id_sets = [set(r["sample_id"] for r in rows) for _, rows in byalpha.values()]
        if any(s != id_sets[0] for s in id_sets):
            die(f"{mdl}: cells do not all share the same sample_id set; "
                "cannot pair them")
        ids_sorted = sorted(id_sets[0])

        scored_by_alpha = {al: {r["sample_id"]: r for r in score_cell(rows, gold)}
                            for al, (_, rows) in byalpha.items()}

        def acc_vec(al):
            s = scored_by_alpha[al]
            return [1 if s[i]["correct"] else 0 for i in ids_sorted]

        acc0 = acc_vec(0)
        model_res = {
            "alphas_present": sorted(byalpha),
            "n_items": len(ids_sorted),
            "cells": {},
        }
        for al in sorted(byalpha):
            rows_scored = [scored_by_alpha[al][i] for i in ids_sorted]
            model_res["cells"][str(al)] = summarize(rows_scored)
            model_res["cells"][str(al)]["commitment"] = aggregate_commitment(
                [r["commit"] for r in rows_scored])

        non_zero = [al for al in byalpha if al != 0]
        pair_results = {}
        for al in non_zero:
            accA = acc_vec(al)
            b01, b10, p = mcnemar_exact(acc0, accA)
            lo, hi = question_paired_bootstrap_ci(acc0, accA)
            pair_results[al] = {
                "alpha": al,
                "acc_base": sum(acc0) / len(acc0),
                "acc_alpha": sum(accA) / len(accA),
                "dAcc_pp": (sum(accA) - sum(acc0)) / len(acc0) * 100,
                "discordant_0to1": b01, "discordant_1to0": b10,
                "p_raw": p, "ci95_pp": [lo, hi],
            }
        holm_pairs = [(al, r["p_raw"]) for al, r in pair_results.items()]
        adj = holm(holm_pairs) if holm_pairs else {}
        for al in pair_results:
            pair_results[al]["p_holm_adj"] = adj.get(al)
            pair_results[al]["holm_family_m"] = a.holm_m

        model_res["mcnemar_vs_alpha0"] = {str(al): v for al, v in pair_results.items()}

        # ---- exploratory D3/D5 and per-label subgroup McNemar (NOT pooled
        # into the primary Holm family above; reported in full)
        subgroup = {}
        for al in non_zero:
            subgroup[str(al)] = {}
            for ds in ("D3", "D5"):
                ids_ds = [i for i in ids_sorted if scored_by_alpha[0][i]["dataset"] == ds]
                if not ids_ds:
                    continue
                a0 = [1 if scored_by_alpha[0][i]["correct"] else 0 for i in ids_ds]
                aA = [1 if scored_by_alpha[al][i]["correct"] else 0 for i in ids_ds]
                b01, b10, p = mcnemar_exact(a0, aA)
                subgroup[str(al)][ds] = {
                    "n": len(ids_ds), "acc_base": sum(a0) / len(a0),
                    "acc_alpha": sum(aA) / len(aA), "p_raw": p,
                    "discordant_0to1": b01, "discordant_1to0": b10,
                    "scope": "exploratory subgroup, not in the primary Holm family",
                }
            for lab in LABELS:
                ids_lab = [i for i in ids_sorted if scored_by_alpha[0][i]["gold"] == lab]
                if not ids_lab:
                    continue
                a0 = [1 if scored_by_alpha[0][i]["correct"] else 0 for i in ids_lab]
                aA = [1 if scored_by_alpha[al][i]["correct"] else 0 for i in ids_lab]
                b01, b10, p = mcnemar_exact(a0, aA)
                subgroup[str(al)][f"label_{lab}"] = {
                    "n": len(ids_lab), "acc_base": sum(a0) / len(a0),
                    "acc_alpha": sum(aA) / len(aA), "p_raw": p,
                    "discordant_0to1": b01, "discordant_1to0": b10,
                    "scope": "exploratory subgroup, not in the primary Holm family",
                }
        model_res["exploratory_subgroups"] = subgroup

        # ---- workpoint reporting
        alpha_to_acc = {al: pair_results[al]["acc_alpha"] if al in pair_results
                        else sum(acc0) / len(acc0) for al in byalpha}
        argmax_alpha = discrete_argmax(alpha_to_acc)
        region = near_optimal_region(alpha_to_acc, tolerance_pp=0.0)

        # A dose only qualifies as a workpoint candidate if it is BOTH
        # Holm-significant AND an IMPROVEMENT (dAcc_pp > 0) vs alpha=0. The
        # previous `any_holm_pass = any(p_holm_adj < 0.05 ...)` accepted a
        # Holm-significant DEGRADATION just as readily as an improvement, and
        # then reported argmax_alpha (computed independently over raw
        # accuracy) as "the workpoint candidate" without ever checking that
        # argmax_alpha was itself among the Holm-passing doses -- so a
        # significant degradation at one alpha could report an unrelated,
        # non-significant alpha as the verdict, or even report alpha=0 itself
        # if every non-zero dose happened to score lower.
        qualifying = [al for al, v in pair_results.items()
                     if v["p_holm_adj"] is not None and v["p_holm_adj"] < 0.05
                     and v["dAcc_pp"] > 0]
        any_holm_pass_either_direction = any(
            v["p_holm_adj"] is not None and v["p_holm_adj"] < 0.05
            for v in pair_results.values())
        if qualifying:
            # Highest accuracy among the qualifying set; ties -> smaller
            # |alpha| (matches zebralogic/eval_zebralogic.py's convention).
            workpoint_alpha = max(qualifying,
                                  key=lambda al: (alpha_to_acc[al], -abs(al)))
        else:
            workpoint_alpha = None
        model_res["workpoint"] = {
            "alpha_to_accuracy": alpha_to_acc,
            "discrete_argmax_alpha": argmax_alpha,
            "near_optimal_region_exact_ties": region,
            "any_dose_holm_significant_either_direction": any_holm_pass_either_direction,
            "holm_significant_improvement_alphas": qualifying,
            "workpoint_alpha": workpoint_alpha,
            "verdict": (NO_WORKPOINT_SENTENCE if workpoint_alpha is None else
                       f"workpoint candidate alpha={workpoint_alpha} "
                       "(Holm-significant IMPROVEMENT vs this model's "
                       "alpha=0; see mcnemar_vs_alpha0 for the full family)"),
        }

        results[mdl] = model_res

    out = {
        "protocol": PROTOCOL,
        "not_a_transfer_test": True,
        "gold_manifest_sha256_16": gmeta.get("manifest_sha256_16"),
        "n_gold_items": N,
        "labels": list(LABELS),
        "owa_semantics": gmeta.get("owa_semantics"),
        "holm_family_m": a.holm_m,
        "results": results,
        "note": ("D3/D5 and per-label breakdowns are exploratory subgroups, "
                "not pooled into the primary per-model Holm(m=3) family. "
                "Commitment-extractor fields are descriptive co-occurrence "
                "statistics only, never causal mediation evidence."),
    }
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"\n=== PROOFWRITER OWA — task-specific dose exploration ===")
    for mdl, r in sorted(results.items()):
        print(f"\n[{mdl}]  n_items={r['n_items']}  alphas={r['alphas_present']}")
        for al in r["alphas_present"]:
            c = r["cells"][str(al)]
            print(f"  alpha={al:>3}  acc={c['accuracy']:.4f}  "
                  f"D3={c['by_dataset']['D3']['accuracy']}  "
                  f"D5={c['by_dataset']['D5']['accuracy']}  "
                  f"parse_fail={c['parse_failure_rate']:.3f}  "
                  f"trunc={c['truncation_rate']}")
        print(f"  workpoint verdict: {r['workpoint']['verdict']}")

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
