#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_gsm_symbolic.py — scoring + statistics for the GSM-Symbolic formal sweep.

Reads the JSON files written by get_answer_gsm_symbolic.py (one per
model x gsm_config x alpha), reproduces accuracy via the SAME frozen
utils.extract_gsm8k_answer / utils.is_correct_gsm8k (never re-derives), and
reports:
  - per (model, config) accuracy at each alpha, paired exact McNemar vs that
    model's own alpha=0, Holm correction over the 3 non-zero alphas (m=3,
    per model, per this experiment's own family -- NOT pooled with any other
    task's Holm family).
  - pooled (main+p1+p2 concatenated) per model, same McNemar/Holm procedure.
  - per-instance accuracy: mean and spread of per-instance accuracy across
    original_id groups, so a config's headline accuracy isn't read off one
    instantiation.
  - a workpoint verdict: an alpha counts as an established workpoint ONLY if
    ITS Holm-adjusted p is significant AND the direction is an INCREASE. The
    raw argmax alpha is reported separately regardless of significance.

No LLM judge anywhere. Alignment is by item `id` within each (config) file,
which is stable because both the loader and generation preserve row order
and id.

Usage:
  python eval_gsm_symbolic.py --model llama3 \
      --base_dir /data1/paveen/Dopamine/components \
      --out docs_gsm_symbolic_llama3_evaluation.json
"""

import argparse
import itertools
import json
import math
import os
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import extract_gsm8k_answer, is_correct_gsm8k  # noqa: E402

CONFIGS = ["main", "p1", "p2"]

ALPHAS = {
    "llama3": [-6, -4, 0, 4],
    "qwen2.5": [-6, 0, 6, 8],
}
LAYERS = {
    "llama3": (11, 20),
    "qwen2.5": (16, 22),
}


def binom_cdf(k, n, p=0.5):
    """Stdlib exact binomial CDF (no scipy dependency)."""
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return total


def exact_mcnemar(b, c):
    """Two-sided exact McNemar test on discordant pairs b (0->1... i.e. base
    wrong, steered right) and c (base right, steered wrong). Returns p-value."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * binom_cdf(k, n, 0.5)
    return min(1.0, p)


def holm_correct(pvals):
    """Holm-Bonferroni step-down correction. pvals: list of raw p-values.
    Returns list of adjusted p-values in the SAME order as input."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        val = min(1.0, pvals[idx] * factor)
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


def load_cell(base_dir, model, gsm_config, alpha, size, layers):
    st, en = layers
    out_dir = os.path.join(base_dir, model, "answer_gsm_symbolic", gsm_config, f"mdf_{alpha}")
    path = os.path.join(out_dir, f"gsm_symbolic_{gsm_config}_{size}_{st}_{en}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing cell: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload


def score_cell(payload):
    """Recompute correctness via the frozen extractor -- never trust the
    generation-time inline `correct` field as MAIN (it is process-state
    provenance, same convention as GSM8K/MATH/GSM-Hard). Returns dict
    id -> (correct: bool, no_answer, multi_marker, first_last_disagree,
    is_loop, truncated, gen_chars)."""
    rows = {}
    for s in payload["data"]:
        pred = extract_gsm8k_answer(s["generated"])
        correct = is_correct_gsm8k(pred, s["answer"])
        rows[s["id"]] = {
            "correct": correct,
            "no_answer": pred == "",
            "multi_marker": s.get("multi_marker", False),
            "first_last_disagree": not s.get("first_last_agree", True),
            "is_loop": s.get("is_loop", False),
            "truncated": s.get("truncated", False),
            "gen_chars": s.get("gen_chars", len(s["generated"])),
            "original_id": s.get("original_id"),
            "instance": s.get("instance"),
        }
    return rows


def paired_mcnemar_vs_base(base_rows, steer_rows):
    ids = sorted(set(base_rows) & set(steer_rows))
    assert len(ids) == len(base_rows) == len(steer_rows), (
        f"id mismatch: base={len(base_rows)} steer={len(steer_rows)} shared={len(ids)}"
    )
    b = c = 0  # b: base wrong steer right; c: base right steer wrong
    n_base_correct = n_steer_correct = 0
    for i in ids:
        br, sr = base_rows[i]["correct"], steer_rows[i]["correct"]
        n_base_correct += br
        n_steer_correct += sr
        if not br and sr:
            b += 1
        elif br and not sr:
            c += 1
    n = len(ids)
    acc_base = n_base_correct / n * 100
    acc_steer = n_steer_correct / n * 100
    p = exact_mcnemar(b, c)
    return {
        "n": n,
        "acc_base_pct": round(acc_base, 4),
        "acc_steer_pct": round(acc_steer, 4),
        "delta_pp": round(acc_steer - acc_base, 4),
        "discordant_b_base_wrong_steer_right": b,
        "discordant_c_base_right_steer_wrong": c,
        "p_raw": p,
    }


def per_instance_breakdown(rows: dict):
    """Group by original_id, compute per-original_id accuracy across its
    instances, then report mean/std across original_id groups (so a config's
    headline number is not dominated by one instantiation)."""
    by_orig = defaultdict(list)
    for r in rows.values():
        by_orig[r["original_id"]].append(1.0 if r["correct"] else 0.0)
    per_orig_acc = [sum(v) / len(v) for v in by_orig.values() if v]
    if not per_orig_acc:
        return {"n_original_ids": 0, "mean_pct": None, "std_pct": None}
    mean = sum(per_orig_acc) / len(per_orig_acc)
    var = sum((x - mean) ** 2 for x in per_orig_acc) / len(per_orig_acc)
    std = math.sqrt(var)
    return {
        "n_original_ids": len(per_orig_acc),
        "mean_pct": round(mean * 100, 4),
        "std_pct": round(std * 100, 4),
    }


def diagnostics_summary(rows: dict):
    n = len(rows)
    if n == 0:
        return {}
    return {
        "no_answer_rate": round(sum(r["no_answer"] for r in rows.values()) / n, 4),
        "multi_marker_rate": round(sum(r["multi_marker"] for r in rows.values()) / n, 4),
        "first_last_disagree_rate": round(sum(r["first_last_disagree"] for r in rows.values()) / n, 4),
        "loop_rate": round(sum(r["is_loop"] for r in rows.values()) / n, 4),
        "truncated_rate": round(sum(r["truncated"] for r in rows.values()) / n, 4),
        "gen_chars_median": sorted(r["gen_chars"] for r in rows.values())[n // 2],
    }


def evaluate_model(model, base_dir, size):
    alphas = ALPHAS[model]
    layers = LAYERS[model]
    assert alphas[alphas.index(0)] == 0
    non_zero = [a for a in alphas if a != 0]
    assert len(non_zero) == 3, f"expected 3 non-zero alphas for Holm m=3, got {non_zero}"

    per_config = {}
    pooled_base_rows = {}
    pooled_steer_rows = {a: {} for a in non_zero}

    for cfg in CONFIGS:
        cells = {}
        for a in alphas:
            payload = load_cell(base_dir, model, cfg, a, size, layers)
            cells[a] = score_cell(payload)
        base_rows = cells[0]
        results = {}
        pvals = []
        for a in non_zero:
            steer_rows = cells[a]
            mc = paired_mcnemar_vs_base(base_rows, steer_rows)
            results[a] = mc
            pvals.append(mc["p_raw"])
        adj = holm_correct(pvals)
        for a, p_adj in zip(non_zero, adj):
            results[a]["p_holm_adj"] = round(p_adj, 6)
            results[a]["significant_holm"] = p_adj < 0.05
            results[a]["is_improvement"] = results[a]["delta_pp"] > 0
            results[a]["established_workpoint"] = (
                results[a]["significant_holm"] and results[a]["is_improvement"]
            )

        acc_by_alpha = {a: (results[a]["acc_steer_pct"] if a != 0 else results[non_zero[0]]["acc_base_pct"])
                         for a in alphas}
        argmax_alpha = max(acc_by_alpha, key=lambda a: acc_by_alpha[a])

        per_config[cfg] = {
            "acc_by_alpha_pct": acc_by_alpha,
            "vs_alpha0": results,
            "argmax_alpha_numeric_only": argmax_alpha,
            "argmax_note": "raw argmax over accuracy; NOT itself a significance claim",
            "diagnostics_by_alpha": {a: diagnostics_summary(cells[a]) for a in alphas},
            "per_instance_by_alpha": {a: per_instance_breakdown(cells[a]) for a in alphas},
        }

        # accumulate pooled rows, id-namespaced by config to avoid collisions
        for i, r in base_rows.items():
            pooled_base_rows[(cfg, i)] = r
        for a in non_zero:
            for i, r in cells[a].items():
                pooled_steer_rows[a][(cfg, i)] = r

    # pooled (main+p1+p2 concatenated)
    pooled_results = {}
    pooled_pvals = []
    for a in non_zero:
        mc = paired_mcnemar_vs_base(pooled_base_rows, pooled_steer_rows[a])
        pooled_results[a] = mc
        pooled_pvals.append(mc["p_raw"])
    pooled_adj = holm_correct(pooled_pvals)
    for a, p_adj in zip(non_zero, pooled_adj):
        pooled_results[a]["p_holm_adj"] = round(p_adj, 6)
        pooled_results[a]["significant_holm"] = p_adj < 0.05
        pooled_results[a]["is_improvement"] = pooled_results[a]["delta_pp"] > 0
        pooled_results[a]["established_workpoint"] = (
            pooled_results[a]["significant_holm"] and pooled_results[a]["is_improvement"]
        )
    pooled_acc = {0: pooled_results[non_zero[0]]["acc_base_pct"],
                  **{a: pooled_results[a]["acc_steer_pct"] for a in non_zero}}
    pooled_argmax = max(pooled_acc, key=lambda a: pooled_acc[a])

    return {
        "model": model,
        "size": size,
        "layers": layers,
        "alphas": alphas,
        "holm_family": "3 non-zero alphas per model per (config or pooled), "
                        "NEVER pooled across configs into one m=9 family, and "
                        "NEVER pooled across models",
        "per_config": per_config,
        "pooled_main_p1_p2": {
            "acc_by_alpha_pct": pooled_acc,
            "vs_alpha0": pooled_results,
            "argmax_alpha_numeric_only": pooled_argmax,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama3", "qwen2.5"])
    ap.add_argument("--size", default=None)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    size = args.size or ("8B" if args.model == "llama3" else "7B")
    if os.path.exists(args.out):
        raise FileExistsError(f"{args.out} already exists -- refusing to overwrite.")

    result = evaluate_model(args.model, args.base_dir, size)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")

    for cfg in CONFIGS:
        pc = result["per_config"][cfg]
        print(f"\n[{cfg}] acc_by_alpha: {pc['acc_by_alpha_pct']}")
        for a, r in pc["vs_alpha0"].items():
            print(f"  alpha={a:>3}: delta={r['delta_pp']:+.2f}pp  "
                  f"p_raw={r['p_raw']:.4g}  p_holm_adj={r['p_holm_adj']:.4g}  "
                  f"workpoint={r['established_workpoint']}")
    pooled = result["pooled_main_p1_p2"]
    print(f"\n[pooled main+p1+p2] acc_by_alpha: {pooled['acc_by_alpha_pct']}")
    for a, r in pooled["vs_alpha0"].items():
        print(f"  alpha={a:>3}: delta={r['delta_pp']:+.2f}pp  "
              f"p_raw={r['p_raw']:.4g}  p_holm_adj={r['p_holm_adj']:.4g}  "
              f"workpoint={r['established_workpoint']}")


if __name__ == "__main__":
    main()
