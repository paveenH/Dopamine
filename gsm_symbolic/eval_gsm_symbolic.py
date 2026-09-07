#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_gsm_symbolic.py — scoring + statistics for the GSM-Symbolic formal sweep.

Reads the JSON files written by get_answer_gsm_symbolic.py (one per
model x gsm_config x alpha), reproduces accuracy via the SAME frozen
utils.extract_gsm8k_answer / utils.is_correct_gsm8k (never re-derives), and
reports:

  - PRIMARY: pooled (main+p1+p2 concatenated) accuracy at each alpha, paired
    exact McNemar vs that model's own alpha=0, Holm correction over the 3
    non-zero alphas (m=3). This is the ONLY family whose p-values are
    Holm-adjusted and whose "established_workpoint" verdict is meaningful.
    NEVER pooled with any other task's Holm family, and NEVER pooled across
    models.
  - DESCRIPTIVE per (model, config): accuracy at each alpha and the SAME
    McNemar test vs that model's own alpha=0, but reported with RAW p only
    (explicitly no Holm adjustment, no significance verdict) -- these three
    are exploratory context for the pooled primary result, not three more
    hypothesis tests. (Running Holm separately on main/p1/p2/pooled would be
    four overlapping, non-independent families sharing the same alpha=0
    baseline and largely the same rows; that is avoided by making pooled the
    one confirmatory family.)
  - per-instance accuracy: mean and spread of per-instance accuracy across
    original_id groups, so a config's headline accuracy isn't read off one
    instantiation.
  - a CROSS-CELL CONSISTENCY CHECK before any statistics are computed: every
    cell for a given model must agree on model/model_dir/size/layer band/
    prompt_template(+hash)/cot/role/max_new_tokens/temperature, and every
    non-zero-alpha cell's mask_sha256 must match that of the SAME model's
    other non-zero cells that share a layer band (mask content is alpha-
    independent -- only the alpha scalar multiplies it), and every cell's
    steering_fires must equal its own recorded steering_fires_expected. A
    config's item id set (by sample_id) must be IDENTICAL across all 4 alpha
    cells of that config. Any violation is a hard stop, not a warning.

No LLM judge anywhere. Alignment is by `sample_id` ("{config}:{id}"), which is
stable because both the loader and generation preserve row order and the
official HF row id.

Usage:
  python eval_gsm_symbolic.py --model llama3 \
      --base_dir /data1/paveen/Dopamine/components \
      --out docs_gsm_symbolic_llama3_evaluation.json
"""

import argparse
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

# Fields every cell of one model must agree on (checked before any statistics
# are computed). alpha/layer_start/layer_end/gsm_config/mask_sha256/
# steering_fires* are checked separately since they legitimately vary by cell.
CONSISTENT_META_FIELDS = [
    "model", "model_dir", "size", "prompt_template", "prompt_template_sha256",
    "cot", "role", "max_new_tokens", "temperature", "batch_size",
    "prefill_only", "prefill_tail_len", "n_layers_band",
]


def binom_cdf(k, n, p=0.5):
    """Stdlib exact binomial CDF (no scipy dependency)."""
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return total


def exact_mcnemar(b, c):
    """Two-sided exact McNemar test on discordant pairs b (base wrong, steered
    right) and c (base right, steered wrong). Returns p-value."""
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
    provenance, same convention as GSM8K/MATH/GSM-Hard). Keyed by sample_id
    (stable across configs; raw HF `id` alone repeats across configs)."""
    rows = {}
    for s in payload["data"]:
        pred = extract_gsm8k_answer(s["generated"])
        correct = is_correct_gsm8k(pred, s["answer"])
        has_marker = "####" in s["generated"]
        sid = s.get("sample_id")
        if sid is None:
            raise KeyError(
                "sample missing 'sample_id' -- this cell predates the loader "
                "fix that preserves the official HF row id; regenerate it."
            )
        rows[sid] = {
            "correct": correct,
            "no_answer": pred == "",
            "no_marker": s.get("no_marker", not has_marker),
            "marker_unparsed": s.get("marker_unparsed", False),
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
        "no_marker_rate": round(sum(r["no_marker"] for r in rows.values()) / n, 4),
        "marker_unparsed_rate": round(sum(r["marker_unparsed"] for r in rows.values()) / n, 4),
        "no_answer_rate": round(sum(r["no_answer"] for r in rows.values()) / n, 4),
        "multi_marker_rate": round(sum(r["multi_marker"] for r in rows.values()) / n, 4),
        "first_last_disagree_rate": round(sum(r["first_last_disagree"] for r in rows.values()) / n, 4),
        "loop_rate": round(sum(r["is_loop"] for r in rows.values()) / n, 4),
        "truncated_rate": round(sum(r["truncated"] for r in rows.values()) / n, 4),
        "gen_chars_median": sorted(r["gen_chars"] for r in rows.values())[n // 2],
    }


def check_cell_consistency(model, payloads: dict):
    """payloads: {(gsm_config, alpha): payload}. Hard-stops on any mismatch.
    Runs BEFORE any statistics are computed, so a wiring problem is caught
    before it can silently bias a p-value."""
    metas = {k: p["meta"] for k, p in payloads.items()}
    ref_key = next(iter(metas))
    ref_meta = metas[ref_key]

    for key, meta in metas.items():
        for field in CONSISTENT_META_FIELDS:
            if meta.get(field) != ref_meta.get(field):
                raise ValueError(
                    f"cell {key} disagrees with {ref_key} on meta field "
                    f"'{field}': {meta.get(field)!r} != {ref_meta.get(field)!r}. "
                    "Cells for one model must share model/prompt/generation "
                    "config; stopping before computing any statistics."
                )
        if meta["model"] != model:
            raise ValueError(f"cell {key} meta['model']={meta['model']!r} != requested {model!r}")
        cfg, alpha = key
        if meta["gsm_config"] != cfg:
            raise ValueError(f"cell {key} meta['gsm_config']={meta['gsm_config']!r} != {cfg!r}")
        if meta["alpha"] != alpha:
            raise ValueError(f"cell {key} meta['alpha']={meta['alpha']!r} != {alpha!r}")
        expected_fires = meta.get("steering_fires_expected")
        actual_fires = meta.get("steering_fires")
        if expected_fires is not None and actual_fires != expected_fires:
            raise ValueError(
                f"cell {key}: steering_fires={actual_fires} != "
                f"steering_fires_expected={expected_fires} recorded at generation "
                "time -- this cell's injection did not fire as expected."
            )
        if alpha == 0 and actual_fires not in (0, None):
            raise ValueError(f"cell {key}: alpha=0 but steering_fires={actual_fires} (expected 0)")

    # mask_sha256 must be identical across ALL non-zero-alpha cells of one
    # model (the mask content is alpha-independent; only the scalar multiply
    # differs) -- a mismatch means two cells silently used different masks
    # (wrong band, stale file, or a mid-experiment mask edit).
    non_zero_masks = {
        key: meta.get("mask_sha256")
        for key, meta in metas.items()
        if key[1] != 0 and meta.get("mask_sha256") is not None
    }
    if non_zero_masks:
        ref_mask = next(iter(non_zero_masks.values()))
        for key, h in non_zero_masks.items():
            if h != ref_mask:
                raise ValueError(
                    f"cell {key} mask_sha256={h} differs from other non-zero-alpha "
                    f"cells of this model (expected {ref_mask}) -- inconsistent mask."
                )

    # every config's 4 alpha cells must cover the IDENTICAL sample_id set
    by_config = defaultdict(dict)
    for (cfg, alpha), payload in payloads.items():
        ids = {s["sample_id"] for s in payload["data"] if s.get("sample_id") is not None}
        by_config[cfg][alpha] = ids
    for cfg, by_alpha in by_config.items():
        ref_alpha = next(iter(by_alpha))
        ref_ids = by_alpha[ref_alpha]
        for alpha, ids in by_alpha.items():
            if ids != ref_ids:
                sym = ids.symmetric_difference(ref_ids)
                raise ValueError(
                    f"config={cfg} alpha={alpha} sample_id set differs from "
                    f"alpha={ref_alpha} ({len(sym)} ids differ) -- cells of the "
                    "same config must run on identical items."
                )


def evaluate_model(model, base_dir, size):
    alphas = ALPHAS[model]
    layers = LAYERS[model]
    assert alphas[alphas.index(0)] == 0
    non_zero = [a for a in alphas if a != 0]
    assert len(non_zero) == 3, f"expected 3 non-zero alphas for Holm m=3, got {non_zero}"

    payloads = {}
    for cfg in CONFIGS:
        for a in alphas:
            payloads[(cfg, a)] = load_cell(base_dir, model, cfg, a, size, layers)

    check_cell_consistency(model, payloads)

    per_config = {}
    pooled_base_rows = {}
    pooled_steer_rows = {a: {} for a in non_zero}

    for cfg in CONFIGS:
        cells = {a: score_cell(payloads[(cfg, a)]) for a in alphas}
        base_rows = cells[0]
        results = {}
        for a in non_zero:
            mc = paired_mcnemar_vs_base(base_rows, cells[a])
            mc["is_improvement"] = mc["delta_pp"] > 0
            mc["note"] = ("DESCRIPTIVE ONLY -- raw p, not Holm-adjusted; see the "
                           "pooled_main_p1_p2 block for the primary confirmatory test")
            results[a] = mc

        acc_by_alpha = {a: (results[a]["acc_steer_pct"] if a != 0 else results[non_zero[0]]["acc_base_pct"])
                         for a in alphas}
        argmax_alpha = max(acc_by_alpha, key=lambda a: acc_by_alpha[a])

        per_config[cfg] = {
            "acc_by_alpha_pct": acc_by_alpha,
            "vs_alpha0_descriptive": results,
            "argmax_alpha_numeric_only": argmax_alpha,
            "argmax_note": "raw argmax over accuracy; NOT itself a significance claim",
            "diagnostics_by_alpha": {a: diagnostics_summary(cells[a]) for a in alphas},
            "per_instance_by_alpha": {a: per_instance_breakdown(cells[a]) for a in alphas},
        }

        # accumulate pooled rows, config-namespaced by sample_id (already
        # unique across configs, but namespacing here too costs nothing and
        # removes any doubt).
        for sid, r in base_rows.items():
            pooled_base_rows[(cfg, sid)] = r
        for a in non_zero:
            for sid, r in cells[a].items():
                pooled_steer_rows[a][(cfg, sid)] = r

    # ---- PRIMARY: pooled (main+p1+p2 concatenated), Holm m=3 ----
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
        "holm_family": (
            "PRIMARY confirmatory family = pooled(main+p1+p2), m=3 non-zero "
            "alphas, this model only. Per-config (main/p1/p2) results are "
            "DESCRIPTIVE -- raw p only, no Holm adjustment, no "
            "'established_workpoint' verdict -- to avoid four overlapping, "
            "non-independent test families sharing one alpha=0 baseline. "
            "Never pooled with any other task's Holm family or across models."
        ),
        "per_config_descriptive": per_config,
        "pooled_main_p1_p2_PRIMARY": {
            "acc_by_alpha_pct": pooled_acc,
            "vs_alpha0": pooled_results,
            "argmax_alpha_numeric_only": pooled_argmax,
            "argmax_note": "raw argmax over accuracy; NOT itself a significance claim",
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
        pc = result["per_config_descriptive"][cfg]
        print(f"\n[{cfg}] (descriptive) acc_by_alpha: {pc['acc_by_alpha_pct']}")
        for a, r in pc["vs_alpha0_descriptive"].items():
            print(f"  alpha={a:>3}: delta={r['delta_pp']:+.2f}pp  p_raw={r['p_raw']:.4g}")
    pooled = result["pooled_main_p1_p2_PRIMARY"]
    print(f"\n[PRIMARY pooled main+p1+p2] acc_by_alpha: {pooled['acc_by_alpha_pct']}")
    for a, r in pooled["vs_alpha0"].items():
        print(f"  alpha={a:>3}: delta={r['delta_pp']:+.2f}pp  "
              f"p_raw={r['p_raw']:.4g}  p_holm_adj={r['p_holm_adj']:.4g}  "
              f"workpoint={r['established_workpoint']}")


if __name__ == "__main__":
    main()
