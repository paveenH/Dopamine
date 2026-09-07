#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_gsm_symbolic.py — scoring + statistics for the GSM-Symbolic formal sweep.

Reads the JSON files written by get_answer_gsm_symbolic.py (one per
model x gsm_config x alpha), reproduces accuracy via the SAME frozen
utils.extract_gsm8k_answer / utils.is_correct_gsm8k (never re-derives), and
reports:

  - PRIMARY: pooled (main+p1+p2 combined) paired CLUSTER BOOTSTRAP vs that
    model's own alpha=0, Holm correction over the 3 non-zero alphas (m=3).
    This is the ONLY family whose result carries an "established_workpoint"
    verdict. NEVER pooled with any other task's Holm family, and NEVER
    pooled across models.

    Why cluster bootstrap and not exact McNemar on ~8819 pooled rows: p1/p2
    each re-instantiate the SAME original_id multiple times with different
    symbolic values (p1 ~5000 rows over far fewer distinct original_id,
    p2 ~2500 similarly) -- rows sharing an original_id are NOT independent
    draws, they are correlated re-samples of one underlying template/model
    error mode. Treating all ~8819 pooled rows as independent Bernoulli
    trials (what exact McNemar assumes) manufactures pseudo-replication and
    inflates significance. The resampling unit is therefore ORIGINAL_ID, not
    the row, and pooling is done by CONFIG-EQUAL WEIGHTING (each of
    main/p1/p2 contributes one equally-weighted stratum), not by raw row
    count -- otherwise p1 (the largest config by rows) would dominate a
    pooled result that is supposed to summarize three configs, not p1 alone.

  - SENSITIVITY (per row, NOT the primary evidence): exact McNemar on the
    pooled rows and on each config separately, reported explicitly labeled
    as a sensitivity check that ignores clustering -- useful only to see
    whether the cluster-aware and naive-row verdicts diverge, never cited as
    the significance result on its own.
  - DESCRIPTIVE per (model, config): accuracy at each alpha, the same
    per-config cluster bootstrap (own Holm family is NOT run per config --
    see "holm_family" in the output), and per-original_id accuracy mean/std
    so a config's headline number isn't read off one instantiation.
  - a CROSS-CELL CONSISTENCY CHECK before any statistics are computed: every
    cell for a given model must agree on model/model_dir/size/layer band/
    prompt_template(+hash)/cot/role/max_new_tokens/temperature, every
    non-zero-alpha cell's mask_sha256 must match every other non-zero-alpha
    cell of that model, every cell's steering_fires must equal its own
    recorded steering_fires_expected, and every config's 4 alpha cells must
    cover the IDENTICAL sample_id set. Any violation is a hard stop.

No LLM judge anywhere. Alignment is by `sample_id` ("{config}:{id}"), which is
stable because both the loader and generation preserve row order and the
official HF row id.

Usage:
  python eval_gsm_symbolic.py --model llama3 \
      --base_dir /data1/paveen/Dopamine/components \
      --out docs_gsm_symbolic_llama3_evaluation.json \
      --n_bootstrap 10000 --seed 0
"""

import argparse
import json
import math
import os
import random
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

CONSISTENT_META_FIELDS = [
    "model", "model_dir", "size", "prompt_template", "prompt_template_sha256",
    "cot", "role", "max_new_tokens", "temperature", "batch_size",
    "prefill_only", "prefill_tail_len", "n_layers_band",
]


# ─────────────────────── exact McNemar (sensitivity only) ───────────────────

def binom_cdf(k, n, p=0.5):
    """Stdlib exact binomial CDF (no scipy dependency)."""
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return total


def exact_mcnemar(b, c):
    """Two-sided exact McNemar test on discordant pairs b (base wrong, steered
    right) and c (base right, steered wrong). Returns p-value. SENSITIVITY
    ONLY on pooled/multi-instance data -- treats every row as an independent
    trial, which p1/p2's repeated original_id violates."""
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


# ─────────────────────── loading / scoring ───────────────────────

def load_cell(base_dir, model, gsm_config, alpha, size, layers, ans_root="answer_gsm_symbolic"):
    """ans_root MUST match whatever --ans_root the generation launcher
    actually used (default answer_gsm_symbolic; the run this repo's
    launchers currently write wrote to plain "gsm_symbolic" instead --
    pass --ans_root explicitly rather than silently trying multiple
    candidate directories, so it stays visible which directory a cited
    result actually came from)."""
    st, en = layers
    out_dir = os.path.join(base_dir, model, ans_root, gsm_config, f"mdf_{alpha}")
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
    """Row-level paired exact McNemar. SENSITIVITY ONLY when rows share
    original_id clusters (p1/p2, and the pooled set) -- see module docstring."""
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


# ─────────────────────── cluster bootstrap (PRIMARY) ───────────────────────

def build_clusters(base_rows: dict, steer_rows: dict, config_of: dict):
    """Group paired (base, steer) rows by (config, original_id) -- the
    resampling unit. Returns {config: {original_id: [(base_correct,
    steer_correct), ...]}}. Every row must appear in both base_rows and
    steer_rows (same assumption paired_mcnemar_vs_base makes)."""
    assert set(base_rows) == set(steer_rows), (
        "base/steer row sets differ -- cannot pair for a cluster bootstrap"
    )
    clusters = defaultdict(lambda: defaultdict(list))
    for sid in base_rows:
        cfg = config_of[sid]
        oid = base_rows[sid]["original_id"]
        clusters[cfg][oid].append((base_rows[sid]["correct"], steer_rows[sid]["correct"]))
    return clusters


def config_equal_weighted_delta(clusters: dict) -> float:
    """One bootstrap-replicate's point estimate: mean per-config delta
    (steer_acc - base_acc), configs weighted EQUALLY regardless of row/cluster
    count, so a config with more rows (p1) cannot dominate the pooled number."""
    config_deltas = []
    for cfg, by_oid in clusters.items():
        pairs = [pair for pairs in by_oid.values() for pair in pairs]
        if not pairs:
            continue
        base_acc = sum(p[0] for p in pairs) / len(pairs)
        steer_acc = sum(p[1] for p in pairs) / len(pairs)
        config_deltas.append(steer_acc - base_acc)
    if not config_deltas:
        return 0.0
    return sum(config_deltas) / len(config_deltas)


def cluster_bootstrap_vs_base(base_rows: dict, steer_rows: dict, config_of: dict,
                               n_bootstrap: int, rng: random.Random):
    """PRIMARY inference: paired cluster bootstrap, resampling ORIGINAL_ID
    clusters WITHIN each config (never across configs, so each config's own
    original_id population is preserved), then combining configs with EQUAL
    weight per replicate (see config_equal_weighted_delta). Also reports the
    naive row-weighted (pooled-by-row) delta for comparison, but the
    equal-weighted number is what "established_workpoint" is judged on.

    p-value: two-sided, from the fraction of bootstrap replicates whose sign
    disagrees with the observed sign (a standard percentile-bootstrap
    significance proxy) -- reported alongside the 95% percentile CI, which is
    the primary evidence a reader should actually look at.
    """
    clusters = build_clusters(base_rows, steer_rows, config_of)

    # observed (point estimate on the real data, not a bootstrap draw)
    observed_equal = config_equal_weighted_delta(clusters)

    # naive row-weighted pooled accuracy, for comparison only (this is what
    # the OLD pooled-McNemar implicitly assumed, and is exactly what
    # over-weights p1 by row count)
    all_pairs = [pair for by_oid in clusters.values() for pairs in by_oid.values() for pair in pairs]
    n_rows_pooled = len(all_pairs)
    base_acc_row_weighted = sum(p[0] for p in all_pairs) / n_rows_pooled * 100
    steer_acc_row_weighted = sum(p[1] for p in all_pairs) / n_rows_pooled * 100

    # per-config cluster (original_id) lists, frozen once so every bootstrap
    # replicate resamples from the SAME pool
    per_config_oids = {cfg: list(by_oid.keys()) for cfg, by_oid in clusters.items()}
    per_config_oid_pairs = {
        cfg: {oid: pairs for oid, pairs in by_oid.items()}
        for cfg, by_oid in clusters.items()
    }
    n_clusters_by_config = {cfg: len(oids) for cfg, oids in per_config_oids.items()}

    replicate_deltas = []
    for _ in range(n_bootstrap):
        config_deltas_rep = []
        for cfg, oids in per_config_oids.items():
            if not oids:
                continue
            resampled = [rng.choice(oids) for _ in range(len(oids))]
            pairs = [p for oid in resampled for p in per_config_oid_pairs[cfg][oid]]
            if not pairs:
                continue
            base_acc = sum(p[0] for p in pairs) / len(pairs)
            steer_acc = sum(p[1] for p in pairs) / len(pairs)
            config_deltas_rep.append(steer_acc - base_acc)
        replicate_deltas.append(
            sum(config_deltas_rep) / len(config_deltas_rep) if config_deltas_rep else 0.0
        )

    replicate_deltas.sort()
    lo_idx = int(0.025 * n_bootstrap)
    hi_idx = min(n_bootstrap - 1, int(0.975 * n_bootstrap))
    ci_lo = replicate_deltas[lo_idx] * 100
    ci_hi = replicate_deltas[hi_idx] * 100

    # two-sided bootstrap p-value: fraction of replicates on the opposite
    # side of 0 from the observed delta, doubled (standard percentile-based
    # sign-test proxy), floored so 0 reads as "< 1/n_bootstrap" rather than
    # literal zero.
    if observed_equal >= 0:
        opposite = sum(1 for d in replicate_deltas if d < 0)
    else:
        opposite = sum(1 for d in replicate_deltas if d > 0)
    p_boot = min(1.0, 2.0 * opposite / n_bootstrap)
    p_boot = max(p_boot, 1.0 / n_bootstrap)

    return {
        "n_bootstrap": n_bootstrap,
        "n_configs_contributing": len(per_config_oids),
        "n_clusters_by_config": n_clusters_by_config,
        "n_rows_pooled": n_rows_pooled,
        "delta_pp_config_equal_weighted": round(observed_equal * 100, 4),
        "delta_pp_ci95_equal_weighted": [round(ci_lo, 4), round(ci_hi, 4)],
        "p_bootstrap": p_boot,
        "delta_pp_naive_row_weighted": round(steer_acc_row_weighted - base_acc_row_weighted, 4),
        "acc_base_pct_row_weighted": round(base_acc_row_weighted, 4),
        "acc_steer_pct_row_weighted": round(steer_acc_row_weighted, 4),
        "note": (
            "PRIMARY estimand is delta_pp_config_equal_weighted (main/p1/p2 "
            "weighted equally, resampled by original_id cluster WITHIN each "
            "config). delta_pp_naive_row_weighted is shown only to expose how "
            "much p1's larger row count would otherwise dominate a plain "
            "pooled-by-row number -- it is NOT the estimand a workpoint "
            "verdict is based on."
        ),
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
    before it can silently bias a result."""
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


def evaluate_model(model, base_dir, size, n_bootstrap, seed, ans_root="answer_gsm_symbolic"):
    alphas = ALPHAS[model]
    layers = LAYERS[model]
    assert alphas[alphas.index(0)] == 0
    non_zero = [a for a in alphas if a != 0]
    assert len(non_zero) == 3, f"expected 3 non-zero alphas for Holm m=3, got {non_zero}"

    payloads = {}
    for cfg in CONFIGS:
        for a in alphas:
            payloads[(cfg, a)] = load_cell(base_dir, model, cfg, a, size, layers, ans_root=ans_root)

    check_cell_consistency(model, payloads)

    per_config = {}
    pooled_base_rows = {}
    pooled_steer_rows = {a: {} for a in non_zero}
    pooled_config_of = {}

    for cfg in CONFIGS:
        cells = {a: score_cell(payloads[(cfg, a)]) for a in alphas}
        base_rows = cells[0]
        results_sensitivity = {}
        for a in non_zero:
            mc = paired_mcnemar_vs_base(base_rows, cells[a])
            mc["is_improvement"] = mc["delta_pp"] > 0
            mc["note"] = ("SENSITIVITY ONLY -- naive per-row exact McNemar, ignores "
                           "original_id clustering within this config; see the "
                           "PRIMARY pooled cluster bootstrap for the real inference")
            results_sensitivity[a] = mc

        acc_by_alpha = {a: (results_sensitivity[a]["acc_steer_pct"] if a != 0
                             else results_sensitivity[non_zero[0]]["acc_base_pct"])
                         for a in alphas}
        argmax_alpha = max(acc_by_alpha, key=lambda a: acc_by_alpha[a])

        per_config[cfg] = {
            "acc_by_alpha_pct": acc_by_alpha,
            "vs_alpha0_row_level_sensitivity": results_sensitivity,
            "argmax_alpha_numeric_only": argmax_alpha,
            "argmax_note": "raw argmax over accuracy; NOT itself a significance claim",
            "diagnostics_by_alpha": {a: diagnostics_summary(cells[a]) for a in alphas},
            "per_instance_by_alpha": {a: per_instance_breakdown(cells[a]) for a in alphas},
        }

        for sid, r in base_rows.items():
            key = (cfg, sid)
            pooled_base_rows[key] = r
            pooled_config_of[key] = cfg
        for a in non_zero:
            for sid, r in cells[a].items():
                pooled_steer_rows[a][(cfg, sid)] = r

    # ---- SENSITIVITY: naive pooled exact McNemar (row-level, ignores clusters) ----
    pooled_sensitivity = {}
    pooled_sensitivity_pvals = []
    for a in non_zero:
        mc = paired_mcnemar_vs_base(pooled_base_rows, pooled_steer_rows[a])
        mc["note"] = ("SENSITIVITY ONLY -- naive per-row exact McNemar on ~"
                       f"{mc['n']} pooled rows, ignores original_id clustering "
                       "(p1/p2 re-instantiate the same original_id many times) "
                       "and weights configs by row count (p1 has far more rows "
                       "than main/p2) -- NOT the primary evidence.")
        pooled_sensitivity[a] = mc
        pooled_sensitivity_pvals.append(mc["p_raw"])
    pooled_sensitivity_adj = holm_correct(pooled_sensitivity_pvals)
    for a, p_adj in zip(non_zero, pooled_sensitivity_adj):
        pooled_sensitivity[a]["p_holm_adj_SENSITIVITY_ONLY"] = round(p_adj, 6)

    # ---- PRIMARY: config-equal-weighted, original_id cluster bootstrap ----
    rng = random.Random(seed)
    pooled_primary = {}
    pooled_primary_pvals = []
    for a in non_zero:
        cb = cluster_bootstrap_vs_base(
            pooled_base_rows, pooled_steer_rows[a], pooled_config_of,
            n_bootstrap=n_bootstrap, rng=rng,
        )
        pooled_primary[a] = cb
        pooled_primary_pvals.append(cb["p_bootstrap"])
    pooled_primary_adj = holm_correct(pooled_primary_pvals)
    for a, p_adj in zip(non_zero, pooled_primary_adj):
        pooled_primary[a]["p_holm_adj"] = round(p_adj, 6)
        pooled_primary[a]["significant_holm"] = p_adj < 0.05
        pooled_primary[a]["is_improvement"] = pooled_primary[a]["delta_pp_config_equal_weighted"] > 0
        pooled_primary[a]["established_workpoint"] = (
            pooled_primary[a]["significant_holm"] and pooled_primary[a]["is_improvement"]
        )

    pooled_acc_row_weighted = {
        0: pooled_sensitivity[non_zero[0]]["acc_base_pct"],
        **{a: pooled_sensitivity[a]["acc_steer_pct"] for a in non_zero},
    }
    pooled_argmax = max(pooled_acc_row_weighted, key=lambda a: pooled_acc_row_weighted[a])

    return {
        "model": model,
        "size": size,
        "layers": layers,
        "alphas": alphas,
        "holm_family": (
            "PRIMARY confirmatory family = pooled(main+p1+p2) config-equal-"
            "weighted, original_id CLUSTER BOOTSTRAP, m=3 non-zero alphas, "
            "this model only ('pooled_main_p1_p2_PRIMARY_cluster_bootstrap'). "
            "Per-config (main/p1/p2) breakdowns and the naive row-level "
            "pooled McNemar ('pooled_main_p1_p2_SENSITIVITY_naive_row_mcnemar') "
            "are SENSITIVITY/DESCRIPTIVE ONLY -- raw p (or Holm-adjusted for "
            "reference only, labeled SENSITIVITY_ONLY), no "
            "'established_workpoint' verdict outside the primary cluster-"
            "bootstrap family. Never pooled with any other task's Holm "
            "family or across models."
        ),
        "per_config_descriptive": per_config,
        "pooled_main_p1_p2_SENSITIVITY_naive_row_mcnemar": {
            "acc_by_alpha_pct_row_weighted": pooled_acc_row_weighted,
            "vs_alpha0": pooled_sensitivity,
            "argmax_alpha_numeric_only": pooled_argmax,
            "argmax_note": "raw argmax over row-weighted accuracy; NOT itself a significance claim",
        },
        "pooled_main_p1_p2_PRIMARY_cluster_bootstrap": {
            "vs_alpha0": pooled_primary,
            "method": (
                "Paired cluster bootstrap resampling original_id clusters "
                "WITHIN each config (main/p1/p2 kept separate during "
                "resampling), then averaging the three configs' deltas with "
                "EQUAL weight per replicate -- so p1's larger row count "
                "cannot dominate the pooled estimate. p-value is a two-sided "
                "percentile-bootstrap sign proxy; the 95% CI "
                "(delta_pp_ci95_equal_weighted) is the primary evidence to "
                "read, not the p-value alone."
            ),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama3", "qwen2.5"])
    ap.add_argument("--size", default=None)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ans_root", default="answer_gsm_symbolic",
                     help="Must match whatever --ans_root the generation "
                          "launcher actually used. Default matches "
                          "run_gsm_symbolic_formal.sh's own default; pass "
                          "explicitly if a run used a different one (e.g. "
                          "the on-disk dirs are literally 'gsm_symbolic' "
                          "for this project's synced results).")
    args = ap.parse_args()

    size = args.size or ("8B" if args.model == "llama3" else "7B")
    if os.path.exists(args.out):
        raise FileExistsError(f"{args.out} already exists -- refusing to overwrite.")

    result = evaluate_model(args.model, args.base_dir, size, args.n_bootstrap,
                             args.seed, ans_root=args.ans_root)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")

    for cfg in CONFIGS:
        pc = result["per_config_descriptive"][cfg]
        print(f"\n[{cfg}] (descriptive) acc_by_alpha: {pc['acc_by_alpha_pct']}")
        for a, r in pc["vs_alpha0_row_level_sensitivity"].items():
            print(f"  alpha={a:>3}: delta={r['delta_pp']:+.2f}pp  p_raw(row-level, SENSITIVITY)={r['p_raw']:.4g}")

    sens = result["pooled_main_p1_p2_SENSITIVITY_naive_row_mcnemar"]
    print(f"\n[SENSITIVITY naive row-level pooled] acc_by_alpha (row-weighted): {sens['acc_by_alpha_pct_row_weighted']}")
    for a, r in sens["vs_alpha0"].items():
        print(f"  alpha={a:>3}: delta={r['delta_pp']:+.2f}pp  p_raw={r['p_raw']:.4g}  "
              f"p_holm_adj(SENSITIVITY_ONLY)={r['p_holm_adj_SENSITIVITY_ONLY']:.4g}")

    prim = result["pooled_main_p1_p2_PRIMARY_cluster_bootstrap"]
    print(f"\n[PRIMARY config-equal-weighted cluster bootstrap]")
    for a, r in prim["vs_alpha0"].items():
        print(f"  alpha={a:>3}: delta={r['delta_pp_config_equal_weighted']:+.2f}pp  "
              f"CI95={r['delta_pp_ci95_equal_weighted']}  "
              f"p_boot={r['p_bootstrap']:.4g}  p_holm_adj={r['p_holm_adj']:.4g}  "
              f"workpoint={r['established_workpoint']}")


if __name__ == "__main__":
    main()
