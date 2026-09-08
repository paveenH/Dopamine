#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis for the SIMPLIFIED Role self-steering (RR) / Confidence
self-steering (CC) MMLU-E experiment, REVISED 2026-09-08 formal dose set
alpha in {0, +2, +4, +6} (RSN-paper-style positive-dose-only comparison;
negative doses -2/-4 are not run this round -- they verify bidirectional
control and are not needed for this round's positive-effect comparison).

Only RR and CC are IN SCOPE conceptually here -- NO RC/CR, NO random control,
NO cross-steering internal comparisons (that matrix is deferred to a later
stage). As of 2026-09-08, RR is NOT ACTUALLY RUN this round (no verified,
protocol-matching historical RR result exists to reuse), so the RR columns
in this analysis's output will be EMPTY/absent unless RR cells are separately
generated later with get_answer_rr_cc_mmlue.py --condition RR. This script
still supports analyzing RR if/when those cells exist -- it does not
hardcode CC-only. This script answers exactly the two questions this stage
asks (the second one only once RR data exists):
  1. Does the original Role mask (RR) reproduce a stable, directionally
     consistent MMLU-E steering effect? (PENDING -- RR not run this round)
  2. Does the new Confidence mask (CC) ALSO produce an independent, causal
     effect on confident/unconfident behavior? (this round's actual scope)

It explicitly does NOT compare RR's and CC's effect MAGNITUDES at the same
raw alpha as a strength claim -- the two masks' raw norms differ and are not
norm-matched in this stage (norm-matching is deferred to the cross-steering
stage). Only DIRECTION CONSISTENCY and PRESENCE of a causal effect are
assessed here.

Reads the per-task JSON files written by get_answer_rr_cc_mmlue.py under
{results_dir}/{condition}/alpha_{a}/{task}_8B_answers.json. Question identity
for pairing is (task, sample_index_within_task).

Reports, per (condition, alpha, role):
  - sample-level MICRO accuracy / E-rate / wrong_non_E-rate
  - task-level MACRO-average accuracy / E-rate (PRIMARY, matches the
    historical "Average of Tasks" convention; micro is SUPPLEMENTARY)
  - four-domain (STEM/Humanities/Social Sciences/Other) micro + macro
  - paired question-level change vs the shared alpha=0 baseline (exact sign
    test, sample-level) + task-clustered bootstrap 95% CI on the macro
    metric (resampling 57 tasks with replacement, B=10000)

Does NOT modify get_answer_rr_cc_mmlue.py, any mask, mean, or result file.
Read-only analysis.

Usage: /opt/anaconda3/bin/python analyze_rr_cc_mmlue.py \
    --results_dir /path/to/llama3_confidence/rr_cc_mmlue/results \
    --out_dir     /path/to/llama3_confidence/rr_cc_mmlue/analysis
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for detection.task_list
from mmlu_category_map import TASK_TO_CATEGORY, CATEGORIES  # noqa: E402

CONDITIONS = ["baseline", "RR", "CC"]
ROLES = ["confident", "unconfident"]
LABELS = ["A", "B", "C", "D", "E"]
ALPHAS_FORMAL = [0.0, 2.0, 4.0, 6.0]  # revised 2026-09-08: RSN-paper-style
                                       # positive-dose comparison. Negative
                                       # doses (-2, -4) are not run this round.
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20260908

STRENGTH_COMPARISON_CAVEAT = (
    "RR and CC use their own raw NMD masks, NOT norm-matched to each other in this "
    "stage. Do NOT read a larger effect at the same raw alpha as evidence that one "
    "mask is 'stronger' -- strict magnitude comparison is deferred to the "
    "norm-matched cross-steering stage. This stage only assesses (a) whether each "
    "condition's steering is directionally consistent and stable across alpha, and "
    "(b) whether CC has an independent causal effect at all."
)


def load_task_json(results_dir: Path, condition: str, alpha: float, task: str) -> dict:
    alpha_str = f"{alpha:g}"
    p = results_dir / condition / f"alpha_{alpha_str}" / f"{task}_8B_answers.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cell(results_dir: Path, condition: str, alpha: float, tasks: list) -> dict:
    out = {}
    missing = []
    for task in tasks:
        try:
            out[task] = load_task_json(results_dir, condition, alpha, task)
        except FileNotFoundError:
            missing.append(task)
    if missing:
        raise FileNotFoundError(
            f"{condition}/alpha_{alpha:g}: {len(missing)} task file(s) missing: {missing[:5]}..."
        )
    return out


def per_sample_pred(cell_data: dict, role: str) -> dict:
    role_key = role.replace(" ", "_")
    out = {}
    for task, payload in cell_data.items():
        for idx, sample in enumerate(payload["data"]):
            pred = sample.get(f"answer_{role_key}")
            true_idx = sample.get("label", -1)
            true_lab = LABELS[true_idx] if 0 <= true_idx < len(LABELS) else None
            out[(task, idx)] = (pred, true_lab)
    return out


def flag_correct(pred, true) -> bool:
    return pred is not None and pred == true


def flag_e(pred, _true) -> bool:
    return pred == "E"


def flag_wrong_non_e(pred, true) -> bool:
    return pred is not None and pred != true and pred != "E"


def micro_rates(records: dict) -> dict:
    n = len(records)
    correct = sum(1 for p, t in records.values() if flag_correct(p, t))
    e_count = sum(1 for p, t in records.values() if flag_e(p, t))
    wrong_non_e = sum(1 for p, t in records.values() if flag_wrong_non_e(p, t))
    return {
        "n": n, "correct": correct, "e_count": e_count, "wrong_non_e": wrong_non_e,
        "accuracy": correct / n if n else float("nan"),
        "e_rate": e_count / n if n else float("nan"),
        "wrong_non_e_rate": wrong_non_e / n if n else float("nan"),
    }


def per_task_rates(records: dict, tasks: list) -> dict:
    by_task = {t: {} for t in tasks}
    for (task, idx), val in records.items():
        by_task[task][(task, idx)] = val
    return {t: micro_rates(recs) for t, recs in by_task.items()}


def macro_average(per_task: dict, metric: str) -> float:
    vals = [v[metric] for v in per_task.values() if not np.isnan(v[metric])]
    return float(np.mean(vals)) if vals else float("nan")


def cluster_bootstrap_macro_ci(per_task: dict, metric: str, tasks: list,
                                n_boot: int, rng: np.random.Generator) -> dict:
    task_vals = np.array([per_task[t][metric] for t in tasks], dtype=np.float64)
    valid_mask = ~np.isnan(task_vals)
    task_vals = task_vals[valid_mask]
    n_tasks = len(task_vals)
    if n_tasks == 0:
        return {"point": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_tasks": 0}
    point = float(np.mean(task_vals))
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_tasks, size=n_tasks)
        boot_means[b] = np.mean(task_vals[idx])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return {"point": point, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "n_tasks": n_tasks}


def cluster_bootstrap_paired_diff_ci(per_task_a: dict, per_task_b: dict, metric: str,
                                      tasks: list, n_boot: int, rng: np.random.Generator) -> dict:
    vals_a = np.array([per_task_a[t][metric] for t in tasks], dtype=np.float64)
    vals_b = np.array([per_task_b[t][metric] for t in tasks], dtype=np.float64)
    valid = ~(np.isnan(vals_a) | np.isnan(vals_b))
    vals_a, vals_b = vals_a[valid], vals_b[valid]
    n_tasks = len(vals_a)
    if n_tasks == 0:
        return {"point_diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_tasks": 0}
    point_diff = float(np.mean(vals_a) - np.mean(vals_b))
    boot_diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_tasks, size=n_tasks)
        boot_diffs[b] = np.mean(vals_a[idx]) - np.mean(vals_b[idx])
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    return {"point_diff": point_diff, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "n_tasks": n_tasks}


def exact_sign_test(records_a: dict, records_b: dict, flag_fn) -> dict:
    common = set(records_a) & set(records_b)
    if not common:
        return {"n_paired": 0}
    a_on_b_off = 0
    b_on_a_off = 0
    for k in common:
        fa = flag_fn(*records_a[k])
        fb = flag_fn(*records_b[k])
        if fa and not fb:
            a_on_b_off += 1
        elif fb and not fa:
            b_on_a_off += 1
    n_disc = a_on_b_off + b_on_a_off
    p = float(binomtest(a_on_b_off, n_disc, 0.5).pvalue) if n_disc > 0 else float("nan")
    return {
        "n_paired": len(common), "a_on_b_off": a_on_b_off, "b_on_a_off": b_on_a_off,
        "n_discordant": n_disc, "exact_sign_test_p_value": p,
    }


def domain_per_task(per_task: dict) -> dict:
    out = {c: {} for c in CATEGORIES}
    for task, rates in per_task.items():
        out[TASK_TO_CATEGORY[task]][task] = rates
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "sample_micro": out_dir / "summary_sample_micro.csv",
        "task_macro": out_dir / "summary_task_macro.csv",
        "domain": out_dir / "summary_domain.csv",
        "vs_baseline": out_dir / "paired_vs_baseline.csv",
        "provenance": out_dir / "provenance.json",
        "fig_png": out_dir / "rr_cc_mmlue_main.png",
        "fig_pdf": out_dir / "rr_cc_mmlue_main.pdf",
    }
    for p in paths.values():
        if p.exists():
            raise FileExistsError(f"{p} already exists -- refusing to overwrite.")

    from detection.task_list import TASKS
    tasks = list(TASKS)

    available_cells = []
    for condition in CONDITIONS:
        cond_dir = results_dir / condition
        if not cond_dir.exists():
            continue
        for alpha in ALPHAS_FORMAL:
            if condition == "baseline" and alpha != 0.0:
                continue
            if condition != "baseline" and alpha == 0.0:
                continue
            alpha_dir = cond_dir / f"alpha_{alpha:g}"
            if alpha_dir.exists() and any(alpha_dir.glob("*_answers.json")):
                available_cells.append((condition, alpha))

    if not available_cells:
        raise FileNotFoundError(f"No formal-dose-set cells found under {results_dir}")

    print(f"Found {len(available_cells)} cells: {available_cells}")

    cell_data = {}
    cell_records = {}
    cell_per_task = {}
    for condition, alpha in available_cells:
        cell_data[(condition, alpha)] = load_cell(results_dir, condition, alpha, tasks)
        for role in ROLES:
            recs = per_sample_pred(cell_data[(condition, alpha)], role)
            cell_records[(condition, alpha, role)] = recs
            cell_per_task[(condition, alpha, role)] = per_task_rates(recs, tasks)

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # ---- 1. sample-level micro ----
    micro_rows = []
    for (condition, alpha, role), records in cell_records.items():
        rates = micro_rates(records)
        micro_rows.append({"condition": condition, "alpha": alpha, "role": role, **rates})
    with open(paths["sample_micro"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(micro_rows[0].keys()))
        w.writeheader()
        for r in micro_rows:
            w.writerow(r)

    # ---- 2. task-macro (PRIMARY) + cluster bootstrap ----
    macro_rows = []
    for (condition, alpha, role), per_task in cell_per_task.items():
        acc_ci = cluster_bootstrap_macro_ci(per_task, "accuracy", tasks, args.n_bootstrap, rng)
        e_ci = cluster_bootstrap_macro_ci(per_task, "e_rate", tasks, args.n_bootstrap, rng)
        wne_ci = cluster_bootstrap_macro_ci(per_task, "wrong_non_e_rate", tasks, args.n_bootstrap, rng)
        macro_rows.append({
            "condition": condition, "alpha": alpha, "role": role,
            "macro_accuracy": acc_ci["point"], "macro_accuracy_ci_lo": acc_ci["ci_lo"],
            "macro_accuracy_ci_hi": acc_ci["ci_hi"],
            "macro_e_rate": e_ci["point"], "macro_e_rate_ci_lo": e_ci["ci_lo"],
            "macro_e_rate_ci_hi": e_ci["ci_hi"],
            "macro_wrong_non_e_rate": wne_ci["point"], "macro_wrong_non_e_rate_ci_lo": wne_ci["ci_lo"],
            "macro_wrong_non_e_rate_ci_hi": wne_ci["ci_hi"],
            "n_tasks": acc_ci["n_tasks"],
        })
    with open(paths["task_macro"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(macro_rows[0].keys()))
        w.writeheader()
        for r in macro_rows:
            w.writerow(r)

    # ---- 3. domain summary ----
    domain_rows = []
    for (condition, alpha, role), per_task in cell_per_task.items():
        by_domain = domain_per_task(per_task)
        records = cell_records[(condition, alpha, role)]
        for cat in CATEGORIES:
            cat_tasks = [t for t in tasks if TASK_TO_CATEGORY[t] == cat]
            cat_records = {k: v for k, v in records.items() if k[0] in cat_tasks}
            micro = micro_rates(cat_records)
            macro_acc = macro_average(by_domain[cat], "accuracy")
            macro_e = macro_average(by_domain[cat], "e_rate")
            domain_rows.append({
                "condition": condition, "alpha": alpha, "role": role, "domain": cat,
                "n_tasks_in_domain": len(cat_tasks),
                "micro_accuracy": micro["accuracy"], "micro_e_rate": micro["e_rate"],
                "micro_wrong_non_e_rate": micro["wrong_non_e_rate"],
                "macro_accuracy": macro_acc, "macro_e_rate": macro_e,
            })
    with open(paths["domain"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(domain_rows[0].keys()))
        w.writeheader()
        for r in domain_rows:
            w.writerow(r)

    # ---- 4. vs alpha=0 baseline ----
    baseline_rows = []
    for role in ROLES:
        base_key = ("baseline", 0.0, role)
        if base_key not in cell_records:
            continue
        base_records = cell_records[base_key]
        base_per_task = cell_per_task[base_key]
        for (condition, alpha, r2), records in cell_records.items():
            if r2 != role or condition == "baseline":
                continue
            per_task = cell_per_task[(condition, alpha, role)]
            acc_boot = cluster_bootstrap_paired_diff_ci(per_task, base_per_task, "accuracy",
                                                         tasks, args.n_bootstrap, rng)
            e_boot = cluster_bootstrap_paired_diff_ci(per_task, base_per_task, "e_rate",
                                                       tasks, args.n_bootstrap, rng)
            acc_sign = exact_sign_test(records, base_records, flag_correct)
            e_sign = exact_sign_test(records, base_records, flag_e)
            baseline_rows.append({
                "condition": condition, "alpha": alpha, "role": role, "vs": "baseline_alpha0",
                "macro_accuracy_diff": acc_boot["point_diff"],
                "macro_accuracy_diff_ci_lo": acc_boot["ci_lo"],
                "macro_accuracy_diff_ci_hi": acc_boot["ci_hi"],
                "macro_e_rate_diff": e_boot["point_diff"],
                "macro_e_rate_diff_ci_lo": e_boot["ci_lo"],
                "macro_e_rate_diff_ci_hi": e_boot["ci_hi"],
                "sample_acc_a_on_b_off": acc_sign.get("a_on_b_off"),
                "sample_acc_b_on_a_off": acc_sign.get("b_on_a_off"),
                "sample_acc_sign_test_p": acc_sign.get("exact_sign_test_p_value"),
                "sample_e_a_on_b_off": e_sign.get("a_on_b_off"),
                "sample_e_b_on_a_off": e_sign.get("b_on_a_off"),
                "sample_e_sign_test_p": e_sign.get("exact_sign_test_p_value"),
            })
    if baseline_rows:
        with open(paths["vs_baseline"], "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(baseline_rows[0].keys()))
            w.writeheader()
            for r in baseline_rows:
                w.writerow(r)

    # ---- Figure ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.0,
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.4,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row_i, role in enumerate(ROLES):
        ax_e = axes[row_i, 0]
        ax_acc = axes[row_i, 1]
        for condition in ["RR", "CC"]:
            xs, e_pts, e_lo, e_hi, acc_pts, acc_lo, acc_hi = [], [], [], [], [], [], []
            for alpha in ALPHAS_FORMAL:
                key = (condition, alpha, role)
                if key not in cell_per_task:
                    continue
                acc_ci = cluster_bootstrap_macro_ci(cell_per_task[key], "accuracy", tasks, 2000, rng)
                e_ci = cluster_bootstrap_macro_ci(cell_per_task[key], "e_rate", tasks, 2000, rng)
                xs.append(alpha)
                e_pts.append(e_ci["point"]); e_lo.append(e_ci["ci_lo"]); e_hi.append(e_ci["ci_hi"])
                acc_pts.append(acc_ci["point"]); acc_lo.append(acc_ci["ci_lo"]); acc_hi.append(acc_ci["ci_hi"])
            base_key = ("baseline", 0.0, role)
            if base_key in cell_per_task:
                acc_ci0 = cluster_bootstrap_macro_ci(cell_per_task[base_key], "accuracy", tasks, 2000, rng)
                e_ci0 = cluster_bootstrap_macro_ci(cell_per_task[base_key], "e_rate", tasks, 2000, rng)
                xs = [0.0] + xs
                e_pts = [e_ci0["point"]] + e_pts; e_lo = [e_ci0["ci_lo"]] + e_lo; e_hi = [e_ci0["ci_hi"]] + e_hi
                acc_pts = [acc_ci0["point"]] + acc_pts; acc_lo = [acc_ci0["ci_lo"]] + acc_lo; acc_hi = [acc_ci0["ci_hi"]] + acc_hi
                order = np.argsort(xs)
                xs = list(np.array(xs)[order])
                e_pts = list(np.array(e_pts)[order]); e_lo = list(np.array(e_lo)[order]); e_hi = list(np.array(e_hi)[order])
                acc_pts = list(np.array(acc_pts)[order]); acc_lo = list(np.array(acc_lo)[order]); acc_hi = list(np.array(acc_hi)[order])
            if xs:
                ax_e.plot(xs, e_pts, marker="o", label=condition)
                ax_e.fill_between(xs, e_lo, e_hi, alpha=0.15)
                ax_acc.plot(xs, acc_pts, marker="o", label=condition)
                ax_acc.fill_between(xs, acc_lo, acc_hi, alpha=0.15)
        ax_e.set_title(f"Macro E-rate ({role}), 95% cluster-bootstrap CI")
        ax_e.set_xlabel("alpha"); ax_e.set_ylabel("E-rate")
        ax_e.legend(frameon=False, fontsize=8)
        ax_e.spines["top"].set_visible(False); ax_e.spines["right"].set_visible(False)

        ax_acc.set_title(f"Macro Accuracy ({role}), 95% cluster-bootstrap CI")
        ax_acc.set_xlabel("alpha"); ax_acc.set_ylabel("Accuracy")
        ax_acc.legend(frameon=False, fontsize=8)
        ax_acc.spines["top"].set_visible(False); ax_acc.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(paths["fig_png"])
    fig.savefig(paths["fig_pdf"])
    plt.close(fig)

    # ---- Provenance ----
    provenance = {
        "results_dir": str(results_dir),
        "available_cells": [(c, a) for c, a in available_cells],
        "formal_dose_set": ALPHAS_FORMAL,
        "roles": ROLES,
        "categories": CATEGORIES,
        "primary_summary": "task_macro (matches the historical 'Average of Tasks' convention). "
                            "sample_micro is supplementary.",
        "field_naming": "'wrong_non_E' replaces the old 'invalid' name.",
        "bootstrap": {
            "method": "task-clustered (resample the 57 tasks with replacement), percentile 95% CI",
            "n_bootstrap": args.n_bootstrap, "seed": BOOTSTRAP_SEED,
        },
        "scope_note": "SIMPLIFIED stage: RR and CC only, no RC/CR, no random control, no "
                       "norm-matching, no cross-steering internal comparisons. This is "
                       "explicitly NOT the cross-steering matrix -- it only asks whether "
                       "each mask independently reproduces a directionally consistent, "
                       "stable causal effect on MMLU-E confident/unconfident behavior.",
        "strength_comparison_caveat": STRENGTH_COMPARISON_CAVEAT,
        "output_paths": {k: str(v) for k, v in paths.items() if k not in ("fig_png", "fig_pdf")},
        "figure_paths": {"png": str(paths["fig_png"]), "pdf": str(paths["fig_pdf"])},
    }
    with open(paths["provenance"], "w", encoding="utf-8") as f:
        json.dump(provenance, f, ensure_ascii=False, indent=2)

    print(f"\nSaved sample-micro CSV -> {paths['sample_micro']}")
    print(f"Saved task-macro CSV   -> {paths['task_macro']}")
    print(f"Saved domain CSV       -> {paths['domain']}")
    if baseline_rows:
        print(f"Saved vs-baseline CSV  -> {paths['vs_baseline']}")
    print(f"Saved provenance       -> {paths['provenance']}")
    print(f"Saved figure           -> {paths['fig_png']}")


if __name__ == "__main__":
    main()
