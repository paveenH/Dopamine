#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis for the Role-Confidence cross-steering MMLU-E experiment
(RR/RC/CR/CC + RRand/CRand + alpha=0 baseline).

Reads the per-task JSON files written by get_answer_cross_steering_mmlue.py
under {out_root}/{condition}/alpha_{a}/{task}_{size}_answers.json, matches
samples question-by-question (same task, same index -- the 57 MMLU task JSON
files are read in a fixed order by the generator, so index i within a task
is the same question across every condition/alpha run against the same
mmlu_dir), and reports:
  - overall + per-domain (STEM/Humanities/Social Sciences/Other) E-rate and
    accuracy, per condition/alpha/role
  - paired change vs the alpha=0 baseline (same condition's own alpha=0 where
    available, else the shared baseline)
  - cross-steering vs matched-random-support control comparison

Does NOT modify get_answer_cross_steering_mmlue.py, any mask, mean, or
result file. Read-only analysis.

Usage: /opt/anaconda3/bin/python analyze_cross_steering_mmlue.py \
    --results_dir /path/to/llama3_confidence/cross_steering_mmlue/results \
    --out_dir     /path/to/llama3_confidence/cross_steering_mmlue/analysis
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmlu_category_map import TASK_TO_CATEGORY, CATEGORIES  # noqa: E402

CONDITIONS = ["baseline", "RR", "RC", "CR", "CC", "RRand", "CRand"]
ROLES = ["confident", "unconfident"]
LABELS = ["A", "B", "C", "D", "E"]


def load_condition_alpha(results_dir: Path, condition: str, alpha) -> dict:
    """Load all 57 task JSON files for one (condition, alpha) cell.
    Returns {task: {"data": [...], "accuracy": {...}}} or raises if any
    task file is missing (fail-closed -- a partial cell must not silently
    produce a partial summary)."""
    alpha_str = f"{alpha:g}"
    cell_dir = results_dir / condition / f"alpha_{alpha_str}"
    if not cell_dir.exists():
        raise FileNotFoundError(f"Missing cell directory: {cell_dir}")

    from detection.task_list import TASKS  # local import: needs repo root on sys.path
    out = {}
    missing = []
    for task in TASKS:
        p = cell_dir / f"{task}_8B_answers.json"
        if not p.exists():
            missing.append(task)
            continue
        with open(p, "r", encoding="utf-8") as f:
            out[task] = json.load(f)
    if missing:
        raise FileNotFoundError(f"{cell_dir}: missing {len(missing)} task files: {missing[:5]}...")
    return out


def per_sample_records(cell_data: dict, role: str) -> dict:
    """cell_data: {task: {"data": [...]}}. Returns {(task, idx): (pred_label, true_label)}."""
    role_key = role.replace(" ", "_")
    records = {}
    for task, payload in cell_data.items():
        for idx, sample in enumerate(payload["data"]):
            pred = sample.get(f"answer_{role_key}")
            true_idx = sample.get("label", -1)
            true_lab = LABELS[true_idx] if 0 <= true_idx < len(LABELS) else None
            records[(task, idx)] = (pred, true_lab)
    return records


def compute_rates(records: dict) -> dict:
    total = len(records)
    correct = sum(1 for pred, true in records.values() if pred is not None and pred == true)
    e_count = sum(1 for pred, _ in records.values() if pred == "E")
    invalid = sum(1 for pred, true in records.values()
                  if pred is not None and pred != true and pred != "E")
    return {
        "n": total,
        "correct": correct,
        "e_count": e_count,
        "invalid": invalid,
        "accuracy": correct / total if total else float("nan"),
        "e_rate": e_count / total if total else float("nan"),
    }


def compute_domain_rates(records: dict) -> dict:
    by_domain = {c: {} for c in CATEGORIES}
    for (task, idx), val in records.items():
        cat = TASK_TO_CATEGORY[task]
        by_domain[cat][(task, idx)] = val
    return {cat: compute_rates(recs) for cat, recs in by_domain.items()}


def paired_delta(records_a: dict, records_b: dict, metric: str) -> dict:
    """metric in {'e', 'acc'}. records_a = treatment, records_b = baseline.
    Returns paired counts and an exact binomial sign-test p-value on the
    discordant pairs (treatment flips on vs off relative to baseline)."""
    common_keys = set(records_a.keys()) & set(records_b.keys())
    if not common_keys:
        return {"n_paired": 0, "note": "no common (task, idx) keys"}

    def flag(pred, true):
        if metric == "e":
            return pred == "E"
        elif metric == "acc":
            return pred is not None and pred == true
        else:
            raise ValueError(metric)

    on_to_off = 0
    off_to_on = 0
    for k in common_keys:
        pred_a, true_a = records_a[k]
        pred_b, true_b = records_b[k]
        fa = flag(pred_a, true_a)
        fb = flag(pred_b, true_b)
        if fb and not fa:
            on_to_off += 1
        elif not fb and fa:
            off_to_on += 1

    n_discordant = on_to_off + off_to_on
    if n_discordant > 0:
        p_value = float(binomtest(off_to_on, n_discordant, 0.5).pvalue)
    else:
        p_value = float("nan")

    return {
        "n_paired": len(common_keys),
        "off_to_on": off_to_on,
        "on_to_off": on_to_off,
        "n_discordant": n_discordant,
        "exact_sign_test_p_value": p_value,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_csv_path = out_dir / "overall_summary.csv"
    domain_csv_path = out_dir / "domain_summary.csv"
    paired_csv_path = out_dir / "paired_comparison.csv"
    provenance_path = out_dir / "provenance.json"
    fig_png_path = out_dir / "cross_steering_mmlue_main.png"
    fig_pdf_path = out_dir / "cross_steering_mmlue_main.pdf"

    for p in (overall_csv_path, domain_csv_path, paired_csv_path, provenance_path,
              fig_png_path, fig_pdf_path):
        if p.exists():
            raise FileExistsError(f"{p} already exists -- refusing to overwrite.")

    # ---- Discover available (condition, alpha) cells ----
    available_cells = []
    for condition in CONDITIONS:
        cond_dir = results_dir / condition
        if not cond_dir.exists():
            continue
        for alpha_dir in sorted(cond_dir.glob("alpha_*")):
            alpha_str = alpha_dir.name.replace("alpha_", "")
            try:
                alpha = float(alpha_str)
            except ValueError:
                continue
            available_cells.append((condition, alpha))

    if not available_cells:
        raise FileNotFoundError(f"No (condition, alpha) cells found under {results_dir}")

    print(f"Found {len(available_cells)} cells: {available_cells}")

    cell_records = {}  # (condition, alpha, role) -> records dict
    for condition, alpha in available_cells:
        cell_data = load_condition_alpha(results_dir, condition, alpha)
        for role in ROLES:
            cell_records[(condition, alpha, role)] = per_sample_records(cell_data, role)

    # ---- Overall + domain summaries ----
    overall_rows = []
    domain_rows = []
    for (condition, alpha, role), records in cell_records.items():
        overall = compute_rates(records)
        overall_rows.append({
            "condition": condition, "alpha": alpha, "role": role,
            **overall,
        })
        domain = compute_domain_rates(records)
        for cat, rates in domain.items():
            domain_rows.append({
                "condition": condition, "alpha": alpha, "role": role, "category": cat,
                **rates,
            })

    with open(overall_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(overall_rows[0].keys()))
        writer.writeheader()
        for r in overall_rows:
            writer.writerow(r)

    with open(domain_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(domain_rows[0].keys()))
        writer.writeheader()
        for r in domain_rows:
            writer.writerow(r)

    # ---- Paired comparison vs alpha=0 baseline ----
    baseline_keys = [k for k in cell_records if k[0] == "baseline" and k[1] == 0.0]
    paired_rows = []
    if baseline_keys:
        for role in ROLES:
            base_key = ("baseline", 0.0, role)
            if base_key not in cell_records:
                continue
            base_records = cell_records[base_key]
            for (condition, alpha, r2), records in cell_records.items():
                if r2 != role or condition == "baseline":
                    continue
                e_delta = paired_delta(records, base_records, "e")
                acc_delta = paired_delta(records, base_records, "acc")
                paired_rows.append({
                    "condition": condition, "alpha": alpha, "role": role,
                    "vs": "baseline_alpha0",
                    "e_rate_off_to_on": e_delta.get("off_to_on"),
                    "e_rate_on_to_off": e_delta.get("on_to_off"),
                    "e_rate_n_discordant": e_delta.get("n_discordant"),
                    "e_rate_sign_test_p": e_delta.get("exact_sign_test_p_value"),
                    "acc_off_to_on": acc_delta.get("off_to_on"),
                    "acc_on_to_off": acc_delta.get("on_to_off"),
                    "acc_n_discordant": acc_delta.get("n_discordant"),
                    "acc_sign_test_p": acc_delta.get("exact_sign_test_p_value"),
                })

    # ---- Cross-steering vs matched-random control ----
    control_pairs = [("RR", "RRand"), ("CC", "CRand")]
    for cross_cond, rand_cond in control_pairs:
        for role in ROLES:
            for alpha in sorted({a for (c, a, r) in cell_records if c == cross_cond and r == role}):
                cross_key = (cross_cond, alpha, role)
                rand_key = (rand_cond, alpha, role)
                if cross_key not in cell_records or rand_key not in cell_records:
                    continue
                e_delta = paired_delta(cell_records[cross_key], cell_records[rand_key], "e")
                acc_delta = paired_delta(cell_records[cross_key], cell_records[rand_key], "acc")
                paired_rows.append({
                    "condition": cross_cond, "alpha": alpha, "role": role,
                    "vs": f"{rand_cond}_same_alpha",
                    "e_rate_off_to_on": e_delta.get("off_to_on"),
                    "e_rate_on_to_off": e_delta.get("on_to_off"),
                    "e_rate_n_discordant": e_delta.get("n_discordant"),
                    "e_rate_sign_test_p": e_delta.get("exact_sign_test_p_value"),
                    "acc_off_to_on": acc_delta.get("off_to_on"),
                    "acc_on_to_off": acc_delta.get("on_to_off"),
                    "acc_n_discordant": acc_delta.get("n_discordant"),
                    "acc_sign_test_p": acc_delta.get("exact_sign_test_p_value"),
                })

    if paired_rows:
        with open(paired_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
            writer.writeheader()
            for r in paired_rows:
                writer.writerow(r)

    # ---- Figure: E-rate and accuracy per condition, both roles ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.0,
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9.0,
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
        for condition in CONDITIONS:
            xs, e_rates, accs = [], [], []
            for (c, alpha, r), records in sorted(cell_records.items(), key=lambda kv: kv[0][1]):
                if c != condition or r != role:
                    continue
                rates = compute_rates(records)
                xs.append(alpha)
                e_rates.append(rates["e_rate"])
                accs.append(rates["accuracy"])
            if xs:
                ax_e.plot(xs, e_rates, marker="o", label=condition)
                ax_acc.plot(xs, accs, marker="o", label=condition)
        ax_e.set_title(f"E-option rate ({role})")
        ax_e.set_xlabel("alpha")
        ax_e.set_ylabel("E-rate")
        ax_e.legend(frameon=False, fontsize=7, ncol=2)
        ax_e.spines["top"].set_visible(False)
        ax_e.spines["right"].set_visible(False)

        ax_acc.set_title(f"Accuracy ({role})")
        ax_acc.set_xlabel("alpha")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend(frameon=False, fontsize=7, ncol=2)
        ax_acc.spines["top"].set_visible(False)
        ax_acc.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(fig_png_path)
    fig.savefig(fig_pdf_path)
    plt.close(fig)

    # ---- Provenance ----
    provenance = {
        "results_dir": str(results_dir),
        "available_cells": [(c, a) for c, a in available_cells],
        "roles": ROLES,
        "categories": CATEGORIES,
        "task_to_category_source": "mmlu_category_map.py (public MMLU 57-task grouping, "
                                    "Hendrycks et al. 2021)",
        "output_paths": {
            "overall_csv": str(overall_csv_path),
            "domain_csv": str(domain_csv_path),
            "paired_csv": str(paired_csv_path),
            "figure_png": str(fig_png_path),
            "figure_pdf": str(fig_pdf_path),
        },
    }
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, ensure_ascii=False, indent=2)

    print(f"\nSaved overall CSV -> {overall_csv_path}")
    print(f"Saved domain CSV  -> {domain_csv_path}")
    if paired_rows:
        print(f"Saved paired CSV  -> {paired_csv_path}")
    print(f"Saved provenance  -> {provenance_path}")
    print(f"Saved figure      -> {fig_png_path}")


if __name__ == "__main__":
    main()
