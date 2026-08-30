#!/usr/bin/env python3
"""P3 supplement -- freeze the CoT predictions BEFORE any CoT cell is generated.

Protocol p3-supp-v1 (docs/PREREG_P3_SUPPLEMENT.md).

WHAT MAKES THE CoT HALF LOCKED. The four CoT cells do not exist yet. Writing the
predicted direction of dAcc here, before generation, is what lets Part 1 be
called a locked prospective condition test rather than another post-hoc reading.
The No-CoT half of the interaction is already unsealed and can never be locked;
the protocol says so and the evaluator labels it.

THIS IS NOT A NEW BLIND VALIDATION. Same 300 questions, same gold, already
unsealed 2026-08-30. What is prospective is the CONDITION.

Nothing is refitted. The predicted direction is derived from the frozen P3
result, not from any CoT data (there is none).
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "p3_supp_predictions.json")
if os.path.exists(OUT):
    sys.exit(f"FAIL: {OUT} exists; refusing to overwrite a frozen prediction file")

P3 = json.load(open(os.path.join(HERE, "p3_predictions.json"), encoding="utf-8"))
EVAL = json.load(open(os.path.join(HERE, "p3_evaluation.json"), encoding="utf-8"))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


report = {
    "protocol": "p3-supp-v1",
    "status": "frozen BEFORE any CoT cell was generated",
    "cot_cells_exist_at_freeze_time": False,
    "not_a_blind_dataset_validation": (
        "Same 300 questions and same gold, unsealed 2026-08-30. What is "
        "prospective is the CONDITION (CoT), not the data."),
    "does_not_modify": "docs/p3_result_20260830.json; tag p3-result-unsealed",
    "basis_sha256": {
        "p3_predictions": sha(os.path.join(HERE, "p3_predictions.json")),
        "p3_evaluation": sha(os.path.join(HERE, "p3_evaluation.json")),
    },
    "models": {},
}

for model, alpha in (("llama", -6), ("qwen", 8)):
    fw = EVAL["models"][model]["fixed_workpoint"]
    report["models"][model] = {
        "cells": [f"CoT alpha=0", f"CoT alpha={alpha:+d}"],
        "alpha": alpha,
        "alpha_source": ("frozen GSM8K workpoint, already used in P3; NOT "
                         "re-searched and NOT derived from any CoT data"),
        "predicted_direction_of_dAcc": "positive",
        "justification": (
            f"Under No-CoT the same alpha improved over alpha=0 by "
            f"{fw['diff_pp']:+.2f} pp (McNemar p={fw['mcnemar_p']:.3g}). The "
            f"pre-registered prediction is that the sign carries over to CoT. "
            f"A null or a reversal is a result about condition specificity and "
            f"is reported with equal prominence."),
        "nocot_reference": {
            "acc_alpha": fw["acc_alpha"], "acc_alpha0": fw["acc_alpha0"],
            "diff_pp": fw["diff_pp"], "mcnemar_p": fw["mcnemar_p"],
        },
        "interaction_status": (
            "NOT a locked prediction -- its No-CoT half is already unsealed. "
            "Reported descriptively with a CI and excluded from the Holm family."),
    }

report["statistics"] = {
    "primary": "dAcc = Acc(CoT+alpha) - Acc(CoT), first_acc, paired per question",
    "test": "exact two-sided McNemar per model, with discordant counts",
    "ci": "paired question-level bootstrap, B=10000, seed 0",
    "holm_family": "the two models, m=2, primary metric only",
    "interaction_excluded_from_holm": True,
}

json.dump(report, open(OUT, "w"), indent=2)
print(f"FROZEN -> {os.path.basename(OUT)}  sha256 {sha(OUT)[:16]}")
for m in report["models"]:
    d = report["models"][m]
    print(f"  {m:6s} alpha={d['alpha']:+d}  predicted dAcc direction: "
          f"{d['predicted_direction_of_dAcc']}  "
          f"(No-CoT reference {d['nocot_reference']['diff_pp']:+.2f} pp)")
print("CoT cells may be generated only AFTER this file exists.")
