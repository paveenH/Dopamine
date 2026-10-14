#!/usr/bin/env python3
"""Paired analysis using the frozen RSN GSM8K accuracy/candidate extractors."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from scipy.stats import binomtest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_root", type=Path, required=True, help=".../gsm8k_orthogonal_v1")
    ap.add_argument("--model", choices=["llama3", "qwen2.5"], required=True)
    ap.add_argument("--reference_root", type=Path, required=True, help="directory containing baseline/result.json and rsn/result.json")
    ap.add_argument("--roleanswer", type=Path, default=Path("/Users/paveenhuang/Documents/RSNResult/RoleAnswer"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.roleanswer))
    sys.path.insert(0, str(Path(__file__).parent / "p3"))
    import analyze_first_last_acc as frozen
    from early_candidate_detector import has_early_candidate
    from precandidate_reasoning import first_candidate
    import analyze_loop_anxiety as loop_metrics
    import early_candidate_detector as ec_module
    import precandidate_reasoning as candidate_module

    def sha_file(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def read(path):
        obj = json.loads(path.read_text())
        return obj["data"]

    def key(row):
        return hashlib.sha256(row["question"].encode("utf-8")).hexdigest()

    def index(rows, label):
        out = {}
        for row in rows:
            qid = key(row)
            if qid in out:
                raise ValueError(f"duplicate question identity in {label}: {qid}")
            out[qid] = row
        return out

    def correctness(row):
        pred = frozen.all_hash(row["generated_neutral"])
        pred = pred[0] if pred else frozen.fallback_gsm8k(row["generated_neutral"])
        return frozen.norm_gsm8k(pred) == frozen.norm_gsm8k(row["answer"])

    alphas = {"llama3": -6, "qwen2.5": 8}
    cells = ["baseline", "rsn", "seed1", "seed2", "seed3"]
    data, summaries = {}, {}
    for cell in cells:
        path = a.result_root / cell / "result.json"
        rows = read(path)
        if not rows:
            raise ValueError(f"empty cell: {path}")
        idx = index(rows, cell)
        data[cell] = idx
        texts = [r["generated_neutral"] for r in rows]
        valid_answers = 0
        for row in rows:
            marks = frozen.all_hash(row["generated_neutral"])
            pred = marks[0] if marks else frozen.fallback_gsm8k(row["generated_neutral"])
            valid_answers += int(bool(str(pred).strip()))
        acc = frozen.first_last_stats(rows, lambda x: x["generated_neutral"], lambda x: x["answer"],
                                      frozen.all_hash, frozen.norm_gsm8k, frozen.fallback_gsm8k)
        summaries[cell] = {**acc, "candidate_coverage": sum(first_candidate(t) is not None for t in texts) / len(texts) * 100,
                           "early_candidate_rate": sum(has_early_candidate(t) for t in texts) / len(texts) * 100,
                           "empty_rate": sum(not t.strip() for t in texts) / len(texts) * 100,
                           "valid_answer_coverage": valid_answers / len(texts) * 100,
                           "token_count_median": sorted(r["generated_token_count"] for r in rows)[len(rows)//2],
                           "generation_chars_median": sorted(len(t) for t in texts)[len(texts)//2],
                           "budget_exhausted_rate": sum(r["stop_reason"] == "budget_exhausted" for r in rows) / len(rows) * 100,
                           "natural_eos_rate": sum(r["stop_reason"] == "natural_eos" for r in rows) / len(rows) * 100,
                           "loop_count": loop_metrics.analyze_file(str(path), "generated_neutral", mode="loop")["n_loop"]}
    common = set(data[cells[0]])
    for cell in cells[1:]:
        if set(data[cell]) != common:
            raise ValueError(f"paired question mismatch in {cell}: missing={len(common-set(data[cell]))}, extra={len(set(data[cell])-common)}")
    for cell in cells:
        for qid in common:
            for field in ("question", "answer"):
                if data[cell][qid][field] != data["baseline"][qid][field]:
                    raise ValueError(f"paired text/gold mismatch: {cell}, {qid}, {field}")

    ref_rows = read(a.reference_root / "rsn" / "result.json")
    ref_acc = frozen.first_last_stats(ref_rows, lambda x: x["generated_neutral"], lambda x: x["answer"],
                                      frozen.all_hash, frozen.norm_gsm8k, frozen.fallback_gsm8k)
    if ref_acc["n"] != 300:
        raise ValueError(f"reference is not the published 300-item cell: n={ref_acc['n']}")
    expected = {"llama3": {"baseline": 60.0, "rsn": 78.0}, "qwen2.5": {"baseline": 68.0, "rsn": 86.0}}[a.model]
    for cell, want in expected.items():
        observed = frozen.first_last_stats(read(a.reference_root / cell / "result.json"), lambda x: x["generated_neutral"],
                                           lambda x: x["answer"], frozen.all_hash,
                                           frozen.norm_gsm8k, frozen.fallback_gsm8k)["first_acc"]
        if observed != want:
            raise ValueError(f"frozen {cell} reference first_acc={observed}, expected published {want}; scorer/reference reproduction failed")
    # Per user's rule, direct contrasts are the six-member Holm family.
    rng = __import__("numpy").random.default_rng(0)
    import numpy as np
    ordered = sorted(common)
    pairs = [("rsn", f"seed{s}") for s in (1, 2, 3)]
    results = []
    pvals = []
    for rsn, ctrl in pairs:
        left = np.array([correctness(data[rsn][q]) for q in ordered], dtype=bool)
        right = np.array([correctness(data[ctrl][q]) for q in ordered], dtype=bool)
        d = left.astype(float) - right.astype(float)
        boot = np.array([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(10000)]) * 100
        b = int(np.sum(left & ~right)); c = int(np.sum(~left & right))
        p = float(binomtest(b, b+c, .5, alternative="two-sided").pvalue) if b+c else 1.0
        pvals.append(p)
        results.append({"contrast": f"{rsn}_vs_{ctrl}", "n": len(d), "delta_first_acc_pp": float(d.mean()*100),
                        "bootstrap_ci95_raw_pp": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                        "mcnemar_discordant_rsn_only": b, "mcnemar_discordant_control_only": c, "mcnemar_p_raw": p})
    # adjust jointly with the other model's three p values when supplied through the sibling file;
    # never silently label a per-model adjustment as the fixed six-test family.
    out = {"protocol": "gsm8k-orthogonal-v1", "model": a.model, "workpoint_alpha": alphas[a.model],
           "published_reference_first_last_stats": ref_acc, "cells": summaries,
           "direct_primary_contrasts_partial_family": results,
           "holm_note": "Holm requires all six direct contrasts across both models; these p-values are raw until combined.",
           "n_paired_question_ids": len(common),
           "provenance": {"analysis_script_sha256": sha_file(__file__),
                          "frozen_extractor_sha256": sha_file(frozen.__file__),
                          "early_candidate_detector_sha256": sha_file(ec_module.__file__),
                          "candidate_locator_sha256": sha_file(candidate_module.__file__),
                          "loop_analyzer_sha256": sha_file(loop_metrics.__file__),
                          "result_json_sha256": {c: sha_file(a.result_root / c / "result.json") for c in cells},
                          "result_manifest_sha256": {c: sha_file(a.result_root / c / "manifest.json") for c in cells},
                          "reference_json_sha256": {c: sha_file(a.reference_root / c / "result.json") for c in ("baseline", "rsn")}}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
