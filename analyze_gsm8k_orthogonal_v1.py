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
    ap.add_argument("--baseline_reference", type=Path, required=True)
    ap.add_argument("--rsn_reference", type=Path, required=True)
    ap.add_argument("--roleanswer", type=Path, default=Path("/Users/paveenhuang/Documents/RSNResult/RoleAnswer"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.roleanswer))
    sys.path.insert(0, str(a.roleanswer / "prediction"))
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
    alpha = alphas[a.model]
    expected_band = {"llama3": (11, 20), "qwen2.5": (16, 22)}[a.model]
    cells = ["baseline", "rsn", "seed1", "seed2", "seed3"]
    configs = {}
    expected_fields = ("model", "protocol", "data_sha256", "question_sha256", "prompt_sha256",
                       "question_order_sha256", "batch_size", "max_new_tokens", "temperature", "top_p",
                       "prefill_only", "prefill_tail_len", "host", "platform", "torch", "numpy",
                       "cuda_visible_devices", "gpu", "model_config_sha256", "tokenizer", "transformers",
                       "tokenizers", "python", "cuda", "device_map", "model_dir", "model_weight_revision",
                       "tokenized_prompt_ids_sha256", "tokenized_prompt_lengths", "runner_sha256", "code_sha256")
    for cell in cells:
        mp = a.result_root / cell / "manifest.json"
        rp = a.result_root / cell / "result.json"
        manifest = json.loads(mp.read_text())
        cfg, run = manifest.get("configuration"), manifest.get("run")
        if not isinstance(cfg, dict) or not isinstance(run, dict):
            raise ValueError(f"invalid manifest schema: {mp}")
        want_alpha = 0 if cell == "baseline" else alpha
        if (cfg.get("model") != a.model or cfg.get("protocol") != "gsm8k-orthogonal-v1"
                or cfg.get("condition") != cell or cfg.get("alpha") != want_alpha
                or (cfg.get("start"), cfg.get("end")) != expected_band
                or (cfg.get("batch_size"), cfg.get("max_new_tokens"), cfg.get("temperature"), cfg.get("top_p")) != (24, 768, 0, 0.9)
                or cfg.get("prefill_only") is not True or cfg.get("prefill_tail_len") != 1):
            raise ValueError(f"protocol/configuration mismatch: {mp}")
        expected_fires = 0 if cell == "baseline" else (expected_band[1]-expected_band[0])*300
        if run.get("row_count") != 300 or run.get("steering_fires") != expected_fires or run.get("expected_steering_fires") != expected_fires:
            raise ValueError(f"row count or injection count mismatch: {mp}")
        if run.get("result_sha256") != sha_file(rp):
            raise ValueError(f"result hash mismatch: {rp}")
        mv = cfg.get("mask_validation", {})
        if (mv.get("mask_sha256") is None or mv.get("rsn_mask_sha256") is None
                or mv.get("diff_mean_sha256") is None or mv.get("none_diff_mean_sha256") is None):
            raise ValueError(f"missing source-mask hashes: {mp}")
        if cell == "rsn" and mv["mask_sha256"] != mv["rsn_mask_sha256"]:
            raise ValueError(f"RSN condition does not use the declared RSN mask: {mp}")
        if cell == "rsn" and mv.get("direction_kind") != "rsn_reference":
            raise ValueError(f"RSN condition has unexpected direction type: {mp}")
        if cell.startswith("seed") and mv.get("direction_kind") != "orthogonal":
            raise ValueError(f"seed condition is not marked as orthogonal: {mp}")
        configs[cell] = cfg
    for cell in cells[1:]:
        for field in expected_fields:
            if configs[cell].get(field) != configs["baseline"].get(field):
                raise ValueError(f"source/environment mismatch in {field}: baseline vs {cell}")
        mv, base_mv = configs[cell]["mask_validation"], configs["baseline"]["mask_validation"]
        for field in ("rsn_mask_sha256", "diff_mean_sha256", "none_diff_mean_sha256"):
            if mv.get(field) != base_mv.get(field):
                raise ValueError(f"reference source mismatch in {field}: baseline vs {cell}")
    control_hashes = [configs[c]["mask_validation"]["mask_sha256"] for c in ("seed1", "seed2", "seed3")]
    if len(set(control_hashes)) != 3:
        raise ValueError("orthogonal seeds do not identify three distinct mask files")
    data, summaries = {}, {}
    for cell in cells:
        path = a.result_root / cell / "result.json"
        rows = read(path)
        if len(rows) != 300:
            raise ValueError(f"formal cell must contain exactly 300 rows: {path} has {len(rows)}")
        qids = [key(r) for r in rows]
        if qids != configs[cell]["question_sha256"]:
            raise ValueError(f"question identity/order differs from manifest: {path}")
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

    ref_rows = read(a.rsn_reference)
    ref_acc = frozen.first_last_stats(ref_rows, lambda x: x["generated_neutral"], lambda x: x["answer"],
                                      frozen.all_hash, frozen.norm_gsm8k, frozen.fallback_gsm8k)
    if ref_acc["n"] != 300:
        raise ValueError(f"reference is not the published 300-item cell: n={ref_acc['n']}")
    baseline_ref_rows = read(a.baseline_reference)
    for label, ref in (("historical baseline", baseline_ref_rows), ("historical RSN", ref_rows)):
        if len(ref) != 300 or [key(r) for r in ref] != configs["baseline"]["question_sha256"]:
            raise ValueError(f"{label} does not match the exact 300-question manifest order")
        for qid, row in zip(configs["baseline"]["question_sha256"], ref):
            current = data["baseline"][qid]
            if row["question"] != current["question"] or row["answer"] != current["answer"]:
                raise ValueError(f"{label} question/gold mismatch: {qid}")
    expected = {"llama3": {"baseline": 60.0, "rsn": 78.0}, "qwen2.5": {"baseline": 68.0, "rsn": 86.0}}[a.model]
    for cell, want in expected.items():
        reference_path = a.baseline_reference if cell == "baseline" else a.rsn_reference
        observed = frozen.first_last_stats(read(reference_path), lambda x: x["generated_neutral"],
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
    for rsn, ctrl in pairs:
        left = np.array([correctness(data[rsn][q]) for q in ordered], dtype=bool)
        right = np.array([correctness(data[ctrl][q]) for q in ordered], dtype=bool)
        d = left.astype(float) - right.astype(float)
        boot = np.array([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(10000)]) * 100
        b = int(np.sum(left & ~right)); c = int(np.sum(~left & right))
        p = float(binomtest(b, b+c, .5, alternative="two-sided").pvalue) if b+c else 1.0
        results.append({"contrast": f"{rsn}_vs_{ctrl}", "n": len(d), "delta_first_acc_pp": float(d.mean()*100),
                        "bootstrap_ci95_raw_pp": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                        "mcnemar_discordant_rsn_only": b, "mcnemar_discordant_control_only": c, "mcnemar_p_raw": p})
    baseline_contrasts = []
    for cell in ("rsn", "seed1", "seed2", "seed3"):
        left = np.array([correctness(data[cell][q]) for q in ordered], dtype=bool)
        right = np.array([correctness(data["baseline"][q]) for q in ordered], dtype=bool)
        d = left.astype(float) - right.astype(float)
        boot = np.array([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(10000)]) * 100
        b, c = int(np.sum(left & ~right)), int(np.sum(~left & right))
        baseline_contrasts.append({"contrast": f"{cell}_vs_baseline", "n": len(d),
            "delta_first_acc_pp": float(d.mean()*100),
            "bootstrap_ci95_raw_pp": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "mcnemar_discordant_cell_only": b, "mcnemar_discordant_baseline_only": c,
            "mcnemar_p_raw_descriptive": float(binomtest(b, b+c, .5, alternative="two-sided").pvalue) if b+c else 1.0})
    # adjust jointly with the other model's three p values when supplied through the sibling file;
    # never silently label a per-model adjustment as the fixed six-test family.
    out = {"protocol": "gsm8k-orthogonal-v1", "model": a.model, "workpoint_alpha": alphas[a.model],
           "published_reference_first_last_stats": ref_acc, "cells": summaries,
           "direct_primary_contrasts_partial_family": results, "baseline_contrasts_descriptive": baseline_contrasts,
           "holm_note": "Holm requires all six direct contrasts across both models; these p-values are raw until combined.",
           "n_paired_question_ids": len(common),
           "provenance": {"analysis_script_sha256": sha_file(__file__),
                          "frozen_extractor_sha256": sha_file(frozen.__file__),
                          "early_candidate_detector_sha256": sha_file(ec_module.__file__),
                          "candidate_locator_sha256": sha_file(candidate_module.__file__),
                          "loop_analyzer_sha256": sha_file(loop_metrics.__file__),
                          "result_json_sha256": {c: sha_file(a.result_root / c / "result.json") for c in cells},
                          "result_manifest_sha256": {c: sha_file(a.result_root / c / "manifest.json") for c in cells},
                          "reference_json_sha256": {"baseline": sha_file(a.baseline_reference), "rsn": sha_file(a.rsn_reference)},
                          "manifest_sha256": {c: sha_file(a.result_root / c / "manifest.json") for c in cells}}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
