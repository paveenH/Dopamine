#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mean hidden-state difference for the explicit-confidence MMLU-E collection
(run_hidden_mmlue_confidence_hs.sh / verify_mmlue_confidence_hs.py).

Computes mean(confident_HS) - mean(unconfident_HS) over DIVERGENT samples
ONLY -- i.e. samples where answer_confident != answer_unconfident -- matching
the existing mean/mean_diff.py convention of restricting to samples where the
two role conditions actually disagree.

This is a STANDALONE script, independent of mean/mean_diff.py:
  - reads H5 (hidden_states: (N, 33, 4096) fp16), not per-role .npy files
  - reads answer JSON with keys answer_confident / answer_unconfident
    (get_answer_logits.py: sample[f"answer_{role}"])
  - does NOT touch mean/mean_diff.py or its outputs

Does NOT filter/select neurons, build an NMD mask, or do any steering --
this only computes and saves the raw per-layer mean-difference matrix, plus a
per-task/aggregate summary of how many divergent samples were used.

Usage:
  python3.10 mean/mean_diff_confidence.py \
      --hs_dir /data1/paveen/Dopamine/components/hidden_states/llama3/mmlue_confidence \
      --ans_dir /data1/paveen/Dopamine/components/answer/llama3_confidence \
      --size 8B \
      --out_dir /data1/paveen/Dopamine/components/hidden_states_mean/llama3_confidence
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detection.task_list import TASKS  # noqa: E402

ROLES = ("confident", "unconfident")


def divergent_indices(samples: list) -> list:
    """Indices where answer_confident != answer_unconfident. Samples missing
    either key are skipped (not counted as divergent), matching mean_diff.py's
    'entry.get(key)' None-safe comparison behavior."""
    idxs = []
    for idx, entry in enumerate(samples):
        a_conf = entry.get("answer_confident")
        a_unconf = entry.get("answer_unconfident")
        if a_conf is None or a_unconf is None:
            continue
        if a_conf != a_unconf:
            idxs.append(idx)
    return idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs_dir", required=True, help="Dir containing {role}_{task}_{size}.h5")
    ap.add_argument("--ans_dir", required=True, help="Dir containing {task}_{size}_answers.json")
    ap.add_argument("--size", default="8B")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    hs_dir = Path(args.hs_dir)
    ans_dir = Path(args.ans_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diff_sum = None  # accumulator, float32, shape (33, 4096)
    total_divergent = 0
    per_task_counts = {}
    skipped_tasks = []

    for task in TASKS:
        ans_path = ans_dir / f"{task}_{args.size}_answers.json"
        h5_paths = {r: hs_dir / f"{r}_{task}_{args.size}.h5" for r in ROLES}

        if not ans_path.exists():
            print(f"[skip] {task}: missing answer JSON {ans_path}")
            skipped_tasks.append(task)
            continue
        if not all(p.exists() for p in h5_paths.values()):
            print(f"[skip] {task}: missing H5 for one or both roles")
            skipped_tasks.append(task)
            continue

        try:
            with open(ans_path, "r", encoding="utf-8") as f:
                ans = json.load(f)
            samples = ans["data"]
        except Exception as e:
            print(f"[skip] {task}: failed to parse answer JSON ({type(e).__name__}: {e})")
            skipped_tasks.append(task)
            continue

        div_idx = divergent_indices(samples)
        per_task_counts[task] = len(div_idx)
        if not div_idx:
            print(f"[skip] {task}: no divergent samples")
            continue

        try:
            with h5py.File(h5_paths["confident"], "r") as fc, h5py.File(h5_paths["unconfident"], "r") as fu:
                ds_c = fc["hidden_states"]
                ds_u = fu["hidden_states"]
                if ds_c.shape != ds_u.shape:
                    print(f"[skip] {task}: paired H5 shape mismatch {ds_c.shape} vs {ds_u.shape}")
                    skipped_tasks.append(task)
                    continue
                n_samples = ds_c.shape[0]
                bad_idx = [i for i in div_idx if i >= n_samples]
                if bad_idx:
                    print(f"[skip] {task}: divergent index out of range for H5 (N={n_samples})")
                    skipped_tasks.append(task)
                    continue

                for i in div_idx:
                    hs_c = ds_c[i].astype(np.float32)
                    hs_u = ds_u[i].astype(np.float32)
                    if not (np.isfinite(hs_c).all() and np.isfinite(hs_u).all()):
                        print(f"[warn] {task} sample {i}: NaN/Inf encountered, skipped")
                        continue
                    d = hs_c - hs_u
                    if diff_sum is None:
                        diff_sum = d.copy()
                    else:
                        diff_sum += d
                    total_divergent += 1
        except Exception as e:
            print(f"[skip] {task}: failed to read H5 ({type(e).__name__}: {e})")
            skipped_tasks.append(task)
            continue

        print(f"[ok]   {task}: divergent={len(div_idx)}, cumulative total={total_divergent}")

    if diff_sum is None or total_divergent == 0:
        print("\nNo divergent samples found across any task -- nothing to save.")
        sys.exit(1)

    diff_mean = (diff_sum / total_divergent).astype(np.float16)
    diff_mean = diff_mean[None, ...]  # (1, 33, 4096), matching mean_diff.py's batch-dim convention

    out_path = out_dir / f"diff_mean_confidence_{args.size}.npy"
    np.save(out_path, diff_mean)

    summary = {
        "roles": list(ROLES),
        "sample_selection": "divergent pairs only (answer_confident != answer_unconfident)",
        "size": args.size,
        "hs_dir": str(hs_dir),
        "ans_dir": str(ans_dir),
        "n_tasks_total": len(TASKS),
        "n_tasks_skipped": len(skipped_tasks),
        "skipped_tasks": skipped_tasks,
        "per_task_divergent_count": per_task_counts,
        "total_divergent_samples": total_divergent,
        "diff_mean_shape": list(diff_mean.shape),
        "diff_mean_path": str(out_path),
    }
    summary_path = out_dir / f"mean_diff_confidence_summary_{args.size}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved diff mean -> {out_path} (shape {diff_mean.shape})")
    print(f"Saved summary   -> {summary_path}")
    print(f"Total divergent samples used: {total_divergent}")
    if skipped_tasks:
        print(f"Skipped tasks ({len(skipped_tasks)}): {skipped_tasks}")


if __name__ == "__main__":
    main()
