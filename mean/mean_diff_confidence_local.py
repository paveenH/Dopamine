#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confidence mean / diff-mean computation, matching the RSN paper's Expert /
Non-Expert convention EXACTLY (mean/mean_diff.py), so downstream
correlation / cosine / neuron-overlap analysis can consume it unchanged.

Convention verified against the existing RoleHidden outputs before writing
this script (see the accompanying summary/report):
  - mean/mean_diff.py: diff_mean_{size}.npy / none_diff_mean_{size}.npy are
    shape (1, 1, layers, hidden) float16 -- a leading batch dim + the legacy
    "(N, 1, layers, hidden)" time dim carried over from the original .npy
    per-sample layout.
  - The final combined diff (e.g. llama3_8B_diff.npy) is the SQUEEZED
    (layers, hidden) float16 array: char_mean.squeeze(0).squeeze(0) - none.
  - Accumulation is GLOBAL POOLED across all 57 tasks, weighted by sample
    count (NOT a per-task average of per-task means): one running sum/count
    updated across every task's batches, one division at the very end.
  - Accumulation dtype is float32 (never float16) to avoid overflow/precision
    loss; only the final mean is cast back to float16 for storage.

This script computes the CONFIDENCE analogue:
  confident_mean   = mean over ALL divergent-pair samples of the CONFIDENT HS
  unconfident_mean = mean over ALL divergent-pair samples of the UNCONFIDENT HS
  confidence_diff  = confident_mean - unconfident_mean

Differences from mean/mean_diff.py (by necessity, not by choice):
  - Source data is H5 (`hidden_states`: (N, 33, 4096) fp16), not per-role
    .npy files, and carries no legacy time dim -- one is inserted here
    (`[:, None, :, :]`) purely to reproduce the historical (1, 1, L, H) shape
    for downstream-format compatibility. It carries no other meaning.
  - Divergent-pair keys are answer_confident / answer_unconfident (not
    answer_{task}_expert / answer_non_{task}_expert).
  - This script does NOT touch mean/mean_diff.py or mean/mean_diff_confidence.py
    (the earlier server-side draft); it is a new, separate script targeting
    the LOCAL RoleHidden analysis tree, per the current instructions.

Sample universe (per current instructions): ALL MMLU-E divergent pairs across
all 57 tasks are used as the main result. No filtering by which of A-E was
selected, and no train/held-out split at this stage.

This stage does NOT run NMD / neuron selection / PCA / manifold / steering.

Usage (LOCAL analysis box -- python3.10, not the server's `python`):
  python3.10 mean/mean_diff_confidence_local.py \
      --hs_dir  /path/to/hidden_states/llama3/mmlue_confidence \
      --ans_dir /path/to/answer/llama3_confidence \
      --size 8B \
      --out_dir /Users/paveenhuang/Documents/RSNResult/RoleHidden/llama3_confidence
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
EXPECTED_NUM_HIDDEN_STATES = 33
EXPECTED_HIDDEN_DIM = 4096


def divergent_indices(samples: list) -> list:
    """Indices where answer_confident != answer_unconfident. A sample missing
    either key is excluded (not counted as divergent), matching mean_diff.py's
    None-safe `entry.get(key)` comparison behavior."""
    idxs = []
    for idx, entry in enumerate(samples):
        a_conf = entry.get("answer_confident")
        a_unconf = entry.get("answer_unconfident")
        if a_conf is None or a_unconf is None:
            continue
        if a_conf != a_unconf:
            idxs.append(idx)
    return idxs


def process_task(task, hs_dir, ans_dir, size, accum, report):
    """Update accum (mutable dict: conf_sum, unconf_sum, count, all float32/int)
    and report (mutable dict of per-task diagnostics) for one task.
    Returns True on success, False if the task was skipped."""
    ans_path = ans_dir / f"{task}_{size}_answers.json"
    h5_paths = {r: hs_dir / f"{r}_{task}_{size}.h5" for r in ROLES}

    entry = {"task": task, "ok": False, "n_divergent": 0, "errors": []}
    report["per_task"].append(entry)

    if not ans_path.exists():
        entry["errors"].append(f"missing answer JSON: {ans_path}")
        return False
    if not all(p.exists() for p in h5_paths.values()):
        entry["errors"].append("missing H5 for one or both roles")
        return False

    try:
        with open(ans_path, "r", encoding="utf-8") as f:
            ans = json.load(f)
        samples = ans["data"]
    except Exception as e:
        entry["errors"].append(f"failed to parse answer JSON: {type(e).__name__}: {e}")
        return False

    n_answer_samples = len(samples)
    div_idx = divergent_indices(samples)
    entry["n_answer_samples"] = n_answer_samples
    entry["n_divergent"] = len(div_idx)

    if not div_idx:
        entry["ok"] = True  # not an error, just nothing to accumulate
        entry["errors"].append("no divergent samples (skipped, not a failure)")
        return True

    try:
        with h5py.File(h5_paths["confident"], "r") as fc, h5py.File(h5_paths["unconfident"], "r") as fu:
            ds_c = fc["hidden_states"]
            ds_u = fu["hidden_states"]
            entry["h5_shape_confident"] = list(ds_c.shape)
            entry["h5_shape_unconfident"] = list(ds_u.shape)

            # ---- alignment checks (per current instructions, item 6) ----
            if ds_c.shape != ds_u.shape:
                entry["errors"].append(f"paired H5 shape mismatch: {ds_c.shape} vs {ds_u.shape}")
                return False
            if ds_c.shape[0] != n_answer_samples:
                entry["errors"].append(
                    f"H5 N={ds_c.shape[0]} != answer sample count {n_answer_samples}"
                )
                return False
            if ds_c.shape[1:] != (EXPECTED_NUM_HIDDEN_STATES, EXPECTED_HIDDEN_DIM):
                entry["errors"].append(
                    f"unexpected H5 shape {ds_c.shape}, expected "
                    f"(N, {EXPECTED_NUM_HIDDEN_STATES}, {EXPECTED_HIDDEN_DIM})"
                )
                return False
            bad_idx = [i for i in div_idx if i >= ds_c.shape[0]]
            if bad_idx:
                entry["errors"].append(f"divergent index out of range for H5 (N={ds_c.shape[0]})")
                return False

            # ---- accumulate in float32, batched to bound memory ----
            batch_size = 100
            n_used = 0
            for i in range(0, len(div_idx), batch_size):
                batch_idx = sorted(div_idx[i:i + batch_size])  # h5py fancy-index requires increasing order
                conf_batch = ds_c[batch_idx, ...].astype(np.float32)
                unconf_batch = ds_u[batch_idx, ...].astype(np.float32)

                finite_mask = np.isfinite(conf_batch).all(axis=(1, 2)) & np.isfinite(unconf_batch).all(axis=(1, 2))
                if not finite_mask.all():
                    n_bad = int((~finite_mask).sum())
                    entry["errors"].append(f"{n_bad} sample(s) with NaN/Inf skipped in batch starting {batch_idx[0]}")
                    conf_batch = conf_batch[finite_mask]
                    unconf_batch = unconf_batch[finite_mask]

                if conf_batch.shape[0] == 0:
                    continue

                # Insert legacy "time" axis: (n, layers, hidden) -> (n, 1, layers, hidden)
                conf_batch = conf_batch[:, None, :, :]
                unconf_batch = unconf_batch[:, None, :, :]

                batch_conf_sum = conf_batch.sum(axis=0)
                batch_unconf_sum = unconf_batch.sum(axis=0)

                if accum["conf_sum"] is None:
                    accum["conf_sum"] = batch_conf_sum
                    accum["unconf_sum"] = batch_unconf_sum
                else:
                    accum["conf_sum"] += batch_conf_sum
                    accum["unconf_sum"] += batch_unconf_sum

                accum["count"] += conf_batch.shape[0]
                # Also accumulate the per-sample-diff-then-sum path for the
                # equivalence check (item 8), independent of the above.
                batch_diff_sum = batch_conf_sum - batch_unconf_sum
                if accum["diff_sum_via_persample"] is None:
                    accum["diff_sum_via_persample"] = batch_diff_sum
                else:
                    accum["diff_sum_via_persample"] += batch_diff_sum

                n_used += conf_batch.shape[0]

            entry["n_used"] = n_used

    except Exception as e:
        entry["errors"].append(f"failed to read H5: {type(e).__name__}: {e}")
        return False

    entry["ok"] = True
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs_dir", required=True)
    ap.add_argument("--ans_dir", required=True)
    ap.add_argument("--size", default="8B")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    hs_dir = Path(args.hs_dir)
    ans_dir = Path(args.ans_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_mean_path = out_dir / f"confident_mean_{args.size}.npy"
    unconf_mean_path = out_dir / f"unconfident_mean_{args.size}.npy"
    diff_path = out_dir / f"confidence_diff_{args.size}.npy"
    manifest_path = out_dir / "confidence_mean_manifest.json"

    for p in (conf_mean_path, unconf_mean_path, diff_path, manifest_path):
        if p.exists():
            print(f"[REFUSE] Output already exists, will not overwrite: {p}")
            sys.exit(1)

    accum = {
        "conf_sum": None,
        "unconf_sum": None,
        "diff_sum_via_persample": None,  # for the mean(a-b) vs mean(a)-mean(b) check
        "count": 0,
    }
    report = {"per_task": []}

    n_ok = 0
    n_failed = 0
    for task in TASKS:
        ok = process_task(task, hs_dir, ans_dir, args.size, accum, report)
        entry = report["per_task"][-1]
        if ok:
            n_ok += 1
            print(f"[ok]   {task}: divergent={entry.get('n_divergent', 0)}, "
                  f"used={entry.get('n_used', 0)}, cumulative total={accum['count']}")
        else:
            n_failed += 1
            print(f"[FAIL] {task}: {entry['errors']}")

    if accum["count"] == 0:
        print("\nNo divergent samples accumulated across any task -- nothing to save.")
        sys.exit(1)

    total_n = accum["count"]

    # ---- confident_mean / unconfident_mean (RSN-paper-compatible shape) ----
    confident_mean = (accum["conf_sum"] / total_n).astype(np.float16)
    unconfident_mean = (accum["unconf_sum"] / total_n).astype(np.float16)
    confident_mean = confident_mean[None, ...]     # (1, 1, 33, 4096)
    unconfident_mean = unconfident_mean[None, ...]  # (1, 1, 33, 4096)

    confidence_diff = confident_mean.astype(np.float32) - unconfident_mean.astype(np.float32)
    confidence_diff = confidence_diff.astype(np.float16)  # (1, 1, 33, 4096)

    # ---- equivalence check (item 8): mean(a_i - b_i) vs mean(a)-mean(b) ----
    diff_via_persample_mean = (accum["diff_sum_via_persample"] / total_n)[None, ...]  # float32, (1,1,33,4096)
    diff_via_means = confident_mean.astype(np.float32) - unconfident_mean.astype(np.float32)
    max_abs_error = float(np.max(np.abs(diff_via_persample_mean - diff_via_means)))

    np.save(conf_mean_path, confident_mean)
    np.save(unconf_mean_path, unconfident_mean)
    np.save(diff_path, confidence_diff)

    manifest = {
        "roles": list(ROLES),
        "sample_selection": (
            "ALL divergent pairs (answer_confident != answer_unconfident) across "
            "all 57 MMLU-E tasks; no filtering by selected answer letter (A-E); "
            "no train/held-out split at this stage"
        ),
        "computation_order": "mean(confident) and mean(unconfident) computed independently, "
                              "then subtracted (confidence_diff = confident_mean - unconfident_mean)",
        "accumulation_weighting": "global pooled sum across all tasks, divided once by the total "
                                   "sample count at the end (NOT a per-task average of per-task means), "
                                   "matching mean/mean_diff.py",
        "accumulation_dtype": "float32 (final mean cast to float16 for storage)",
        "size": args.size,
        "hs_dir": str(hs_dir),
        "ans_dir": str(ans_dir),
        "n_tasks_total": len(TASKS),
        "n_tasks_ok": n_ok,
        "n_tasks_failed": n_failed,
        "total_samples_used": total_n,
        "per_task": report["per_task"],
        "alignment_check": {
            "description": "Per task: confident/unconfident H5 shape equality, H5 N == answer "
                            "sample count, H5 shape == (N, 33, 4096), divergent indices within range.",
            "all_tasks_passed": n_failed == 0,
            "failed_tasks": [e["task"] for e in report["per_task"] if not e["ok"]],
        },
        "output_shapes": {
            "confident_mean": list(confident_mean.shape),
            "unconfident_mean": list(unconfident_mean.shape),
            "confidence_diff": list(confidence_diff.shape),
            "dtype": "float16",
            "note": "Shape (1, 1, 33, 4096) matches mean/mean_diff.py's diff_mean_{size}.npy / "
                    "none_diff_mean_{size}.npy convention (leading batch dim + legacy time dim); "
                    "squeeze both leading dims to get the (33, 4096) form used by "
                    "{model}_{size}_diff.npy in RoleHidden.",
        },
        "consistency_check": {
            "description": "mean(confident_i - unconfident_i) vs confident_mean - unconfident_mean, "
                            "both computed in float32 before any float16 cast",
            "max_abs_error": max_abs_error,
        },
        "output_paths": {
            "confident_mean": str(conf_mean_path),
            "unconfident_mean": str(unconf_mean_path),
            "confidence_diff": str(diff_path),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nSaved confident_mean   -> {conf_mean_path} (shape {confident_mean.shape}, dtype {confident_mean.dtype})")
    print(f"Saved unconfident_mean -> {unconf_mean_path} (shape {unconfident_mean.shape}, dtype {unconfident_mean.dtype})")
    print(f"Saved confidence_diff  -> {diff_path} (shape {confidence_diff.shape}, dtype {confidence_diff.dtype})")
    print(f"Saved manifest         -> {manifest_path}")
    print(f"Total samples used: {total_n}")
    print(f"Tasks ok/failed: {n_ok}/{n_failed}")
    print(f"Consistency check max abs error: {max_abs_error}")

    if n_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
