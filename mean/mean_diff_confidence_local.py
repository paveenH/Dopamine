#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confidence mean / diff-mean computation, matching the RSN paper's Expert /
Non-Expert convention EXACTLY (mean/mean_diff.py), so downstream
correlation / cosine / neuron-overlap analysis can consume it unchanged.

Convention verified against the existing RoleHidden outputs before writing
this script:
  - mean/mean_diff.py: diff_mean_{size}.npy / none_diff_mean_{size}.npy are
    shape (1, 1, layers, hidden) float16 -- a leading batch dim + the legacy
    "(N, 1, layers, hidden)" time dim carried over from the original .npy
    per-sample layout.
  - The final combined diff (e.g. llama3_8B_diff.npy) is the SQUEEZED
    (layers, hidden) float16 array.
  - Accumulation is GLOBAL POOLED across all tasks, weighted by sample count
    (NOT a per-task average of per-task means): one running sum/count updated
    across every task's batches, one division at the very end.
  - Accumulation dtype is float32 (never float16) to avoid overflow/precision
    loss; only the final mean is cast back to float16 for storage.

MAIN result (per current instructions -- corrected from an earlier draft):
  confident_mean   = mean over ALL PAIRED samples (no answer-based filtering)
  unconfident_mean = mean over ALL PAIRED samples (same samples, same order)
  confidence_diff  = confident_mean - unconfident_mean
"ALL paired samples" means every sample present in both the confident and
unconfident H5/answer files for a task -- NOT restricted to samples where
answer_confident != answer_unconfident. Divergent-only filtering is computed
SEPARATELY as a sensitivity output and never influences the three main files.

FAIL-CLOSED (corrected from an earlier draft): if ANY task is missing a file,
has a shape/sample-count mismatch, or fails to read, the run aborts with a
non-zero exit and writes NONE of the three main .npy files (nor the
sensitivity file) -- never a partial result computed from the surviving
tasks. All tasks are checked before any file is written.

Consistency audit: mean(confident_i - unconfident_i) is checked against
confident_mean - unconfident_mean in BOTH float32 (the production
accumulation dtype) and float64 (an independent higher-precision audit pass,
computed from the same per-sample data, to characterize how much of any
observed float32 discrepancy is summation-order rounding vs an actual logic
error). The float64 pass is diagnostic only -- it does NOT change how the
three main .npy files are computed or stored (still float32 accumulate ->
float16 store, per the original mean_diff.py convention).

This is a STANDALONE script targeting the LOCAL RoleHidden analysis tree; it
does NOT touch mean/mean_diff.py or the server-side mean/mean_diff_confidence.py.

This stage does NOT run NMD / neuron selection / PCA / manifold / steering.

Usage (run on the SERVER, where the H5/answer data live -- the server
interpreter is `python`, NOT `python3.10`):
  python mean/mean_diff_confidence_local.py \
      --hs_dir  /data1/paveen/Dopamine/components/hidden_states/llama3/mmlue_confidence \
      --ans_dir /data1/paveen/Dopamine/components/answer/llama3_confidence \
      --size 8B \
      --out_dir /data1/paveen/Dopamine/components/hidden_states_mean/llama3_confidence
Then sync --out_dir's contents to the local analysis tree:
  /Users/paveenhuang/Documents/RSNResult/RoleHidden/llama3_confidence/
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
BATCH_SIZE = 100


def divergent_mask(samples: list) -> np.ndarray:
    """Boolean mask, True where answer_confident != answer_unconfident.
    A sample missing either key is treated as NOT divergent (False), matching
    mean_diff.py's None-safe `entry.get(key)` comparison behavior. This is
    used ONLY for the separate sensitivity output, never the main result."""
    mask = np.zeros(len(samples), dtype=bool)
    for idx, entry in enumerate(samples):
        a_conf = entry.get("answer_confident")
        a_unconf = entry.get("answer_unconfident")
        if a_conf is None or a_unconf is None:
            continue
        mask[idx] = a_conf != a_unconf
    return mask


def validate_task(task, hs_dir, ans_dir, size) -> dict:
    """Read + validate one task's answer JSON and paired H5 WITHOUT
    accumulating anything. Returns a dict with 'ok', 'errors', and (if ok)
    the loaded samples + divergent mask + H5 paths, so a second pass can
    accumulate only after EVERY task has already passed validation
    (fail-closed: no accumulation starts until the whole run is known-good)."""
    ans_path = ans_dir / f"{task}_{size}_answers.json"
    h5_paths = {r: hs_dir / f"{r}_{task}_{size}.h5" for r in ROLES}
    result = {"task": task, "ok": False, "errors": []}

    if not ans_path.exists():
        result["errors"].append(f"missing answer JSON: {ans_path}")
        return result
    missing_h5 = [str(p) for p in h5_paths.values() if not p.exists()]
    if missing_h5:
        result["errors"].append(f"missing H5 file(s): {missing_h5}")
        return result

    try:
        with open(ans_path, "r", encoding="utf-8") as f:
            ans = json.load(f)
        samples = ans["data"]
    except Exception as e:
        result["errors"].append(f"failed to parse answer JSON: {type(e).__name__}: {e}")
        return result

    n_answer_samples = len(samples)
    if n_answer_samples == 0:
        result["errors"].append("answer JSON has zero samples")
        return result

    try:
        with h5py.File(h5_paths["confident"], "r") as fc, h5py.File(h5_paths["unconfident"], "r") as fu:
            shape_c = fc["hidden_states"].shape
            shape_u = fu["hidden_states"].shape
    except Exception as e:
        result["errors"].append(f"failed to read H5 header: {type(e).__name__}: {e}")
        return result

    if shape_c != shape_u:
        result["errors"].append(f"paired H5 shape mismatch: {shape_c} vs {shape_u}")
        return result
    if shape_c[0] != n_answer_samples:
        result["errors"].append(f"H5 N={shape_c[0]} != answer sample count {n_answer_samples}")
        return result
    if tuple(shape_c[1:]) != (EXPECTED_NUM_HIDDEN_STATES, EXPECTED_HIDDEN_DIM):
        result["errors"].append(
            f"unexpected H5 shape {shape_c}, expected (N, {EXPECTED_NUM_HIDDEN_STATES}, {EXPECTED_HIDDEN_DIM})"
        )
        return result

    div_mask = divergent_mask(samples)
    result.update({
        "ok": True,
        "n_answer_samples": n_answer_samples,
        "n_divergent": int(div_mask.sum()),
        "h5_shape": list(shape_c),
        "h5_paths": h5_paths,
        "div_mask": div_mask,
    })
    return result


def accumulate_task(task_info, accum, sens_accum) -> list:
    """Second pass: accumulate one already-validated task's contribution to
    (a) the MAIN all-paired-sample accumulators and (b) the SEPARATE
    divergent-only sensitivity accumulators. Both are float32 running sums,
    plus a float64 running-sum shadow for the audit. Returns a list of
    warnings (e.g. NaN/Inf rows skipped); an empty samples/H5-mismatch
    condition here would be a bug in validate_task, so this does not
    re-check shape (already validated)."""
    warnings = []
    h5_paths = task_info["h5_paths"]
    div_mask = task_info["div_mask"]
    n = task_info["n_answer_samples"]

    with h5py.File(h5_paths["confident"], "r") as fc, h5py.File(h5_paths["unconfident"], "r") as fu:
        ds_c = fc["hidden_states"]
        ds_u = fu["hidden_states"]

        for start in range(0, n, BATCH_SIZE):
            idx = list(range(start, min(start + BATCH_SIZE, n)))
            conf_batch_16 = ds_c[idx, ...]
            unconf_batch_16 = ds_u[idx, ...]

            conf_32 = conf_batch_16.astype(np.float32)
            unconf_32 = unconf_batch_16.astype(np.float32)
            finite = np.isfinite(conf_32).all(axis=(1, 2)) & np.isfinite(unconf_32).all(axis=(1, 2))
            if not finite.all():
                bad = int((~finite).sum())
                warnings.append(f"{bad} sample(s) with NaN/Inf skipped in batch starting {start}")

            batch_div_mask = div_mask[start:start + len(idx)]

            for is_main, sel_mask, acc in ((True, np.ones(len(idx), dtype=bool), accum),
                                            (False, batch_div_mask, sens_accum)):
                sel = sel_mask & finite
                if not sel.any():
                    continue
                c32 = conf_32[sel][:, None, :, :]
                u32 = unconf_32[sel][:, None, :, :]
                c64 = c32.astype(np.float64)
                u64 = u32.astype(np.float64)

                if acc["conf_sum32"] is None:
                    acc["conf_sum32"] = c32.sum(axis=0)
                    acc["unconf_sum32"] = u32.sum(axis=0)
                    acc["conf_sum64"] = c64.sum(axis=0)
                    acc["unconf_sum64"] = u64.sum(axis=0)
                    acc["diffsum32"] = (c32 - u32).sum(axis=0)
                    acc["diffsum64"] = (c64 - u64).sum(axis=0)
                else:
                    acc["conf_sum32"] += c32.sum(axis=0)
                    acc["unconf_sum32"] += u32.sum(axis=0)
                    acc["conf_sum64"] += c64.sum(axis=0)
                    acc["unconf_sum64"] += u64.sum(axis=0)
                    acc["diffsum32"] += (c32 - u32).sum(axis=0)
                    acc["diffsum64"] += (c64 - u64).sum(axis=0)
                acc["count"] += int(sel.sum())

    return warnings


def new_accum():
    return {
        "conf_sum32": None, "unconf_sum32": None,
        "conf_sum64": None, "unconf_sum64": None,
        "diffsum32": None, "diffsum64": None,
        "count": 0,
    }


def finalize(accum):
    """Return (confident_mean_fp16, unconfident_mean_fp16, diff_fp16,
    consistency_report) for one accumulator, all shape (1, 1, L, H)."""
    n = accum["count"]
    conf_mean32 = accum["conf_sum32"] / n
    unconf_mean32 = accum["unconf_sum32"] / n
    conf_mean64 = accum["conf_sum64"] / n
    unconf_mean64 = accum["unconf_sum64"] / n

    # MAIN storage path: float32 accumulate -> float16 store (unchanged from
    # the original mean_diff.py convention).
    confident_mean = conf_mean32.astype(np.float16)[None, ...]
    unconfident_mean = unconf_mean32.astype(np.float16)[None, ...]
    diff = (conf_mean32 - unconf_mean32).astype(np.float16)[None, ...]

    # Consistency audit: mean(a_i - b_i) vs mean(a) - mean(b), in float32 AND
    # float64, all BEFORE any float16 cast.
    diff_via_persample32 = accum["diffsum32"] / n
    diff_via_means32 = conf_mean32 - unconf_mean32
    max_abs_err_f32 = float(np.max(np.abs(diff_via_persample32 - diff_via_means32)))

    diff_via_persample64 = accum["diffsum64"] / n
    diff_via_means64 = conf_mean64 - unconf_mean64
    max_abs_err_f64 = float(np.max(np.abs(diff_via_persample64 - diff_via_means64)))

    consistency = {
        "description": "mean(confident_i - unconfident_i) vs confident_mean - unconfident_mean; "
                        "computed twice, once in float32 (production accumulation dtype) and once "
                        "in float64 (independent higher-precision audit), both from the same "
                        "per-sample data and before any float16 cast",
        "max_abs_error_float32": max_abs_err_f32,
        "max_abs_error_float64": max_abs_err_f64,
        "note": "float32 error, if any, is expected to be dominated by summation-order rounding "
                "noise rather than a logic error; the float64 pass should be several orders of "
                "magnitude smaller and serves as the check for that.",
    }
    return confident_mean, unconfident_mean, diff, consistency, n


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

    conf_mean_path = out_dir / f"confident_mean_{args.size}.npy"
    unconf_mean_path = out_dir / f"unconfident_mean_{args.size}.npy"
    diff_path = out_dir / f"confidence_diff_{args.size}.npy"
    manifest_path = out_dir / "confidence_mean_manifest.json"
    sens_dir = out_dir / "sensitivity_divergent_only"
    sens_conf_path = sens_dir / f"confident_mean_divergent_{args.size}.npy"
    sens_unconf_path = sens_dir / f"unconfident_mean_divergent_{args.size}.npy"
    sens_diff_path = sens_dir / f"confidence_diff_divergent_{args.size}.npy"

    for p in (conf_mean_path, unconf_mean_path, diff_path, manifest_path,
              sens_conf_path, sens_unconf_path, sens_diff_path):
        if p.exists():
            print(f"[REFUSE] Output already exists, will not overwrite: {p}")
            sys.exit(1)

    # ---- Pass 1: validate EVERY task BEFORE accumulating anything. ----
    # Fail-closed: if any task fails validation, abort here -- no file is
    # written, and no accumulation is even started on the surviving tasks.
    validations = [validate_task(task, hs_dir, ans_dir, args.size) for task in TASKS]
    failed = [v for v in validations if not v["ok"]]
    if failed:
        print(f"[REFUSE] {len(failed)}/{len(TASKS)} task(s) failed validation -- "
              f"aborting before writing any output (fail-closed).")
        for v in failed:
            print(f"  [FAIL] {v['task']}: {v['errors']}")
        sys.exit(1)

    print(f"All {len(TASKS)} tasks passed validation. Proceeding to accumulate.")

    # ---- Pass 2: accumulate now that every task is known-good. ----
    accum = new_accum()          # MAIN: all paired samples, no filtering
    sens_accum = new_accum()     # SENSITIVITY: divergent-only subset
    per_task_report = []
    for v in validations:
        warnings = accumulate_task(v, accum, sens_accum)
        per_task_report.append({
            "task": v["task"],
            "n_answer_samples": v["n_answer_samples"],
            "n_divergent": v["n_divergent"],
            "h5_shape": v["h5_shape"],
            "warnings": warnings,
        })
        print(f"[ok] {v['task']}: n_answer_samples={v['n_answer_samples']}, "
              f"n_divergent={v['n_divergent']}, cumulative main total={accum['count']}, "
              f"cumulative divergent total={sens_accum['count']}")

    if accum["count"] == 0:
        print("[REFUSE] Zero samples accumulated in the MAIN (all-paired) pass -- aborting.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    confident_mean, unconfident_mean, confidence_diff, consistency, total_n = finalize(accum)
    np.save(conf_mean_path, confident_mean)
    np.save(unconf_mean_path, unconfident_mean)
    np.save(diff_path, confidence_diff)

    sens_result = None
    if sens_accum["count"] > 0:
        sens_dir.mkdir(parents=True, exist_ok=True)
        s_conf, s_unconf, s_diff, s_consistency, s_n = finalize(sens_accum)
        np.save(sens_conf_path, s_conf)
        np.save(sens_unconf_path, s_unconf)
        np.save(sens_diff_path, s_diff)
        sens_result = {
            "total_samples_used": s_n,
            "consistency_check": s_consistency,
            "output_paths": {
                "confident_mean": str(sens_conf_path),
                "unconfident_mean": str(sens_unconf_path),
                "confidence_diff": str(sens_diff_path),
            },
        }
    else:
        print("[note] No divergent samples found anywhere -- sensitivity output skipped.")

    manifest = {
        "roles": list(ROLES),
        "main_sample_selection": (
            "ALL PAIRED samples across all tasks -- every sample present in both the "
            "confident and unconfident H5/answer files for a task. No filtering by which "
            "of A-E was selected. No train/held-out split at this stage."
        ),
        "sensitivity_sample_selection": (
            "Divergent pairs ONLY (answer_confident != answer_unconfident), computed "
            "SEPARATELY and saved under sensitivity_divergent_only/. Does NOT influence "
            "the three main files above."
        ),
        "fail_closed": (
            "Every task is validated (file presence, paired H5 shape equality, H5 N == "
            "answer sample count, H5 shape == (N, 33, 4096)) BEFORE any accumulation or "
            "file write. If any task fails validation the run aborts with exit 1 and "
            "writes nothing -- never a partial result computed from the surviving tasks."
        ),
        "computation_order": "mean(confident) and mean(unconfident) computed independently, "
                              "then subtracted (confidence_diff = confident_mean - unconfident_mean)",
        "accumulation_weighting": "global pooled sum across all tasks, divided once by the total "
                                   "sample count at the end (NOT a per-task average of per-task means), "
                                   "matching mean/mean_diff.py",
        "accumulation_dtype": "float32 for the production mean (final mean cast to float16 for "
                               "storage); a float64 shadow pass is run in parallel for the "
                               "consistency audit only and never used for the saved .npy files",
        "size": args.size,
        "hs_dir": str(hs_dir),
        "ans_dir": str(ans_dir),
        "n_tasks_total": len(TASKS),
        "n_tasks_validated_ok": len(validations),
        "total_samples_used_main": total_n,
        "per_task": per_task_report,
        "alignment_check": {
            "description": "Per task: confident/unconfident H5 shape equality, H5 N == answer "
                            "sample count, H5 shape == (N, 33, 4096); checked for ALL tasks "
                            "before any accumulation (fail-closed).",
            "all_tasks_passed": True,
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
        "consistency_check_main": consistency,
        "sensitivity_divergent_only": sens_result,
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
    print(f"Total samples used (MAIN, all-paired): {total_n}")
    print(f"Consistency check (MAIN): float32 max_abs_err={consistency['max_abs_error_float32']}, "
          f"float64 max_abs_err={consistency['max_abs_error_float64']}")
    if sens_result:
        print(f"Sensitivity (divergent-only) samples used: {sens_result['total_samples_used']}")
        print(f"Sensitivity outputs -> {sens_dir}")


if __name__ == "__main__":
    main()
