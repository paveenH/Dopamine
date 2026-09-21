#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert - Non-expert mean/diff computation for the two GSM8K Role hidden-state
extractions (extract_gsm8k_role_hs.py): gsm8k_role (no abstention exit) and
gsm8k_role_abstention (v2 abstention-enabled prompt), both over the full
1319-question GSM8K test set. Sibling of mean/mean_diff_chat_construction.py
-- same fail-closed conventions, dtype convention, and atomic-write pattern,
NOT a re-derivation -- adapted from a chat/bare axis to an expert/non_expert
axis and a variable, task-selected output subdirectory instead of one fixed
task name.

Actual input schema (as produced by extract_gsm8k_role_hs.py, read directly,
not assumed):
  components/hidden_states/{model}/{task}/{condition}_{size}.h5
    - top-level dataset "hidden_states", shape (1319, n_layers, hidden_size),
      dtype float16. n_layers index 0 = embedding output, 1..N = decoder
      layer outputs.
    - top-level dataset "in_300_sample", shape (1319,), dtype bool --
      provenance-only membership flag (whether this row's question is in the
      original frozen 300-question benchmark/gsm8k_test_sample.json).
    - top-level HDF5 .attrs carry: n_samples, n_samples_done,
      hidden_state_shape (JSON string), ordered_sample_identity_sha256,
      questions_sha256, chat_template_applied (always False here),
      steering_applied (always False), generation_performed (always False),
      prefill_only (always True), model, task, condition, etc.
  components/hidden_states/{model}/{task}/{condition}_{size}.manifest.json
    - sidecar JSON, superset of the H5 attrs.

DIRECTION (frozen, do NOT flip): diff_mean = expert_mean - non_expert_mean.
Symbol convention matches every other Expert-vs-Non-expert direction in this
repo (e.g. mean/mean_diff.py's own role_diff = expert_mean - non_expert_mean).

FAIL-CLOSED: each (model, task) cell is validated BEFORE any accumulation.
Any failure aborts with exit 1 and writes nothing for that cell, including:
missing file, wrong --size for the locked model, n_samples != 1319 (the
locked full-test-set count), shape mismatch, expert/non_expert shape or
pairing-digest mismatch, expert/non_expert questions_sha256 mismatch, a
model/task/condition field in the H5 attrs disagreeing with the file being
read, generation_performed != False, prefill_only != True,
chat_template_applied != False, steering_applied != False, non-finite
values, n_in_300_sample/n_remaining_construction not summing to 1319 or not
equal to 300/1019 respectively, expert/non_expert in_300_sample arrays not
being IDENTICAL (both conditions must flag the SAME rows as in-300, since
both read the same gsm8k_1319_full.json).

THREE MEAN/DIFF COMPUTATIONS PER (model, task) CELL, not one -- this is the
whole point of this script relative to mean_diff_chat_construction.py, which
computes only one full-sample diff. Per the task's own requirement ("分别报
告全量 1319 题，以及原固定 300 题与其余 1019 题的方向一致性，作为稳定性检查，
不用任何子集重新挑选方向"):
  (a) full_1319:              all 1319 rows (the PRIMARY direction)
  (b) subset_300:              only rows with in_300_sample=True (n=300)
  (c) subset_remaining_1019:   only rows with in_300_sample=False (n=1019)
(b) and (c) are NOT alternative directions and are NEVER substituted for (a)
anywhere downstream -- they exist ONLY to report a stability/consistency
check (cosine similarity, per-layer and pooled, of (b) vs (c) vs (a)) on the
SAME already-fixed full-1319 direction's construction, never to re-select a
"better" subset after the fact.

Accumulation dtype: float32 (never float16) for the running sum; final
mean/diff cast to float16 only for the saved .npy, matching
mean_diff_chat_construction.py's stated rationale. A float64 shadow pass
audits the float32 accumulation error, also matching that convention.

Loads NO model, uses NO GPU, never writes to (or modifies) the input H5
source files.

Usage (run on the SERVER -- server interpreter is `python`, NOT `python3.10`):
  python mean/mean_diff_gsm8k_role.py --model llama3 --size 8B \
      --task gsm8k_role \
      --hs_root /data1/paveen/Dopamine/components/hidden_states \
      --out_root /data1/paveen/Dopamine/components/hidden_states_mean
  python mean/mean_diff_gsm8k_role.py --model llama3 --size 8B \
      --task gsm8k_role_abstention \
      --hs_root /data1/paveen/Dopamine/components/hidden_states \
      --out_root /data1/paveen/Dopamine/components/hidden_states_mean

Output layout follows the existing per-model directory organization:
  <out_root>/{model}/{task}/{expert_mean,non_expert_mean,diff_mean}_{size}.npy
  <out_root>/{model}/{task}/manifest.json
    (carries the full_1319 diff's stats PLUS the subset_300/
    subset_remaining_1019 cosine-consistency check; no additional .npy files
    are written for the two subset means -- they are computed in-memory for
    the consistency check only and are not persisted as separate artifacts,
    since they are diagnostic, not primary, directions)
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

SCRIPT_VERSION = "mean_diff_gsm8k_role-v1"

TASKS = ("gsm8k_role", "gsm8k_role_abstention")
CONDITIONS = ("expert", "non_expert")
DIRECTION = "expert_minus_non_expert"

MODEL_SHAPE = {
    "llama3": {"n_layers": 33, "hidden_size": 4096},
    "qwen2.5": {"n_layers": 29, "hidden_size": 3584},
}
MODEL_SIZE = {"llama3": "8B", "qwen2.5": "7B"}

EXPECTED_N_SAMPLES = 1319
EXPECTED_N_IN_300 = 300
EXPECTED_N_REMAINING = 1019


def die(msg: str, code: int = 1):
    print(f"[REFUSE] {msg}", file=sys.stderr)
    sys.exit(code)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def h5_path(hs_root: Path, model: str, task: str, condition: str, size: str) -> Path:
    return hs_root / model / task / f"{condition}_{size}.h5"


def manifest_path(hs_root: Path, model: str, task: str, condition: str, size: str) -> Path:
    return hs_root / model / task / f"{condition}_{size}.manifest.json"


def validate_task(model: str, task: str, hs_root: Path, size: str) -> dict:
    result = {"model": model, "task": task, "ok": False, "errors": []}
    expect = MODEL_SHAPE[model]

    expected_size = MODEL_SIZE[model]
    if size != expected_size:
        result["errors"].append(
            f"--size={size!r} passed but model={model} is locked to "
            f"size={expected_size!r}.")
        return result

    h5_paths = {c: h5_path(hs_root, model, task, c, size) for c in CONDITIONS}
    manifest_paths = {c: manifest_path(hs_root, model, task, c, size) for c in CONDITIONS}
    missing = [str(p) for p in list(h5_paths.values()) + list(manifest_paths.values())
               if not p.exists()]
    if missing:
        result["errors"].append(f"missing file(s): {missing}")
        return result

    manifests = {}
    for c in CONDITIONS:
        try:
            with open(manifest_paths[c], "r", encoding="utf-8") as f:
                manifests[c] = json.load(f)
        except Exception as e:
            result["errors"].append(
                f"failed to parse manifest {manifest_paths[c]}: {type(e).__name__}: {e}")
            return result

    shapes = {}
    attrs = {}
    in_300_flags = {}
    try:
        for c in CONDITIONS:
            with h5py.File(h5_paths[c], "r") as f:
                ds = f["hidden_states"]
                shapes[c] = ds.shape
                attrs[c] = dict(f.attrs)
                if "in_300_sample" not in f:
                    result["errors"].append(
                        f"condition={c}: H5 has no 'in_300_sample' dataset -- "
                        "this must be an extract_gsm8k_role_hs.py output.")
                    return result
                in_300_flags[c] = f["in_300_sample"][...]
    except Exception as e:
        result["errors"].append(f"failed to read H5 header: {type(e).__name__}: {e}")
        return result

    if shapes["expert"] != shapes["non_expert"]:
        result["errors"].append(
            f"expert/non_expert H5 shape mismatch: {shapes['expert']} vs "
            f"{shapes['non_expert']}")
        return result

    n, n_layers, hidden = shapes["expert"]
    if n != EXPECTED_N_SAMPLES:
        result["errors"].append(
            f"n_samples={n}, expected exactly {EXPECTED_N_SAMPLES} (the "
            "locked full-GSM8K-test-set count).")
        return result
    if (n_layers, hidden) != (expect["n_layers"], expect["hidden_size"]):
        result["errors"].append(
            f"unexpected shape ({n_layers}, {hidden}), expected "
            f"({expect['n_layers']}, {expect['hidden_size']}) for model={model}")
        return result

    for c in CONDITIONS:
        a = attrs[c]
        n_done = a.get("n_samples_done")
        if n_done is None or int(n_done) != n:
            result["errors"].append(
                f"condition={c}: n_samples_done={n_done} != n={n} in H5 attrs.")
            return result
        if bool(a.get("steering_applied", True)) is not False:
            result["errors"].append(
                f"condition={c}: steering_applied is not False in H5 attrs.")
            return result
        a_model = a.get("model")
        if a_model != model:
            result["errors"].append(
                f"condition={c}: H5 attrs record model={a_model!r}, "
                f"expected {model!r}.")
            return result
        a_task = a.get("task")
        if a_task != task:
            result["errors"].append(
                f"condition={c}: H5 attrs record task={a_task!r}, "
                f"expected {task!r}.")
            return result
        a_condition = a.get("condition")
        if a_condition != c:
            result["errors"].append(
                f"file at the '{c}' path has H5 attrs recording "
                f"condition={a_condition!r}.")
            return result
        if bool(a.get("generation_performed", True)) is not False:
            result["errors"].append(
                f"condition={c}: generation_performed is not False in H5 attrs.")
            return result
        if bool(a.get("prefill_only", False)) is not True:
            result["errors"].append(
                f"condition={c}: prefill_only is not True in H5 attrs.")
            return result
        if bool(a.get("chat_template_applied", True)) is not False:
            result["errors"].append(
                f"condition={c}: chat_template_applied is not False in H5 "
                "attrs (this pipeline never applies a chat template).")
            return result

        n_in_300 = int(a.get("n_in_300_sample", -1))
        n_remaining = int(a.get("n_remaining_construction", -1))
        if n_in_300 != EXPECTED_N_IN_300 or n_remaining != EXPECTED_N_REMAINING:
            result["errors"].append(
                f"condition={c}: n_in_300_sample={n_in_300} "
                f"n_remaining_construction={n_remaining}, expected "
                f"{EXPECTED_N_IN_300}/{EXPECTED_N_REMAINING}.")
            return result
        if n_in_300 + n_remaining != n:
            result["errors"].append(
                f"condition={c}: n_in_300_sample + n_remaining_construction "
                f"({n_in_300}+{n_remaining}) != n_samples ({n}).")
            return result
        if int(in_300_flags[c].sum()) != n_in_300:
            result["errors"].append(
                f"condition={c}: in_300_sample dataset has "
                f"{int(in_300_flags[c].sum())} True rows, attrs claim "
                f"n_in_300_sample={n_in_300} -- disagreement.")
            return result

    if not np.array_equal(in_300_flags["expert"], in_300_flags["non_expert"]):
        result["errors"].append(
            "expert/non_expert in_300_sample arrays differ -- both "
            "conditions must flag the identical set of rows as in-300 "
            "(they were extracted from the same gsm8k_1319_full.json).")
        return result

    digest_expert = attrs["expert"].get("ordered_sample_identity_sha256")
    digest_non = attrs["non_expert"].get("ordered_sample_identity_sha256")
    if not digest_expert or digest_expert != digest_non:
        result["errors"].append(
            f"expert/non_expert ordered_sample_identity_sha256 mismatch: "
            f"{digest_expert!r} vs {digest_non!r}.")
        return result

    q_sha_expert = attrs["expert"].get("questions_sha256")
    q_sha_non = attrs["non_expert"].get("questions_sha256")
    if not q_sha_expert or q_sha_expert != q_sha_non:
        result["errors"].append(
            f"expert/non_expert questions_sha256 mismatch: "
            f"{q_sha_expert!r} vs {q_sha_non!r}.")
        return result

    manifest_digest_expert = manifests["expert"].get("ordered_sample_identity_sha256")
    manifest_digest_non = manifests["non_expert"].get("ordered_sample_identity_sha256")
    if manifest_digest_expert != digest_expert or manifest_digest_non != digest_non:
        result["errors"].append(
            "manifest.json ordered_sample_identity_sha256 disagrees with the "
            "H5 attrs of the same file.")
        return result

    result.update({
        "ok": True,
        "n_samples": n,
        "shape": [n, n_layers, hidden],
        "h5_paths": {c: str(h5_paths[c]) for c in CONDITIONS},
        "manifest_paths": {c: str(manifest_paths[c]) for c in CONDITIONS},
        "pairing_digest": digest_expert,
        "h5_sha256": {c: sha256_of_file(h5_paths[c]) for c in CONDITIONS},
        "in_300_mask": in_300_flags["expert"],  # identical between conditions
    })
    return result


def load_and_check_finite(h5_file_path: Path) -> np.ndarray:
    with h5py.File(h5_file_path, "r") as f:
        arr = f["hidden_states"][...]
    if not np.isfinite(arr.astype(np.float32)).all():
        raise ValueError(f"non-finite value(s) found in {h5_file_path}")
    return arr


def cosine_per_layer_and_pooled(a: np.ndarray, b: np.ndarray):
    """a, b: (n_layers, hidden). Returns (per_layer_cosine (n_layers,),
    pooled_cosine over the flattened vector)."""
    eps = 1e-12
    an = np.linalg.norm(a, axis=-1)
    bn = np.linalg.norm(b, axis=-1)
    dots = np.sum(a * b, axis=-1)
    per_layer = dots / np.maximum(an * bn, eps)

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    pooled = float(np.dot(a_flat, b_flat) /
                   max(np.linalg.norm(a_flat) * np.linalg.norm(b_flat), eps))
    return per_layer.tolist(), pooled


def compute_task_means(task_info: dict) -> dict:
    expert_16 = load_and_check_finite(Path(task_info["h5_paths"]["expert"]))
    non_16 = load_and_check_finite(Path(task_info["h5_paths"]["non_expert"]))
    in_300_mask = task_info["in_300_mask"]  # (n,) bool

    def diff_for_mask(mask):
        e32 = expert_16[mask].astype(np.float32)
        n32 = non_16[mask].astype(np.float32)
        e_mean = e32.mean(axis=0)
        n_mean = n32.mean(axis=0)
        d_mean = e_mean - n_mean
        return e_mean, n_mean, d_mean, int(mask.sum())

    # (a) full 1319 -- PRIMARY
    full_mask = np.ones(expert_16.shape[0], dtype=bool)
    e_mean_full32, n_mean_full32, d_mean_full32, n_full = diff_for_mask(full_mask)

    diff_via_persample32 = (
        expert_16.astype(np.float32) - non_16.astype(np.float32)).mean(axis=0)
    max_abs_err_f32 = float(np.max(np.abs(diff_via_persample32 - d_mean_full32)))

    expert_64 = expert_16.astype(np.float64)
    non_64 = non_16.astype(np.float64)
    e_mean_full64 = expert_64.mean(axis=0)
    n_mean_full64 = non_64.mean(axis=0)
    d_mean_full64 = e_mean_full64 - n_mean_full64
    diff_via_persample64 = (expert_64 - non_64).mean(axis=0)
    max_abs_err_f64 = float(np.max(np.abs(diff_via_persample64 - d_mean_full64)))

    direct_diff = expert_16.astype(np.float32).mean(axis=0) - \
        non_16.astype(np.float32).mean(axis=0)
    max_abs_diff_identity_err = float(
        np.max(np.abs(direct_diff - d_mean_full32)))
    if max_abs_diff_identity_err > 1e-3:
        raise ValueError(
            "full_1319 diff_mean != expert_mean - non_expert_mean identity "
            f"check FAILED: max_abs_err={max_abs_diff_identity_err}")

    # (b) subset_300, (c) subset_remaining_1019 -- DIAGNOSTIC ONLY, computed
    # in-memory for the stability/consistency check, never persisted as
    # separate .npy artifacts and never substituted for (a).
    e_mean_300_32, n_mean_300_32, d_mean_300_32, n_300 = \
        diff_for_mask(in_300_mask)
    e_mean_rem_32, n_mean_rem_32, d_mean_rem_32, n_rem = \
        diff_for_mask(~in_300_mask)

    if n_300 != EXPECTED_N_IN_300 or n_rem != EXPECTED_N_REMAINING:
        raise ValueError(
            f"subset sizes n_300={n_300} n_remaining={n_rem} do not match "
            f"expected {EXPECTED_N_IN_300}/{EXPECTED_N_REMAINING}.")

    per_layer_cos_300_vs_full, pooled_cos_300_vs_full = \
        cosine_per_layer_and_pooled(d_mean_300_32, d_mean_full32)
    per_layer_cos_rem_vs_full, pooled_cos_rem_vs_full = \
        cosine_per_layer_and_pooled(d_mean_rem_32, d_mean_full32)
    per_layer_cos_300_vs_rem, pooled_cos_300_vs_rem = \
        cosine_per_layer_and_pooled(d_mean_300_32, d_mean_rem_32)

    stability_check = {
        "description": (
            "Direction-consistency (NOT re-selection) check: cosine "
            "similarity of the diff_mean computed on the subset_300 rows "
            "and on the subset_remaining_1019 rows, each compared against "
            "the PRIMARY full_1319 diff_mean and against each other. Both "
            "subsets are read off the SAME already-fixed full-1319 "
            "extraction/direction; no subset was re-selected or "
            "re-weighted based on this result."
        ),
        "n_subset_300": n_300,
        "n_subset_remaining_1019": n_rem,
        "pooled_cosine_subset_300_vs_full_1319": pooled_cos_300_vs_full,
        "pooled_cosine_subset_remaining_1019_vs_full_1319": pooled_cos_rem_vs_full,
        "pooled_cosine_subset_300_vs_subset_remaining_1019": pooled_cos_300_vs_rem,
        "per_layer_cosine_subset_300_vs_full_1319": per_layer_cos_300_vs_full,
        "per_layer_cosine_subset_remaining_1019_vs_full_1319": per_layer_cos_rem_vs_full,
        "per_layer_cosine_subset_300_vs_subset_remaining_1019": per_layer_cos_300_vs_rem,
    }

    e_mean_out = e_mean_full32.astype(np.float16)[None, None, ...]
    n_mean_out = n_mean_full32.astype(np.float16)[None, None, ...]
    d_mean_out = d_mean_full32.astype(np.float16)[None, None, ...]

    return {
        "expert_mean": e_mean_out,
        "non_expert_mean": n_mean_out,
        "diff_mean": d_mean_out,
        "n_samples": n_full,
        "consistency_check": {
            "description": "mean(expert_i - non_expert_i) vs expert_mean - "
                            "non_expert_mean over the full 1319, computed "
                            "in float32 (production accumulation dtype) and "
                            "float64 (independent higher-precision audit), "
                            "both before any float16 cast",
            "max_abs_error_float32": max_abs_err_f32,
            "max_abs_error_float64": max_abs_err_f64,
        },
        "diff_identity_check": {
            "description": "diff_mean (float32, pre-cast) vs expert_mean32 "
                            "- non_expert_mean32, independently re-derived",
            "max_abs_error_float32": max_abs_diff_identity_err,
            "passed": True,
        },
        "stability_check": stability_check,
    }


def atomic_save_npy(path: Path, arr: np.ndarray):
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".npy", dir=str(path.parent), prefix=path.stem + "_")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            np.save(f, arr)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def atomic_write_json(path: Path, obj: dict):
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".json.tmp", dir=str(path.parent), prefix=path.stem + "_")
    os.close(tmp_fd)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SHAPE.keys()))
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--size", required=True)
    ap.add_argument("--hs_root", required=True,
                     help="e.g. /data1/paveen/Dopamine/components/hidden_states")
    ap.add_argument("--out_root", required=True,
                     help="e.g. /data1/paveen/Dopamine/components/hidden_states_mean")
    args = ap.parse_args()

    model = args.model
    task = args.task
    size = args.size
    hs_root = Path(args.hs_root)
    out_root = Path(args.out_root)
    task_dir = out_root / model / task

    all_out_paths = [
        task_dir / f"expert_mean_{size}.npy",
        task_dir / f"non_expert_mean_{size}.npy",
        task_dir / f"diff_mean_{size}.npy",
        task_dir / "manifest.json",
    ]
    existing = [str(p) for p in all_out_paths if p.exists()]
    if existing:
        die(f"refusing to run model={model} task={task}: output file(s) "
            "already exist:\n  " + "\n  ".join(existing) +
            "\nDelete them deliberately first if a re-run is truly intended.")

    v = validate_task(model, task, hs_root, size)
    if not v["ok"]:
        print(f"[REFUSE] task={task} failed validation for model={model} -- "
              "aborting before writing any output (fail-closed).")
        print(f"  [FAIL] {v['errors']}")
        sys.exit(1)

    print(f"Task {task} passed validation for model={model}, size={size}, "
          f"n_samples={v['n_samples']}.")

    print(f"[compute] model={model} task={task} n_samples={v['n_samples']} "
          f"shape={v['shape']}")
    means = compute_task_means(v)

    task_dir.mkdir(parents=True, exist_ok=True)
    expert_path = task_dir / f"expert_mean_{size}.npy"
    non_expert_path = task_dir / f"non_expert_mean_{size}.npy"
    diff_path = task_dir / f"diff_mean_{size}.npy"
    manifest_out_path = task_dir / "manifest.json"

    atomic_save_npy(expert_path, means["expert_mean"])
    atomic_save_npy(non_expert_path, means["non_expert_mean"])
    atomic_save_npy(diff_path, means["diff_mean"])

    manifest = {
        "script_version": SCRIPT_VERSION,
        "model": model,
        "task": task,
        "size": size,
        "n_samples": v["n_samples"],
        "input_shape": v["shape"],
        "output_shape": list(means["expert_mean"].shape),
        "accumulation_dtype": "float32 (float64 shadow pass for audit only)",
        "output_dtype": "float16",
        "difference_direction": DIRECTION,
        "pairing_digest": v["pairing_digest"],
        "input_h5_sha256": v["h5_sha256"],
        "input_h5_paths": v["h5_paths"],
        "input_manifest_paths": v["manifest_paths"],
        "output_paths": {
            "expert_mean": str(expert_path),
            "non_expert_mean": str(non_expert_path),
            "diff_mean": str(diff_path),
        },
        "consistency_check": means["consistency_check"],
        "diff_identity_check": means["diff_identity_check"],
        "stability_check": means["stability_check"],
        "fail_closed": (
            "The task is validated (H5+manifest presence, expert/non_expert "
            "shape equality, n_samples==1319, expected (n_layers, "
            "hidden_size), n_samples_done==n, steering_applied==False, "
            "generation_performed==False, prefill_only==True, "
            "chat_template_applied==False, n_in_300_sample==300, "
            "n_remaining_construction==1019, expert/non_expert "
            "in_300_sample arrays identical, ordered_sample_identity_sha256 "
            "equality between expert/non_expert AND between H5 attrs/"
            "manifest.json) BEFORE any accumulation or file write; a "
            "failing validation aborts the run and writes nothing."
        ),
        "computation": (
            "expert_mean/non_expert_mean = elementwise mean over the sample "
            "axis (per layer, per hidden dim), computed over the FULL 1319 "
            "rows (primary direction); diff_mean = expert_mean - "
            "non_expert_mean. The subset_300/subset_remaining_1019 means "
            "are computed in-memory ONLY for the stability_check cosine "
            "comparison above and are not saved as separate .npy files."
        ),
        "note": "GSM8K Role hidden-state pipeline (extract_gsm8k_role_hs.py). "
                "Deliberately NOT named RRSN/ARRSN -- see CLAUDE.md's "
                "'Reasoning RSN (RRSN) history' entry (both prior lines "
                "retired 2026-09-18); this is a plainly-named, independent "
                "GSM8K Role direction. gsm8k_role has no abstention exit "
                "(matches the existing Bare behavior-experiment prompt); "
                "gsm8k_role_abstention uses the v2 abstention-enabled prompt "
                "(get_answer_gsm8k_role_abstention_v2.py's PROMPT_TEMPLATE, "
                "imported not retyped).",
        "extraction_timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(manifest_out_path, manifest)

    print(f"[ok] model={model} task={task}: wrote expert_mean/non_expert_mean/"
          f"diff_mean + manifest.json under {task_dir}")
    print({
        "task": task,
        "n_samples": v["n_samples"],
        "pairing_digest": v["pairing_digest"],
        "consistency_max_abs_error_float32":
            means["consistency_check"]["max_abs_error_float32"],
        "consistency_max_abs_error_float64":
            means["consistency_check"]["max_abs_error_float64"],
        "diff_identity_passed": means["diff_identity_check"]["passed"],
        "pooled_cosine_subset_300_vs_full_1319":
            means["stability_check"]["pooled_cosine_subset_300_vs_full_1319"],
        "pooled_cosine_subset_remaining_1019_vs_full_1319":
            means["stability_check"]["pooled_cosine_subset_remaining_1019_vs_full_1319"],
        "pooled_cosine_subset_300_vs_subset_remaining_1019":
            means["stability_check"]["pooled_cosine_subset_300_vs_subset_remaining_1019"],
    })


if __name__ == "__main__":
    main()
