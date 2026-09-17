#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert / Non-expert mean + diff computation for the GSM8K abstention-enabled
Role HS extraction (extract_gsm8k_abstention_role_hs.py). Server-side,
GSM8K-only, single (expert, non_expert) H5 pair for one model.

Follows the SAME fail-closed conventions as mean/mean_role_reasoning.py
(validate-before-load, load-before-write, atomic writes), scoped down to
this experiment's single cell instead of that script's 3-task loop. Unlike
mean_role_reasoning.py, this script DOES compute and save the diff (per
explicit instruction for this experiment):

  gsm8k_abstention_role_diff = expert_mean - non_expert_mean

raw sign, NO L2 normalization anywhere.

Input schema (as produced by extract_gsm8k_abstention_role_hs.py, read
directly, not assumed):
  components/hidden_states/{model}/gsm8k_abstention_role/{condition}_{size}.h5
    top-level dataset "hidden_states", shape (300, n_layers, hidden_size),
    dtype float16. n_layers index 0 = embedding output, 1..N = decoder layer
    outputs.
    top-level HDF5 .attrs carry (among others): n_samples, n_samples_done,
    ordered_sample_identity_sha256, steering_applied (always False here),
    generation_performed (always False here), prefill_only (always True).
  components/hidden_states/{model}/gsm8k_abstention_role/manifest.json
    experiment-level manifest (ONE file, both conditions), written by
    extract_gsm8k_abstention_role_hs.py. This script reads it only for a
    pairing-digest sanity cross-check (second, independent confirmation
    channel), matching mean_role_reasoning.py's "trust the H5, not the
    sidecar" style -- all gating checks are read directly from each H5's
    own attrs + dataset.

OUTPUT (same directory as the input H5 pair, per instruction -- this is a
small-file, server-computed step whose output syncs to the local analysis
workspace, not the large H5 themselves):
  components/hidden_states/{model}/gsm8k_abstention_role/
    expert_mean_{size}.npy
    non_expert_mean_{size}.npy
    gsm8k_abstention_role_diff_{size}.npy

ACCUMULATION DTYPE: float32 (never float16) for the running sum, matching
mean_diff.py / mean_diff_confidence_local.py / mean_diff_chat.py /
mean_role_reasoning.py's stated rationale. NO L2 normalization anywhere --
direct arithmetic mean over raw hidden states, and a direct subtraction for
the diff.

OUTPUT DTYPE, TWO DIFFERENT VALUES BY DESIGN (not a typo, not uniform):
expert_mean/non_expert_mean are cast to FLOAT16 for saving, per the legacy
(1, 1, n_layers, hidden_size) mean-file convention. The DIRECTION
(gsm8k_abstention_role_diff) is then computed from those SAME SAVED float16
mean arrays -- upcast back to float32, THEN subtracted -- and saved as
FLOAT32, matching the existing formal reasoning Role-direction construction
and the old GSM8K Role direction file's dtype. This is deliberately NOT the
same as subtracting the two mean arrays before their float16 cast: the two
orders (round-then-subtract vs subtract-then-round) differ by float16
rounding, which could otherwise shift an exact-NMD top-k selection against
the existing pipeline. See compute_means_and_diff()'s own docstring for the
exact construction.

FAIL-CLOSED, THREE PASSES: (1) validate the expert+non_expert H5/manifest
headers+attrs (existence, shape, n_samples==300, n_samples_done==300,
steering_applied==False, generation_performed==False, pairing-digest
equality between the two H5, and cross-check against the experiment-level
manifest.json's OWN recorded digest) WITHOUT loading the hidden_states
arrays; (2) load both full arrays, check finiteness, compute
expert_mean/non_expert_mean/diff IN MEMORY, still without writing; (3) only
once both conditions have passed BOTH pass 1 and pass 2 are any files
written. This mirrors mean_role_reasoning.py's own three-pass discipline
exactly (this script has only one "task" -- GSM8K -- where that script
loops over three).

Loads NO model, uses NO GPU. Never writes to, or modifies, the input
expert_{size}.h5 / non_expert_{size}.h5 files themselves -- those are only
ever opened read-only.

The experiment-level manifest.json IS updated, but NOT by rewriting or
removing anything extract_gsm8k_abstention_role_hs.py wrote to it: this
script performs one ADDITIVE, atomic (tempfile+os.replace) update that adds
a new "mean_diff" key recording the mean/diff output paths, dtypes, shapes,
and the consistency-audit numbers. Every key already written by
extract_gsm8k_abstention_role_hs.py is read back unchanged and re-written
byte-for-byte as part of the same JSON object -- none of them is altered or
removed.

Usage (run on the SERVER -- server interpreter is `python`, NOT `python3.10`,
matching every other server-side script in this family):
  python mean/mean_gsm8k_abstention_role.py --model llama3  --size 8B \
      --hs_root /data1/paveen/Dopamine/components/hidden_states
  python mean/mean_gsm8k_abstention_role.py --model qwen2.5 --size 7B \
      --hs_root /data1/paveen/Dopamine/components/hidden_states

Then sync ONLY the small outputs (the two mean .npy, the diff .npy, and the
updated manifest.json) to the local analysis workspace -- NOT the large
expert_{size}.h5 / non_expert_{size}.h5 files themselves.
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

SCRIPT_VERSION = "mean_gsm8k_abstention_role-v1"

CONDITIONS = ("expert", "non_expert")

MODEL_SHAPE = {
    "llama3": {"n_layers": 33, "hidden_size": 4096},
    "qwen2.5": {"n_layers": 29, "hidden_size": 3584},
}

# --size is locked to --model, same rationale as mean_role_reasoning.py: a
# typo'd/mismatched size cannot silently construct a misleading path.
MODEL_EXPECTED_SIZE = {
    "llama3": "8B",
    "qwen2.5": "7B",
}


def die(msg: str, code: int = 1):
    print(f"[REFUSE] {msg}", file=sys.stderr)
    sys.exit(code)


def assert_size_matches_model(model: str, size: str) -> None:
    expected = MODEL_EXPECTED_SIZE[model]
    if size != expected:
        die(f"--size {size!r} does not match --model {model!r} (expected "
            f"--size {expected!r}). --size is locked to the model to avoid "
            "constructing a misleading output path under the wrong size "
            "suffix.")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def h5_path(out_dir: Path, condition: str, size: str) -> Path:
    return out_dir / f"{condition}_{size}.h5"


def manifest_path(out_dir: Path) -> Path:
    return out_dir / "manifest.json"


# ---------------------------------------------------------------------------
# Pass 1: validate BOTH conditions' H5 headers/attrs + the experiment
# manifest, WITHOUT loading the hidden_states arrays.
# ---------------------------------------------------------------------------

def validate(model: str, out_dir: Path, size: str) -> dict:
    result = {"model": model, "ok": False, "errors": []}
    expect = MODEL_SHAPE[model]

    h5_paths = {c: h5_path(out_dir, c, size) for c in CONDITIONS}
    exp_manifest_path = manifest_path(out_dir)
    missing = [str(p) for p in list(h5_paths.values()) + [exp_manifest_path]
               if not p.exists()]
    if missing:
        result["errors"].append(f"missing file(s): {missing}")
        return result

    try:
        with open(exp_manifest_path, "r", encoding="utf-8") as f:
            exp_manifest = json.load(f)
    except Exception as e:
        result["errors"].append(
            f"failed to parse experiment manifest {exp_manifest_path}: "
            f"{type(e).__name__}: {e}")
        return result

    if exp_manifest.get("experiment") != "gsm8k_abstention_role":
        result["errors"].append(
            "experiment manifest does not declare "
            "experiment='gsm8k_abstention_role' "
            f"(got {exp_manifest.get('experiment')!r}) -- refusing to "
            "compute means from an unrecognized manifest.")
        return result

    shapes = {}
    attrs = {}
    try:
        for c in CONDITIONS:
            with h5py.File(h5_paths[c], "r") as f:
                ds = f["hidden_states"]
                shapes[c] = ds.shape
                attrs[c] = dict(f.attrs)
    except Exception as e:
        result["errors"].append(f"failed to read H5 header: {type(e).__name__}: {e}")
        return result

    if shapes["expert"] != shapes["non_expert"]:
        result["errors"].append(
            f"expert/non_expert H5 shape mismatch: {shapes['expert']} vs "
            f"{shapes['non_expert']}")
        return result

    n, n_layers, hidden = shapes["expert"]
    if n != 300:
        result["errors"].append(f"n_samples={n}, expected exactly 300")
        return result
    if (n_layers, hidden) != (expect["n_layers"], expect["hidden_size"]):
        result["errors"].append(
            f"unexpected shape ({n_layers}, {hidden}), expected "
            f"({expect['n_layers']}, {expect['hidden_size']}) for model={model}")
        return result

    for c in CONDITIONS:
        n_done = attrs[c].get("n_samples_done")
        if n_done is None or int(n_done) != n:
            result["errors"].append(
                f"condition={c}: n_samples_done={n_done} != n={n} in H5 "
                "attrs -- output may be partial/incomplete, refusing to "
                "trust it.")
            return result
        if bool(attrs[c].get("steering_applied", True)) is not False:
            result["errors"].append(
                f"condition={c}: steering_applied is not False in H5 attrs "
                "-- refusing to use a steered file as a role-only baseline.")
            return result
        if bool(attrs[c].get("generation_performed", True)) is not False:
            result["errors"].append(
                f"condition={c}: generation_performed is not False in H5 "
                "attrs -- refusing to use a generation-tainted file as a "
                "prefill-only baseline.")
            return result
        if bool(attrs[c].get("prefill_only", False)) is not True:
            result["errors"].append(
                f"condition={c}: prefill_only is not True in H5 attrs.")
            return result

    digest_expert = attrs["expert"].get("ordered_sample_identity_sha256")
    digest_nonexpert = attrs["non_expert"].get("ordered_sample_identity_sha256")
    if not digest_expert or digest_expert != digest_nonexpert:
        result["errors"].append(
            "expert/non_expert ordered_sample_identity_sha256 mismatch in "
            f"H5 attrs: {digest_expert!r} vs {digest_nonexpert!r} -- expert "
            "and non_expert must be over the identical ordered question "
            "list.")
        return result

    manifest_digest = exp_manifest.get("ordered_sample_identity_sha256")
    if manifest_digest != digest_expert:
        result["errors"].append(
            "experiment manifest.json's ordered_sample_identity_sha256 "
            f"({manifest_digest!r}) disagrees with the H5 attrs "
            f"({digest_expert!r}) -- refusing to trust either.")
        return result

    result.update({
        "ok": True,
        "n_samples": n,
        "shape": [n, n_layers, hidden],
        "h5_paths": {c: h5_paths[c] for c in CONDITIONS},
        "pairing_digest": digest_expert,
        "h5_sha256": {c: sha256_of_file(h5_paths[c]) for c in CONDITIONS},
        "exp_manifest_path": exp_manifest_path,
        "exp_manifest": exp_manifest,
    })
    return result


# ---------------------------------------------------------------------------
# Pass 2: load both arrays, check finiteness, compute means + diff IN
# MEMORY. Still no write.
# ---------------------------------------------------------------------------

def load_and_check_finite(h5_file_path: Path) -> np.ndarray:
    with h5py.File(h5_file_path, "r") as f:
        arr = f["hidden_states"][...]  # (n, n_layers, hidden), float16
    if not np.isfinite(arr.astype(np.float32)).all():
        raise ValueError(f"non-finite value(s) found in {h5_file_path}")
    return arr


def compute_means_and_diff(v: dict) -> dict:
    """Mean/diff dtype convention, MATCHING the existing formal reasoning
    Role direction construction (and the old GSM8K Role direction file),
    NOT independently re-derived:
      - expert_mean / non_expert_mean are float32-accumulated, then cast to
        float16 for saving (mean_role_reasoning.py's convention).
      - the DIRECTION (diff) is computed from those SAME saved float16 mean
        arrays, upcast back to float32, THEN subtracted, and saved as
        float32 -- NOT computed from the pre-cast float32 means. The two
        differ by float16 rounding order (round-then-subtract vs
        subtract-then-round), and only the round-then-subtract order
        matches what the existing exact-NMD/Role-direction pipeline
        actually consumes (a stored float16 mean pair, subtracted after
        loading). This ordering is small but was explicitly requested to
        avoid an exact-NMD top-k discrepancy against the existing pipeline.
    """
    expert_16 = load_and_check_finite(v["h5_paths"]["expert"])
    nonexpert_16 = load_and_check_finite(v["h5_paths"]["non_expert"])

    expert_32 = expert_16.astype(np.float32)
    nonexpert_32 = nonexpert_16.astype(np.float32)

    expert_mean32 = expert_32.mean(axis=0)   # (L, H)
    nonexpert_mean32 = nonexpert_32.mean(axis=0)

    # ---- Final on-disk mean dtype: float16. Everything downstream of this
    # ---- point (the diff) is built from THESE arrays, not from
    # ---- expert_mean32/nonexpert_mean32 directly. ----
    expert_mean_out = expert_mean32.astype(np.float16)[None, None, ...]
    nonexpert_mean_out = nonexpert_mean32.astype(np.float16)[None, None, ...]

    # ---- Direction: reconstructed from the SAVED float16 means, upcast to
    # ---- float32, then subtracted -- matches the existing Role direction
    # ---- construction (expert_mean_f16.astype(float32) -
    # ---- nonexpert_mean_f16.astype(float32)). Saved as float32, NOT
    # ---- float16 -- matches the old GSM8K Role direction file's dtype. NO
    # ---- L2 normalization anywhere. ----
    diff_out = (
        expert_mean_out.astype(np.float32)
        - nonexpert_mean_out.astype(np.float32)
    )

    # float64 shadow pass for a consistency audit (accumulation precision
    # only), matching mean_role_reasoning.py's pattern. The float64
    # reference diff is likewise built the round-then-subtract way (via a
    # float64 mean rounded to float16 and back), so it audits the SAME
    # on-disk construction, not the alternate subtract-then-round one.
    expert_64 = expert_16.astype(np.float64)
    nonexpert_64 = nonexpert_16.astype(np.float64)
    expert_mean64 = expert_64.mean(axis=0)
    nonexpert_mean64 = nonexpert_64.mean(axis=0)
    diff64_ref = (
        expert_mean64.astype(np.float16).astype(np.float64)
        - nonexpert_mean64.astype(np.float16).astype(np.float64)
    )

    max_abs_err_expert = float(np.max(np.abs(
        expert_mean64.astype(np.float32) - expert_mean32)))
    max_abs_err_nonexpert = float(np.max(np.abs(
        nonexpert_mean64.astype(np.float32) - nonexpert_mean32)))
    max_abs_err_diff = float(np.max(np.abs(
        diff64_ref.astype(np.float32) - diff_out[0, 0])))

    # Identity check: diff_out must equal, by direct recomputation from the
    # FINAL on-disk mean arrays (not from the pre-cast float32 means), a
    # fresh (expert_mean_out.astype(float32) - nonexpert_mean_out.astype
    # (float32)) -- catches a copy/paste/reordering bug where diff was
    # accidentally computed from something else than the saved means.
    recomputed = (
        expert_mean_out.astype(np.float32)
        - nonexpert_mean_out.astype(np.float32)
    )
    if not np.array_equal(diff_out, recomputed):
        die("internal error: diff_out does not equal "
            "expert_mean_out.astype(float32) - "
            "nonexpert_mean_out.astype(float32) by direct recomputation; "
            "refusing to write an inconsistent diff.")

    return {
        "expert_mean": expert_mean_out,
        "non_expert_mean": nonexpert_mean_out,
        "diff": diff_out,
        "mean_dtype": "float16",
        "diff_dtype": "float32",
        "consistency_check": {
            "description": "float32 (production accumulation dtype) means "
                           "vs an independent float64 (higher-precision "
                           "audit) recomputation, both before any float16 "
                           "cast; NO L2 normalization applied anywhere. "
                           "diff is computed from the SAVED float16 means "
                           "(round-then-subtract), upcast to float32, "
                           "matching the existing formal Role-direction "
                           "construction and the old GSM8K Role direction "
                           "file's dtype -- NOT from the pre-cast float32 "
                           "means (subtract-then-round), which would differ "
                           "by float16 rounding order. Verified by direct "
                           "recomputation from the final on-disk mean "
                           "arrays before writing.",
            "max_abs_error_float64_expert": max_abs_err_expert,
            "max_abs_error_float64_non_expert": max_abs_err_nonexpert,
            "max_abs_error_float64_diff": max_abs_err_diff,
        },
    }


# ---------------------------------------------------------------------------
# Pass 3: write, now that both conditions are known-good.
# ---------------------------------------------------------------------------

def atomic_save_npy(path: Path, arr: np.ndarray):
    # np.save() only skips appending ".npy" when the given name's suffix is
    # EXACTLY ".npy" -- a mkstemp name ending in ".npy.tmp" does NOT
    # qualify, so np.save would silently write to a SECOND, different path
    # and leave the mkstemp-created file behind empty (this bug was found
    # and fixed in mean_diff_chat.py / mean_role_reasoning.py; replicated
    # here fixed from the start). Fixed by suffixing with plain ".npy" and
    # writing into the already-open file descriptor directly.
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


def run(args):
    model = args.model
    size = args.size
    hs_root = Path(args.hs_root)
    out_dir = hs_root / model / "gsm8k_abstention_role"

    expert_mean_path = out_dir / f"expert_mean_{size}.npy"
    nonexpert_mean_path = out_dir / f"non_expert_mean_{size}.npy"
    diff_path = out_dir / f"gsm8k_abstention_role_diff_{size}.npy"

    # ---- Overwrite preflight for all 3 outputs, BEFORE any read. ----
    existing = [str(p) for p in (expert_mean_path, nonexpert_mean_path, diff_path)
               if p.exists()]
    if existing:
        die(f"refusing to run model={model}: output file(s) already exist:\n  " +
            "\n  ".join(existing) +
            "\nDelete them deliberately first if a re-run is truly intended.")

    # ---- Pass 1: validate BOTH conditions + the experiment manifest. ----
    v = validate(model, out_dir, size)
    if not v["ok"]:
        print(f"[REFUSE] validation failed for model={model} -- aborting "
              "before writing any output (fail-closed).")
        for e in v["errors"]:
            print(f"  [FAIL] {e}")
        sys.exit(1)

    print(f"Validation passed for model={model}, size={size}, "
          f"n_samples={v['n_samples']}, shape={v['shape']}, "
          f"pairing_digest={v['pairing_digest']}.")

    # ---- Pass 2: load, check finite, compute -- still no write. ----
    try:
        means = compute_means_and_diff(v)
    except Exception as e:
        print(f"[REFUSE] load/finite/compute failed for model={model}: "
              f"{type(e).__name__}: {e} -- aborting before writing any "
              "output (fail-closed).")
        sys.exit(1)

    print(f"Compute passed for model={model}: expert_mean shape="
          f"{means['expert_mean'].shape}, non_expert_mean shape="
          f"{means['non_expert_mean'].shape}, diff shape="
          f"{means['diff'].shape}.")
    print(f"  consistency: {means['consistency_check']}")

    # ---- Pass 3: write. ----
    atomic_save_npy(expert_mean_path, means["expert_mean"])
    atomic_save_npy(nonexpert_mean_path, means["non_expert_mean"])
    atomic_save_npy(diff_path, means["diff"])
    print(f"wrote {expert_mean_path}")
    print(f"wrote {nonexpert_mean_path}")
    print(f"wrote {diff_path}")

    # ---- Update the experiment-level manifest IN PLACE (additive only). ----
    exp_manifest = dict(v["exp_manifest"])
    exp_manifest["mean_diff"] = {
        "script_version": SCRIPT_VERSION,
        "computation": (
            "expert_mean/non_expert_mean = elementwise arithmetic mean over "
            "the sample axis (per layer, per hidden dim), NO L2 "
            "normalization, accumulated in float32 then cast to float16 for "
            "saving. gsm8k_abstention_role_diff = expert_mean.astype("
            "float32) - non_expert_mean.astype(float32), computed from "
            "those SAME SAVED float16 mean arrays (round-then-subtract, "
            "matching the existing formal Role-direction construction and "
            "the old GSM8K Role direction file's dtype), NOT from the "
            "pre-cast float32 means -- raw sign, NO L2 normalization, "
            "verified by direct recomputation from the on-disk mean arrays "
            "before writing."
        ),
        "accumulation_dtype": "float32 (float64 shadow pass for audit only)",
        "mean_dtype": means["mean_dtype"],
        "diff_dtype": means["diff_dtype"],
        "n_samples": v["n_samples"],
        "input_shape": v["shape"],
        "output_shape": list(means["expert_mean"].shape),
        "pairing_digest": v["pairing_digest"],
        "input_h5_sha256": {c: v["h5_sha256"][c] for c in CONDITIONS},
        "output_paths": {
            "expert_mean": str(expert_mean_path),
            "non_expert_mean": str(nonexpert_mean_path),
            "gsm8k_abstention_role_diff": str(diff_path),
        },
        "consistency_check": means["consistency_check"],
        "computed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(v["exp_manifest_path"], exp_manifest)
    print(f"updated experiment manifest: {v['exp_manifest_path']}")

    print(f"[DONE] model={model}: mean/diff computation complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SHAPE.keys()))
    ap.add_argument("--size", required=True,
                    help=f"Locked to --model: {MODEL_EXPECTED_SIZE}")
    ap.add_argument("--hs_root", required=True,
                    help="e.g. /data1/paveen/Dopamine/components/hidden_states "
                         "-- reads <hs_root>/<model>/gsm8k_abstention_role/, "
                         "writes into the SAME directory.")
    args = ap.parse_args()
    assert_size_matches_model(args.model, args.size)
    run(args)


if __name__ == "__main__":
    main()
