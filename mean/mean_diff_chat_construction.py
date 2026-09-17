#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat-Bare mean / diff-mean computation for the GSM8K-only construction-split
interface hidden-state extraction (extract_interface_hidden_states_construction.py).
Sibling of mean/mean_diff_chat.py, scoped to ONE task (gsm8k_construction)
instead of 3, and to a variable sample count instead of a hardcoded 300 --
same fail-closed conventions and dtype convention otherwise, not a
re-derivation.

Actual input schema (as produced by
extract_interface_hidden_states_construction.py, read directly, not
assumed):
  components/hidden_states/{model}/gsm8k_construction/{condition}_{size}.h5
    - top-level dataset "hidden_states", shape (n_samples, n_layers,
      hidden_size), dtype float16. n_layers index 0 = embedding output,
      1..N = decoder layer outputs.
    - top-level HDF5 .attrs carry: n_samples, n_samples_done,
      hidden_state_shape (JSON string), ordered_sample_identity_sha256,
      questions_sha256, chat_template_applied, steering_applied (bool,
      always False here), etc. n_samples is NOT hardcoded to 300 here --
      read from the H5 itself, and chat/bare are cross-checked equal to
      each other (not to a fixed constant).
  components/hidden_states/{model}/gsm8k_construction/{condition}_{size}.manifest.json
    - sidecar JSON, superset of the H5 attrs, same field names.

This script does NOT read the manifest.json for anything gating -- all
gating checks (shape, n_samples, pairing digest, finiteness) are read
directly from the H5 itself, matching mean_diff_chat.py's "trust the data,
not the sidecar" style. The manifest.json is consulted only for the SHA256
pairing_digest sanity cross-check.

DIRECTION (frozen, do not flip, matching mean_diff_chat.py exactly):
  diff_mean = chat_mean - bare_mean, i.e. "Chat minus Bare".

FAIL-CLOSED: the single (model, gsm8k_construction) cell is validated BEFORE
any accumulation. Any failure aborts with exit 1 and writes nothing,
including: missing file, wrong --size for the locked model, n_samples !=
1019 (the locked construction-split count), shape mismatch, chat/bare shape
or pairing-digest mismatch, chat/bare questions_sha256 mismatch, chat/bare
construction_split_manifest_digest mismatch, a model/task/condition field in
the H5 attrs disagreeing with the file being read, generation_performed !=
False, prefill_only != True, chat_template_applied not matching its own
condition (chat=True, bare=False), non-finite values.

Accumulation dtype: float32 (never float16) for the running sum; final
mean/diff cast to float16 only for the saved .npy, matching mean_diff_chat.py
and mean_diff_confidence_local.py's stated rationale.

This stage ONLY computes chat_mean / bare_mean / diff_mean. No cosine
similarity, split-half reliability, bootstrap CIs, or RSN/NMD mask
alignment.

Loads NO model, uses NO GPU, never writes to (or modifies) the input H5
source files.

Usage (run on the SERVER -- server interpreter is `python`, NOT `python3.10`):
  python mean/mean_diff_chat_construction.py --model llama3  --size 8B \
      --hs_root /data1/paveen/Dopamine/components/hidden_states \
      --out_root /data1/paveen/Dopamine/components/hidden_states_mean
  python mean/mean_diff_chat_construction.py --model qwen2.5 --size 7B \
      --hs_root /data1/paveen/Dopamine/components/hidden_states \
      --out_root /data1/paveen/Dopamine/components/hidden_states_mean
Output layout follows the existing per-model directory organization (NOT a
separate {model}_chat_construction top level -- the gsm8k_construction task
subdirectory already disambiguates this from any other task):
  <out_root>/{model}/gsm8k_construction/{chat_mean,bare_mean,diff_mean}_{size}.npy
Then sync --out_root's per-model subtree to the local analysis tree, e.g.:
  ~/Documents/RSNResult/RoleHidden/llama3/gsm8k_construction/
  ~/Documents/RSNResult/RoleHidden/qwen2.5/gsm8k_construction/
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

SCRIPT_VERSION = "mean_diff_chat_construction-v1"

TASK = "gsm8k_construction"
CONDITIONS = ("chat", "bare")
DIRECTION = "chat_minus_bare"

MODEL_SHAPE = {
    "llama3": {"n_layers": 33, "hidden_size": 4096},
    "qwen2.5": {"n_layers": 29, "hidden_size": 3584},
}

# Locked per-model expected size string AND expected sample count, matching
# the construction split this pipeline is built to read
# (components/benchmark/gsm8k_construction_split.json, verified n=1019 --
# build_gsm8k_construction_split.py's own EXPECTED_CONSTRUCTION_N). A
# wrongly-named H5 (e.g. the 300-question chat/bare tree copied into this
# path by mistake, or a different model's file placed under the wrong
# model_key directory) must not silently pass validation on shape/digest
# checks alone.
MODEL_SIZE = {"llama3": "8B", "qwen2.5": "7B"}
EXPECTED_N_SAMPLES = 1019


def die(msg: str, code: int = 1):
    print(f"[REFUSE] {msg}", file=sys.stderr)
    sys.exit(code)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def h5_path(hs_root: Path, model: str, condition: str, size: str) -> Path:
    return hs_root / model / TASK / f"{condition}_{size}.h5"


def manifest_path(hs_root: Path, model: str, condition: str, size: str) -> Path:
    return hs_root / model / TASK / f"{condition}_{size}.manifest.json"


def validate_task(model: str, hs_root: Path, size: str) -> dict:
    """Validate the ONE (model, gsm8k_construction) chat+bare pair, WITHOUT
    loading the full hidden_states arrays into memory yet. Mirrors
    validate_task() in mean_diff_chat.py, but n is checked against the
    LOCKED construction-split count (1019) rather than a fixed 300."""
    result = {"model": model, "task": TASK, "ok": False, "errors": []}
    expect = MODEL_SHAPE[model]

    expected_size = MODEL_SIZE[model]
    if size != expected_size:
        result["errors"].append(
            f"--size={size!r} passed but model={model} is locked to "
            f"size={expected_size!r} -- refusing a size mismatch that "
            "would otherwise resolve to a plausible-looking but wrong "
            "file path.")
        return result

    h5_paths = {c: h5_path(hs_root, model, c, size) for c in CONDITIONS}
    manifest_paths = {c: manifest_path(hs_root, model, c, size) for c in CONDITIONS}
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
    try:
        for c in CONDITIONS:
            with h5py.File(h5_paths[c], "r") as f:
                ds = f["hidden_states"]
                shapes[c] = ds.shape
                attrs[c] = dict(f.attrs)
    except Exception as e:
        result["errors"].append(f"failed to read H5 header: {type(e).__name__}: {e}")
        return result

    if shapes["chat"] != shapes["bare"]:
        result["errors"].append(
            f"chat/bare H5 shape mismatch: {shapes['chat']} vs {shapes['bare']}")
        return result

    n, n_layers, hidden = shapes["chat"]
    if n != EXPECTED_N_SAMPLES:
        result["errors"].append(
            f"n_samples={n}, expected exactly {EXPECTED_N_SAMPLES} (the "
            "locked GSM8K construction-split count) -- refusing a wrong "
            "sample count rather than silently accepting whatever the file "
            "happens to hold.")
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
                f"condition={c}: n_samples_done={n_done} != n={n} in H5 attrs -- "
                "output may be partial/incomplete, refusing to trust it.")
            return result
        if bool(a.get("steering_applied", True)) is not False:
            result["errors"].append(
                f"condition={c}: steering_applied is not False in H5 attrs -- "
                "refusing to use a steered file as an interface-only baseline.")
            return result
        # ---- Field-identity checks: a wrongly-named/misplaced H5 (e.g. the
        # ---- 300-question tree, a different model's file, or a file
        # ---- carrying a different condition than its own filename claims)
        # ---- must not pass on shape+digest alone.
        a_model = a.get("model")
        if a_model != model:
            result["errors"].append(
                f"condition={c}: H5 attrs record model={a_model!r}, "
                f"expected {model!r} -- refusing a model-identity mismatch.")
            return result
        a_task = a.get("task")
        if a_task != TASK:
            result["errors"].append(
                f"condition={c}: H5 attrs record task={a_task!r}, "
                f"expected {TASK!r} -- refusing a task-identity mismatch "
                "(this must be the gsm8k_construction file, not the "
                "300-question gsm8k tree).")
            return result
        a_condition = a.get("condition")
        if a_condition != c:
            result["errors"].append(
                f"file at the '{c}' path has H5 attrs recording "
                f"condition={a_condition!r} -- refusing a condition-"
                "identity mismatch (filename/path and attrs disagree on "
                "whether this is chat or bare).")
            return result
        if bool(a.get("generation_performed", True)) is not False:
            result["errors"].append(
                f"condition={c}: generation_performed is not False in H5 "
                "attrs -- refusing a file that may not be the pure "
                "prefill-only extraction this pipeline expects.")
            return result
        if bool(a.get("prefill_only", False)) is not True:
            result["errors"].append(
                f"condition={c}: prefill_only is not True in H5 attrs -- "
                "refusing a file that may not be the pure prefill-only "
                "extraction this pipeline expects.")
            return result
        expected_chat_applied = (c == "chat")
        actual_chat_applied = bool(a.get("chat_template_applied"))
        if actual_chat_applied != expected_chat_applied:
            result["errors"].append(
                f"condition={c}: chat_template_applied={actual_chat_applied} "
                f"in H5 attrs, expected {expected_chat_applied} -- refusing "
                "a chat/bare labeling mismatch (this would silently compute "
                "the diff in the wrong direction or from two bare/two chat "
                "files).")
            return result

    digest_chat = attrs["chat"].get("ordered_sample_identity_sha256")
    digest_bare = attrs["bare"].get("ordered_sample_identity_sha256")
    if not digest_chat or digest_chat != digest_bare:
        result["errors"].append(
            f"chat/bare ordered_sample_identity_sha256 mismatch in H5 attrs: "
            f"{digest_chat!r} vs {digest_bare!r} -- chat and bare must be over "
            "the identical ordered question list.")
        return result

    q_sha_chat = attrs["chat"].get("questions_sha256")
    q_sha_bare = attrs["bare"].get("questions_sha256")
    if not q_sha_chat or q_sha_chat != q_sha_bare:
        result["errors"].append(
            f"chat/bare questions_sha256 mismatch in H5 attrs: "
            f"{q_sha_chat!r} vs {q_sha_bare!r} -- chat and bare must have "
            "been extracted from the byte-identical construction-split "
            "question file.")
        return result

    csm_digest_chat = attrs["chat"].get("construction_split_manifest_digest")
    csm_digest_bare = attrs["bare"].get("construction_split_manifest_digest")
    if not csm_digest_chat or csm_digest_chat != csm_digest_bare:
        result["errors"].append(
            f"chat/bare construction_split_manifest_digest mismatch in H5 "
            f"attrs: {csm_digest_chat!r} vs {csm_digest_bare!r} -- chat and "
            "bare must have been extracted against the same construction-"
            "split manifest digest.")
        return result

    manifest_digest_chat = manifests["chat"].get("ordered_sample_identity_sha256")
    manifest_digest_bare = manifests["bare"].get("ordered_sample_identity_sha256")
    if manifest_digest_chat != digest_chat or manifest_digest_bare != digest_bare:
        result["errors"].append(
            "manifest.json ordered_sample_identity_sha256 disagrees with the "
            "H5 attrs of the same file -- refusing to trust either.")
        return result

    result.update({
        "ok": True,
        "n_samples": n,
        "shape": [n, n_layers, hidden],
        "h5_paths": {c: str(h5_paths[c]) for c in CONDITIONS},
        "manifest_paths": {c: str(manifest_paths[c]) for c in CONDITIONS},
        "pairing_digest": digest_chat,
        "h5_sha256": {c: sha256_of_file(h5_paths[c]) for c in CONDITIONS},
    })
    return result


def load_and_check_finite(h5_file_path: Path) -> np.ndarray:
    with h5py.File(h5_file_path, "r") as f:
        arr = f["hidden_states"][...]  # (n, n_layers, hidden), float16
    if not np.isfinite(arr.astype(np.float32)).all():
        raise ValueError(f"non-finite value(s) found in {h5_file_path}")
    return arr


def compute_task_means(task_info: dict) -> dict:
    """Load chat+bare arrays for the already-validated task, accumulate in
    float32, and return chat_mean/bare_mean/diff_mean (all (1, 1, L, H)
    float16) plus a consistency-audit report. Identical logic to
    mean_diff_chat.py's compute_task_means()."""
    chat_16 = load_and_check_finite(Path(task_info["h5_paths"]["chat"]))
    bare_16 = load_and_check_finite(Path(task_info["h5_paths"]["bare"]))

    chat_32 = chat_16.astype(np.float32)
    bare_32 = bare_16.astype(np.float32)

    chat_mean32 = chat_32.mean(axis=0)   # (L, H)
    bare_mean32 = bare_32.mean(axis=0)
    diff_mean32 = chat_mean32 - bare_mean32

    diff_via_persample32 = (chat_32 - bare_32).mean(axis=0)
    max_abs_err_f32 = float(np.max(np.abs(diff_via_persample32 - diff_mean32)))

    chat_64 = chat_16.astype(np.float64)
    bare_64 = bare_16.astype(np.float64)
    chat_mean64 = chat_64.mean(axis=0)
    bare_mean64 = bare_64.mean(axis=0)
    diff_mean64 = chat_mean64 - bare_mean64
    diff_via_persample64 = (chat_64 - bare_64).mean(axis=0)
    max_abs_err_f64 = float(np.max(np.abs(diff_via_persample64 - diff_mean64)))

    direct_diff = chat_32.mean(axis=0) - bare_32.mean(axis=0)
    max_abs_diff_identity_err = float(np.max(np.abs(direct_diff - diff_mean32)))
    if max_abs_diff_identity_err > 1e-3:
        raise ValueError(
            f"diff_mean != chat_mean - bare_mean identity check FAILED: "
            f"max_abs_err={max_abs_diff_identity_err}")

    chat_mean_out = chat_mean32.astype(np.float16)[None, None, ...]
    bare_mean_out = bare_mean32.astype(np.float16)[None, None, ...]
    diff_mean_out = diff_mean32.astype(np.float16)[None, None, ...]

    return {
        "chat_mean": chat_mean_out,
        "bare_mean": bare_mean_out,
        "diff_mean": diff_mean_out,
        "n_samples": chat_16.shape[0],
        "consistency_check": {
            "description": "mean(chat_i - bare_i) vs chat_mean - bare_mean, "
                            "computed in float32 (production accumulation "
                            "dtype) and float64 (independent higher-"
                            "precision audit), both before any float16 cast",
            "max_abs_error_float32": max_abs_err_f32,
            "max_abs_error_float64": max_abs_err_f64,
        },
        "diff_identity_check": {
            "description": "diff_mean (float32, pre-cast) vs "
                            "chat_mean32 - bare_mean32, independently "
                            "re-derived",
            "max_abs_error_float32": max_abs_diff_identity_err,
            "passed": True,
        },
    }


def atomic_save_npy(path: Path, arr: np.ndarray):
    # np.save() only skips appending ".npy" when the given name's suffix is
    # EXACTLY ".npy" -- see mean_diff_chat.py's identical comment for the
    # bug this avoids.
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
    ap.add_argument("--size", required=True)
    ap.add_argument("--hs_root", required=True,
                     help="e.g. /data1/paveen/Dopamine/components/hidden_states")
    ap.add_argument("--out_root", required=True,
                     help="e.g. /data1/paveen/Dopamine/components/hidden_states_mean")
    args = ap.parse_args()

    model = args.model
    size = args.size
    hs_root = Path(args.hs_root)
    out_root = Path(args.out_root)
    model_out_dir = out_root / model

    # ---- Overwrite preflight for this task's outputs, BEFORE any read. ----
    task_dir = model_out_dir / TASK
    all_out_paths = [
        task_dir / f"chat_mean_{size}.npy",
        task_dir / f"bare_mean_{size}.npy",
        task_dir / f"diff_mean_{size}.npy",
        task_dir / "manifest.json",
    ]
    existing = [str(p) for p in all_out_paths if p.exists()]
    if existing:
        die(f"refusing to run model={model}: output file(s) already exist:\n  " +
            "\n  ".join(existing) +
            "\nDelete them deliberately first if a re-run is truly intended.")

    # ---- Pass 1: validate BEFORE any accumulation/write. ----
    v = validate_task(model, hs_root, size)
    if not v["ok"]:
        print(f"[REFUSE] task={TASK} failed validation for model={model} -- "
              "aborting before writing any output (fail-closed).")
        print(f"  [FAIL] {v['errors']}")
        sys.exit(1)

    print(f"Task {TASK} passed validation for model={model}, size={size}, "
          f"n_samples={v['n_samples']}.")

    # ---- Pass 2: compute + write, now that the task is known-good. ----
    print(f"[compute] model={model} task={TASK} n_samples={v['n_samples']} "
          f"shape={v['shape']}")
    means = compute_task_means(v)

    task_dir.mkdir(parents=True, exist_ok=True)
    chat_path = task_dir / f"chat_mean_{size}.npy"
    bare_path = task_dir / f"bare_mean_{size}.npy"
    diff_path = task_dir / f"diff_mean_{size}.npy"
    manifest_out_path = task_dir / "manifest.json"

    atomic_save_npy(chat_path, means["chat_mean"])
    atomic_save_npy(bare_path, means["bare_mean"])
    atomic_save_npy(diff_path, means["diff_mean"])

    manifest = {
        "script_version": SCRIPT_VERSION,
        "model": model,
        "task": TASK,
        "size": size,
        "n_samples": v["n_samples"],
        "input_shape": v["shape"],
        "output_shape": list(means["chat_mean"].shape),
        "accumulation_dtype": "float32 (float64 shadow pass for audit only)",
        "output_dtype": "float16",
        "difference_direction": DIRECTION,
        "pairing_digest": v["pairing_digest"],
        "input_h5_sha256": v["h5_sha256"],
        "input_h5_paths": v["h5_paths"],
        "input_manifest_paths": v["manifest_paths"],
        "output_paths": {
            "chat_mean": str(chat_path),
            "bare_mean": str(bare_path),
            "diff_mean": str(diff_path),
        },
        "consistency_check": means["consistency_check"],
        "diff_identity_check": means["diff_identity_check"],
        "fail_closed": (
            "The gsm8k_construction task is validated (H5+manifest presence, "
            "chat/bare shape equality, n_samples>0, expected "
            "(n_layers, hidden_size), n_samples_done==n, steering_applied==False, "
            "ordered_sample_identity_sha256 equality between chat/bare AND "
            "between H5 attrs/manifest.json) BEFORE any accumulation or file "
            "write; a failing validation aborts the run and writes nothing."
        ),
        "computation": (
            "chat_mean/bare_mean = elementwise mean over the sample axis "
            "(per layer, per hidden dim); diff_mean = chat_mean - bare_mean. "
            "No cosine, split-half, bootstrap, or RSN/NMD alignment computed "
            "at this stage."
        ),
        "note": "GSM8K-only construction-split sibling of mean_diff_chat.py, "
                "scoped to gsm8k_construction only. Built to sample-"
                "composition-match the formal construction-split ARRSN. "
                "MATH/GSM-Hard deliberately not included.",
        "extraction_timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(manifest_out_path, manifest)

    print(f"[ok] model={model} task={TASK}: wrote chat_mean/bare_mean/diff_mean + "
          f"manifest.json under {task_dir}")
    print(f"[DONE] model={model}: 1/1 task complete.")
    print({
        "task": TASK,
        "n_samples": v["n_samples"],
        "pairing_digest": v["pairing_digest"],
        "consistency_max_abs_error_float32":
            means["consistency_check"]["max_abs_error_float32"],
        "consistency_max_abs_error_float64":
            means["consistency_check"]["max_abs_error_float64"],
        "diff_identity_passed": means["diff_identity_check"]["passed"],
    })


if __name__ == "__main__":
    main()
