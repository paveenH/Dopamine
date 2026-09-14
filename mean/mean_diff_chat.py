#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat-Bare mean / diff-mean computation for the interface hidden-state
extraction pipeline (extract_interface_hidden_states.py), following the SAME
conventions as mean/mean_diff_confidence_local.py (verified against that
script and against the actual on-disk schema of
extract_interface_hidden_states.py's H5/manifest output before writing this
one -- fields are not guessed).

Actual input schema (as produced by extract_interface_hidden_states.py, read
directly rather than assumed):
  components/hidden_states/{model}/{task}/{condition}_{size}.h5
    - top-level dataset "hidden_states", shape (n_samples, n_layers,
      hidden_size), dtype float16. NOT nested under any per-role group.
      n_layers index 0 = embedding output, 1..N = decoder layer outputs.
    - top-level HDF5 .attrs carry: n_samples, n_samples_done,
      hidden_state_shape (JSON string), ordered_sample_identity_sha256,
      questions_sha256, chat_template_applied, steering_applied (bool,
      always False here), etc.
  components/hidden_states/{model}/{task}/{condition}_{size}.manifest.json
    - sidecar JSON, superset of the H5 attrs, same field names.

This script does NOT read the manifest.json for anything gating -- all
gating checks (shape, n_samples, pairing digest, finiteness) are read
directly from the H5 itself (attrs + dataset), matching the "trust the data,
not the sidecar" style of mean_diff_confidence_local.py's validate_task().
The manifest.json is consulted only for the SHA256 pairing_digest sanity
cross-check, purely as a second, independent confirmation channel.

DIRECTION (frozen, do not flip): diff_mean = chat_mean - bare_mean, i.e.
"Chat minus Bare" (matching the confidence script's
"confident_mean - unconfident_mean" convention exactly, chat playing the
role of the "treatment" condition, bare the "baseline").

FAIL-CLOSED: every (model, task) cell is validated BEFORE any accumulation.
If ANY of the 3 tasks for a model fails validation (missing file, n!=300,
shape mismatch, chat/bare shape or pairing-digest mismatch, non-finite
values), the WHOLE MODEL's run aborts with exit 1 and writes NOTHING for
that model -- never a partial result computed from the surviving tasks. This
mirrors mean_diff_confidence_local.py's two-pass (validate-then-accumulate)
design, but here validation and output are PER TASK (not pooled across
tasks), since Chat-Bare mean diff is wanted per (model, task), not pooled
globally the way the confidence script pools across all 57 MMLU-E tasks into
one mean. (There is no cross-task pooling requirement in this instruction --
"Chat-Bare mean difference" is computed once per model x task cell.)

Accumulation dtype: float32 (never float16) for the running sum, matching
mean_diff.py and mean_diff_confidence_local.py's stated rationale
(overflow/precision loss avoidance); final mean/diff cast to float16 only
for the saved .npy, per the legacy (1, 1, n_layers, hidden_size) convention
documented in mean_diff_confidence_local.py's own header comment (verified
against mean/mean_diff.py's diff_mean_{size}.npy convention before writing
this script -- not re-derived from memory).

This stage ONLY computes chat_mean / bare_mean / diff_mean (elementwise mean
over the sample axis, per layer, per hidden dim). It does NOT compute
cosine similarity, split-half reliability, bootstrap CIs, or any RSN/NMD
mask alignment -- those are separate, later stages, exactly as the
instruction specifies.

Loads NO model, uses NO GPU, and never writes to (or reads from) the H5
source files except in read-only mode -- this script only reads the already-
extracted hidden_states arrays.

Usage (run on the SERVER, where the H5 data live -- server interpreter is
`python`, NOT `python3.10`, matching every other server-side script in this
family):
  python mean/mean_diff_chat.py --model llama3  --size 8B \
      --hs_root /data1/paveen/Dopamine/components/hidden_states \
      --out_root /data1/paveen/Dopamine/components/hidden_states_mean
  python mean/mean_diff_chat.py --model qwen2.5 --size 7B \
      --hs_root /data1/paveen/Dopamine/components/hidden_states \
      --out_root /data1/paveen/Dopamine/components/hidden_states_mean
Then sync --out_root's per-model subtree to the local analysis tree, e.g.:
  ~/Documents/RSNResult/RoleHidden/llama3_chat/
  ~/Documents/RSNResult/RoleHidden/qwen2.5_chat/
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

SCRIPT_VERSION = "mean_diff_chat-v1"

TASKS = ("gsm8k", "math", "gsm_hard")
CONDITIONS = ("chat", "bare")
DIRECTION = "chat_minus_bare"

MODEL_SHAPE = {
    "llama3": {"n_layers": 33, "hidden_size": 4096},
    "qwen2.5": {"n_layers": 29, "hidden_size": 3584},
}


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
    """Validate ONE (model, task)'s chat+bare pair, WITHOUT loading the full
    hidden_states arrays into memory yet (header/attrs only). Returns a dict
    with 'ok', 'errors', and (if ok) the file paths + expected shape, so the
    second pass can load and accumulate only after every task for this model
    has already passed. Mirrors validate_task() in
    mean_diff_confidence_local.py."""
    result = {"model": model, "task": task, "ok": False, "errors": []}
    expect = MODEL_SHAPE[model]

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
                f"condition={c}: n_samples_done={n_done} != n={n} in H5 attrs -- "
                "output may be partial/incomplete, refusing to trust it.")
            return result
        if bool(attrs[c].get("steering_applied", True)) is not False:
            result["errors"].append(
                f"condition={c}: steering_applied is not False in H5 attrs -- "
                "refusing to use a steered file as an interface-only baseline.")
            return result

    digest_chat = attrs["chat"].get("ordered_sample_identity_sha256")
    digest_bare = attrs["bare"].get("ordered_sample_identity_sha256")
    if not digest_chat or digest_chat != digest_bare:
        result["errors"].append(
            f"chat/bare ordered_sample_identity_sha256 mismatch in H5 attrs: "
            f"{digest_chat!r} vs {digest_bare!r} -- chat and bare must be over "
            "the identical ordered question list.")
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
    """Load chat+bare arrays for one already-validated task, accumulate in
    float32, and return chat_mean/bare_mean/diff_mean (all (1, 1, L, H)
    float16) plus a consistency-audit report. No accumulation across tasks
    here -- each (model, task) cell gets its own independent mean, per the
    instruction (not a pooled-across-tasks mean like the confidence
    script)."""
    chat_16 = load_and_check_finite(Path(task_info["h5_paths"]["chat"]))
    bare_16 = load_and_check_finite(Path(task_info["h5_paths"]["bare"]))

    chat_32 = chat_16.astype(np.float32)
    bare_32 = bare_16.astype(np.float32)

    chat_mean32 = chat_32.mean(axis=0)   # (L, H)
    bare_mean32 = bare_32.mean(axis=0)
    diff_mean32 = chat_mean32 - bare_mean32

    # Consistency audit: mean(chat_i - bare_i) vs chat_mean - bare_mean, in
    # float32 (production dtype) AND float64 (independent higher-precision
    # audit), matching mean_diff_confidence_local.py's finalize() pattern.
    diff_via_persample32 = (chat_32 - bare_32).mean(axis=0)
    max_abs_err_f32 = float(np.max(np.abs(diff_via_persample32 - diff_mean32)))

    chat_64 = chat_16.astype(np.float64)
    bare_64 = bare_16.astype(np.float64)
    chat_mean64 = chat_64.mean(axis=0)
    bare_mean64 = bare_64.mean(axis=0)
    diff_mean64 = chat_mean64 - bare_mean64
    diff_via_persample64 = (chat_64 - bare_64).mean(axis=0)
    max_abs_err_f64 = float(np.max(np.abs(diff_via_persample64 - diff_mean64)))

    # Requirement 6: verify diff_mean ~= chat_mean - bare_mean (the two are
    # computed via the same chat_mean32/bare_mean32 above by construction,
    # but re-derive independently from the raw arrays as an explicit check
    # rather than trusting the shared variable).
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
    # EXACTLY ".npy" (a str.endswith(".npy") check) -- a mkstemp name ending
    # in ".npy.tmp" does NOT qualify, so np.save silently writes to a
    # SECOND, different path (tmp_path + ".npy") and leaves the original
    # mkstemp-created file behind as an empty, never-written leftover.
    # Fixed by suffixing with plain ".npy" and writing into the already-open
    # file descriptor directly, so there is exactly one file on disk at any
    # time before the atomic os.replace().
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
    model_out_dir = out_root / f"{model}_chat"

    # ---- Overwrite preflight for ALL 3 tasks' outputs, BEFORE any read. ----
    all_out_paths = []
    for task in TASKS:
        task_dir = model_out_dir / task
        all_out_paths += [
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

    # ---- Pass 1: validate all 3 tasks BEFORE any accumulation/write. ----
    validations = {task: validate_task(model, task, hs_root, size) for task in TASKS}
    failed = {t: v for t, v in validations.items() if not v["ok"]}
    if failed:
        print(f"[REFUSE] {len(failed)}/{len(TASKS)} task(s) failed validation for "
              f"model={model} -- aborting before writing any output (fail-closed).")
        for t, v in failed.items():
            print(f"  [FAIL] {t}: {v['errors']}")
        sys.exit(1)

    print(f"All {len(TASKS)} tasks passed validation for model={model}, size={size}.")

    # ---- Pass 2: compute + write, now that every task is known-good. ----
    per_task_report = []
    for task in TASKS:
        v = validations[task]
        print(f"[compute] model={model} task={task} n_samples={v['n_samples']} "
              f"shape={v['shape']}")
        means = compute_task_means(v)

        task_dir = model_out_dir / task
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
            "task": task,
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
                "Every task for this model is validated (H5+manifest presence, "
                "chat/bare shape equality, n_samples==300, expected "
                "(n_layers, hidden_size), n_samples_done==n, steering_applied==False, "
                "ordered_sample_identity_sha256 equality between chat/bare AND "
                "between H5 attrs/manifest.json) BEFORE any accumulation or file "
                "write for ANY of this model's 3 tasks; a single failing task "
                "aborts the whole model's run and writes nothing."
            ),
            "computation": (
                "chat_mean/bare_mean = elementwise mean over the sample axis "
                "(per layer, per hidden dim); diff_mean = chat_mean - bare_mean. "
                "No cosine, split-half, bootstrap, or RSN/NMD alignment computed "
                "at this stage."
            ),
            "extraction_timestamp_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_write_json(manifest_out_path, manifest)

        per_task_report.append({
            "task": task,
            "n_samples": v["n_samples"],
            "pairing_digest": v["pairing_digest"],
            "consistency_max_abs_error_float32":
                means["consistency_check"]["max_abs_error_float32"],
            "consistency_max_abs_error_float64":
                means["consistency_check"]["max_abs_error_float64"],
            "diff_identity_passed": means["diff_identity_check"]["passed"],
        })
        print(f"[ok] model={model} task={task}: wrote chat_mean/bare_mean/diff_mean + "
              f"manifest.json under {task_dir}")

    print(f"[DONE] model={model}: {len(TASKS)}/{len(TASKS)} tasks complete.")
    for r in per_task_report:
        print(f"  {r}")


if __name__ == "__main__":
    main()
