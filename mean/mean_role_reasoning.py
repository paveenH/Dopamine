#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert / Non-expert mean computation for the role hidden-state extraction
pipeline (extract_role_hidden_states.py), reasoning tasks ONLY (GSM8K, MATH,
GSM-Hard). Follows the SAME conventions as mean/mean_diff_chat.py (verified
against that script and against the actual on-disk schema of
extract_role_hidden_states.py's H5/manifest output before writing this one --
fields are not guessed).

MMLU-E is explicitly OUT OF SCOPE for this script -- the RSN paper's existing,
answer-filtered MMLU-E direction continues to be used for that task; this
script never reads components/hidden_states/{model}/mmlue/.

Actual input schema (as produced by extract_role_hidden_states.py, read
directly rather than assumed):
  components/hidden_states/{model}/{task}/{condition}_{size}.h5
    condition in {"expert", "non_expert"}
    - top-level dataset "hidden_states", shape (n_samples, n_layers,
      hidden_size), dtype float16. NOT nested under any per-role group.
      n_layers index 0 = embedding output, 1..N = decoder layer outputs.
    - top-level HDF5 .attrs carry (among others): n_samples, n_samples_done,
      hidden_state_shape (JSON string), ordered_sample_identity_sha256,
      questions_sha256, steering_applied (bool, always False here),
      generation_performed (bool, always False here), prefill_only (bool,
      always True here).
  components/hidden_states/{model}/{task}/{condition}_{size}.manifest.json
    - sidecar JSON, superset of the H5 attrs, same field names.

This script does NOT read the manifest.json for anything gating -- all
gating checks (shape, n_samples, pairing digest, finiteness) are read
directly from the H5 itself (attrs + dataset), matching mean_diff_chat.py's
"trust the data, not the sidecar" style. The manifest.json is consulted only
for the SHA256 pairing_digest sanity cross-check, purely as a second,
independent confirmation channel.

SCOPE OF THIS STAGE (server-side): compute and save ONLY
  expert_mean_{size}.npy
  non_expert_mean_{size}.npy
per (model, task). This script does NOT compute or save any Role diff_mean
(expert_mean - non_expert_mean) -- per the instruction, that subtraction is
done LATER, LOCALLY, after syncing these two mean files to the analysis
workspace. Do not add a diff_mean output here even though mean_diff_chat.py
(the structural template) does compute one -- that is a deliberate
difference from the template, not an oversight.

Output layout: these Expert/Non-expert means are written into the EXISTING
Chat-Bare mean directory tree (components/hidden_states_mean/{model}_chat/
{task}/), alongside the already-existing chat_bare_diff_mean_{size}.npy (see
the separate --rename_chat_bare step below). They do NOT get their own
"{model}_role/" subtree.

ACCUMULATION DTYPE: float32 (never float16) for the running sum, matching
mean_diff.py / mean_diff_confidence_local.py / mean_diff_chat.py's stated
rationale (overflow/precision loss avoidance); final mean cast to float16
only for the saved .npy, per the legacy (1, 1, n_layers, hidden_size)
convention. No L2 normalization anywhere in this script -- per instruction,
this is a direct arithmetic mean over raw hidden states.

FAIL-CLOSED, THREE PASSES (not two): (1) validate all 3 tasks' H5/manifest
headers+attrs (existence, shape, n_samples==300, n_samples_done, pairing
digest) for a model, WITHOUT loading the hidden_states arrays; (2) THEN load
every task's full arrays and check finiteness + compute means IN MEMORY for
ALL 3 tasks, still without writing anything; (3) only once ALL 3 tasks have
passed BOTH pass 1 and pass 2 does ANY task's output get written. This
three-pass split (rather than interleaving pass 2's compute with pass 3's
write per task) is what makes "a single failing task writes nothing for any
task" actually true: the finite check happens during array loading (pass 2),
and if it were interleaved with writing, a later task's NaN/Inf would only
be discovered AFTER earlier tasks had already been written to disk. If ANY
of the 3 tasks fails pass 1 OR pass 2 (missing file, n!=300, shape mismatch,
expert/non_expert shape or pairing-digest mismatch, non-finite values), the
WHOLE MODEL's run aborts with exit 1 and writes NOTHING for that model --
never a partial result computed from the surviving tasks.

Loads NO model, uses NO GPU, and never writes to (or reads from) the H5
source files except in read-only mode -- this script only reads the
already-extracted hidden_states arrays. Never touches the original H5,
MMLU-E files, or any existing numeric result.

--- SECOND MODE: --rename_chat_bare -----------------------------------------

A SEPARATE, independently-invoked mode of this same script (never run
implicitly alongside the mean computation above) that renames the EXISTING
Chat-Bare diff_mean_{size}.npy file in each (model, task) output directory
to chat_bare_diff_mean_{size}.npy, to avoid semantic confusion with the new
Expert/Non-expert files that will later live in the same directory. This is
a RENAME ONLY:
  - The file's bytes/values are never read into a numeric computation, never
    recomputed, never modified.
  - Before renaming, this mode reads that task's existing manifest.json and
    asserts manifest["difference_direction"] == "chat_minus_bare" -- if that
    field is missing or has any other value, the rename for that (model,
    task) is REFUSED and nothing is touched, precisely because the
    instruction requires confirming this BEFORE renaming.
  - The manifest.json's own "diff_mean" entry under "output_paths" (and any
    other place the filename appears within that JSON) is updated in place
    to name the new "chat_bare_diff_mean_{size}.npy" filename -- values are
    never touched.
  - Every (model, task) cell is validated (file exists, manifest exists and
    parses, direction confirmed, target name not already taken) BEFORE any
    rename is performed, same fail-closed two-pass discipline as the mean
    computation above.
  - Uses tempfile + os.replace for the manifest rewrite (atomic); the .npy
    rename itself uses os.replace directly (atomic rename, no temp copy
    needed since no bytes are altered).
  - Refuses (fail closed) if the target chat_bare_diff_mean_{size}.npy
    already exists, or if the source diff_mean_{size}.npy is missing.

Usage (run on the SERVER, where the H5 data live -- server interpreter is
`python`, NOT `python3.10`, matching every other server-side script in this
family):
  # Step 1: rename the existing Chat-Bare diff_mean file FIRST (independent
  # of, and safe to run before or after, the mean computation below).
  python mean/mean_role_reasoning.py --rename_chat_bare --model llama3 --size 8B \
      --mean_root /data1/paveen/Dopamine/components/hidden_states_mean
  python mean/mean_role_reasoning.py --rename_chat_bare --model qwen2.5 --size 7B \
      --mean_root /data1/paveen/Dopamine/components/hidden_states_mean

  # Step 2: compute Expert/Non-expert means.
  python mean/mean_role_reasoning.py --model llama3  --size 8B \
      --hs_root /data1/paveen/Dopamine/components/hidden_states \
      --out_root /data1/paveen/Dopamine/components/hidden_states_mean
  python mean/mean_role_reasoning.py --model qwen2.5 --size 7B \
      --hs_root /data1/paveen/Dopamine/components/hidden_states \
      --out_root /data1/paveen/Dopamine/components/hidden_states_mean

Then sync --out_root's per-model subtree to the local analysis tree (e.g.
~/Documents/RSNResult/RoleHidden/llama3_chat/,
~/Documents/RSNResult/RoleHidden/qwen2.5_chat/) and compute
role_diff = expert_mean - non_expert_mean LOCALLY -- not on the server, not
by this script.
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

SCRIPT_VERSION = "mean_role_reasoning-v1"

TASKS = ("gsm8k", "math", "gsm_hard")
CONDITIONS = ("expert", "non_expert")

MODEL_SHAPE = {
    "llama3": {"n_layers": 33, "hidden_size": 4096},
    "qwen2.5": {"n_layers": 29, "hidden_size": 3584},
}

# --size is locked to --model rather than accepted as a free string, so a
# typo'd/mismatched size (e.g. --model llama3 --size 7B) cannot silently
# construct a misleading path (llama3_chat/gsm8k/expert_mean_7B.npy) that
# looks plausible but names a size no llama3 file ever used.
MODEL_EXPECTED_SIZE = {
    "llama3": "8B",
    "qwen2.5": "7B",
}

EXPECTED_CHAT_BARE_DIRECTION = "chat_minus_bare"


def assert_size_matches_model(model: str, size: str) -> None:
    expected = MODEL_EXPECTED_SIZE[model]
    if size != expected:
        die(f"--size {size!r} does not match --model {model!r} (expected "
            f"--size {expected!r}). --size is locked to the model to avoid "
            "constructing a misleading output path under the wrong size "
            "suffix.")


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


# ---------------------------------------------------------------------------
# Mean computation (mirrors mean_diff_chat.py's validate_task /
# compute_task_means, with chat/bare renamed to expert/non_expert and NO
# diff_mean output).
# ---------------------------------------------------------------------------

def validate_task(model: str, task: str, hs_root: Path, size: str) -> dict:
    """Validate ONE (model, task)'s expert+non_expert pair, WITHOUT loading
    the full hidden_states arrays into memory yet (header/attrs only).
    Returns a dict with 'ok', 'errors', and (if ok) the file paths +
    expected shape, so the second pass can load and accumulate only after
    every task for this model has already passed."""
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
                f"condition={c}: n_samples_done={n_done} != n={n} in H5 attrs -- "
                "output may be partial/incomplete, refusing to trust it.")
            return result
        if bool(attrs[c].get("steering_applied", True)) is not False:
            result["errors"].append(
                f"condition={c}: steering_applied is not False in H5 attrs -- "
                "refusing to use a steered file as a role-only baseline.")
            return result
        if bool(attrs[c].get("generation_performed", True)) is not False:
            result["errors"].append(
                f"condition={c}: generation_performed is not False in H5 "
                "attrs -- refusing to use a generation-tainted file as a "
                "prefill-only baseline.")
            return result

    digest_expert = attrs["expert"].get("ordered_sample_identity_sha256")
    digest_nonexpert = attrs["non_expert"].get("ordered_sample_identity_sha256")
    if not digest_expert or digest_expert != digest_nonexpert:
        result["errors"].append(
            f"expert/non_expert ordered_sample_identity_sha256 mismatch in "
            f"H5 attrs: {digest_expert!r} vs {digest_nonexpert!r} -- expert "
            "and non_expert must be over the identical ordered question "
            "list.")
        return result

    manifest_digest_expert = manifests["expert"].get("ordered_sample_identity_sha256")
    manifest_digest_nonexpert = manifests["non_expert"].get("ordered_sample_identity_sha256")
    if manifest_digest_expert != digest_expert or manifest_digest_nonexpert != digest_nonexpert:
        result["errors"].append(
            "manifest.json ordered_sample_identity_sha256 disagrees with "
            "the H5 attrs of the same file -- refusing to trust either.")
        return result

    result.update({
        "ok": True,
        "n_samples": n,
        "shape": [n, n_layers, hidden],
        "h5_paths": {c: str(h5_paths[c]) for c in CONDITIONS},
        "manifest_paths": {c: str(manifest_paths[c]) for c in CONDITIONS},
        "pairing_digest": digest_expert,
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
    """Load expert+non_expert arrays for one already-validated task,
    accumulate in float32, and return expert_mean/non_expert_mean (both
    (1, 1, L, H) float16) plus a consistency-audit report. Direct
    ARITHMETIC mean over the sample axis, NO L2 normalization anywhere.
    No diff_mean is computed or returned here -- see the module docstring
    for why."""
    expert_16 = load_and_check_finite(Path(task_info["h5_paths"]["expert"]))
    nonexpert_16 = load_and_check_finite(Path(task_info["h5_paths"]["non_expert"]))

    expert_32 = expert_16.astype(np.float32)
    nonexpert_32 = nonexpert_16.astype(np.float32)

    expert_mean32 = expert_32.mean(axis=0)   # (L, H)
    nonexpert_mean32 = nonexpert_32.mean(axis=0)

    # Consistency audit: float32 (production) mean vs an independent float64
    # (higher-precision) recomputation of the SAME mean, matching
    # mean_diff_chat.py's finalize() pattern -- this checks accumulation
    # precision only, there is no per-sample-difference identity to check
    # here since there is no diff_mean output at this stage.
    expert_64 = expert_16.astype(np.float64)
    nonexpert_64 = nonexpert_16.astype(np.float64)
    expert_mean64 = expert_64.mean(axis=0)
    nonexpert_mean64 = nonexpert_64.mean(axis=0)
    max_abs_err_expert_f64 = float(np.max(np.abs(
        expert_mean64.astype(np.float32) - expert_mean32)))
    max_abs_err_nonexpert_f64 = float(np.max(np.abs(
        nonexpert_mean64.astype(np.float32) - nonexpert_mean32)))

    expert_mean_out = expert_mean32.astype(np.float16)[None, None, ...]
    nonexpert_mean_out = nonexpert_mean32.astype(np.float16)[None, None, ...]

    return {
        "expert_mean": expert_mean_out,
        "non_expert_mean": nonexpert_mean_out,
        "n_samples": expert_16.shape[0],
        "consistency_check": {
            "description": "float32 (production accumulation dtype) mean "
                            "vs an independent float64 (higher-precision "
                            "audit) recomputation of the same mean, both "
                            "before any float16 cast; NO L2 normalization "
                            "applied anywhere -- this is a direct "
                            "arithmetic mean over raw hidden states.",
            "max_abs_error_float64_expert": max_abs_err_expert_f64,
            "max_abs_error_float64_non_expert": max_abs_err_nonexpert_f64,
        },
    }


def atomic_save_npy(path: Path, arr: np.ndarray):
    # np.save() only skips appending ".npy" when the given name's suffix is
    # EXACTLY ".npy" (a str.endswith(".npy") check) -- a mkstemp name ending
    # in ".npy.tmp" does NOT qualify, so np.save silently writes to a
    # SECOND, different path (tmp_path + ".npy") and leaves the original
    # mkstemp-created file behind as an empty, never-written leftover (this
    # bug was found and fixed in mean_diff_chat.py; replicated here fixed
    # from the start). Fixed by suffixing with plain ".npy" and writing into
    # the already-open file descriptor directly, so there is exactly one
    # file on disk at any time before the atomic os.replace().
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


def scan_leftover_tmp_files(root: Path) -> list:
    """Report (never delete) any leftover *.npy.tmp / *.json.tmp residue
    under root, per instruction 9. This is a read-only scan run at the very
    start of main(), before any validation or write."""
    if not root.exists():
        return []
    hits = []
    for p in root.rglob("*.tmp"):
        if p.is_file():
            hits.append(str(p))
    return sorted(hits)


def run_compute_means(args):
    model = args.model
    size = args.size
    hs_root = Path(args.hs_root)
    out_root = Path(args.out_root)
    model_out_dir = out_root / f"{model}_chat"

    leftover = scan_leftover_tmp_files(out_root)
    if leftover:
        print(f"[REPORT] {len(leftover)} leftover *.tmp file(s) found under "
              f"{out_root} -- NOT deleted automatically, report only:")
        for p in leftover:
            print(f"  {p}")

    # ---- Overwrite preflight for ALL 3 tasks' outputs, BEFORE any read. ----
    all_out_paths = []
    for task in TASKS:
        task_dir = model_out_dir / task
        all_out_paths += [
            task_dir / f"expert_mean_{size}.npy",
            task_dir / f"non_expert_mean_{size}.npy",
            task_dir / "role_manifest.json",
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

    print(f"All {len(TASKS)} tasks passed header/attrs validation for "
          f"model={model}, size={size}.")

    # ---- Pass 2: load every task's H5 arrays, check finiteness, and
    # ---- compute means IN MEMORY for ALL 3 tasks -- still NO file write.
    # ---- This is what makes "a single failing task writes nothing for
    # ---- any task" actually true: compute_task_means() is where the
    # ---- finite check happens (load_and_check_finite), and previously it
    # ---- ran interleaved with Pass-3's writes, so a later task's NaN/Inf
    # ---- would be discovered only AFTER earlier tasks had already been
    # ---- written to disk. Now every task's finite check completes before
    # ---- any task's write begins. ----
    means_by_task = {}
    compute_errors = {}
    for task in TASKS:
        v = validations[task]
        print(f"[compute] model={model} task={task} n_samples={v['n_samples']} "
              f"shape={v['shape']}")
        try:
            means_by_task[task] = compute_task_means(v)
        except Exception as e:
            compute_errors[task] = f"{type(e).__name__}: {e}"

    if compute_errors:
        print(f"[REFUSE] {len(compute_errors)}/{len(TASKS)} task(s) failed "
              f"load/finite/compute for model={model} -- aborting before "
              "writing any output (fail-closed).")
        for t, msg in compute_errors.items():
            print(f"  [FAIL] {t}: {msg}")
        sys.exit(1)

    print(f"All {len(TASKS)} tasks passed load/finite/compute for "
          f"model={model}, size={size}.")

    # ---- Pass 3: write, now that every task is fully known-good
    # ---- (validated AND loaded AND finite AND computed). ----
    per_task_report = []
    for task in TASKS:
        v = validations[task]
        means = means_by_task[task]

        task_dir = model_out_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        expert_path = task_dir / f"expert_mean_{size}.npy"
        nonexpert_path = task_dir / f"non_expert_mean_{size}.npy"
        manifest_out_path = task_dir / "role_manifest.json"

        atomic_save_npy(expert_path, means["expert_mean"])
        atomic_save_npy(nonexpert_path, means["non_expert_mean"])

        manifest = {
            "script_version": SCRIPT_VERSION,
            "model": model,
            "task": task,
            "size": size,
            "content": "Expert/Non-expert MEANS ONLY -- this manifest does "
                       "NOT describe a Role direction/diff_mean. "
                       "role_diff = expert_mean - non_expert_mean is "
                       "computed LOCALLY, later, after syncing these two "
                       ".npy files; it is not computed or stored by this "
                       "script.",
            "n_samples": v["n_samples"],
            "input_shape": v["shape"],
            "output_shape": list(means["expert_mean"].shape),
            "normalization": "none -- direct arithmetic mean over raw "
                             "hidden states, no L2 normalization",
            "accumulation_dtype": "float32 (float64 shadow pass for audit only)",
            "output_dtype": "float16",
            "pairing_digest": v["pairing_digest"],
            "input_h5_sha256": v["h5_sha256"],
            "input_h5_paths": v["h5_paths"],
            "input_manifest_paths": v["manifest_paths"],
            "output_paths": {
                "expert_mean": str(expert_path),
                "non_expert_mean": str(nonexpert_path),
            },
            "consistency_check": means["consistency_check"],
            "fail_closed": (
                "Every task for this model is validated (H5+manifest presence, "
                "expert/non_expert shape equality, n_samples==300, expected "
                "(n_layers, hidden_size), n_samples_done==n, "
                "steering_applied==False, generation_performed==False, "
                "ordered_sample_identity_sha256 equality between "
                "expert/non_expert AND between H5 attrs/manifest.json) BEFORE "
                "any accumulation or file write for ANY of this model's 3 "
                "tasks; a single failing task aborts the whole model's run "
                "and writes nothing."
            ),
            "computation": (
                "expert_mean/non_expert_mean = elementwise arithmetic mean "
                "over the sample axis (per layer, per hidden dim), NO L2 "
                "normalization. No diff_mean, cosine, split-half, "
                "bootstrap, or RSN/NMD alignment computed at this stage."
            ),
            "extraction_timestamp_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_write_json(manifest_out_path, manifest)

        per_task_report.append({
            "task": task,
            "n_samples": v["n_samples"],
            "pairing_digest": v["pairing_digest"],
            "consistency_max_abs_error_float64_expert":
                means["consistency_check"]["max_abs_error_float64_expert"],
            "consistency_max_abs_error_float64_non_expert":
                means["consistency_check"]["max_abs_error_float64_non_expert"],
        })
        print(f"[ok] model={model} task={task}: wrote expert_mean/non_expert_mean + "
              f"role_manifest.json under {task_dir}")

    print(f"[DONE] model={model}: {len(TASKS)}/{len(TASKS)} tasks complete.")
    for r in per_task_report:
        print(f"  {r}")


# ---------------------------------------------------------------------------
# --rename_chat_bare mode: rename the existing Chat-Bare diff_mean_{size}.npy
# to chat_bare_diff_mean_{size}.npy in each (model, task) directory, updating
# that task's manifest.json in place. RENAME ONLY -- no value ever read into
# a computation, no recomputation.
# ---------------------------------------------------------------------------

def validate_rename_task(model: str, task: str, mean_root: Path, size: str) -> dict:
    result = {"model": model, "task": task, "ok": False, "errors": []}
    task_dir = mean_root / f"{model}_chat" / task
    src_npy = task_dir / f"diff_mean_{size}.npy"
    dst_npy = task_dir / f"chat_bare_diff_mean_{size}.npy"
    manifest_p = task_dir / "manifest.json"

    src_exists = src_npy.exists()
    dst_exists = dst_npy.exists()

    if not src_exists and dst_exists:
        # This is exactly the half-done state a crash between os.replace()
        # (npy) and atomic_write_json() (manifest) in a PRIOR run would
        # leave behind: the .npy was already renamed, but this task's own
        # manifest was never updated to point at the new name (that update
        # is the LAST step of a per-task rename; a crash before it fires
        # leaves this state, never the reverse). Detected explicitly here
        # rather than silently proceeding -- the caller must inspect
        # manifest_p by hand and confirm whether it is safe to treat this
        # task as already-done before re-running.
        result["errors"].append(
            f"HALF-DONE STATE DETECTED: {dst_npy} exists but {src_npy} does "
            "not, and this looks like an interrupted prior rename for this "
            "task (the .npy was renamed but the manifest update step never "
            "completed or was not verified). Refusing to guess -- inspect "
            f"{manifest_p} by hand: if it already refers to "
            f"{dst_npy.name}, this task is actually already fully done and "
            "can be left alone (or the whole run re-launched, which will "
            "then see this same file layout and can be told to skip it); "
            "if it still refers to the OLD filename, the manifest needs a "
            "manual fix before this script is trusted on it again.")
        return result
    if not src_exists:
        result["errors"].append(f"source file missing: {src_npy}")
        return result
    if not manifest_p.exists():
        result["errors"].append(f"manifest missing: {manifest_p}")
        return result
    if dst_exists:
        result["errors"].append(
            f"target already exists, refusing to overwrite: {dst_npy}")
        return result

    try:
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        result["errors"].append(
            f"failed to parse manifest {manifest_p}: {type(e).__name__}: {e}")
        return result

    direction = manifest.get("difference_direction")
    if direction != EXPECTED_CHAT_BARE_DIRECTION:
        result["errors"].append(
            f"manifest difference_direction={direction!r}, expected "
            f"{EXPECTED_CHAT_BARE_DIRECTION!r} -- refusing to rename a file "
            "whose manifest does not explicitly confirm this is the "
            "Chat-Bare (chat_minus_bare) diff_mean. This check MUST pass "
            "before any rename, per instruction.")
        return result

    result.update({
        "ok": True,
        "src_npy": str(src_npy),
        "dst_npy": str(dst_npy),
        "manifest_path": str(manifest_p),
        "manifest": manifest,
    })
    return result


def rewrite_manifest_filename_refs(manifest: dict, old_name: str, new_name: str) -> dict:
    """Recursively replace any string value equal to, or ending with,
    '/' + old_name (or exactly old_name) with the equivalent new_name path,
    anywhere in the manifest dict/list structure. Only the FILENAME
    component is changed; directory prefixes are preserved verbatim. Never
    touches numeric values."""
    def replace_one(s: str) -> str:
        if s == old_name:
            return new_name
        if s.endswith("/" + old_name):
            return s[: -len(old_name)] + new_name
        return s

    def walk(obj):
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        if isinstance(obj, str):
            return replace_one(obj)
        return obj

    return walk(manifest)


def run_rename_chat_bare(args):
    model = args.model
    size = args.size
    mean_root = Path(args.mean_root)

    leftover = scan_leftover_tmp_files(mean_root)
    if leftover:
        print(f"[REPORT] {len(leftover)} leftover *.tmp file(s) found under "
              f"{mean_root} -- NOT deleted automatically, report only:")
        for p in leftover:
            print(f"  {p}")

    # ---- Pass 1: validate all 3 tasks' rename preconditions BEFORE any
    # ---- rename or manifest write. ----
    validations = {task: validate_rename_task(model, task, mean_root, size)
                   for task in TASKS}
    failed = {t: v for t, v in validations.items() if not v["ok"]}
    if failed:
        print(f"[REFUSE] {len(failed)}/{len(TASKS)} task(s) failed rename "
              f"validation for model={model} -- aborting before renaming "
              "anything (fail-closed).")
        for t, v in failed.items():
            print(f"  [FAIL] {t}: {v['errors']}")
        sys.exit(1)

    print(f"All {len(TASKS)} tasks confirmed difference_direction="
          f"{EXPECTED_CHAT_BARE_DIRECTION!r} for model={model}, size={size}; "
          "proceeding with rename.")

    # ---- Pass 2: rename .npy (atomic os.replace, bytes untouched) + rewrite
    # ---- manifest.json (atomic tempfile + os.replace, values untouched).
    # ----
    # ---- NOTE ON TRANSACTIONALITY (honest limitation, not claimed to be
    # ---- stronger than it is): each task does TWO separate filesystem
    # ---- operations (os.replace the .npy, THEN atomic_write_json the
    # ---- manifest). Each operation is individually atomic, but the PAIR
    # ---- is not -- a crash/disk-error between them leaves that one task's
    # ---- .npy renamed while its manifest still names the old file. This
    # ---- is a real, accepted limitation (a true cross-file transaction
    # ---- would need a WAL or a two-phase-commit protocol, out of scope
    # ---- here), NOT described as strict all-or-nothing. Two mitigations
    # ---- ARE in place: (1) the .npy rename happens FIRST and the manifest
    # ---- write SECOND, so a crash between them always leaves the harmless
    # ---- direction (file physically present under its new name, manifest
    # ---- merely stale) rather than a manifest claiming a file exists that
    # ---- does not; (2) validate_rename_task() explicitly DETECTS this
    # ---- exact half-done state on any subsequent run (dst_npy exists,
    # ---- src_npy does not) and refuses with a clear message rather than
    # ---- silently reprocessing or silently skipping it. Tasks are also
    # ---- still processed one at a time with per-task progress printed
    # ---- immediately below, so a crash's blast radius (which tasks
    # ---- completed vs which did not) is visible in the log rather than
    # ---- discovered later. ----
    old_name = f"diff_mean_{size}.npy"
    new_name = f"chat_bare_diff_mean_{size}.npy"
    for task in TASKS:
        v = validations[task]
        os.replace(v["src_npy"], v["dst_npy"])
        print(f"[step] model={model} task={task}: renamed .npy "
              f"({v['src_npy']} -> {v['dst_npy']}); updating manifest next")

        updated_manifest = rewrite_manifest_filename_refs(
            v["manifest"], old_name, new_name)
        updated_manifest["renamed_by"] = {
            "script_version": SCRIPT_VERSION,
            "old_filename": old_name,
            "new_filename": new_name,
            "renamed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": "avoid semantic confusion with the newly-added "
                     "Expert/Non-expert mean files in the same directory; "
                     "values NOT recomputed or modified -- rename only, "
                     "direction confirmed chat_minus_bare before renaming.",
        }
        atomic_write_json(Path(v["manifest_path"]), updated_manifest)

        print(f"[ok] model={model} task={task}: renamed "
              f"{v['src_npy']} -> {v['dst_npy']}; updated {v['manifest_path']}")

    print(f"[DONE] model={model}: {len(TASKS)}/{len(TASKS)} chat-bare files renamed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SHAPE.keys()))
    ap.add_argument("--size", required=True,
                    help=f"Locked to --model: {MODEL_EXPECTED_SIZE}")
    ap.add_argument("--rename_chat_bare", action="store_true",
                    help="Run the rename-only mode instead of computing "
                         "Expert/Non-expert means. See module docstring.")
    ap.add_argument("--hs_root", default=None,
                    help="Required for the default (mean-computation) mode, "
                         "e.g. /data1/paveen/Dopamine/components/hidden_states")
    ap.add_argument("--out_root", default=None,
                    help="Required for the default (mean-computation) mode, "
                         "e.g. /data1/paveen/Dopamine/components/hidden_states_mean")
    ap.add_argument("--mean_root", default=None,
                    help="Required for --rename_chat_bare mode, e.g. "
                         "/data1/paveen/Dopamine/components/hidden_states_mean")
    args = ap.parse_args()
    assert_size_matches_model(args.model, args.size)

    if args.rename_chat_bare:
        if not args.mean_root:
            die("--rename_chat_bare requires --mean_root")
        run_rename_chat_bare(args)
    else:
        if not args.hs_root or not args.out_root:
            die("mean-computation mode requires --hs_root and --out_root")
        run_compute_means(args)


if __name__ == "__main__":
    main()
