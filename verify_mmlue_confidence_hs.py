#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Acceptance check + manifest generator for the MMLU-E Confidence HS collection
(run_hidden_mmlue_confidence_hs.sh). Read-only: does not touch HS/answers.

Checks per task:
  - confident + unconfident H5 both exist
  - H5 N matches the answer JSON sample count
  - paired H5 shapes match exactly
  - dtype == float16
  - hidden_states has no NaN/Inf (read per-sample/chunk, never the whole
    dataset at once)
  - EVERY sample (not just the first) in the answer JSON carries, for both
    roles: answer_/prob_/softmax_/logits_ keys; softmax and logits have
    length 5 (A-E); answer is one of A/B/C/D/E; prob/softmax/logits are all
    finite

A missing task/H5/answer file is recorded as a failure for that task and does
NOT crash the script or get reported as an overall success; the run keeps
checking every other task and reports full detail at the end.

Exit code is 1 if all_ok is False, 0 otherwise, so the check can gate a
pipeline step.

Usage (run on the SERVER, in the project's conda env -- the server interpreter
is `python`, NOT `python3.10`; `python3.10` is the local analysis-box
convention only and does not exist on the server):
  python verify_mmlue_confidence_hs.py \
      --hs_dir /data1/paveen/ConfSteer/HiddenStates/llama3/mmlue_confidence \
      --ans_dir /data1/paveen/Dopamine/components/llama3/answer_hs_mmlue_confidence \
      --size 8B \
      --out manifest_mmlue_confidence.json
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROLES = ["confident", "unconfident"]
VALID_ANSWERS = {"A", "B", "C", "D", "E"}
N_OPTIONS = 5
# Llama3.1-8B-Instruct: 32 decoder layers + embedding layer = 33 hidden_states
# entries per forward pass (output_hidden_states=True), hidden_dim=4096.
EXPECTED_NUM_HIDDEN_STATES = 33
EXPECTED_HIDDEN_DIM = 4096


def check_h5_finite(h5_path: Path, n_samples: int, errors: list) -> tuple:
    """Read the dataset in per-sample chunks (never ds[:] whole-file) and
    check for NaN/Inf, dtype, and the exact expected shape.
    Returns the dataset shape, or None on hard failure (missing dataset or
    a parse/read exception, which is recorded as an error rather than
    propagated)."""
    try:
        with h5py.File(h5_path, "r") as f:
            if "hidden_states" not in f:
                errors.append(f"{h5_path}: no 'hidden_states' dataset")
                return None
            ds = f["hidden_states"]
            shape = ds.shape
            dtype = ds.dtype
            if dtype != np.float16:
                errors.append(f"{h5_path}: dtype={dtype}, expected float16")
            if shape[0] != n_samples:
                errors.append(f"{h5_path}: N={shape[0]} != answer sample count {n_samples}")
            if len(shape) != 3 or shape[1:] != (EXPECTED_NUM_HIDDEN_STATES, EXPECTED_HIDDEN_DIM):
                errors.append(
                    f"{h5_path}: shape={shape}, expected "
                    f"(N, {EXPECTED_NUM_HIDDEN_STATES}, {EXPECTED_HIDDEN_DIM})"
                )
            # Per-sample chunked read so a large H5 is never loaded whole.
            n_to_scan = min(shape[0], n_samples) if shape[0] else 0
            for i in range(n_to_scan):
                chunk = ds[i].astype(np.float32)
                if not np.isfinite(chunk).all():
                    errors.append(f"{h5_path}: NaN/Inf at sample index {i}")
            return shape
    except Exception as e:
        errors.append(f"{h5_path}: failed to read H5 ({type(e).__name__}: {e})")
        return None


def check_answer_samples(task: str, role: str, samples: list, errors: list) -> None:
    """Check EVERY sample (not just samples[0]) for this role's fields."""
    key_answer = f"answer_{role}"
    key_prob = f"prob_{role}"
    key_softmax = f"softmax_{role}"
    key_logits = f"logits_{role}"

    for idx, sample in enumerate(samples):
        missing = [key for key in (key_answer, key_prob, key_softmax, key_logits) if key not in sample]
        if missing:
            for key in missing:
                errors.append(f"{task}/{role}: sample {idx} missing key '{key}'")
            continue  # skip further checks on THIS sample only, keep checking the rest

        ans = sample[key_answer]
        if ans not in VALID_ANSWERS:
            errors.append(f"{task}/{role}: sample {idx} answer '{ans}' not in A-E")

        prob = sample[key_prob]
        if not np.isfinite(prob):
            errors.append(f"{task}/{role}: sample {idx} prob is not finite ({prob})")

        softmax = sample[key_softmax]
        if len(softmax) != N_OPTIONS:
            errors.append(
                f"{task}/{role}: sample {idx} softmax length {len(softmax)} != {N_OPTIONS}"
            )
        elif not all(np.isfinite(v) for v in softmax):
            errors.append(f"{task}/{role}: sample {idx} softmax contains non-finite value")

        logits = sample[key_logits]
        if len(logits) != N_OPTIONS:
            errors.append(
                f"{task}/{role}: sample {idx} logits length {len(logits)} != {N_OPTIONS}"
            )
        elif not all(np.isfinite(v) for v in logits):
            errors.append(f"{task}/{role}: sample {idx} logits contains non-finite value")


def check_task(task: str, hs_dir: Path, ans_dir: Path, size: str) -> dict:
    result = {"task": task, "ok": True, "errors": []}

    ans_path = ans_dir / f"{task}_{size}_answers.json"
    if not ans_path.exists():
        result["ok"] = False
        result["errors"].append(f"missing answer JSON: {ans_path}")
        return result

    try:
        with open(ans_path, "r", encoding="utf-8") as f:
            ans = json.load(f)
        samples = ans["data"]
    except Exception as e:
        result["ok"] = False
        result["errors"].append(f"{ans_path}: failed to parse ({type(e).__name__}: {e})")
        return result

    n_samples = len(samples)
    result["n_samples"] = n_samples
    result["accuracy"] = ans.get("accuracy", {})

    for role in ROLES:
        check_answer_samples(task, role, samples, result["errors"])

    role_shapes = {}
    for role in ROLES:
        h5_path = hs_dir / f"{role}_{task}_{size}.h5"
        if not h5_path.exists():
            result["errors"].append(f"missing H5: {h5_path}")
            continue
        shape = check_h5_finite(h5_path, n_samples, result["errors"])
        if shape is not None:
            role_shapes[role] = shape

    if len(role_shapes) == 2:
        shapes = list(role_shapes.values())
        if shapes[0] != shapes[1]:
            result["errors"].append(f"paired shape mismatch: {role_shapes}")
        result["hs_shape"] = {k: list(v) for k, v in role_shapes.items()}
    elif len(role_shapes) < 2:
        # Already recorded as missing above; nothing more to compare.
        pass

    if result["errors"]:
        result["ok"] = False

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs_dir", required=True)
    ap.add_argument("--ans_dir", required=True)
    ap.add_argument("--size", default="8B")
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--out", default="manifest_mmlue_confidence.json")
    args = ap.parse_args()

    hs_dir = Path(args.hs_dir)
    ans_dir = Path(args.ans_dir)

    from detection.task_list import TASKS

    task_results = []
    all_ok = True
    total_correct = {r: 0 for r in ROLES}
    total_E = {r: 0 for r in ROLES}
    total_n = {r: 0 for r in ROLES}
    shape_seen = set()

    for task in TASKS:
        try:
            r = check_task(task, hs_dir, ans_dir, args.size)
        except Exception as e:
            r = {
                "task": task,
                "ok": False,
                "errors": [f"unexpected exception during check: {type(e).__name__}: {e}"],
            }
        task_results.append(r)
        if not r["ok"]:
            all_ok = False
            print(f"[FAIL] {task}: {r['errors']}")
        else:
            print(f"[OK]   {task}: n={r['n_samples']}")
        acc = r.get("accuracy", {})
        for role in ROLES:
            s = acc.get(role, {})
            total_correct[role] += s.get("correct", 0)
            total_E[role] += s.get("E_count", 0)
            total_n[role] += s.get("total", 0)
        if "hs_shape" in r:
            for v in r["hs_shape"].values():
                shape_seen.add(tuple(v[1:]))  # drop N dim

    e_ratio = {
        role: (total_E[role] / total_n[role] if total_n[role] else None)
        for role in ROLES
    }
    acc_overall = {
        role: (total_correct[role] / total_n[role] if total_n[role] else None)
        for role in ROLES
    }

    manifest = {
        "model": args.model,
        "model_dir": args.model_dir,
        "size": args.size,
        "n_tasks": len(TASKS),
        "tasks": TASKS,
        "roles": ROLES,
        "prompt_mode": "bare-string (no chat template)",
        "use_chat": False,
        "cot": False,
        "suite": "default",
        "use_E": True,
        "hs_dtype": "float16",
        "hs_shapes_seen": [list(s) for s in shape_seen],
        "hs_dir": str(hs_dir),
        "ans_dir": str(ans_dir),
        "per_task": task_results,
        "all_ok": all_ok,
        "aggregate": {
            "total_n": total_n,
            "total_correct": total_correct,
            "total_E_count": total_E,
            "accuracy": acc_overall,
            "E_ratio": e_ratio,
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nWrote manifest -> {args.out}")
    print(f"all_ok = {all_ok}")
    print(f"E_ratio: {e_ratio}")
    print(f"accuracy: {acc_overall}")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
