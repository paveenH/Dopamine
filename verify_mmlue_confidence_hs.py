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
  - hidden_states has no NaN/Inf
  - answer JSON has A-E logits/softmax for both roles

Then writes a manifest JSON summarizing model/checkpoint, task count, per-task
sample counts, prompt mode, cot/chat flags, H5 shape/dtype, output paths, and
aggregate confident/unconfident accuracy + E-ratio (manipulation check only —
not used to filter neurons).

Usage:
  python3.10 verify_mmlue_confidence_hs.py \
      --hs_dir /data1/paveen/ConfSteer/HiddenStates/llama3/mmlue_confidence \
      --ans_dir /data1/paveen/Dopamine/components/llama3/answer_hs_mmlue_confidence \
      --size 8B \
      --out manifest_mmlue_confidence.json
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ROLES = ["confident", "unconfident"]


def check_task(task: str, hs_dir: Path, ans_dir: Path, size: str) -> dict:
    result = {"task": task, "ok": True, "errors": []}

    ans_path = ans_dir / f"{task}_{size}_answers.json"
    if not ans_path.exists():
        result["ok"] = False
        result["errors"].append(f"missing answer JSON: {ans_path}")
        return result

    with open(ans_path, "r", encoding="utf-8") as f:
        ans = json.load(f)
    samples = ans["data"]
    n_samples = len(samples)
    result["n_samples"] = n_samples
    result["accuracy"] = ans.get("accuracy", {})

    role_shapes = {}
    for role in ROLES:
        h5_path = hs_dir / f"{role}_{task}_{size}.h5"
        if not h5_path.exists():
            result["ok"] = False
            result["errors"].append(f"missing H5: {h5_path}")
            continue
        with h5py.File(h5_path, "r") as f:
            if "hidden_states" not in f:
                result["ok"] = False
                result["errors"].append(f"{h5_path}: no 'hidden_states' dataset")
                continue
            ds = f["hidden_states"]
            role_shapes[role] = ds.shape
            if ds.shape[0] != n_samples:
                result["ok"] = False
                result["errors"].append(
                    f"{h5_path}: N={ds.shape[0]} != answer sample count {n_samples}"
                )
            if ds.dtype != np.float16:
                result["ok"] = False
                result["errors"].append(f"{h5_path}: dtype={ds.dtype}, expected float16")
            arr = ds[:]
            if not np.isfinite(arr.astype(np.float32)).all():
                result["ok"] = False
                result["errors"].append(f"{h5_path}: contains NaN/Inf")

        # answer JSON must carry answer/prob/softmax/logits for this role
        safe_role = role
        for key_prefix in ["answer_", "prob_", "softmax_", "logits_"]:
            key = f"{key_prefix}{safe_role}"
            if not samples or key not in samples[0]:
                result["ok"] = False
                result["errors"].append(f"answer JSON missing key '{key}' for role {role}")

    if len(role_shapes) == 2:
        shapes = list(role_shapes.values())
        if shapes[0] != shapes[1]:
            result["ok"] = False
            result["errors"].append(f"paired shape mismatch: {role_shapes}")
        result["hs_shape"] = {k: list(v) for k, v in role_shapes.items()}

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
    dtype_seen = set()
    shape_seen = set()

    for task in TASKS:
        r = check_task(task, hs_dir, ans_dir, args.size)
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
                shape_seen.add(tuple(v[1:]))  # drop N dim (varies little but should match)

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


if __name__ == "__main__":
    main()
