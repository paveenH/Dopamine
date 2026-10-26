#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:54:17 2024
@description:
    Mean NMD for Expert - NoExpert in diff answer-pairs
    Uses incremental mean calculation to avoid memory issues.

@author: paveenhuang
"""

## Calculate Mean for All Tasks

import os
import sys
import numpy as np
import json
import argparse

# Add parent directory to path to import task_list from detection/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'detection'))
from task_list import TASKS

# ==================== Argument Parsing ====================
parser = argparse.ArgumentParser(description="Compute mean hidden states for inconsistent samples")
parser.add_argument("--model", type=str, default="llama3", help="Model name")
parser.add_argument("--size", type=str, default="70B", help="Model size (e.g., 1B, 7B, 70B)")
parser.add_argument("--type", type=str, default="non", help="Type: 'non' or 'exp'")
parser.add_argument("--base_dir", type=str, default="/work/d12922004/RolePlaying/components",
                    help="Base directory for data")
parser.add_argument("--input_format", choices=["npy", "h5"], default="npy",
                    help="npy = legacy layout (default, unchanged); h5 = get_answer_logits.py --save layout")
parser.add_argument("--answer_dir", type=str, default=None, help="[h5] directory with {task}_{size}_answers.json")
parser.add_argument("--hs_dir", type=str, default=None, help="[h5] directory with the per-role .h5 files")
parser.add_argument("--out_dir", type=str, default=None, help="[h5] output directory for the mean files + manifest.json")
args = parser.parse_args()


def run_h5():
    """Paired Expert/Non-expert means over items whose two answers differ. Fail-closed: any problem exits with
    status 1 BEFORE anything is written (nothing is saved until every task has been read)."""
    import h5py
    for k in ("answer_dir", "hs_dir", "out_dir"):
        if not getattr(args, k):
            sys.exit(f"[x] --{k} is required with --input_format h5")
    size = args.size
    out_files = [os.path.join(args.out_dir, f"diff_mean_{size}.npy"), os.path.join(args.out_dir, f"none_diff_mean_{size}.npy"),
                 os.path.join(args.out_dir, "manifest.json")]
    if any(os.path.exists(f) for f in out_files):
        sys.exit(f"[x] output files already exist in {args.out_dir}; refusing to overwrite")
    sums, count, shape, per_task = [None, None], 0, None, {}
    for task in TASKS:
        tk = task.replace("_", " ")
        ek, nk = f"answer_{tk}_expert".replace(" ", "_"), f"answer_non_{tk}_expert".replace(" ", "_")
        jf = os.path.join(args.answer_dir, f"{task}_{size}_answers.json")
        hf = [os.path.join(args.hs_dir, f"{task}_expert_{task}_{size}.h5"), os.path.join(args.hs_dir, f"non_{task}_expert_{task}_{size}.h5")]
        for f in [jf] + hf:
            if not os.path.exists(f):
                sys.exit(f"[x] missing file: {f}")
        data = json.load(open(jf, encoding="utf-8"))["data"]
        sel = []
        for i, e in enumerate(data):
            a, b = e.get(ek), e.get(nk)
            if a is None and b is None:
                if 0 <= e.get("label", -1) < 5:
                    sys.exit(f"[x] {task}[{i}]: role answer keys missing ({ek}, {nk}) for a valid item")
                continue                                   # item skipped by the extractor (label out of range)
            if a is None or b is None:
                sys.exit(f"[x] {task}[{i}]: only one role key present")
            if a != b:
                sel.append(i)
        files = [h5py.File(f, "r") for f in hf]
        try:
            ds = []
            for fh, f in zip(files, hf):
                if "hidden_states" not in fh:
                    sys.exit(f"[x] no 'hidden_states' dataset in {f}")
                ds.append(fh["hidden_states"])
            if ds[0].shape != ds[1].shape or ds[0].ndim != 3:
                sys.exit(f"[x] {task}: H5 shapes {ds[0].shape} / {ds[1].shape} (need identical (N,L,H))")
            if ds[0].shape[0] != len(data):
                sys.exit(f"[x] {task}: H5 rows {ds[0].shape[0]} != answers {len(data)}")
            if shape is None:
                shape = ds[0].shape[1:]
                sums = [np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)]
            elif ds[0].shape[1:] != shape:
                sys.exit(f"[x] {task}: per-sample shape {ds[0].shape[1:]} != {shape}")
            for j in range(0, len(sel), 100):
                idx = sel[j:j + 100]
                for r_ in (0, 1):
                    blk = ds[r_][idx, ...].astype(np.float32)
                    if not np.isfinite(blk).all():
                        sys.exit(f"[x] {task}: non-finite hidden states in role {r_}")
                    sums[r_] += blk.sum(axis=0)
        finally:
            for fh in files:
                fh.close()
        count += len(sel)
        per_task[task] = {"items": len(data), "selected": len(sel)}
        print(f"{task}: selected {len(sel)}/{len(data)}  (total {count})")
    if count == 0:
        sys.exit("[x] no differing samples; nothing to average")
    os.makedirs(args.out_dir, exist_ok=True)
    exp = (sums[0] / count).astype(np.float16)[None, None, ...]
    non = (sums[1] / count).astype(np.float16)[None, None, ...]
    np.save(out_files[0], exp); np.save(out_files[1], non)
    json.dump({"model": args.model, "size": size, "source_answer_dir": args.answer_dir, "source_hs_dir": args.hs_dir,
               "selection_rule": "answer_{task}_expert != answer_non_{task}_expert (same item index); no filter on correctness or E direction",
               "aggregation": "pooled over selected samples across tasks (not a mean of task means); float32 accumulation, saved float16",
               "diff_mean_8B_meaning": "Expert mean (file diff_mean_<size>.npy)", "none_diff_mean_meaning": "Non-expert mean",
               "shape": list(exp.shape), "dtype": "float16", "includes_embedding": True, "n_tasks": len(per_task),
               "total_selected": count, "per_task": per_task}, open(out_files[2], "w"), indent=1)
    print(f"saved {out_files[0]} {exp.shape}, {out_files[1]}, manifest; total selected {count}")


if args.input_format == "h5":
    run_h5()
    sys.exit(0)

# ==================== Configuration ====================
model = args.model
size = args.size
TYPE = args.type
AnswerName = f"answer_{TYPE}_logits"  # "answer_non_logits" or "answer_non"

# Base directory (NCHC or local)
DIR = args.base_dir

# Paths
json_path = os.path.join(DIR, AnswerName, model)
hidden_states_path = os.path.join(DIR, f"hidden_states_{TYPE}", model)

if "logits" in AnswerName:
    save_path = os.path.join(DIR, "hidden_states_mean", f"{model}_{TYPE}_logits")
else:
    save_path = os.path.join(DIR, "hidden_states_mean", f"{model}_{TYPE}")
os.makedirs(save_path, exist_ok=True)

print(f"Model: {model}, Size: {size}, Type: {TYPE}")
print(f"JSON path: {json_path}")
print(f"Hidden states path: {hidden_states_path}")
print(f"Save path: {save_path}")

# Use incremental mean calculation to avoid memory issues
# mean = sum / count, computed incrementally as: new_mean = old_mean + (new_data - old_mean) / new_count
char_sum = None
none_char_sum = None
char_count = 0
none_char_count = 0

for task in TASKS:
    try:
        print(f"Processing task: {task}")

        # Construct file paths
        # File naming: {task}_expert_{task}_{size}.npy, non_{task}_expert_{task}_{size}.npy
        data_char_filepath = os.path.join(hidden_states_path, f"{task}_expert_{task}_{size}.npy")
        data_none_char_filepath = os.path.join(hidden_states_path, f"{TYPE}_{task}_expert_{task}_{size}.npy")
        json_filepath = os.path.join(json_path, f"{task}_{size}_answers.json")

        # Check if NPY files exist
        if not os.path.exists(data_char_filepath):
            print(f"Data char file not found: {data_char_filepath}")
            continue
        if not os.path.exists(data_none_char_filepath):
            print(f"Data none-char file not found: {data_none_char_filepath}")
            continue

        # Check if JSON file exists first (before loading large npy files)
        if not os.path.exists(json_filepath):
            print(f"JSON file not found: {json_filepath}")
            continue

        # Load inconsistent indices from JSON
        with open(json_filepath, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        inconsistent_indices = []
        for idx, entry in enumerate(data.get("data", [])):
            # Keys based on role names: "{task} expert" -> "answer_{task}_expert"
            #                           "non {task} expert" -> "answer_non_{task}_expert"
            task_key = task.replace("_", " ")  # e.g., "college_math" -> "college math"
            expert_key = f"answer_{task_key}_expert".replace(" ", "_")  # "answer_college_math_expert"
            non_expert_key = f"answer_non_{task_key}_expert".replace(" ", "_")  # "answer_non_college_math_expert"

            ans_expert = entry.get(expert_key)
            ans_non_expert = entry.get(non_expert_key)
            if ans_expert != ans_non_expert:
                inconsistent_indices.append(idx)

        if not inconsistent_indices:
            print(f"No inconsistent samples found for task: {task}")
            continue

        # Load npy files with memory mapping to reduce memory usage
        data_char = np.load(data_char_filepath, mmap_mode='r')
        data_none_char = np.load(data_none_char_filepath, mmap_mode='r')

        # Get only inconsistent samples and compute their sum
        # Process in smaller batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(inconsistent_indices), batch_size):
            batch_indices = inconsistent_indices[i:i+batch_size]

            # Load batch data
            char_batch = data_char[batch_indices, ...]
            none_char_batch = data_none_char[batch_indices, ...]

            # Ensure correct shape (N, 1, layers, hidden)
            if char_batch.ndim == 3:
                char_batch = char_batch[:, None, :, :]
            if none_char_batch.ndim == 3:
                none_char_batch = none_char_batch[:, None, :, :]

            # Convert to float32 for accumulation to avoid overflow
            char_batch = char_batch.astype(np.float32)
            none_char_batch = none_char_batch.astype(np.float32)

            # Accumulate sum
            batch_char_sum = char_batch.sum(axis=0)
            batch_none_char_sum = none_char_batch.sum(axis=0)

            if char_sum is None:
                char_sum = batch_char_sum
                none_char_sum = batch_none_char_sum
            else:
                char_sum += batch_char_sum
                none_char_sum += batch_none_char_sum

            char_count += len(batch_indices)
            none_char_count += len(batch_indices)

        print(f"Processed task: {task}, inconsistent samples: {len(inconsistent_indices)}, total so far: {char_count}")

    except Exception as e:
        print(f"Error processing task {task}: {e}")

# Compute final means
if char_count > 0:
    char_mean = (char_sum / char_count).astype(np.float16)  # Convert back to float16 for storage
    char_mean = char_mean[None, ...]  # Add batch dimension: (1, 1, layers, hidden)
    char_mean_filepath = os.path.join(save_path, f"diff_mean_{size}.npy")
    np.save(char_mean_filepath, char_mean)
    print(f"Expert mean saved to {char_mean_filepath}, shape: {char_mean.shape}, total samples: {char_count}")
else:
    print("No expert differences found across tasks.")

if none_char_count > 0:
    none_char_mean = (none_char_sum / none_char_count).astype(np.float16)
    none_char_mean = none_char_mean[None, ...]
    none_char_mean_filepath = os.path.join(save_path, f"none_diff_mean_{size}.npy")
    np.save(none_char_mean_filepath, none_char_mean)
    print(f"Non-expert mean saved to {none_char_mean_filepath}, shape: {none_char_mean.shape}, total samples: {none_char_count}")
else:
    print("No non-expert differences found across tasks.")
