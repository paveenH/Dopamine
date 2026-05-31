#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and format MATH dataset (300-sample version) for effort-choice experiment.

Samples 300 examples from the test set (all difficulty levels, all subjects).

Output JSON format:
[
  {
    "task": "math",
    "question": "...",
    "answer": "...",       # ground truth answer (often LaTeX expression)
    "solution": "...",     # full solution
    "level": "Level 4",   # difficulty level
    "type": "Algebra"     # subject type
  },
  ...
]
"""

import os
import json
import random
from datasets import load_dataset


if __name__ == "__main__":
    SAMPLE_SIZE = 300
    RANDOM_SEED = 42

    cache_dir = "/data1/paveen/RolePlaying/.cache"
    save_dir = "/data1/paveen/RolePlaying/components/benchmark"
    os.makedirs(save_dir, exist_ok=True)

    # Try standard Parquet dataset first; fields: problem, solution, level, type
    # nlile/hendrycks-MATH-benchmark is a parquet mirror with no loading script
    try:
        ds = load_dataset(
            "nlile/hendrycks-MATH-benchmark",
            split="test",
            cache_dir=cache_dir,
        )
    except Exception:
        # Fallback: original repo with loading script
        ds = load_dataset(
            "hendrycks/competition_math",
            "all",
            split="test",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    random.seed(RANDOM_SEED)
    total = len(ds)
    if total <= SAMPLE_SIZE:
        indices = list(range(total))
    else:
        indices = random.sample(range(total), SAMPLE_SIZE)
        indices.sort()

    export = []
    for idx in indices:
        row = ds[idx]
        export.append({
            "task": "math",
            "question": row["problem"].strip(),
            "answer": row["solution"].strip(),  # full solution; ground truth answer is last \boxed{}
            "level": row.get("level", ""),
            "type": row.get("type", ""),
        })

    out_path = os.path.join(save_dir, "math_test_sample.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    print(f"Saved MATH test sample: {len(export)}/{total} samples -> {out_path}")
    print(f"  Random seed: {RANDOM_SEED}")
    if export:
        levels = {}
        for item in export:
            levels[item["level"]] = levels.get(item["level"], 0) + 1
        print(f"  Level distribution: {dict(sorted(levels.items()))}")
