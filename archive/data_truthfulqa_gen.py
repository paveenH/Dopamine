#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TruthfulQA-Generation -> single JSON for the open-ended over-generation / truthfulness probe.

- Loads HF `truthfulqa/truthful_qa` config `generation` (817 Q, validation; pure
  Parquet). Fields: question / best_answer / correct_answers[] / incorrect_answers[].
- Samples N questions (default 100). Carries the reference answer lists into the
  output so truthfulness can be judged offline against them (no fine-tuned
  GPT-judge needed — judgments are made by reading raw generations vs. these refs).

Output schema:
  {
    "data": [
      {
        "task": "TruthfulQA-Gen",
        "text": "<question>",                # the bare question (prompt body)
        "question": "<question>",
        "best_answer": "...",
        "correct_answers": [...],
        "incorrect_answers": [...],
        "category": "...",
        "src_idx": <0-based row index>
      },
      ...
    ]
  }

Usage:
  python data_truthfulqa_gen.py --n 100 --out /data1/paveen/Dopamine/components/benchmark/truthfulqa_gen.json
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of questions to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default="/data1/paveen/Dopamine/components/benchmark/truthfulqa_gen.json")
    ap.add_argument("--cache_dir", type=str, default=None)
    args = ap.parse_args()

    print("[LOAD] truthfulqa/truthful_qa :: generation")
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation",
                      cache_dir=args.cache_dir)
    n_total = len(ds)
    print(f"[INFO] generation total rows: {n_total}")

    rng = random.Random(args.seed)
    idxs = list(range(n_total))
    rng.shuffle(idxs)
    idxs = sorted(idxs[: args.n])   # sorted for stable reading order

    data: List[Dict[str, Any]] = []
    for src_idx in idxs:
        row = ds[src_idx]
        q = str(row.get("question", "")).strip()
        if not q:
            continue
        data.append({
            "task": "TruthfulQA-Gen",
            "text": q,
            "question": q,
            "best_answer": str(row.get("best_answer", "")).strip(),
            "correct_answers": list(row.get("correct_answers", []) or []),
            "incorrect_answers": list(row.get("incorrect_answers", []) or []),
            "category": str(row.get("category", "")).strip(),
            "src_idx": src_idx,
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"data": data}, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved TruthfulQA-Gen → {args.out}")
    print(f"[INFO] kept {len(data)} questions from {len(idxs)} sampled rows")
    print("[Preview top-2]")
    print(json.dumps(data[:2], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
