#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HaluEval QA (discrimination) -> single JSON for the signed-bias hallucination probe.

- Loads HF `pminervini/HaluEval` config `qa` (pure Parquet; the new `datasets`
  reads it directly). Each source row has: knowledge / question / right_answer /
  hallucinated_answer.
- We sample N source rows (default 300) and expand EACH into TWO discrimination
  items — one showing the right_answer (gold = "No", not hallucinated) and one
  showing the hallucinated_answer (gold = "Yes", hallucinated). So 300 rows ->
  600 balanced items (half gold=Yes, half gold=No).
- This balance is what lets the runner read a SIGNED bias (FPR vs FNR / overall
  "accept-as-true" tendency) instead of a single accuracy number.

Output schema:
  {
    "data": [
      {
        "task": "HaluEval-QA",
        "knowledge": "...",
        "question": "...",
        "answer": "<the answer being judged>",
        "is_hallucination": 0|1,   # gold: 1 = hallucinated, 0 = right
        "src_idx": <0-based source row index>,
        "variant": "right" | "hallucinated"
      },
      ...
    ]
  }

Usage:
  python data_halueval.py --n 300 --out /data1/paveen/Dopamine/components/benchmark/halueval_qa_disc.json
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of source rows to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default="/data1/paveen/Dopamine/components/benchmark/halueval_qa_disc.json")
    ap.add_argument("--cache_dir", type=str, default=None)
    args = ap.parse_args()

    print("[LOAD] pminervini/HaluEval :: qa")
    ds = load_dataset("pminervini/HaluEval", "qa", split="data", cache_dir=args.cache_dir)
    n_total = len(ds)
    print(f"[INFO] qa total rows: {n_total}")

    rng = random.Random(args.seed)
    idxs = list(range(n_total))
    rng.shuffle(idxs)
    idxs = idxs[: args.n]

    data: List[Dict[str, Any]] = []
    for src_idx in idxs:
        row = ds[src_idx]
        knowledge = str(row.get("knowledge", "")).strip()
        question = str(row.get("question", "")).strip()
        right = str(row.get("right_answer", "")).strip()
        hall = str(row.get("hallucinated_answer", "")).strip()
        if not question or not right or not hall:
            continue
        data.append({
            "task": "HaluEval-QA", "knowledge": knowledge, "question": question,
            "answer": right, "is_hallucination": 0, "src_idx": src_idx, "variant": "right",
        })
        data.append({
            "task": "HaluEval-QA", "knowledge": knowledge, "question": question,
            "answer": hall, "is_hallucination": 1, "src_idx": src_idx, "variant": "hallucinated",
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"data": data}, f, ensure_ascii=False, indent=2)

    n_pos = sum(d["is_hallucination"] for d in data)
    print(f"✅ Saved HaluEval-QA discrimination → {args.out}")
    print(f"[INFO] {len(data)} items ({len(data)-n_pos} right / {n_pos} hallucinated), "
          f"from {len(idxs)} source rows")
    print("[Preview top-2]")
    print(json.dumps(data[:2], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
