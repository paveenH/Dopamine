#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SocialIQA (lighteval/siqa) -> single MMLU-Pro-like JSON (3-choice).

- Loads the HF `lighteval/siqa` validation split (1954 items WITH gold labels).
  lighteval/siqa is a pure-Parquet mirror of allenai/social_i_qa (identical
  fields); the script-based originals are rejected by newer `datasets`.
  NOTE: the test split (~2224 items) is unlabeled, so it cannot be used for
  accuracy — validation is the only labeled eval set.
- Each item: context + question + 3 answer choices (A/B/C).
- Output schema matches the other data_*.py loaders so the SocialIQA runner
  can reuse utils.load_json / the {text,label,num_options} convention:

  {
    "data": [
      {
        "task": "SocialIQA",
        "text": "<context>\n\n<question>\nA) ...\nB) ...\nC) ...\n",
        "label": <0-based correct index in {0,1,2}>,
        "num_options": 3
      },
      ...
    ]
  }

Usage:
  python data_socialiqa.py --out /data1/paveen/Dopamine/components/socialiqa/socialiqa.json
"""

import os
import json
import argparse
from typing import List, Dict, Any

from datasets import load_dataset

LETTER = [chr(ord("A") + i) for i in range(26)]


def build_text(context: str, question: str, options: List[str]) -> str:
    context = (context or "").strip()
    question = (question or "").strip()
    lines: List[str] = []
    if context:
        lines.append(context)
        lines.append("")  # blank line between context and question
    lines.append(question)
    for i, opt in enumerate(options):
        lines.append(f"{LETTER[i]}) {str(opt).strip()}")
    return "\n".join(lines) + "\n"


def row_to_item(row: Dict[str, Any]) -> Dict[str, Any]:
    context = str(row.get("context", "")).strip()
    question = str(row.get("question", "")).strip()
    options = [
        str(row.get("answerA", "")).strip(),
        str(row.get("answerB", "")).strip(),
        str(row.get("answerC", "")).strip(),
    ]
    # HF social_i_qa stores label as "1"/"2"/"3" (1-based string)
    gold_1based = int(str(row.get("label", "")).strip())
    gold = gold_1based - 1
    if not question or any(o == "" for o in options) or not (0 <= gold < 3):
        return None
    return {
        "task": "SocialIQA",
        "text": build_text(context, question, options),
        "label": int(gold),
        "num_options": 3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="validation",
                    help="HF split with gold labels (validation = 1954 items).")
    ap.add_argument("--out", type=str,
                    default="/data1/paveen/Dopamine/components/benchmark/socialiqa.json")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    # lighteval/siqa is a pure-Parquet mirror (same fields as allenai/social_i_qa:
    # context/question/answerA/answerB/answerC/label[1-based str]); the original
    # social_i_qa / allenai/social_i_qa are loading-script datasets that newer
    # `datasets` versions reject ("Dataset scripts are no longer supported").
    print(f"[LOAD] lighteval/siqa :: {args.split}")
    ds = load_dataset("lighteval/siqa", split=args.split, cache_dir=args.cache_dir)
    n = len(ds)
    print(f"[INFO] {args.split}: {n} rows")

    merged: List[Dict[str, Any]] = []
    skipped = 0
    for i in range(n):
        if args.limit and len(merged) >= args.limit:
            break
        item = row_to_item(ds[i])
        if item is None:
            skipped += 1
            continue
        merged.append(item)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"data": merged}, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved SocialIQA ({args.split}) → {args.out}")
    print(f"[INFO] kept {len(merged)}, skipped {skipped}")
    print("[Preview top-2]")
    print(json.dumps(merged[:2], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
