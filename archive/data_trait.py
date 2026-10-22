#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download mirlab/TRAIT from HuggingFace and export to pipeline-compatible JSON.

Output: {save_dir}/trait_test.json
Format: [{"task": "BFI_Extraversion", "text": "...", "label": -1, "num_options": 4}, ...]

Dataset structure (confirmed from schema check):
  - splits: one per trait full name (e.g. "Openness", "Extraversion", ...)
  - fields: personality, question, response_high1, response_high2, response_low1, response_low2
  - no 'situation' field — 'question' contains full scenario + question

Option order (fixed by TRAIT paper, Section 3.4):
  A = response_high1, B = response_low1, C = response_high2, D = response_low2
  → option_scores = [1, 0, 1, 0] (hardcoded in get_answer_trait.py)
"""

import os
import json
import argparse
import datasets

# Full trait name → instrument + task name
TRAIT_SPLITS = [
    "Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism",
    "Machiavellianism", "Narcissism", "Psychopathy",
]

BFI_TRAITS = {"Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"}
SD3_TRAITS = {"Machiavellianism", "Narcissism", "Psychopathy"}


def export(cache_dir: str, save_dir: str, token: str = None):
    mcq_list = []

    for trait_name in TRAIT_SPLITS:
        instrument = "BFI" if trait_name in BFI_TRAITS else "SD3"
        task = f"{instrument}_{trait_name}"

        print(f"Loading split '{trait_name}' ...")
        ds = datasets.load_dataset(
            "mirlab/TRAIT",
            split=trait_name,
            cache_dir=cache_dir,
            token=token,
        )
        print(f"  {len(ds)} samples")

        for row in ds:
            question = row.get("question", "").strip()
            h1 = row.get("response_high1", "").strip()
            l1 = row.get("response_low1", "").strip()
            h2 = row.get("response_high2", "").strip()
            l2 = row.get("response_low2", "").strip()

            # A=High1, B=Low1, C=High2, D=Low2
            text = f"{question}\nA) {h1}\nB) {l1}\nC) {h2}\nD) {l2}\n"

            mcq_list.append({
                "task": task,
                "text": text,
                "label": -1,
                "num_options": 4,
            })

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "trait_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mcq_list, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(mcq_list)} samples to: {out_path}")

    from collections import Counter
    counts = Counter(s["task"] for s in mcq_list)
    print("\nTask distribution:")
    for t, n in sorted(counts.items()):
        print(f"  {t:<35} {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export mirlab/TRAIT to pipeline JSON.")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (falls back to HF_TOKEN env var)")
    args = parser.parse_args()

    import os as _os
    token = args.token or _os.environ.get("HF_TOKEN")
    export(cache_dir=args.cache_dir, save_dir=args.save_dir, token=token)
