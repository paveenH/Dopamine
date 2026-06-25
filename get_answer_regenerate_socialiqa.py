#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SocialIQA (3-choice) generation runner with RSN α steering.

Standalone — does NOT import template.py or touch any existing task's loader /
runner, so the established MMLU / GSM8K / betting pipelines are unaffected.

- Reads the single SocialIQA JSON produced by data_socialiqa.py
  ({"data":[{text,label,num_options:3}]}).
- bare-string neutral prompt (no role, no chat template) → matches the
  activation distribution the NMD mask was extracted in (default-bare convention).
- GENERATION (vc.regenerate, prefill-only tail=1) then parse the A/B/C letter,
  rather than single-token logit argmax — per the betting/CGT convention.
- Reuses the standard α-hook: diff = list(nmd_mask * alpha), full-length list
  (one row per decoder layer), exactly like get_answer_cgt_seq.py.

Output per α: {SAVE_ROOT}/mdf_{alpha}/socialiqa_{size}_answers_{st}_{en}.json
"""

import os
import re
import gc
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

from llms import VicundaModel
import utils

# ───────────────────── Prompt (3-choice, bare-string, neutral) ─────────────────────

SIQA_PREFIX = (
    "Would you answer the following question with A, B or C?\n"
)
SIQA_SUFFIX = (
    "\nAnswer with a single letter (A, B, or C).\nAnswer: "
)


def build_prompt(ctx: str) -> str:
    return SIQA_PREFIX + ctx.strip() + SIQA_SUFFIX


# ───────────────────── Parse generated answer → A/B/C ─────────────────────

VALID = ("A", "B", "C")


def parse_letter(raw: str) -> str | None:
    """Extract the chosen letter from generated text.

    Priority:
      1) opening bare letter (the Answer: anchor makes 'A'/'A)'/'A.' the norm)
      2) 'Answer: X' anywhere
      3) first standalone A/B/C token in the text
    Returns None if nothing parseable (counted as invalid).
    """
    if not raw:
        return None
    s = raw.strip()
    # 1) opening letter, optionally followed by ) . : or whitespace/EOS
    m = re.match(r"^\(?\s*([ABC])\b", s)
    if m:
        return m.group(1)
    # 2) explicit Answer: X
    m = re.search(r"(?i)answer\s*[:\-]?\s*\(?\s*([ABC])\b", s)
    if m:
        return m.group(1).upper()
    # 3) first standalone capital A/B/C
    m = re.search(r"\b([ABC])\b", s)
    if m:
        return m.group(1)
    return None


# ───────────────────── One α-cell over the whole dataset ─────────────────────

def run_alpha(vc: VicundaModel, data: list, diff_mtx: list, alpha: float,
              max_new_tokens: int, save_all_raw: bool):
    stats = {"correct": 0, "invalid": 0, "total": 0}
    for sample in tqdm(data, desc=f"α={alpha}"):
        ctx = sample.get("text", "")
        true_idx = sample.get("label", -1)
        if not (0 <= true_idx < 3):
            continue
        true_lab = VALID[true_idx]

        prompt = build_prompt(ctx)
        out = vc.regenerate(
            inputs=[prompt],
            diff_matrices=diff_mtx,
            prefill_only=True,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )[0]

        pred = parse_letter(out)

        sample["answer_neutral"] = pred if pred is not None else "INVALID"
        if save_all_raw:
            sample["raw_neutral"] = out

        stats["total"] += 1
        if pred is None:
            stats["invalid"] += 1
        elif pred == true_lab:
            stats["correct"] += 1

    tot = stats["total"]
    acc = stats["correct"] / tot * 100 if tot else 0.0
    inv = stats["invalid"] / tot * 100 if tot else 0.0
    stats["accuracy_percentage"] = round(acc, 2)
    stats["invalid_percentage"] = round(inv, 2)
    print(f"  α={alpha}: acc={acc:.2f}%  invalid={inv:.2f}%  "
          f"(correct {stats['correct']}/{tot})")
    return stats


# ───────────────────── Main ─────────────────────

def main():
    ALPHAS = utils.parse_configs(args.configs)
    print("Configs:", ALPHAS)

    data_path = SIQA_PATH
    blob = utils.load_json(data_path)
    base_data = blob["data"] if isinstance(blob, dict) and "data" in blob else blob
    print(f"[INFO] loaded {len(base_data)} SocialIQA items from {data_path}")

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    for alpha, (st, en) in ALPHAS:
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}.npy"
        raw_mask = np.load(os.path.join(MASK_DIR, mask_name))
        diff_mtx = list(raw_mask * alpha)   # full-length: one row per decoder layer
        print(f"\n=== α={alpha} | layers={st}-{en} ===")

        # fresh copy so per-α answer fields don't collide
        data = json.loads(json.dumps(base_data))

        with torch.no_grad():
            accuracy = run_alpha(vc, data, diff_mtx, alpha,
                                 args.max_new_tokens, args.save_all_raw)

        out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"socialiqa_{args.size}_answers_{st}_{en}.json")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump({"data": data, "accuracy": {"neutral": accuracy},
                       "prompt_template": SIQA_PREFIX + "{ctx}" + SIQA_SUFFIX},
                      fw, ensure_ascii=False, indent=2)
        print("Saved →", out_path)

        del data, accuracy
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll α finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SocialIQA 3-choice generation runner with RSN α steering.")
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--model_dir", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hs", type=str, default="llama3")
    parser.add_argument("--size", type=str, default="8B")
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["0-11-20", "4-11-20", "neg4-11-20"],
                        help="alpha-start-end triplets, e.g. 4-11-20")
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--ans_file", type=str, default="answer_socialiqa")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--save_all_raw", action="store_true",
                        help="store the raw generated text per sample for offline diagnosis")
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir", type=str, default=None)
    args = parser.parse_args()

    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/Dopamine/components"

    MASK_DIR = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SIQA_PATH = os.path.join(BASE, "benchmark", "socialiqa.json")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    main()
