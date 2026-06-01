#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MATH open-ended generation evaluation with RSN steering.

Three conditions in one run: alpha=0 (baseline), alpha=+4, alpha=-4.
- alpha=0: vc.generate() (no steering)
- alpha!=0: vc.regenerate() with diff_matrices

Answer extraction: \\boxed{...} → last number → last non-empty line.
Ground truth is the full solution text; we extract \\boxed{} from it for matching.
"""

import os
import re
import gc
import csv
import json
import copy
import argparse

import numpy as np
import torch
from tqdm import tqdm

from llms import VicundaModel
from template import build_math_suite
import utils


# Answer extraction / correctness imported from utils (canonical, shared across scripts).
from utils import extract_boxed, extract_math_answer, is_correct_math


# ──────────────────────────────────────────────
# Per-config runner
# ──────────────────────────────────────────────

def run_math(vc, samples, diff_mtx, template, alpha, st, en, role):
    # GSM8K/MATH have no E-option → roles must NOT carry "honest" framing.
    # neutral → neutral template; any other role → neg ("Now you are {char}.").
    # Mirrors get_answer_regenerate_gsm8k.py routing so prompts align.
    if role in ("neutral", "norole"):
        prompts = [template["neutral"].format(context=s["question"]) for s in samples]
    else:
        character = utils.ROLE_TO_CHARACTER[role]
        prompts = [template["neg"].format(character=character, context=s["question"]) for s in samples]
    print(f"  [{role}] Generating {len(prompts)} samples (batch_size={args.batch_size})...")

    # Always regenerate (diff_mtx = mask×alpha; alpha=0 is a no-op mask×0).
    # Single generation path keeps α=0 baseline comparable to ±4 steering.
    generated_texts = vc.regenerate(
        prompts,
        diff_matrices=diff_mtx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )

    correct_count = 0
    results = []
    for s, generated in zip(samples, generated_texts):
        gold_boxed = extract_boxed(s["answer"])
        pred = extract_math_answer(generated)
        ok = is_correct_math(pred, gold_boxed) if gold_boxed else False
        correct_count += int(ok)
        results.append({
            "question": s["question"],
            "gold_solution": s["answer"],
            "gold_answer": gold_boxed,
            "level": s.get("level", ""),
            "type": s.get("type", ""),
            "generated": generated,
            "pred_answer": pred,
            "correct": ok,
        })

    acc = correct_count / len(results) * 100 if results else 0.0
    print(f"  alpha={alpha} | role={role} | acc={acc:.2f}% ({correct_count}/{len(results)})")
    return results, acc


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("Configs:", ALPHAS_START_END_PAIRS)

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    all_samples = utils.load_json(DATA_DIR)
    n = min(len(all_samples), args.n_samples)
    all_samples = all_samples[:n]
    print(f"Loaded {n} MATH samples.")

    template = build_math_suite(cot=args.cot)

    # Roles to evaluate. Default = neutral only (back-compat). Pass --roles
    # "neutral,expert,non_expert,math_expert" for the role comparison.
    if args.roles:
        roles = [r.strip() for r in args.roles.split(",")]
    else:
        roles = ["neutral"]
    print("Roles:", roles)

    csv_rows = []

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        print(f"\n=== alpha={alpha} | layers={st}-{en} ===")

        # Always load the mask and scale by alpha (alpha=0 → mask×0 = no-op),
        # so α=0 baseline and ±4 steering go through the SAME generation path
        # (vc.regenerate). Mirrors get_answer_regenerate_gsm8k.py — avoids the
        # generate-vs-regenerate ~2% padding gap noted in CLAUDE.md, keeping
        # baseline and steering strictly comparable.
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, mask_name)
        diff_mtx = np.load(mask_path) * alpha

        for role in roles:
            samples = copy.deepcopy(all_samples)
            with torch.no_grad():
                results, acc = run_math(vc, samples, diff_mtx, template, alpha, st, en, role)

            # Save JSON. Output path encodes role so roles don't overwrite each
            # other; neutral keeps the legacy name for back-compat.
            alpha_tag = f"neg{abs(alpha)}" if alpha < 0 else str(alpha)
            role_tag = "" if role in ("neutral", "norole") else f"_{role}"
            out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha_tag}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"math_{args.size}_{st}_{en}{role_tag}.json")
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump({"data": results, "accuracy": round(acc, 4), "role": role},
                          fw, ensure_ascii=False, indent=2)
            print(f"[Saved JSON] {out_path}")

            csv_rows.append({
                "model": args.model,
                "size": args.size,
                "alpha": alpha,
                "start": st,
                "end": en,
                "role": role,
                "n_total": len(results),
                "correct": sum(1 for r in results if r["correct"]),
                "accuracy_percentage": round(acc, 4),
            })

            del results, samples
            gc.collect()
            torch.cuda.empty_cache()

    # Save CSV summary
    csv_path = os.path.join(SAVE_ROOT, f"summary_math_{args.model}_{args.size}.csv")
    fieldnames = ["model", "size", "alpha", "start", "end", "role", "n_total", "correct", "accuracy_percentage"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[Saved CSV] {csv_path}")
    print("All configs finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MATH generation with RSN steering")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--hs", type=str, required=True)
    parser.add_argument("--size", type=str, required=True)
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["0-11-20", "4-11-20", "neg4-11-20"])
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--test_file", type=str, default="benchmark/math_test_sample.json")
    parser.add_argument("--ans_file", type=str, default="answer_math")
    parser.add_argument("--roles", type=str, default=None,
                        help="Comma-separated roles, e.g. 'neutral,expert,non_expert,math_expert'. "
                             "neutral → neutral template; others → neg ('Now you are {char}.'). "
                             "Default: neutral only.")
    parser.add_argument("--cot", action="store_true",
                        help="Enable CoT ('Let's think step by step.'). Default OFF "
                             "(No-CoT is the main line, matching GSM8K convention).")
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir", type=str, default=None)
    args = parser.parse_args()

    print("Model:", args.model)
    print("Model dir:", args.model_dir)

    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/Dopamine/components"

    DATA_DIR = os.path.join(BASE, args.test_file)
    MASK_DIR = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()
