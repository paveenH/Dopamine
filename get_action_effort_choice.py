#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effort-based Task Choice experiment with RSN steering.

Round 1 only: Present paired GSM8K (easy) + MATH (hard) questions.
              Collect P(A) / P(B) via logit extraction.

Three steering conditions: alpha=0 / +4 / -4 (controlled by --configs).
"""

import os
import gc
import json
import csv
import argparse

import numpy as np
import torch
from tqdm import tqdm

from llms import VicundaModel
from template import build_effort_choice_suite
import utils


# ──────────────────────────────────────────────
# Round 1: choice logit extraction
# ──────────────────────────────────────────────

def run_round1(vc, pairs, diff_mtx, choice_templates):
    """
    For each (gsm8k, math) pair, compute P(A) and P(B) from logits.
    Returns list of dicts with choice probabilities.
    """
    LABELS = choice_templates["labels"]  # ["A", "B"]
    opt_ids = utils.option_token_ids(vc, LABELS)
    results = []
    bs = args.batch_size

    for i in tqdm(range(0, len(pairs), bs), desc="round1-choice"):
        batch = pairs[i: i + bs]
        prompts = [
            choice_templates["neutral"].format(
                context_a=gsm["question"], context_b=math["question"]
            )
            for gsm, math in batch
        ]

        if diff_mtx is None:
            logits = vc.get_logits(prompts, return_hidden=False)
            raws = logits[:, -1].detach().cpu().float().numpy()
        else:
            raws = vc.regenerate_logits(prompts, diff_mtx, tail_len=args.tail_len)

        for j, (gsm_sample, math_sample) in enumerate(batch):
            raw = raws[j]
            opt_logits = np.array([raw[k] for k in opt_ids], dtype=float)
            opt_logits -= opt_logits.max()
            probs = np.exp(opt_logits)
            probs /= probs.sum()

            choice_idx = int(np.argmax(opt_logits))
            choice = LABELS[choice_idx]

            results.append({
                "gsm8k_question": gsm_sample["question"],
                "math_question": math_sample["question"],
                "gsm8k_answer": gsm_sample["answer"],
                "math_answer": math_sample["answer"],
                "prob_A": float(probs[0]),
                "prob_B": float(probs[1]),
                "choice": choice,
            })

    return results


# ──────────────────────────────────────────────
# Summary stats
# ──────────────────────────────────────────────

def summarize(results, alpha, st, en):
    n = len(results)
    n_A = sum(1 for r in results if r["choice"] == "A")
    n_B = n - n_A
    rate_B = n_B / n if n > 0 else 0.0
    mean_prob_B = float(np.mean([r["prob_B"] for r in results]))

    print(f"  n={n}  choice_B_rate={rate_B:.3f}  mean_P(B)={mean_prob_B:.3f}")
    print(f"  n_A={n_A}  n_B={n_B}")

    return {
        "model": args.model,
        "size": args.size,
        "alpha": alpha,
        "start": st,
        "end": en,
        "n_total": n,
        "n_A": n_A,
        "n_B": n_B,
        "choice_B_rate": round(rate_B, 6),
        "mean_prob_B": round(mean_prob_B, 6),
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("Configs:", ALPHAS_START_END_PAIRS)

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    gsm8k_samples = utils.load_json(GSM8K_FILE)
    math_samples = utils.load_json(MATH_FILE)

    # Pair samples by index (both are seed=42 random samples of 300)
    n = min(len(gsm8k_samples), len(math_samples), 300)
    gsm8k_samples = gsm8k_samples[:n]
    math_samples = math_samples[:n]
    pairs = list(zip(gsm8k_samples, math_samples))
    print(f"Paired {n} GSM8K + MATH samples.")

    choice_templates = build_effort_choice_suite()

    all_csv_rows = []

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        print(f"\n=== alpha={alpha} | layers={st}-{en} ===")

        if alpha == 0:
            diff_mtx = None
        else:
            mask_suffix = "_abs" if args.abs else ""
            mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
            mask_path = os.path.join(MASK_DIR, mask_name)
            diff_mtx = np.load(mask_path) * alpha

        with torch.no_grad():
            results = run_round1(vc, pairs, diff_mtx, choice_templates)

        csv_row = summarize(results, alpha, st, en)
        all_csv_rows.append(csv_row)

        # Save per-config JSON
        alpha_tag = f"neg{abs(alpha)}" if alpha < 0 else str(alpha)
        out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha_tag}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"effort_choice_{args.size}_{st}_{en}.json")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump(results, fw, ensure_ascii=False, indent=2)
        print(f"[Saved JSON] {out_path}")

        del results
        gc.collect()
        torch.cuda.empty_cache()

    # Save CSV summary
    fieldnames = [
        "model", "size", "alpha", "start", "end",
        "n_total", "n_A", "n_B", "choice_B_rate", "mean_prob_B",
    ]
    csv_path = os.path.join(SAVE_ROOT, f"summary_effort_choice_{args.model}_{args.size}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"\n[Saved CSV] {csv_path}")
    print("All configs finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Effort-based Task Choice with RSN steering")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--hs", type=str, required=True)
    parser.add_argument("--size", type=str, required=True)
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["0-11-20", "4-11-20", "neg4-11-20"])
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--gsm8k_file", type=str, default="benchmark/gsm8k_test_sample.json")
    parser.add_argument("--math_file", type=str, default="benchmark/math_test_sample.json")
    parser.add_argument("--ans_file", type=str, default="answer_effort_choice")
    parser.add_argument("--tail_len", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir", type=str, default=None)
    args = parser.parse_args()

    print("Model:", args.model)
    print("Model dir:", args.model_dir)

    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/RolePlaying/components"

    GSM8K_FILE = os.path.join(BASE, args.gsm8k_file)
    MATH_FILE = os.path.join(BASE, args.math_file)
    MASK_DIR = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()
