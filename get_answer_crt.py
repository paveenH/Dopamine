#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRT (Cognitive Reflection Test) experiment with RSN steering.

For each question, compute joint P(S1) and joint P(S2) via teacher forcing.
Steering is applied only at the last prompt token; answer tokens are unaffected.

Three conditions: alpha=0 (baseline), alpha=+4, alpha=-4.
"""

import os
import gc
import json
import csv
import math
import argparse

import numpy as np
import torch
from tqdm import tqdm

from llms import VicundaModel
import utils


PROMPT_TEMPLATE = (
    "Answer the following question with a number or word only.\n"
    "Question: {question}\n"
    "Answer:"
)


def tokenize_answer(tokenizer, answer: str) -> list[int]:
    """Tokenize an answer string (no special tokens, with a leading space)."""
    token_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    return token_ids


def joint_log_prob(logits_per_token: np.ndarray, token_ids: list[int]) -> float:
    """
    Compute sum of log P(token_k | context) for each answer token.
    logits_per_token: (n_tokens, V)
    token_ids: list of length n_tokens
    """
    total = 0.0
    for k, tid in enumerate(token_ids):
        logit_row = logits_per_token[k].astype(np.float64)
        logit_row -= logit_row.max()
        log_softmax = logit_row - math.log(np.exp(logit_row).sum())
        total += log_softmax[tid]
    return total


def run_crt(vc, samples, diff_mtx):
    prompts = [PROMPT_TEMPLATE.format(question=s["question"]) for s in samples]

    s1_ids_list = [tokenize_answer(vc.tokenizer, s["s1_answer"]) for s in samples]
    s2_ids_list = [tokenize_answer(vc.tokenizer, s["s2_answer"]) for s in samples]

    # Teacher-forcing: one pass per answer candidate
    with torch.no_grad():
        logits_s1 = vc.regenerate_logits_teacher_forcing(prompts, s1_ids_list, diff_mtx)
        logits_s2 = vc.regenerate_logits_teacher_forcing(prompts, s2_ids_list, diff_mtx)

    results = []
    for i, s in enumerate(samples):
        lp_s1 = joint_log_prob(logits_s1[i], s1_ids_list[i])
        lp_s2 = joint_log_prob(logits_s2[i], s2_ids_list[i])

        # Softmax over {S1, S2} to get normalised probabilities
        lp_max = max(lp_s1, lp_s2)
        exp_s1 = math.exp(lp_s1 - lp_max)
        exp_s2 = math.exp(lp_s2 - lp_max)
        denom = exp_s1 + exp_s2
        prob_s1 = exp_s1 / denom
        prob_s2 = exp_s2 / denom

        choice = "S2" if lp_s2 > lp_s1 else "S1"

        results.append({
            "id": s["id"],
            "question": s["question"],
            "s1_answer": s["s1_answer"],
            "s2_answer": s["s2_answer"],
            "s1_token_ids": s1_ids_list[i],
            "s2_token_ids": s2_ids_list[i],
            "log_prob_s1": round(lp_s1, 6),
            "log_prob_s2": round(lp_s2, 6),
            "prob_s1": round(prob_s1, 6),
            "prob_s2": round(prob_s2, 6),
            "choice": choice,
            "source": s.get("source", ""),
        })

    n_s2 = sum(1 for r in results if r["choice"] == "S2")
    mean_prob_s2 = float(np.mean([r["prob_s2"] for r in results]))
    print(f"  n={len(results)}  choice_S2_rate={n_s2/len(results):.3f}  mean_P(S2)={mean_prob_s2:.3f}")
    return results


def summarize(results, alpha, st, en):
    n = len(results)
    n_s2 = sum(1 for r in results if r["choice"] == "S2")
    n_s1 = n - n_s2
    rate_s2 = n_s2 / n if n > 0 else 0.0
    mean_prob_s2 = float(np.mean([r["prob_s2"] for r in results]))
    mean_prob_s1 = float(np.mean([r["prob_s1"] for r in results]))
    return {
        "model": args.model,
        "size": args.size,
        "alpha": alpha,
        "start": st,
        "end": en,
        "n_total": n,
        "n_s1": n_s1,
        "n_s2": n_s2,
        "choice_s2_rate": round(rate_s2, 6),
        "mean_prob_s1": round(mean_prob_s1, 6),
        "mean_prob_s2": round(mean_prob_s2, 6),
    }


def main():
    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("Configs:", ALPHAS_START_END_PAIRS)

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    samples = utils.load_json(CRT_FILE)
    print(f"Loaded {len(samples)} CRT questions.")
    for s in samples:
        print(f"  [{s['id']}] S1={s['s1_answer']!r}  S2={s['s2_answer']!r}")

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

        results = run_crt(vc, samples, diff_mtx)

        csv_row = summarize(results, alpha, st, en)
        all_csv_rows.append(csv_row)

        alpha_tag = f"neg{abs(alpha)}" if alpha < 0 else str(alpha)
        out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha_tag}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"crt_{args.size}_{st}_{en}.json")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump({"data": results, "summary": csv_row}, fw, ensure_ascii=False, indent=2)
        print(f"[Saved JSON] {out_path}")

        gc.collect()
        torch.cuda.empty_cache()

    fieldnames = [
        "model", "size", "alpha", "start", "end",
        "n_total", "n_s1", "n_s2", "choice_s2_rate", "mean_prob_s1", "mean_prob_s2",
    ]
    csv_path = os.path.join(SAVE_ROOT, f"summary_crt_{args.model}_{args.size}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"\n[Saved CSV] {csv_path}")
    print("All configs finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRT experiment with RSN steering")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--hs", type=str, required=True)
    parser.add_argument("--size", type=str, required=True)
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["0-11-20", "4-11-20", "neg4-11-20"])
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--crt_file", type=str, default="benchmark/crt_questions.json")
    parser.add_argument("--ans_file", type=str, default="answer_crt")
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir", type=str, default=None)
    args = parser.parse_args()

    print("Model:", args.model)
    print("Model dir:", args.model_dir)

    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/RolePlaying/components"

    CRT_FILE = os.path.join(BASE, args.crt_file)
    MASK_DIR = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()
