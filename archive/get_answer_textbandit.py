#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextBandit replication with RSN diff injection.
Follows TextBandit (ACL EthicalLLMs 2025) design (baseline.py source):
  - K arms named "Slot machine 1..K" (fixed, no shuffle)
  - Fixed reward probs per K configuration (same as paper)
  - History feedback: "Slot Machine N won" / "Slot Machine N lost"
  - Few-shot examples in prompt (K examples, one per arm as best)
  - Full history in prompt each round
  - Single-shot numeric output; invalid → random arm
  - T=25 rounds per episode (TextBandit default)
  - RSN diff injection via vc.regenerate (same as get_answer_bandit.py)

K=5 reward probs (TextBandit): [0.20, 0.75, 0.35, 0.25, 0.55]
  best arm = Slot machine 2 (0.75), worst = Slot machine 1 (0.20)
  Note: paper source code sets best_machine=3 for K=5 (paper bug);
        we use the true best arm (machine 2, prob=0.75).

Metrics (same as get_answer_bandit.py):
  - opt_frac        proportion of rounds choosing the best arm
  - explore_rate    1 − opt_frac
  - worst_frac      proportion choosing the worst arm
  - cum_regret      Σ(best_prob − chosen_prob)
  - early_opt_frac  opt_frac for rounds 1–min(12, T)
  - late_opt_frac   opt_frac for rounds max(0,T−12)..T
  - invalid_rate    proportion of rounds with unparseable output

Usage:
  python get_answer_textbandit.py \
    --model llama3 --model_dir meta-llama/Llama-3.1-8B-Instruct \
    --hs llama3 --size 8B --type non \
    --percentage 0.5 --mask_type nmd \
    --configs 0-11-20 4-11-20 neg4-11-20 \
    --num_runs 30 --num_rounds 25 --K 5 \
    --data data1
"""

import os
import gc
import csv
import json
import random
import argparse
import numpy as np
import torch

from llms import VicundaModel
import utils

# ── TextBandit reward probabilities per K (from paper Table 1) ───────────────
TEXTBANDIT_PROBS = {
    2: [0.30, 0.65],
    3: [0.40, 0.30, 0.70],
    4: [0.80, 0.60, 0.35, 0.25],
    5: [0.20, 0.75, 0.35, 0.25, 0.55],
}


def make_arm_names(K: int) -> list[str]:
    return [f"Slot machine {i+1}" for i in range(K)]


# ── Few-shot examples per K (from TextBandit baseline.py) ────────────────────
# Each example shows one machine as the clear winner (3 wins/1 loss pattern).
# History uses "Slot Machine N won/lost" format (matching source code exactly).

FEW_SHOT_EXAMPLES = {
    5: """\
Example 1:
# Analysis: Machine 1 has 3 wins/1 loss (75%). Others are worse. Machine 1 is the best.
History:
Slot Machine 1 won
Slot Machine 2 lost
Slot Machine 3 lost
Slot Machine 4 lost
Slot Machine 5 lost
Slot Machine 1 won
Slot Machine 2 won
Slot Machine 3 lost
Slot Machine 4 won
Slot Machine 5 lost
Slot Machine 1 won
Slot Machine 2 lost
Slot Machine 3 won
Slot Machine 4 lost
Slot Machine 5 won
Your choice (1, 2, 3, 4, or 5): 1

Example 2:
# Analysis: Machine 2 has 3 wins/1 loss (75%). Others are worse. Machine 2 is the best.
History:
Slot Machine 2 won
Slot Machine 1 lost
Slot Machine 3 lost
Slot Machine 4 lost
Slot Machine 5 lost
Slot Machine 2 won
Slot Machine 1 won
Slot Machine 3 lost
Slot Machine 4 won
Slot Machine 5 lost
Slot Machine 2 won
Slot Machine 1 lost
Slot Machine 3 won
Slot Machine 4 lost
Slot Machine 5 won
Your choice (1, 2, 3, 4, or 5): 2

Example 3:
# Analysis: Machine 3 has 3 wins/1 loss (75%). Others are worse. Machine 3 is the best.
History:
Slot Machine 3 won
Slot Machine 1 lost
Slot Machine 2 lost
Slot Machine 4 lost
Slot Machine 5 lost
Slot Machine 3 won
Slot Machine 1 won
Slot Machine 2 lost
Slot Machine 4 won
Slot Machine 5 lost
Slot Machine 3 won
Slot Machine 1 lost
Slot Machine 2 won
Slot Machine 4 lost
Slot Machine 5 won
Your choice (1, 2, 3, 4, or 5): 3

Example 4:
# Analysis: Machine 4 has 3 wins/1 loss (75%). Others are worse. Machine 4 is the best.
History:
Slot Machine 4 won
Slot Machine 1 lost
Slot Machine 2 lost
Slot Machine 3 lost
Slot Machine 5 lost
Slot Machine 4 won
Slot Machine 1 won
Slot Machine 2 lost
Slot Machine 3 won
Slot Machine 5 lost
Slot Machine 4 won
Slot Machine 1 lost
Slot Machine 2 won
Slot Machine 3 lost
Slot Machine 5 won
Your choice (1, 2, 3, 4, or 5): 4

Example 5:
# Analysis: Machine 5 has 3 wins/1 loss (75%). Others are worse. Machine 5 is the best.
History:
Slot Machine 5 won
Slot Machine 1 lost
Slot Machine 2 lost
Slot Machine 3 lost
Slot Machine 4 lost
Slot Machine 5 won
Slot Machine 1 won
Slot Machine 2 lost
Slot Machine 3 won
Slot Machine 4 lost
Slot Machine 5 won
Slot Machine 1 lost
Slot Machine 2 won
Slot Machine 3 lost
Slot Machine 4 won
Your choice (1, 2, 3, 4, or 5): 5""",
}


def build_prompt(arm_names: list[str], history: list[tuple[str, str]]) -> str:
    """
    TextBandit-style prompt matching baseline.py exactly:
      - Role instruction + few-shot examples
      - "Current situation: History:" section
      - history entries: "Slot Machine N won/lost"
      - Query: "Your choice (1, 2, ..., or K):"
    """
    K = len(arm_names)
    choices_str = ", ".join(str(i) for i in range(1, K)) + f", or {K}"
    few_shot = FEW_SHOT_EXAMPLES.get(K, "")

    lines = [
        "You are a decision-making agent. Your task is to choose the slot machine most likely to win.",
        "Based on the history of wins and losses, decide which machine to play next.",
        f"Output ONLY the number 1 or {' or '.join(str(i) for i in range(2, K+1))}. Do not include any other words or formatting.",
        "",
    ]
    if few_shot:
        lines.append(few_shot)
        lines.append("")

    lines.append("Current situation:")
    lines.append("History:")
    for arm, outcome in history:
        lines.append(f"{arm} {outcome}")
    lines.append(f"Your choice ({choices_str}):")
    return "\n".join(lines)


def parse_choice(output: str, arm_names: list[str]) -> tuple[str, bool]:
    """
    Two-stage regex matching (mirrors TextBandit baseline.py):
      1. Start-anchored: look for digit at beginning of output
      2. Fallback: look for digit anywhere in output
    Returns (chosen_arm_name, is_valid). Invalid → uniform random.
    """
    import re
    K = len(arm_names)
    pattern_start    = rf'^\s*([1-{K}])\b'
    pattern_fallback = rf'\b([1-{K}])\b'

    m = re.match(pattern_start, output.strip())
    if m:
        return arm_names[int(m.group(1)) - 1], True
    m = re.search(pattern_fallback, output.strip())
    if m:
        return arm_names[int(m.group(1)) - 1], True
    return random.choice(arm_names), False


def get_feedback_text(won: bool) -> str:
    return "won" if won else "lost"


def run_episode(
    vc: VicundaModel,
    diff_mtx,
    arm_names: list[str],
    reward_probs: list[float],
    num_rounds: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    arm_map = {name: prob for name, prob in zip(arm_names, reward_probs)}
    opt   = max(arm_map, key=arm_map.get)
    worst = min(arm_map, key=arm_map.get)

    history   = []   # list of (arm_name, outcome_str)
    choices   = []
    feedbacks = []
    invalids  = []

    for _ in range(num_rounds):
        prompt = build_prompt(arm_names, history)
        output = vc.regenerate(
            inputs=[prompt],
            diff_matrices=diff_mtx,
            max_new_tokens=10,
            temperature=1.0,
        )
        raw = output[0] if isinstance(output, list) else output
        arm, valid = parse_choice(raw, arm_names)

        won    = rng.random() < arm_map[arm]
        reward = 1 if won else 0
        outcome_str = get_feedback_text(won)   # "won" or "lost"

        choices.append(arm)
        feedbacks.append(reward)
        invalids.append(0 if valid else 1)
        # History format matches few-shot examples: "Slot Machine N won/lost"
        history.append((arm.replace("Slot machine", "Slot Machine"), outcome_str))

    early_end  = min(num_rounds // 2, 12)
    late_start = max(0, num_rounds - 12)

    opt_frac       = sum(1 for c in choices if c == opt) / num_rounds
    explore_rate   = 1.0 - opt_frac
    worst_frac     = sum(1 for c in choices if c == worst) / num_rounds
    cum_regret     = float(sum(arm_map[opt] - arm_map[c] for c in choices))
    early_opt_frac = sum(1 for c in choices[:early_end] if c == opt) / early_end
    late_opt_frac  = sum(1 for c in choices[late_start:] if c == opt) / (num_rounds - late_start)
    invalid_rate   = sum(invalids) / num_rounds

    return {
        "arm_map":        arm_map,
        "best_arm":       opt,
        "worst_arm":      worst,
        "choices":        choices,
        "feedbacks":      feedbacks,
        "invalid_rate":   float(invalid_rate),
        "opt_frac":       float(opt_frac),
        "explore_rate":   float(explore_rate),
        "worst_frac":     float(worst_frac),
        "cum_regret":     float(cum_regret),
        "early_opt_frac": float(early_opt_frac),
        "late_opt_frac":  float(late_opt_frac),
    }


def main():
    K            = args.K
    reward_probs = TEXTBANDIT_PROBS[K]
    arm_names    = make_arm_names(K)
    best_idx     = reward_probs.index(max(reward_probs))

    print(f"K={K}  arm_names={arm_names}")
    print(f"reward_probs={reward_probs}  best={arm_names[best_idx]}")

    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("Configs:", ALPHAS_START_END_PAIRS)

    FIELDNAMES = [
        "model", "size", "K", "alpha", "start", "end", "TOP",
        "num_runs", "num_rounds",
        "mean_opt_frac", "mean_explore_rate", "mean_worst_frac",
        "mean_cum_regret", "mean_early_opt_frac", "mean_late_opt_frac",
        "mean_invalid_rate",
        "std_opt_frac", "std_explore_rate", "std_worst_frac",
        "std_cum_regret", "std_early_opt_frac", "std_late_opt_frac",
        "std_invalid_rate",
    ]

    os.makedirs(SAVE_ROOT, exist_ok=True)
    csv_path = os.path.join(SAVE_ROOT, f"summary_{args.model}_{args.size}_K{K}.csv")

    done_keys = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done_keys.add(float(r["alpha"]))
        print(f"[Resume] {len(done_keys)} alpha configs already done, skipping.")

    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        if float(alpha) in done_keys:
            print(f"[Skip] α={alpha} already done.")
            continue

        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, mask_name)
        raw_mask  = np.load(mask_path)
        diff_mtx  = list(raw_mask * alpha)
        TOP       = max(1, int(args.percentage / 100 * raw_mask.shape[1]))
        print(f"\n=== α={alpha} | layers={st}-{en} | TOP={TOP} ===")

        run_results = []
        for run_idx in range(args.num_runs):
            print(f"  Run {run_idx + 1}/{args.num_runs}", end=" ... ", flush=True)
            with torch.no_grad():
                result = run_episode(
                    vc=vc,
                    diff_mtx=diff_mtx,
                    arm_names=arm_names,
                    reward_probs=reward_probs,
                    num_rounds=args.num_rounds,
                    seed=run_idx,
                )
            run_results.append(result)
            print(
                f"opt={result['opt_frac']:.2f}  "
                f"worst={result['worst_frac']:.2f}  "
                f"regret={result['cum_regret']:.1f}  "
                f"invalid={result['invalid_rate']:.2f}"
            )
            gc.collect()
            torch.cuda.empty_cache()

        opt  = [r["opt_frac"]       for r in run_results]
        expl = [r["explore_rate"]   for r in run_results]
        wf   = [r["worst_frac"]     for r in run_results]
        regr = [r["cum_regret"]     for r in run_results]
        eopt = [r["early_opt_frac"] for r in run_results]
        lopt = [r["late_opt_frac"]  for r in run_results]
        inv  = [r["invalid_rate"]   for r in run_results]

        row = {
            "model": args.model, "size": args.size, "K": K,
            "alpha": alpha, "start": st, "end": en, "TOP": TOP,
            "num_runs": args.num_runs, "num_rounds": args.num_rounds,
            "mean_opt_frac":        round(float(np.mean(opt)),  4),
            "mean_explore_rate":    round(float(np.mean(expl)), 4),
            "mean_worst_frac":      round(float(np.mean(wf)),   4),
            "mean_cum_regret":      round(float(np.mean(regr)), 4),
            "mean_early_opt_frac":  round(float(np.mean(eopt)), 4),
            "mean_late_opt_frac":   round(float(np.mean(lopt)), 4),
            "mean_invalid_rate":    round(float(np.mean(inv)),  4),
            "std_opt_frac":         round(float(np.std(opt)),   4),
            "std_explore_rate":     round(float(np.std(expl)),  4),
            "std_worst_frac":       round(float(np.std(wf)),    4),
            "std_cum_regret":       round(float(np.std(regr)),  4),
            "std_early_opt_frac":   round(float(np.std(eopt)),  4),
            "std_late_opt_frac":    round(float(np.std(lopt)),  4),
            "std_invalid_rate":     round(float(np.std(inv)),   4),
        }

        writer.writerow(row)
        csv_file.flush()

        out_dir     = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
        os.makedirs(out_dir, exist_ok=True)
        detail_path = os.path.join(out_dir, f"textbandit_{args.size}_K{K}_{TOP}_{st}_{en}.json")
        with open(detail_path, "w", encoding="utf-8") as fw:
            json.dump({
                "alpha": alpha, "K": K,
                "reward_probs": reward_probs,
                "arm_names": arm_names,
                "runs": run_results,
            }, fw, indent=2)
        print(f"  → {detail_path}")

        gc.collect()
        torch.cuda.empty_cache()

    csv_file.close()
    print("\n✅ TextBandit run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TextBandit replication with RSN steering.")
    parser.add_argument("--model",      type=str, default="llama3")
    parser.add_argument("--model_dir",  type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hs",         type=str, default="llama3")
    parser.add_argument("--size",       type=str, default="8B")
    parser.add_argument("--type",       type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--mask_type",  type=str, default="nmd")
    parser.add_argument("--abs",        action="store_true")
    parser.add_argument("--configs",    nargs="+", default=["0-11-20", "4-11-20", "neg4-11-20"])
    parser.add_argument("--num_runs",   type=int, default=30)
    parser.add_argument("--num_rounds", type=int, default=25)
    parser.add_argument("--K",          type=int, default=5, choices=[2, 3, 4, 5])
    parser.add_argument("--ans_file",   type=str, default="answer_textbandit_v2")
    parser.add_argument("--data",       type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir",   type=str, default=None)

    args = parser.parse_args()

    print("Model:", args.model)
    print("Model dir:", args.model_dir)
    print(f"K={args.K}  Rounds={args.num_rounds}  Runs={args.num_runs}")

    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/RolePlaying/components"

    MASK_DIR  = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()
