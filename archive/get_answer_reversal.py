#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probabilistic Reversal Learning benchmark with RSN diff injection.

Design:
  - 40-round binary choice (A vs B), noisy feedback (80/20)
  - Phase 1 (rounds 1-phase_switch): first_correct is correct
  - Phase 2 (rounds phase_switch+1-end): second_correct is correct
  - Counterbalancing: even runs use A→B, odd runs use B→A
  - Limited history window (default 5) to emphasize recent feedback
  - All optimal-rate metrics are based on choosing the correct option

Metrics:
  - win_stay_rate:       P(same choice | prev feedback = Correct)
  - lose_shift_rate:     P(switch choice | prev feedback = Incorrect)
  - phase1_optimal_rate: proportion of Phase 1 rounds choosing correct option
  - phase2_early_optimal: proportion of Phase 2 first 10 rounds choosing correct (reversal speed proxy)
  - phase2_optimal_rate:  proportion of Phase 2 last 10 rounds choosing correct

Usage:
  python get_answer_reversal.py \
    --model llama3 --model_dir meta-llama/Llama-3.1-8B-Instruct \
    --hs llama3 --size 8B --type non \
    --percentage 0.5 --mask_type nmd \
    --configs 0-11-20 4-11-20 neg4-11-20 \
    --num_runs 30 --num_rounds 40 --phase_switch 20 \
    --reward_prob 0.8 --history_window 5 --data data1
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

LABELS = ["A", "B"]


def build_prompt(history: list[tuple[str, str]], history_window: int) -> str:
    lines = [
        "You are making repeated choices between two options: A and B.",
        "At each round you will see your recent choice history and must pick A or B.",
        "Adjust your strategy based on the feedback you receive.",
        "Reply with ONLY the letter A or B — nothing else.",
        "",
    ]
    windowed = history[-history_window:] if history_window > 0 else history
    if windowed:
        start_idx = len(history) - len(windowed) + 1
        lines.append("Recent rounds:")
        for i, (choice, feedback) in enumerate(windowed, start_idx):
            lines.append(f"  Round {i}: You chose {choice}, feedback: {feedback}")
        lines.append("")
    next_round = len(history) + 1
    lines.append(f"Round {next_round}: Which option do you choose? Reply with A or B.")
    return "\n".join(lines)


def get_choice(vc: VicundaModel, prompt: str, diff_mtx, token_ids: list[int]) -> tuple[str, np.ndarray]:
    """Return chosen label and softmax probs over [A, B]."""
    logits = vc.regenerate_logits([prompt], diff_matrices=diff_mtx)  # (1, V)
    ab_logits = logits[0, token_ids]  # (2,)
    probs = utils.softmax_1d(ab_logits)
    choice = LABELS[int(np.argmax(probs))]
    return choice, probs


def get_feedback(choice: str, correct: str, reward_prob: float, rng: random.Random) -> str:
    """Generate noisy feedback: correct with reward_prob if choice==correct."""
    if choice == correct:
        return "Correct" if rng.random() < reward_prob else "Incorrect"
    else:
        return "Incorrect" if rng.random() < reward_prob else "Correct"


def run_episode(
    vc: VicundaModel,
    diff_mtx,
    token_ids: list[int],
    num_rounds: int,
    phase_switch: int,
    reward_prob: float,
    seed: int,
    history_window: int,
    counterbalance: bool,
) -> dict:
    rng = random.Random(seed)

    # Counterbalancing: even seeds → A-first, odd seeds → B-first
    if counterbalance and (seed % 2 == 1):
        phase1_correct, phase2_correct = "B", "A"
    else:
        phase1_correct, phase2_correct = "A", "B"

    history = []
    choices = []
    feedbacks = []
    corrects = []
    probs_log = []

    for round_idx in range(num_rounds):
        correct = phase1_correct if round_idx < phase_switch else phase2_correct
        corrects.append(correct)

        prompt = build_prompt(history, history_window)
        choice, probs = get_choice(vc, prompt, diff_mtx, token_ids)
        feedback = get_feedback(choice, correct, reward_prob, rng)

        choices.append(choice)
        feedbacks.append(feedback)
        probs_log.append(probs.tolist())
        history.append((choice, feedback))

    # --- Compute metrics ---
    win_stay, lose_shift = [], []
    for i in range(1, num_rounds):
        prev_fb = feedbacks[i - 1]
        stayed = choices[i] == choices[i - 1]
        if prev_fb == "Correct":
            win_stay.append(1 if stayed else 0)
        else:
            lose_shift.append(1 if not stayed else 0)

    # Phase 1 optimal rate (chose correct option in phase 1)
    phase1_optimal = sum(
        1 for i in range(phase_switch) if choices[i] == corrects[i]
    ) / phase_switch

    # Phase 2 early optimal rate (first 10 rounds of phase 2) — reversal speed proxy
    phase2_early_end = min(phase_switch + 10, num_rounds)
    phase2_early = [
        1 for i in range(phase_switch, phase2_early_end) if choices[i] == corrects[i]
    ]
    phase2_early_optimal = sum(phase2_early) / (phase2_early_end - phase_switch)

    # Phase 2 late optimal rate (last 10 rounds)
    phase2_tail_start = num_rounds - 10
    phase2_optimal = sum(
        1 for i in range(phase2_tail_start, num_rounds) if choices[i] == corrects[i]
    ) / 10

    return {
        "phase1_correct": phase1_correct,
        "phase2_correct": phase2_correct,
        "choices": choices,
        "feedbacks": feedbacks,
        "probs": probs_log,
        "win_stay_rate":         float(np.mean(win_stay)) if win_stay else 0.0,
        "lose_shift_rate":       float(np.mean(lose_shift)) if lose_shift else 0.0,
        "phase1_optimal_rate":   phase1_optimal,
        "phase2_early_optimal":  phase2_early_optimal,
        "phase2_optimal_rate":   phase2_optimal,
    }


def main():
    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("Configs:", ALPHAS_START_END_PAIRS)

    FIELDNAMES = [
        "model", "size", "alpha", "start", "end", "TOP",
        "num_runs", "num_rounds", "phase_switch", "reward_prob", "history_window",
        "mean_win_stay", "mean_lose_shift",
        "mean_phase1_optimal", "mean_phase2_early_optimal", "mean_phase2_optimal",
        "std_win_stay", "std_lose_shift",
        "std_phase1_optimal", "std_phase2_early_optimal", "std_phase2_optimal",
    ]

    os.makedirs(SAVE_ROOT, exist_ok=True)
    csv_path = os.path.join(SAVE_ROOT, f"summary_{args.model}_{args.size}.csv")

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

    token_ids = utils.option_token_ids(vc, LABELS)
    print(f"Token IDs: A={token_ids[0]}, B={token_ids[1]}")

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        if float(alpha) in done_keys:
            print(f"[Skip] α={alpha} already done.")
            continue

        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, mask_name)
        raw_mask = np.load(mask_path)
        diff_mtx = list(raw_mask * alpha)
        TOP = max(1, int(args.percentage / 100 * raw_mask.shape[1]))
        print(f"\n=== α={alpha} | layers={st}-{en} | TOP={TOP} | window={args.history_window} ===")

        run_results = []
        for run_idx in range(args.num_runs):
            cb_tag = "B→A" if (args.counterbalance and run_idx % 2 == 1) else "A→B"
            print(f"  Run {run_idx + 1}/{args.num_runs} [{cb_tag}]", end=" ... ", flush=True)
            with torch.no_grad():
                result = run_episode(
                    vc=vc,
                    diff_mtx=diff_mtx,
                    token_ids=token_ids,
                    num_rounds=args.num_rounds,
                    phase_switch=args.phase_switch,
                    reward_prob=args.reward_prob,
                    seed=run_idx,
                    history_window=args.history_window,
                    counterbalance=args.counterbalance,
                )
            run_results.append(result)
            print(
                f"win_stay={result['win_stay_rate']:.2f}  "
                f"lose_shift={result['lose_shift_rate']:.2f}  "
                f"p1_opt={result['phase1_optimal_rate']:.2f}  "
                f"p2_early={result['phase2_early_optimal']:.2f}  "
                f"p2_opt={result['phase2_optimal_rate']:.2f}"
            )
            gc.collect()
            torch.cuda.empty_cache()

        # Aggregate
        ws   = [r["win_stay_rate"] for r in run_results]
        ls   = [r["lose_shift_rate"] for r in run_results]
        p1   = [r["phase1_optimal_rate"] for r in run_results]
        p2e  = [r["phase2_early_optimal"] for r in run_results]
        p2   = [r["phase2_optimal_rate"] for r in run_results]

        row = {
            "model": args.model,
            "size": args.size,
            "alpha": alpha,
            "start": st,
            "end": en,
            "TOP": TOP,
            "num_runs": args.num_runs,
            "num_rounds": args.num_rounds,
            "phase_switch": args.phase_switch,
            "reward_prob": args.reward_prob,
            "history_window": args.history_window,
            "mean_win_stay":              round(float(np.mean(ws)),   4),
            "mean_lose_shift":            round(float(np.mean(ls)),   4),
            "mean_phase1_optimal":        round(float(np.mean(p1)),   4),
            "mean_phase2_early_optimal":  round(float(np.mean(p2e)),  4),
            "mean_phase2_optimal":        round(float(np.mean(p2)),   4),
            "std_win_stay":               round(float(np.std(ws)),    4),
            "std_lose_shift":             round(float(np.std(ls)),    4),
            "std_phase1_optimal":         round(float(np.std(p1)),    4),
            "std_phase2_early_optimal":   round(float(np.std(p2e)),   4),
            "std_phase2_optimal":         round(float(np.std(p2)),    4),
        }

        writer.writerow(row)
        csv_file.flush()

        # Save per-run details
        out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
        os.makedirs(out_dir, exist_ok=True)
        detail_path = os.path.join(out_dir, f"reversal_{args.size}_{TOP}_{st}_{en}.json")
        with open(detail_path, "w", encoding="utf-8") as fw:
            json.dump({"alpha": alpha, "runs": run_results}, fw, indent=2)
        print(f"  → {detail_path}")

        gc.collect()
        torch.cuda.empty_cache()

    csv_file.close()
    print("\n✅ Reversal Learning run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probabilistic Reversal Learning with RSN steering.")
    parser.add_argument("--model",           type=str, default="llama3")
    parser.add_argument("--model_dir",       type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hs",              type=str, default="llama3")
    parser.add_argument("--size",            type=str, default="8B")
    parser.add_argument("--type",            type=str, default="non")
    parser.add_argument("--percentage",      type=float, default=0.5)
    parser.add_argument("--mask_type",       type=str, default="nmd")
    parser.add_argument("--abs",             action="store_true")
    parser.add_argument("--configs",         nargs="+", default=["0-11-20", "4-11-20", "neg4-11-20"])
    parser.add_argument("--num_runs",        type=int, default=30)
    parser.add_argument("--num_rounds",      type=int, default=40)
    parser.add_argument("--phase_switch",    type=int, default=20)
    parser.add_argument("--reward_prob",     type=float, default=0.8)
    parser.add_argument("--history_window",  type=int, default=5,
                        help="Number of recent rounds to show (0 = full history)")
    parser.add_argument("--counterbalance",  action="store_true", default=True,
                        help="Alternate A→B and B→A across runs to cancel position bias")
    parser.add_argument("--no_counterbalance", dest="counterbalance", action="store_false")
    parser.add_argument("--ans_file",        type=str, default="answer_reversal")
    parser.add_argument("--data",            type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir",        type=str, default=None)

    args = parser.parse_args()

    print("Model:", args.model)
    print("Model dir:", args.model_dir)
    print(f"History window: {args.history_window}, Counterbalance: {args.counterbalance}")

    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/RolePlaying/components"

    MASK_DIR  = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()
