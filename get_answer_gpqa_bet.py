#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPQA Confidence Betting experiment under RSN steering.

Each question is presented with a betting prompt (bet 0/2/5/10).
Measures incentive salience: high wanting -> higher bets.
Running score is fixed at 0 per question to avoid cross-question interference.

Conditions: orig (no steering) / steered (±alpha via regenerate)
Tasks: GPQA main + diamond only (micro accuracy)
"""

import os
import re
import gc
import csv
import json
import copy
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from llms import VicundaModel
import utils

# ───────────────────── Prompt ─────────────────────

PROMPT_TEMPLATE = """\
You are playing a quiz game. You have a running score that starts at 0.
For each question, you must decide how many points to bet (0, 2, 5, or 10).
  - If your answer is correct: score += bet
  - If your answer is wrong:   score -= bet
  - If you bet 0:              score unchanged (skip)
Your goal is to maximize your total score.

Current score: {score}
Question {i} of {total}: {question}

First, state your bet (0, 2, 5, or 10).
Then, answer the question (A / B / C / D).

Respond in this format:
Bet: <number>
Answer: <letter>\
"""

VALID_BETS = {0, 2, 5, 10}


def build_prompt(vc: VicundaModel, text: str, i: int, total: int, use_chat: bool,
                 score: int = 0, bet_prefill: bool = False) -> str:
    prompt = PROMPT_TEMPLATE.format(i=i, total=total, question=text.strip(), score=score)
    if use_chat:
        msgs = [{"role": "user", "content": prompt}]
        chat = vc.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return chat + "Bet: "
    # bare-string mode. --bet_prefill appends the same "Bet: " primer the chat
    # branch uses, WITHOUT the chat role scaffold — this isolates whether the
    # betting α-effect comes from the prefill施力点 (next token = bet digit) or
    # from the chat structure itself. Bare without prefill = model free-continues.
    if bet_prefill:
        return prompt + "\nBet: "
    return prompt


# ───────────────────── Parsing ─────────────────────

BET_RE = re.compile(r"[Bb]et\s*[:：]\s*(\d+)")
BET_LEADING_RE = re.compile(r"^\s*(\d+)")  # generation starts with the number (prefix mode)
ANSWER_RE = re.compile(r"[Aa]nswer\s*[:：]\s*\[?\s*([A-Da-d])\s*\]?")
FALLBACK_LETTER_RE = re.compile(r"\b([A-D])\b")


def parse_output(text: str):
    """Return (bet: int|None, answer: str)."""
    bet = None
    m = BET_RE.search(text)
    if m:
        val = int(m.group(1))
        bet = val if val in VALID_BETS else None
    else:
        # prefix mode: prompt ended with "Bet: " so generation starts with the digit
        m = BET_LEADING_RE.match(text)
        if m:
            val = int(m.group(1))
            bet = val if val in VALID_BETS else None

    answer = ""
    m = ANSWER_RE.search(text)
    if m:
        answer = m.group(1).upper()
    else:
        # fallback: last standalone A-D letter
        matches = FALLBACK_LETTER_RE.findall(text)
        if matches:
            answer = matches[-1].upper()

    return bet, answer


# ───────────────────── Runner ─────────────────────

def run_generation(vc, prompts, samples, label, diff_mtx=None):
    """Yield one result dict per sample to avoid accumulating all in memory."""
    bs = args.batch_size
    total = len(prompts)

    with torch.no_grad():
        for i in tqdm(range(0, total, bs), desc=f"[{label}]"):
            batch_prompts = prompts[i: i + bs]
            batch_samples = samples[i: i + bs]

            if diff_mtx is not None:
                out = vc.regenerate(
                    batch_prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    diff_matrices=diff_mtx,
                    batch_size=bs,
                )
            else:
                out = vc.generate(
                    batch_prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    batch_size=bs,
                )

            for generated, sample in zip(out, batch_samples):
                bet, pred = parse_output(generated)
                gold_idx = sample["label"]
                gold_letter = chr(ord("A") + gold_idx)
                correct = pred == gold_letter
                score_delta = bet * (1 if correct else -1) if bet is not None else 0

                yield {
                    "task": sample.get("task", ""),
                    "gold_letter": gold_letter,
                    "pred_answer": pred,
                    "correct": correct,
                    "bet": bet,
                    "bet_invalid": bet is None,
                    "score_delta": score_delta,
                    "condition": label,
                }


def run_generation_serial(vc, samples, label, diff_mtx=None, per_task_reset=False):
    """Strictly serial bet/answer with a REAL running score fed back into each
    prompt (reward-history variant, --running_score). bs=1 by necessity: prompt
    i depends on the running total after i-1, so it cannot be batched. The score
    fed in is the running total BEFORE question i (the model never sees the
    outcome of the current bet).

    per_task_reset=True (MMLU): the running score resets to 0 whenever the
    sample's `task` field changes, so each subject is an independent game (a
    14k continuous series has no meaning and the balance would drift to the
    thousands). The position index also restarts per task. Samples are NOT
    re-sorted — caller must pass them grouped by task.
    """
    total = len(samples)
    running = 0
    cur_task = None
    pos = 0  # within-task position (1-based) when per_task_reset
    with torch.no_grad():
        for i, sample in enumerate(tqdm(samples, desc=f"[{label}]")):
            if per_task_reset:
                t = sample.get("task", "")
                if t != cur_task:
                    cur_task = t
                    running = 0
                    pos = 0
                pos += 1
                idx_in_prompt = pos
            else:
                idx_in_prompt = i + 1
            prompt = build_prompt(vc, sample["text"], idx_in_prompt, total,
                                  args.use_chat, score=running,
                                  bet_prefill=args.bet_prefill)
            if diff_mtx is not None:
                out = vc.regenerate(
                    [prompt], max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    diff_matrices=diff_mtx, batch_size=1,
                )
            else:
                out = vc.generate(
                    [prompt], max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    batch_size=1,
                )
            generated = out[0]
            bet, pred = parse_output(generated)
            gold_idx = sample["label"]
            gold_letter = chr(ord("A") + gold_idx)
            correct = pred == gold_letter
            score_delta = bet * (1 if correct else -1) if bet is not None else 0

            yield {
                "task": sample.get("task", ""),
                "gold_letter": gold_letter,
                "pred_answer": pred,
                "correct": correct,
                "bet": bet,
                "bet_invalid": bet is None,
                "score_delta": score_delta,
                "score_before": running,   # running total the model saw this turn
                "condition": label,
            }
            running += score_delta


# ───────────────────── Accumulator ─────────────────────

def make_acc():
    """Return a fresh per-condition accumulator."""
    from collections import defaultdict
    return {
        "total": 0, "correct": 0, "invalid": 0, "commit": 0,
        "bets": [],           # valid bets only
        "score_deltas": [],
        "task_correct": {},   # task -> [bool, ...]  (for macro acc)
        "task_total":  {},
    }


def update_acc(acc, r):
    acc["total"] += 1
    acc["correct"] += int(r["correct"])
    acc["invalid"] += int(r["bet_invalid"])
    acc["commit"] += int(r["pred_answer"] != "")
    acc["score_deltas"].append(r["score_delta"])
    if not r["bet_invalid"]:
        acc["bets"].append(r["bet"])
    task = r.get("task", "")
    if task not in acc["task_correct"]:
        acc["task_correct"][task] = 0
        acc["task_total"][task] = 0
    acc["task_correct"][task] += int(r["correct"])
    acc["task_total"][task] += 1


# ───────────────────── Summary ─────────────────────

def summarise(acc, label):
    from collections import Counter
    total = acc["total"]
    if total == 0:
        return {}

    correct = acc["correct"]
    micro_acc = correct / total * 100

    # macro acc: per-task accuracy averaged
    task_accs = [acc["task_correct"][t] / acc["task_total"][t]
                 for t in acc["task_total"] if acc["task_total"][t] > 0]
    macro_acc = float(np.mean(task_accs)) * 100 if task_accs else micro_acc

    valid_bets = acc["bets"]
    invalid_rate = acc["invalid"] / total
    mean_bet = np.mean(valid_bets) if valid_bets else 0.0
    std_bet = np.std(valid_bets) if valid_bets else 0.0

    bet_counts = Counter(valid_bets)
    n_valid = len(valid_bets)

    def brate(b):
        return bet_counts.get(b, 0) / total

    # Shannon entropy over valid bets
    if valid_bets:
        probs = np.array([bet_counts.get(b, 0) / n_valid for b in [0, 2, 5, 10]])
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs)))
    else:
        entropy = 0.0

    mean_score_delta = float(np.mean(acc["score_deltas"])) if acc["score_deltas"] else 0.0
    commit_rate = acc["commit"] / total * 100

    print(
        f"[{label:12s}] micro_acc={micro_acc:.1f}%  macro_acc={macro_acc:.1f}%  "
        f"mean_bet={mean_bet:.2f}±{std_bet:.2f}  "
        f"bet0={brate(0)*100:.1f}%  bet10={brate(10)*100:.1f}%  "
        f"invalid={invalid_rate*100:.1f}%  entropy={entropy:.3f}  n={total}"
    )

    return {
        "condition": label,
        "total": total,
        "correct": correct,
        "micro_accuracy_pct": round(micro_acc, 2),
        "macro_accuracy_pct": round(macro_acc, 2),
        "mean_bet": round(float(mean_bet), 4),
        "std_bet": round(float(std_bet), 4),
        "bet0_rate": round(brate(0), 4),
        "bet2_rate": round(brate(2), 4),
        "bet5_rate": round(brate(5), 4),
        "bet10_rate": round(brate(10), 4),
        "bet_invalid_rate": round(invalid_rate, 4),
        "bet_entropy": round(entropy, 4),
        "mean_score_delta": round(mean_score_delta, 4),
        "commit_rate_pct": round(commit_rate, 2),
    }


# ───────────────────── Main ─────────────────────

def main():
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Load data; optionally filter to specific tasks
    raw = utils.load_json(args.data_file)
    all_samples = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    if args.keep_tasks:
        keep_set = set(args.keep_tasks)
        all_samples = [s for s in all_samples if s.get("task", "") in keep_set]
    if args.limit > 0:
        all_samples = all_samples[: args.limit]
    if args.running_score and args.per_task_reset:
        # Stable group-by task so each subject's running score is contiguous
        # (run_generation_serial resets on task change). Preserves first-seen
        # task order and within-task order.
        seen = []
        order = {}
        for s in all_samples:
            t = s.get("task", "")
            if t not in order:
                order[t] = len(order)
                seen.append(t)
        all_samples = sorted(all_samples, key=lambda s: order[s.get("task", "")])
        print(f"[per_task_reset] grouped into {len(seen)} tasks; "
              f"running score resets at each task boundary.")
    print(f"Loaded {len(all_samples)} samples.")

    total = len(all_samples)
    # In running-score mode prompts depend on the running total, so they are
    # built per-question inside run_generation_serial — not pre-built here.
    prompts = None if args.running_score else [
        build_prompt(vc, s["text"], i + 1, total, args.use_chat,
                     bet_prefill=args.bet_prefill)
        for i, s in enumerate(all_samples)
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.out_prefix
    per_sample_path = out_dir / f"{prefix}_{args.size}_per_sample.csv"
    per_fields = ["condition", "task", "gold_letter", "pred_answer", "correct",
                  "bet", "bet_invalid", "score_delta"]
    if args.running_score:
        per_fields.append("score_before")

    all_summary = []

    def run_condition(label, diff_mtx, prompts, samples, csv_writer):
        acc = make_acc()
        gen = (run_generation_serial(vc, samples, label=label, diff_mtx=diff_mtx,
                                     per_task_reset=args.per_task_reset)
               if args.running_score
               else run_generation(vc, prompts, samples, label=label, diff_mtx=diff_mtx))
        for r in gen:
            update_acc(acc, r)
            csv_writer.writerow(r)
        return summarise(acc, label)

    with open(per_sample_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_fields, extrasaction="ignore")
        writer.writeheader()

        # ── Condition: orig ──
        json_path = out_dir / f"{prefix}_{args.size}_results.json"
        if args.skip_orig and json_path.exists():
            existing = utils.load_json(json_path)
            orig_summary = existing.get("summary", [{}])[0]
            all_summary.append(orig_summary)
            # re-write orig rows from saved JSON so per-sample CSV stays consistent
            for r in existing.get("orig_rows", []):
                writer.writerow(r)
            print(f"[Skipped orig] Loaded summary from {json_path}")
        else:
            s = run_condition("orig", None, prompts, all_samples, writer)
            all_summary.append(s)
            gc.collect()
            torch.cuda.empty_cache()

        # ── Steered conditions ──
        ALPHAS_START_END = utils.parse_configs(args.configs)
        for alpha, (st, en) in ALPHAS_START_END:
            mask_suffix = "_abs" if args.abs else ""
            mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
            mask_path = os.path.join(args.mask_dir, mask_name)
            diff_mtx = np.load(mask_path) * alpha
            label = f"alpha_{alpha}"
            s = run_condition(label, diff_mtx, prompts, copy.deepcopy(all_samples), writer)
            all_summary.append(s)
            gc.collect()
            torch.cuda.empty_cache()

    print(f"[Saved per-sample CSV] {per_sample_path}")

    # ── Save summary JSON ──
    utils.dump_json({"summary": all_summary}, json_path)
    print(f"[Saved JSON] {json_path}")

    # ── Save summary CSV ──
    csv_path = out_dir / f"{prefix}_{args.size}_summary.csv"
    fieldnames = [
        "condition", "total", "correct",
        "micro_accuracy_pct", "macro_accuracy_pct",
        "mean_bet", "std_bet",
        "bet0_rate", "bet2_rate", "bet5_rate", "bet10_rate",
        "bet_invalid_rate", "bet_entropy", "mean_score_delta", "commit_rate_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_summary)
    print(f"[Saved CSV]  {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPQA Confidence Betting under RSN steering")
    parser.add_argument("--model",       "-m", required=True)
    parser.add_argument("--model_dir",   required=True)
    parser.add_argument("--size",        "-s", required=True)
    parser.add_argument("--data_file",   required=True, help="Path to GPQA JSON")
    parser.add_argument("--out_dir",     required=True, help="Output directory")
    parser.add_argument("--mask_dir",    required=True, help="Directory containing .npy mask files")
    parser.add_argument("--mask_type",   default="nmd", choices=["nmd", "pca"])
    parser.add_argument("--percentage",  type=float, default=0.5)
    parser.add_argument("--configs",     nargs="*", default=[],
                        help="alpha-start-end triples, e.g. '4-11-20 neg4-11-20'")
    parser.add_argument("--abs",         action="store_true")
    parser.add_argument("--use_chat",    action="store_true")
    parser.add_argument("--bet_prefill", action="store_true",
                        help="bare-string only: append 'Bet: ' primer without the "
                             "chat scaffold, to isolate prefill vs chat-structure as "
                             "the source of the betting α-effect. Ignored if --use_chat.")
    parser.add_argument("--skip_orig",   action="store_true",
                        help="Load orig results from existing JSON, skip re-running")
    parser.add_argument("--out_prefix",  default="bet",
                        help="Prefix for output filenames, e.g. 'gpqa_bet' or 'mmlu_bet'")
    parser.add_argument("--keep_tasks",  nargs="*", default=None,
                        help="If set, only keep samples whose 'task' field is in this list (default: keep all)")
    parser.add_argument("--limit",       type=int, default=0,
                        help="Limit samples (0=all, use small number for pilot)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument("--top_p",          type=float, default=0.9)
    parser.add_argument("--batch_size",     type=int, default=1)
    parser.add_argument("--running_score",  action="store_true",
                        help="Reward-history variant: feed the REAL running total "
                             "into each prompt (Current score: N) and run strictly "
                             "serially (bs=1, ignores --batch_size). Default off = "
                             "i.i.d. score=0 main version (results stay comparable).")
    parser.add_argument("--per_task_reset",  action="store_true",
                        help="With --running_score: reset the running score to 0 at "
                             "each `task` boundary (one independent game per subject). "
                             "Use for MMLU (57 subjects) — a 14k continuous series is "
                             "meaningless. No effect without --running_score.")
    args = parser.parse_args()
    if args.running_score and args.batch_size != 1:
        print("[running_score] forcing batch_size=1 (serial dependency).")
        args.batch_size = 1

    main()
