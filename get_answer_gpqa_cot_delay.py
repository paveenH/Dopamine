#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPQA Delay Discounting experiment: adaptive CoT length under RSN steering.

Prompt lets the model decide when to commit ("Final Answer: X").
Measures CoT token length before commit as a proxy for delay tolerance.

Conditions: orig (no steering) / mdf (±alpha steering via regenerate)
"""

import os
import re
import gc
import csv
import json
import copy
import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from llms import VicundaModel
import utils

# ───────────────────── Prompt ─────────────────────

PROMPT_TEMPLATE = (
    "Solve the following problem. You may think as much or as little as you need "
    "before giving your final answer. When you are ready to commit, write "
    '"Final Answer: [X]" where X is the letter of your choice.\n\n'
    "{context}\n\n"
    "Think step by step, then give your Final Answer:"
)

ROLE = "neutral"


def build_prompt(vc: VicundaModel, text: str, use_chat: bool) -> str:
    prompt = PROMPT_TEMPLATE.format(context=text.strip())
    if use_chat:
        msgs = [{"role": "user", "content": prompt}]
        return vc.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return prompt


# ───────────────────── Answer & CoT extraction ─────────────────────

LETTER_RE = re.compile(
    r"[Ff]inal\s+[Aa]nswer\s*[:\-]?\s*\[?\s*([A-Da-d])\s*\]?", re.IGNORECASE
)
FALLBACK_RE = re.compile(r"\b([A-D])\)", re.IGNORECASE)


def extract_final_answer(text: str) -> str:
    m = LETTER_RE.search(text)
    if m:
        return m.group(1).upper()
    # fallback: last standalone letter
    m = FALLBACK_RE.search(text[::-1])  # search from end
    if m:
        return m.group(1).upper()
    return ""


def cot_token_count(text: str, tokenizer) -> int:
    """Tokens before 'Final Answer:' marker (or full text if missing)."""
    match = LETTER_RE.search(text)
    if match:
        before = text[: match.start()]
    else:
        before = text
    return len(tokenizer.encode(before, add_special_tokens=False))


# ───────────────────── Runner ─────────────────────

def run_generation(vc, prompts, samples, label, diff_mtx=None):
    results = []
    bs = args.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), bs), desc=f"[{label}]"):
            batch_prompts = prompts[i : i + bs]
            batch_samples = samples[i : i + bs]
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
                pred = extract_final_answer(generated)
                gold_idx = sample["label"]
                gold_letter = chr(ord("A") + gold_idx)
                correct = pred == gold_letter
                cot_len = cot_token_count(generated, vc.tokenizer)
                results.append({
                    "task": sample.get("task", "GPQA"),
                    "text": sample["text"],
                    "label": gold_idx,
                    "gold_letter": gold_letter,
                    "generated": generated,
                    "pred_answer": pred,
                    "correct": correct,
                    "cot_tokens": cot_len,
                    "condition": label,
                })

    return results


# ───────────────────── Main ─────────────────────

def summarise(results, label):
    total = len(results)
    correct = sum(r["correct"] for r in results)
    acc = correct / total * 100 if total else 0
    cot_lens = [r["cot_tokens"] for r in results]
    mean_cot = np.mean(cot_lens)
    std_cot = np.std(cot_lens)
    commit_rate = sum(1 for r in results if r["pred_answer"] != "") / total * 100
    print(
        f"[{label:12s}] acc={acc:.1f}%  cot_mean={mean_cot:.1f}  "
        f"cot_std={std_cot:.1f}  commit_rate={commit_rate:.1f}%  n={total}"
    )
    return {
        "condition": label,
        "total": total,
        "correct": correct,
        "accuracy_pct": round(acc, 2),
        "cot_mean_tokens": round(float(mean_cot), 2),
        "cot_std_tokens": round(float(std_cot), 2),
        "commit_rate_pct": round(commit_rate, 2),
    }


def main():
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Load GPQA data
    raw = utils.load_json(args.data_file)
    if isinstance(raw, dict) and "data" in raw:
        all_samples = raw["data"]
    else:
        all_samples = raw

    # Stratified sampling: main + diamond only, proportional to 200 total
    KEEP_TASKS = {"GPQA (gpqa_main)", "GPQA (gpqa_diamond)"}
    if args.sample > 0:
        import random
        random.seed(args.seed)
        by_task = {}
        for s in all_samples:
            t = s.get("task", "")
            if t in KEEP_TASKS:
                by_task.setdefault(t, []).append(s)
        total_pool = sum(len(v) for v in by_task.values())
        sampled = []
        for t, pool in by_task.items():
            n = round(args.sample * len(pool) / total_pool)
            sampled.extend(random.sample(pool, min(n, len(pool))))
        all_samples = sampled
        task_counts = {t: sum(1 for s in all_samples if s["task"] == t) for t in by_task}
        print(f"Stratified sample: {len(all_samples)} samples — " +
              ", ".join(f"{t}: {n}" for t, n in task_counts.items()))
    else:
        all_samples = [s for s in all_samples if s.get("task", "") in KEEP_TASKS]
        if args.limit > 0:
            all_samples = all_samples[: args.limit]
        print(f"Loaded {len(all_samples)} GPQA samples (main + diamond).")

    print(f"Total samples: {len(all_samples)}")

    prompts = [build_prompt(vc, s["text"], args.use_chat) for s in all_samples]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    all_results = {}

    # ── Condition 1: orig ──
    json_path = out_dir / f"gpqa_delay_{args.size}_results.json"
    if args.skip_orig and json_path.exists():
        existing = utils.load_json(json_path)
        orig_results = existing["data"]["orig"]
        all_results["orig"] = orig_results
        all_summary.append(summarise(orig_results, "orig"))
        print(f"[Skipped orig] Loaded {len(orig_results)} samples from {json_path}")
    else:
        orig_results = run_generation(vc, prompts, all_samples, label="orig", diff_mtx=None)
        all_results["orig"] = orig_results
        all_summary.append(summarise(orig_results, "orig"))

    # ── Conditions 2+: steered ──
    ALPHAS_START_END = utils.parse_configs(args.configs)
    for alpha, (st, en) in ALPHAS_START_END:
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(args.mask_dir, mask_name)
        diff_mtx = np.load(mask_path) * alpha
        label = f"alpha_{alpha}"
        steered = run_generation(
            vc, prompts, copy.deepcopy(all_samples), label=label, diff_mtx=diff_mtx
        )
        all_results[label] = steered
        all_summary.append(summarise(steered, label))

        gc.collect()
        torch.cuda.empty_cache()

    # ── Save JSON ──
    json_path = out_dir / f"gpqa_delay_{args.size}_results.json"
    utils.dump_json({"summary": all_summary, "data": all_results}, json_path)
    print(f"[Saved JSON] {json_path}")

    # ── Save CSV summary ──
    csv_path = out_dir / f"gpqa_delay_{args.size}_summary.csv"
    fieldnames = [
        "condition", "total", "correct", "accuracy_pct",
        "cot_mean_tokens", "cot_std_tokens", "commit_rate_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_summary)
    print(f"[Saved CSV]  {csv_path}")

    # ── Save per-sample CSV ──
    per_sample_path = out_dir / f"gpqa_delay_{args.size}_per_sample.csv"
    flat = [r for v in all_results.values() for r in v]
    per_fields = ["condition", "task", "gold_letter", "pred_answer", "correct", "cot_tokens"]
    with open(per_sample_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat)
    print(f"[Saved per-sample CSV] {per_sample_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPQA Delay Discounting (adaptive CoT length)")
    parser.add_argument("--model",     "-m", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--size",      "-s", required=True)
    parser.add_argument("--data_file", required=True, help="Path to GPQA JSON")
    parser.add_argument("--out_dir",   required=True, help="Output directory")
    parser.add_argument("--mask_dir",  required=True, help="Directory containing .npy mask files")
    parser.add_argument("--mask_type", default="nmd", choices=["nmd", "pca"])
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs",   nargs="*", default=[],
                        help="Space-separated alpha-start-end triples, e.g. '4-11-20 neg4-11-20'")
    parser.add_argument("--abs",       action="store_true")
    parser.add_argument("--use_chat",  action="store_true")
    parser.add_argument("--limit",     type=int, default=0,
                        help="Limit number of samples (0 = all, use 20 for pilot)")
    parser.add_argument("--sample",    type=int, default=0,
                        help="Stratified sample size from main+diamond (0 = use all main+diamond)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed for stratified sampling")
    parser.add_argument("--skip_orig", action="store_true",
                        help="Load orig results from existing JSON, skip re-running")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature",    type=float, default=0.0)
    parser.add_argument("--top_p",          type=float, default=0.9)
    parser.add_argument("--batch_size",     type=int, default=1,
                        help="Prompts per forward pass (set >1 for speedup, watch VRAM)")
    args = parser.parse_args()

    main()
