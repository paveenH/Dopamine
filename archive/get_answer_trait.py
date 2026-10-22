#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for VicundaModel on TRAIT personality benchmark with neuron editing.
- Loads trait_test.json (produced by data_trait.py)
- Groups samples by task (e.g. BFI_Extraversion, SD3_Narcissism)
- For each sample: computes softmax over [A,B,C,D] logits, then
  trait_score = sum(softmax[i] * OPTION_SCORES[i]) where OPTION_SCORES=[1,0,1,0]
- Outputs per-task JSON + summary CSV per alpha condition

Option scoring (fixed by TRAIT paper Section 3.4):
  A=High1(1), B=Low1(0), C=High2(1), D=Low2(0) → OPTION_SCORES = [1, 0, 1, 0]
  trait_score ∈ [0, 1]: 1.0 = fully high-trait, 0.0 = fully low-trait
"""

import os
import gc
import copy
import json
import csv
import numpy as np
import torch
from tqdm import tqdm
import argparse

from llms import VicundaModel
from template import select_templates_pro
import utils

OPTION_SCORES = [1.0, 0.0, 1.0, 0.0]  # A=High, B=Low, C=High, D=Low
LABELS = ["A", "B", "C", "D"]


def run_task_trait(
    vc: VicundaModel,
    task: str,
    samples: list,
    diff_mtx: np.ndarray,
    suite: str,
):
    """
    Run one TRAIT task (all its samples) with a fixed diff_mtx.
    Returns updated samples, per-role stats, and recorded templates.
    """
    custom_roles = None
    if args.roles:
        custom_roles = [r.strip() for r in args.roles.split(",")]
    roles = utils.make_characters(task.replace(" ", "_"), custom_roles)

    templates = select_templates_pro(suite=suite, labels=LABELS, use_E=False)
    opt_ids = utils.option_token_ids(vc, LABELS)

    stats = {r: {"trait_scores": [], "total": 0} for r in roles}

    for sample in tqdm(samples, desc=task):
        ctx = sample.get("text", "")

        for role in roles:
            prompt = utils.construct_prompt(vc, templates, ctx, role, args.use_chat)

            raw_logits = vc.regenerate_logits([prompt], diff_mtx, tail_len=args.tail_len)[0]
            opt_logits = np.array([raw_logits[i] for i in opt_ids])

            exp = np.exp(opt_logits - opt_logits.max())
            soft = exp / exp.sum()

            trait_score = float(sum(soft[i] * OPTION_SCORES[i] for i in range(4)))

            role_key = role.replace(" ", "_")
            sample[f"trait_score_{role_key}"] = trait_score
            sample[f"softmax_{role_key}"] = [float(p) for p in soft]
            sample[f"logits_{role_key}"] = [float(x) for x in opt_logits]

            stats[role]["trait_scores"].append(trait_score)
            stats[role]["total"] += 1

    # Summarize per role
    summary = {}
    for role, s in stats.items():
        scores = np.array(s["trait_scores"])
        mean_ts = float(np.mean(scores)) if len(scores) else 0.0
        std_ts = float(np.std(scores)) if len(scores) else 0.0
        summary[role] = {
            "n": s["total"],
            "mean_trait_score": round(mean_ts, 4),
            "std_trait_score": round(std_ts, 4),
        }
        print(f"{role:<25} mean_trait={mean_ts:.4f}  std={std_ts:.4f}  (n={s['total']})")

    tmp_record = utils.record_template(roles, templates)
    return samples, summary, tmp_record


def main():
    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("ALPHAS_START_END_PAIRS:", ALPHAS_START_END_PAIRS)

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    all_samples = utils.load_json(DATA_DIR)
    tasks = sorted({s["task"] for s in all_samples})
    print(f"Found {len(tasks)} TRAIT tasks: {tasks}")

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, mask_name)
        diff_mtx = np.load(mask_path) * alpha
        TOP = max(1, int(args.percentage / 100 * diff_mtx.shape[1]))
        print(f"\n=== α={alpha} | layers={st}-{en} | TOP={TOP} ===")

        csv_rows = []

        for task in tasks:
            task_samples = [copy.deepcopy(s) for s in all_samples if s["task"] == task]
            if not task_samples:
                print("[Skip] empty task:", task)
                continue

            # Parse instrument and dimension from task name (e.g. "BFI_Extraversion")
            parts = task.split("_", 1)
            instrument = parts[0] if len(parts) == 2 else "BFI"
            dimension = parts[1] if len(parts) == 2 else task

            print(f"\n--- Task: {task} ({instrument} / {dimension}) ---")
            with torch.no_grad():
                updated_data, summary, tmp_record = run_task_trait(
                    vc=vc,
                    task=task,
                    samples=task_samples,
                    diff_mtx=diff_mtx,
                    suite=args.suite,
                )
            del task_samples

            out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{task}_{args.size}_answers_{TOP}_{st}_{en}.json")
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump(
                    {"data": updated_data, "stats": summary, "template": tmp_record},
                    fw, ensure_ascii=False, indent=2,
                )
            print("Saved →", out_path)
            del updated_data
            gc.collect()

            for role, s in summary.items():
                csv_rows.append({
                    "model": args.model,
                    "size": args.size,
                    "alpha": alpha,
                    "start": st,
                    "end": en,
                    "TOP": TOP,
                    "task": task,
                    "instrument": instrument,
                    "dimension": dimension,
                    "role": role,
                    "n": s["n"],
                    "mean_trait_score": s["mean_trait_score"],
                    "std_trait_score": s["std_trait_score"],
                    "suite": args.suite,
                })

        out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"summary_{args.model}_{args.size}_{TOP}_{st}_{en}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model", "size", "alpha", "start", "end", "TOP",
                "task", "instrument", "dimension", "role",
                "n", "mean_trait_score", "std_trait_score", "suite",
            ])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[Saved CSV] {csv_path}")

        del csv_rows
        gc.collect()
        torch.cuda.empty_cache()

    print("\n✅  All TRAIT tasks finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VicundaModel on TRAIT personality benchmark.")
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--model_dir", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--hs", type=str, default="llama3")
    parser.add_argument("--size", type=str, default="8B")
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["0-11-20", "4-11-20", "neg4-11-20"])
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--ans_file", type=str, default="answer_trait")
    parser.add_argument("--use_chat", action="store_true")
    parser.add_argument("--tail_len", type=int, default=1)
    parser.add_argument("--suite", type=str, default="vanilla", choices=["default", "vanilla"])
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--roles", type=str, default=None)

    args = parser.parse_args()

    print("Model:", args.model)
    print("Model dir:", args.model_dir)
    print("HS:", args.hs)
    print("Mask type:", args.mask_type)

    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/RolePlaying/components"

    DATA_DIR = os.path.join(BASE, args.test_file)
    MASK_DIR = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()
