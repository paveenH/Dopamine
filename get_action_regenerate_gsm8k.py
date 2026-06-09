#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSM8K willingness self-evaluation (0-9) with neuron editing (mdf).

Combines:
- Steering logic from get_action_regenerate_logits_pro.py (diff_mtx * alpha)
- 0-9 logit extraction from get_action_logits_pro.py
- GSM8K data/template from get_answer_regenerate_gsm8k.py

Three conditions per run (controlled by --configs):
  neutral (alpha=0), alpha=+4, alpha=-4
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
from template import select_templates_gsm8k
import utils


def run_gsm8k_action(vc, samples, diff_mtx, templates, roles):
    """Extract 0-9 willingness scores for all samples and roles."""
    LABELS = templates["labels"]
    opt_ids = utils.option_token_ids(vc, LABELS)
    role_stats = {r: {str(i): 0 for i in range(10)} | {"total": 0} for r in roles}

    for sample in tqdm(samples, desc="gsm8k-action"):
        ctx = sample["question"]

        for role in roles:
            prompt = utils.construct_prompt(vc, templates, ctx, role, args.use_chat)

            if diff_mtx is None:
                logits = vc.get_logits([prompt], return_hidden=False)
                raw_logits = logits[0, -1].detach().cpu().float().numpy()
            else:
                raw_logits = vc.regenerate_logits([prompt], diff_mtx, tail_len=args.tail_len)[0]

            opt_logits = np.array([raw_logits[i] for i in opt_ids], dtype=float)
            opt_logits -= opt_logits.max()
            probs = np.exp(opt_logits)
            probs /= probs.sum()

            pred_idx = int(np.argmax(opt_logits))
            pred_label = LABELS[pred_idx]
            pred_prob = float(probs[pred_idx])

            rk = role.replace(" ", "_")
            sample[f"score_{rk}"] = pred_label
            sample[f"score_prob_{rk}"] = pred_prob
            sample[f"score_dist_{rk}"] = probs.tolist()

            rs = role_stats[role]
            rs["total"] += 1
            rs[pred_label] = rs.get(pred_label, 0) + 1

    return samples, role_stats


def summarize(role_stats, alpha, st, en, roles):
    rows = []
    for role in roles:
        s = role_stats[role]
        total = int(s["total"])
        counts = [int(s[str(i)]) for i in range(10)]
        if total > 0:
            mean = sum(i * c for i, c in enumerate(counts)) / total
            var = sum(((i - mean) ** 2) * c for i, c in enumerate(counts)) / total
            std = math.sqrt(var)
            dist = np.array(counts, dtype=float) / total
            ent = utils.entropy_bits(dist)
            top_score = int(np.argmax(counts))
            top_ratio = max(counts) / total
            tail_low = sum(counts[0:3]) / total
            tail_high = sum(counts[8:10]) / total
        else:
            mean = std = ent = top_ratio = tail_low = tail_high = 0.0
            top_score = 0

        print(f"  {role:<25} avg={mean:.2f} std={std:.2f} top={top_score} counts={counts}")

        row = {
            "model": args.model,
            "size": args.size,
            "alpha": alpha,
            "start": st,
            "end": en,
            "task": f"gsm8k_{args.suite}",
            "role": role,
            "total": total,
            "mean": round(mean, 6),
            "std": round(std, 6),
            "entropy_bits": round(ent, 6),
            "top_score": top_score,
            "top_ratio": round(top_ratio, 6),
            "tail_low_0_2": round(tail_low, 6),
            "tail_high_8_9": round(tail_high, 6),
        }
        for i in range(10):
            row[f"counts_{i}"] = counts[i]
        rows.append(row)
    return rows


def main():
    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("Configs:", ALPHAS_START_END_PAIRS)

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    all_samples = utils.load_json(DATA_DIR)
    print(f"Loaded {len(all_samples)} GSM8K samples.")

    templates = select_templates_gsm8k(suite=args.suite, cot=args.cot)
    roles = utils.make_characters("gsm8k")
    tmp_record = utils.record_template(roles, templates)

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

        config_samples = [dict(s) for s in all_samples]

        with torch.no_grad():
            updated_samples, role_stats = run_gsm8k_action(
                vc, config_samples, diff_mtx, templates, roles
            )

        csv_rows = summarize(role_stats, alpha, st, en, roles)
        all_csv_rows.extend(csv_rows)

        # Save JSON
        alpha_tag = f"neg{abs(alpha)}" if alpha < 0 else str(alpha)
        out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha_tag}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"gsm8k_action_{args.size}_answers_{st}_{en}.json")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump(
                {"data": updated_samples, "role_stats": role_stats, "template": tmp_record},
                fw, ensure_ascii=False, indent=2,
            )
        print(f"[Saved JSON] {out_path}")

        del config_samples, updated_samples
        gc.collect()
        torch.cuda.empty_cache()

    # Save CSV
    fieldnames = [
        "model", "size", "alpha", "start", "end", "task", "role",
        "total", "mean", "std", "entropy_bits",
        "top_score", "top_ratio", "tail_low_0_2", "tail_high_8_9",
    ] + [f"counts_{i}" for i in range(10)]

    csv_path = os.path.join(SAVE_ROOT, f"summary_gsm8k_action_{args.model}_{args.size}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"\n[Saved CSV] {csv_path}")
    print("All configs finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSM8K self-eval (0-9) with neuron editing")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--hs", type=str, required=True)
    parser.add_argument("--size", type=str, required=True)
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["0-11-20", "4-11-20", "neg4-11-20"])
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--ans_file", type=str, default="answer_action_gsm8k")
    parser.add_argument("--suite", type=str, default="action", choices=["action", "confidence"])
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--use_chat", action="store_true")
    parser.add_argument("--tail_len", type=int, default=1)
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
