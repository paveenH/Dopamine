#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSM8K willingness self-evaluation (0-9) — GENERATION variant.

Sibling of get_action_regenerate_gsm8k.py (the LOGITS path, §2.6 authoritative).
That script reads the logits of the "0".."9" tokens at a single position. THIS
script instead has the model GENERATE a "Willingness: <number>" line and parses
the digit out of the text — the betting-style readout (cf. get_answer_gpqa_bet.py).

WHY a generation variant:
  - Betting showed that a free-generation, single-decision readout (Bet: <number>)
    has an α-effect that only appears under the chat wrapper + a "Bet: " assistant
    prefill (system isolation focuses the model on the one decision token). This
    script tests whether the SAME thing happens to willingness once it is phrased
    as generation rather than a logits readout: i.e. does chat + "Willingness: "
    prefill revive a dose-response that the bare logits readout did not show?
  - It is a CONTROL / probe, NOT a replacement for the §2.6 logits numbers, which
    stay authoritative. Output goes to its own dir.

Defaults to --use_chat (the generation form is a free continuation, so it wants
the chat scaffold + prefill exactly like betting). Bare is available for ablation.

Conditions per run via --configs (e.g. "0-11-20 4-11-20 neg4-11-20"); alpha=0 uses
plain generate (no mask), other alphas load diff_mtx * alpha and steer via
regenerate (prefill-only), matching the betting/GSM8K-regenerate convention.
"""

import os
import re
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

# ───────────────────── Prompt (betting-style, generation) ─────────────────────
# Question + an explicit "respond in this format" block, then "Willingness: " is
# pulled out as the assistant prefill (under chat) / appended (bare) so the model
# generates the digit immediately. NOT the logits-suite template — defined here to
# avoid touching template.py's history-bound suites.

# BET-aligned structure (cf. get_answer_gpqa_bet.py PROMPT_TEMPLATE): an imperative
# instruction with the explicit option set, a "Respond in this format" block, then
# "Willingness: " pulled out as the assistant prefill. Mirrors betting's
# "…bet (0,2,5,10)… Respond in this format: Bet: <number>" + "Bet: " prefill, so a
# generation-mode willingness readout sits on the same footing as the betting one.
INSTRUCTION = (
    'Honestly evaluate your "reasoning willingness" (0, 1, 2, 3, 4, 5, 6, 7, 8, 9).\n'
    "Respond in this format:\n"
    "Willingness: <number>"
)
PREFILL = "Willingness: "


def build_prompt(vc: VicundaModel, ctx: str, role: str, use_chat: bool) -> str:
    lines = [f"Here is a question: {ctx}"]
    if role not in ("neutral", "norole"):
        lines.append(f"Now you are {role}.")
    lines.append(INSTRUCTION)
    body = "\n".join(lines)
    if use_chat:
        msgs = [{"role": "user", "content": body}]
        chat = vc.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return chat + PREFILL
    # bare: append the same prefill without the chat scaffold
    return body + "\n" + PREFILL


# ───────────────────── Parsing ─────────────────────
# Prompt ends with "… is: ", so the model continues with the digit (prefix mode).
LEADING_RE = re.compile(r"^\s*(\d)")                       # generation starts with the digit
WILL_RE = re.compile(r"willingness[^0-9]{0,12}(\d)", re.IGNORECASE)  # echoed phrasing fallback


def parse_willingness(text: str):
    """Return (score 0-9 | None, valid: bool)."""
    m = LEADING_RE.match(text)          # prefix mode (the normal case)
    if not m:
        m = WILL_RE.search(text)        # fallback: model echoed "willingness … N"
    if m:
        return int(m.group(1)), True
    return None, False


# ───────────────────── Runner ─────────────────────

def run_gsm8k_action_gen(vc, samples, diff_mtx, roles):
    role_stats = {r: {str(i): 0 for i in range(10)} | {"total": 0, "invalid": 0} for r in roles}

    for role in roles:
        rk = role.replace(" ", "_")
        prompts = [build_prompt(vc, s["question"], role, args.use_chat) for s in samples]
        gen_texts = []
        for i in tqdm(range(0, len(prompts), args.batch_size),
                      desc=f"gsm8k-action-gen [{role}]"):
            batch = prompts[i: i + args.batch_size]
            if diff_mtx is None:
                out = vc.generate(
                    batch, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    batch_size=args.batch_size,
                )
            else:
                out = vc.regenerate(
                    batch, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    diff_matrices=diff_mtx, batch_size=args.batch_size,
                )
            gen_texts.extend(out)

        rs = role_stats[role]
        for sample, gen in zip(samples, gen_texts):
            score, valid = parse_willingness(gen)
            sample[f"generated_{rk}"] = gen
            sample[f"score_{rk}"] = score
            sample[f"valid_{rk}"] = valid
            rs["total"] += 1
            if valid:
                rs[str(score)] += 1
            else:
                rs["invalid"] += 1

    return samples, role_stats


def summarize(role_stats, alpha, st, en, roles):
    rows = []
    for role in roles:
        s = role_stats[role]
        total = int(s["total"])
        invalid = int(s["invalid"])
        valid = total - invalid
        counts = [int(s[str(i)]) for i in range(10)]
        if valid > 0:
            mean = sum(i * c for i, c in enumerate(counts)) / valid
            var = sum(((i - mean) ** 2) * c for i, c in enumerate(counts)) / valid
            std = math.sqrt(var)
            dist = np.array(counts, dtype=float) / valid
            ent = utils.entropy_bits(dist)
            top_score = int(np.argmax(counts))
            top_ratio = max(counts) / valid
            tail_low = sum(counts[0:3]) / valid
            tail_high = sum(counts[8:10]) / valid
        else:
            mean = std = ent = top_ratio = tail_low = tail_high = 0.0
            top_score = 0

        inv_rate = invalid / total if total else 0.0
        print(f"  {role:<25} avg={mean:.2f} std={std:.2f} top={top_score} "
              f"invalid={inv_rate:.1%} counts={counts}")

        row = {
            "model": args.model, "size": args.size, "alpha": alpha,
            "start": st, "end": en, "task": f"gsm8k_{args.suite}_gen", "role": role,
            "total": total, "valid": valid, "invalid_rate": round(inv_rate, 6),
            "mean": round(mean, 6), "std": round(std, 6),
            "entropy_bits": round(ent, 6), "top_score": top_score,
            "top_ratio": round(top_ratio, 6),
            "tail_low_0_2": round(tail_low, 6), "tail_high_8_9": round(tail_high, 6),
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

    custom_roles = None
    if args.roles:
        custom_roles = [r.strip() for r in args.roles.split(",")]
    roles = utils.make_characters("gsm8k", custom_roles)
    print("Roles:", roles)
    print("Prompt (neutral):")
    print(build_prompt(vc, "<Q>", "neutral", args.use_chat))

    all_csv_rows = []

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        print(f"\n=== alpha={alpha} | layers={st}-{en} | chat={args.use_chat} ===")
        if alpha == 0:
            diff_mtx = None
        else:
            mask_suffix = "_abs" if args.abs else ""
            mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
            diff_mtx = np.load(os.path.join(MASK_DIR, mask_name)) * alpha

        config_samples = [dict(s) for s in all_samples]
        with torch.no_grad():
            updated, role_stats = run_gsm8k_action_gen(vc, config_samples, diff_mtx, roles)

        all_csv_rows.extend(summarize(role_stats, alpha, st, en, roles))

        alpha_tag = f"neg{abs(alpha)}" if alpha < 0 else str(alpha)
        out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha_tag}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"gsm8k_action_gen_{args.size}_{st}_{en}.json")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump({"data": updated, "role_stats": role_stats,
                       "prompt_template": build_prompt(vc, "{context}", "neutral", args.use_chat),
                       "use_chat": args.use_chat},
                      fw, ensure_ascii=False, indent=2)
        print(f"[Saved JSON] {out_path}")

        del config_samples, updated
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fieldnames = [
        "model", "size", "alpha", "start", "end", "task", "role",
        "total", "valid", "invalid_rate", "mean", "std", "entropy_bits",
        "top_score", "top_ratio", "tail_low_0_2", "tail_high_8_9",
    ] + [f"counts_{i}" for i in range(10)]
    csv_path = os.path.join(SAVE_ROOT, f"summary_gsm8k_action_gen_{args.model}_{args.size}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"\n[Saved CSV] {csv_path}")
    print("All configs finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSM8K willingness self-eval (0-9), GENERATION variant")
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
    parser.add_argument("--ans_file", type=str, default="answer_action_gen_gsm8k")
    parser.add_argument("--suite", type=str, default="action", choices=["action", "confidence"])
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--roles", type=str, default="")
    parser.add_argument("--use_chat", action="store_true",
                        help="ON by default in run_*.sh — the generation form wants "
                             "the chat scaffold + 'Willingness: ' prefill (betting-style).")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir", type=str, default=None)
    args = parser.parse_args()

    print("Model:", args.model, "| dir:", args.model_dir)
    BASE = args.base_dir if args.base_dir else f"/{args.data}/paveen/Dopamine/components"
    DATA_DIR = os.path.join(BASE, args.test_file)
    MASK_DIR = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()
