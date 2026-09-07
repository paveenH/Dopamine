#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_answer_gsm_symbolic.py — GSM-Symbolic generation + scoring with RSN
steering, one config (main/p1/p2) per call.

This is a FORK of get_answer_regenerate_gsm8k.py, not a reuse -- kept
standalone under gsm_symbolic/ so no existing GSM8K/MATH/other-task runner is
touched. It deliberately reuses only:
  - utils.extract_gsm8k_answer / utils.is_correct_gsm8k  (the frozen scorer,
    imported from the repo root, NOT reimplemented)
  - utils.parse_configs / utils.decoder_layer_range
  - llms.VicundaModel
  - template.select_templates_gsm8k(suite="default", cot=True, wording="plain")
    -- the EXISTING explicit-CoT GSM8K prompt, byte-identical to the frozen
    GSM8K main line. No new prompt protocol.

Role is fixed to "neutral" only (no role sweep) -- this experiment is about
numeric/template robustness under steering, not persona.

Extra diagnostics beyond correct/incorrect (all computed from the SAME
generated text, no LLM judge):
  - no_answer: extract_gsm8k_answer found nothing parseable at all
  - multi_marker: more than one '####' marker in the output
  - first_last_disagree: first-#### vs last-#### extraction disagree
    (both via the same regex family, just scanning from either end)
  - is_loop: strict tail-repetition loop detector (final 40-char block
    recurring >=4x), same convention as analyze_loop_anxiety.py's --mode loop
  - truncated: generated_token_count >= max_new_tokens - 1 (heuristic; exact
    count comes from VicundaModel.regenerate(return_metadata=True))
  - gen_chars / gen_tokens: length diagnostics
"""

import argparse
import csv
import gc
import json
import os
import re
import sys
from typing import List

import numpy as np
import torch
from tqdm import tqdm

# Import the frozen GSM8K prompt builder + steering primitives from the repo
# root (parent of this file's directory), without touching any file there.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from llms import VicundaModel  # noqa: E402
from template import select_templates_gsm8k  # noqa: E402
import utils  # noqa: E402
from utils import extract_gsm8k_answer, is_correct_gsm8k  # noqa: E402


def is_loop(text: str, tail_len: int = 40, min_repeats: int = 4) -> bool:
    """Strict degenerate-tail loop detector: the final `tail_len`-char block
    recurs >= min_repeats times in the whole text. Same convention as
    analyze_loop_anxiety.py's --mode loop (a permissive n-gram proxy was
    verified to read 80-86% false positives on GSM8K; this strict version
    reads single-digit-to-teens percent)."""
    text = text.strip()
    if len(text) < tail_len:
        return False
    tail = text[-tail_len:]
    return text.count(tail) >= min_repeats


def first_last_hash_agree(text: str) -> bool:
    """True if the FIRST and LAST '#### <number>' marker (if both exist)
    extract to the same normalized value. Uses utils.normalize_gsm8k so the
    comparison matches the scorer's own equivalence rule."""
    matches = list(re.finditer(r"####\s*([+-]?[\d,]+\.?\d*)", text))
    if len(matches) < 2:
        return True  # nothing to disagree about
    first = matches[0].group(1).replace(",", "")
    last = matches[-1].group(1).replace(",", "")
    return utils.normalize_gsm8k(first) == utils.normalize_gsm8k(last)


def count_hash_markers(text: str) -> int:
    return len(re.findall(r"####", text))


def run_config(
    vc: VicundaModel,
    samples: List[dict],
    diff_mtx,
    prompt_template: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    prompts = [prompt_template.format(context=s["question"]) for s in samples]
    outs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="GSM-Symbolic-regen"):
        batch_prompts = prompts[i : i + batch_size]
        batch_out = vc.regenerate(
            batch_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            diff_matrices=diff_mtx,
            batch_size=batch_size,
            return_metadata=True,
        )
        outs.extend(batch_out)

    correct = 0
    for sample, meta in zip(samples, outs):
        generated = meta["text"] if isinstance(meta, dict) else meta
        gen_tok = meta.get("generated_token_count") if isinstance(meta, dict) else None
        pred_answer = extract_gsm8k_answer(generated)
        is_ok = is_correct_gsm8k(pred_answer, sample["answer"])
        sample["generated"] = generated
        sample["pred_answer"] = pred_answer
        sample["correct"] = is_ok
        sample["no_answer"] = (pred_answer == "")
        sample["n_hash_markers"] = count_hash_markers(generated)
        sample["multi_marker"] = sample["n_hash_markers"] > 1
        sample["first_last_agree"] = first_last_hash_agree(generated)
        sample["is_loop"] = is_loop(generated)
        sample["gen_chars"] = len(generated)
        sample["gen_tokens"] = gen_tok
        sample["truncated"] = (
            gen_tok is not None and gen_tok >= max_new_tokens - 1
        )
        if is_ok:
            correct += 1

    total = len(samples)
    acc = correct / total * 100 if total else 0.0
    print(f"config accuracy: {acc:5.2f}%  ({correct}/{total})")
    return samples, {"correct": correct, "total": total, "accuracy_percentage": round(acc, 2)}


def main():
    ap = argparse.ArgumentParser(description="GSM-Symbolic generation + steering (one config per call)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--hs", required=True, help="mask-dir prefix, e.g. llama3 / qwen2.5")
    ap.add_argument("--size", required=True)
    ap.add_argument("--type", default="non")
    ap.add_argument("--percentage", type=float, default=0.5)
    ap.add_argument("--mask_type", default="nmd")
    ap.add_argument("--configs", nargs="*", required=True,
                     help="alpha-start-end triplets, e.g. 0-11-20 neg6-11-20")
    ap.add_argument("--gsm_config", required=True, choices=["main", "p1", "p2"],
                     help="which GSM-Symbolic official config to run")
    ap.add_argument("--data_file", required=True,
                     help="path (relative to --base_dir) to the loader's output "
                          "JSON for this gsm_config, e.g. "
                          "benchmark/gsm_symbolic/gsm_symbolic_main_test.json")
    ap.add_argument("--preflight", action="store_true",
                     help="run only on the preflight subset file instead of the "
                          "full config (the file must already be filtered to "
                          "this gsm_config by the caller, or --preflight_file "
                          "used).")
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--ans_root", default="answer_gsm_symbolic")
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--batch_size", type=int, default=24)
    args = ap.parse_args()

    data_path = os.path.join(args.base_dir, args.data_file)
    with open(data_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    all_samples = payload["data"]
    # If the loader's combined preflight file was passed, filter to this config.
    all_samples = [s for s in all_samples if s.get("config", args.gsm_config) == args.gsm_config]
    print(f"Loaded {len(all_samples)} GSM-Symbolic[{args.gsm_config}] samples from {data_path}")

    mask_dir = os.path.join(args.base_dir, "mask", f"{args.hs}_{args.type}_logits")

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Frozen explicit-CoT GSM8K prompt, neutral role, plain wording. No new
    # prompt protocol -- byte-identical builder to the main GSM8K line.
    templates = select_templates_gsm8k(suite="default", cot=True, wording="plain")
    prompt_template = templates["neutral"]

    configs = utils.parse_configs(args.configs)
    print("ALPHAS_START_END_PAIRS:", configs)

    save_root = os.path.join(args.base_dir, args.model, args.ans_root, args.gsm_config)
    os.makedirs(save_root, exist_ok=True)

    for alpha, (st, en) in configs:
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}.npy"
        mask_path = os.path.join(mask_dir, mask_name)
        mask = np.load(mask_path)
        diff_mtx = mask * alpha
        print(f"\n=== gsm_config={args.gsm_config} alpha={alpha} layers={st}-{en} ===")

        vc.steering_fire_count(reset=True)

        import copy
        config_samples = copy.deepcopy(all_samples)
        with torch.no_grad():
            updated, accuracy = run_config(
                vc, config_samples, diff_mtx, prompt_template,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        fires = vc.steering_fire_count()
        print(f"steering_fires: {fires}")

        out_dir = os.path.join(save_root, f"mdf_{alpha}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"gsm_symbolic_{args.gsm_config}_{args.size}_{st}_{en}.json")
        if os.path.exists(out_path) and not args.preflight:
            raise FileExistsError(f"{out_path} already exists -- refusing to overwrite.")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "model": args.model,
                    "model_dir": args.model_dir,
                    "size": args.size,
                    "gsm_config": args.gsm_config,
                    "alpha": alpha,
                    "layer_start": st,
                    "layer_end": en,
                    "prompt_template": prompt_template,
                    "cot": True,
                    "role": "neutral",
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "batch_size": args.batch_size,
                    "steering_fires": fires,
                    "n_samples": len(updated),
                    "accuracy": accuracy,
                    "preflight": bool(args.preflight),
                },
                "data": updated,
            }, f, ensure_ascii=False, indent=2)
        print("Saved ->", out_path)

        csv_path = os.path.join(out_dir, f"summary_gsm_symbolic_{args.model}_{args.size}_{st}_{en}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model", "size", "gsm_config", "alpha", "start", "end",
                "correct", "total", "accuracy_percentage", "steering_fires",
                "no_answer", "multi_marker", "first_last_disagree", "loop",
                "truncated",
            ])
            writer.writeheader()
            writer.writerow({
                "model": args.model, "size": args.size, "gsm_config": args.gsm_config,
                "alpha": alpha, "start": st, "end": en,
                "correct": accuracy["correct"], "total": accuracy["total"],
                "accuracy_percentage": accuracy["accuracy_percentage"],
                "steering_fires": fires,
                "no_answer": sum(1 for s in updated if s["no_answer"]),
                "multi_marker": sum(1 for s in updated if s["multi_marker"]),
                "first_last_disagree": sum(1 for s in updated if not s["first_last_agree"]),
                "loop": sum(1 for s in updated if s["is_loop"]),
                "truncated": sum(1 for s in updated if s["truncated"]),
            })
        print("Saved CSV ->", csv_path)

        del config_samples
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll configs finished.")


if __name__ == "__main__":
    main()
