#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_answer_gsm_symbolic_nocot.py — GSM-Symbolic NO-COT generation + scoring
with RSN steering, one config (main/p1/p2) per call.

This is a FORK of get_answer_gsm_symbolic.py (the CoT sibling), NOT a flag on
it -- the CoT script writes FROZEN cells under --ans_root=gsm_symbolic, and
adding a cot/no-cot branch there would put frozen and follow-up output on one
code path (the same reasoning the project's P3-supp CoT follow-up used for
the opposite direction). This script is READ-ONLY with respect to that CoT
tree: different default --ans_root, different prompt builder call, and its
own meta.cot=False so the evaluator's cross-cell consistency check can never
silently mix a No-CoT cell into a CoT family or vice versa.

This is a FORK of get_answer_regenerate_gsm8k.py, not a reuse -- kept
standalone under gsm_symbolic/ so no existing GSM8K/MATH/other-task runner is
touched. It deliberately reuses only:
  - utils.extract_gsm8k_answer / utils.is_correct_gsm8k  (the frozen scorer,
    imported from the repo root, NOT reimplemented)
  - utils.parse_configs / utils.decoder_layer_range
  - llms.VicundaModel
  - template.select_templates_gsm8k(suite="default", cot=False, wording="plain")
    -- the EXISTING No-CoT GSM8K prompt (cot=False, not a new template), the
    ONLY change from the CoT sibling script is this one argument. The
    resulting neutral prompt is:
        "Solve the following math problem.\n"
        "Question: {context}\n"
        "Provide your final numeric answer after '####'.\n"
        "Answer: "
    i.e. exactly the CoT prompt minus its "Let's think step by step." line --
    everything else (role routing, #### directive wording, symmetry) is
    byte-identical, so any accuracy difference from the CoT sibling
    attributes cleanly to the CoT instruction, not to a reworded protocol.

Role is fixed to "neutral" only (no role sweep) -- this experiment is about
numeric/template robustness under steering, not persona.

Extra diagnostics beyond correct/incorrect (all computed from the SAME
generated text, no LLM judge). Accuracy always uses the frozen
extract_gsm8k_answer/is_correct_gsm8k pair unmodified; the fields below exist
ONLY to describe the generation, not to change scoring:
  - no_marker: the text contains no '####' at all (extract_gsm8k_answer then
    falls back to "the answer is X" / \boxed{} / last-number-in-text, so a
    no_marker sample can still score correct or have a non-empty pred_answer
    -- this field is a FORMAT diagnostic, not "no answer was extracted")
  - marker_unparsed: '####' is present but the digits after it did not parse
    (extract_gsm8k_answer's primary regex found no match even though a '####'
    substring exists -- e.g. "#### unclear")
  - no_answer: extract_gsm8k_answer's full fallback chain (marker -> "the
    answer is" -> \boxed{} -> last number in text) found NOTHING at all.
    Distinct from no_marker/marker_unparsed: a no_marker sample is usually
    NOT no_answer, because the fallback chain typically recovers a number.
  - multi_marker: more than one '####' marker in the output
  - first_last_disagree: first-#### vs last-#### extraction disagree
    (both via the same regex family, just scanning from either end; only
    meaningful when >=2 markers are present)
  - is_loop: strict tail-repetition loop detector (final 40-char block
    recurring >=4x), same convention as analyze_loop_anxiety.py's --mode loop
  - truncated: generated_token_count >= max_new_tokens - 1 (heuristic; exact
    count comes from VicundaModel.regenerate(return_metadata=True))
  - gen_chars / gen_tokens: length diagnostics
"""

import argparse
import csv
import gc
import hashlib
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
from utils import extract_gsm8k_answer, is_correct_gsm8k, decoder_layer_range  # noqa: E402


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


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


def has_parseable_hash(text: str) -> bool:
    """True iff at least one '#### <number>' marker parses -- the SAME regex
    extract_gsm8k_answer's primary branch uses. Used only to distinguish
    no_marker (no '####' substring at all) from marker_unparsed ('####'
    present but never followed by parseable digits)."""
    return re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text) is not None


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
        # Hard check: the model must return exactly one output per prompt in
        # the batch. A silent length mismatch would misalign sample<->output
        # for every row after the first short/long batch.
        if len(batch_out) != len(batch_prompts):
            raise RuntimeError(
                f"regenerate() returned {len(batch_out)} outputs for "
                f"{len(batch_prompts)} prompts (batch starting at index {i}) "
                "-- output would silently misalign with samples."
            )
        outs.extend(batch_out)

    if len(outs) != len(samples):
        raise RuntimeError(
            f"total outputs ({len(outs)}) != total samples ({len(samples)}) "
            "after all batches -- refusing to score a misaligned run."
        )

    correct = 0
    for sample, meta in zip(samples, outs):
        generated = meta["text"] if isinstance(meta, dict) else meta
        gen_tok = meta.get("generated_token_count") if isinstance(meta, dict) else None
        pred_answer = extract_gsm8k_answer(generated)
        is_ok = is_correct_gsm8k(pred_answer, sample["answer"])
        has_marker = "####" in generated
        marker_parses = has_parseable_hash(generated)
        sample["generated"] = generated
        sample["pred_answer"] = pred_answer
        sample["correct"] = is_ok
        # no_answer: the FULL fallback chain (marker -> "answer is" ->
        # \boxed{} -> last number) found nothing. Distinct from no_marker.
        sample["no_answer"] = (pred_answer == "")
        sample["no_marker"] = not has_marker
        sample["marker_unparsed"] = has_marker and not marker_parses
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
    ap.add_argument("--ans_root", default="gsm_symbolic_nocot",
                     help="MUST differ from the CoT sibling's tree "
                          "('gsm_symbolic' on this project's synced results, "
                          "'answer_gsm_symbolic' as that script's own "
                          "argparse default) -- never write No-CoT cells "
                          "into the frozen CoT directory.")
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--batch_size", type=int, default=24)
    args = ap.parse_args()

    # Hard guard: never let this No-CoT fork write into (or read a stale
    # cell from) the frozen CoT tree, whichever name that happens to have on
    # this machine.
    _FORBIDDEN_ANS_ROOTS = {"gsm_symbolic", "answer_gsm_symbolic"}
    if args.ans_root in _FORBIDDEN_ANS_ROOTS:
        raise ValueError(
            f"--ans_root={args.ans_root!r} is one of the CoT sibling's known "
            f"output directory names {_FORBIDDEN_ANS_ROOTS}. This No-CoT "
            "fork must write to its own separate tree (default "
            "'gsm_symbolic_nocot') -- refusing to risk overwriting or "
            "silently mixing with frozen CoT cells."
        )

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

    # Existing No-CoT GSM8K prompt, neutral role, plain wording -- same
    # builder as the CoT sibling script, cot=False instead of cot=True. This
    # is the ONLY prompt change in this fork: drops "Let's think step by
    # step." and nothing else.
    templates = select_templates_gsm8k(suite="default", cot=False, wording="plain")
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
        mask_sha256 = sha256_array(mask)
        n_layers_band = len(list(decoder_layer_range(st, en)))
        print(f"\n=== gsm_config={args.gsm_config} alpha={alpha} layers={st}-{en} "
              f"(L={n_layers_band}) mask_sha256={mask_sha256[:12]}... ===")

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
        n_samples = len(updated)
        # Hard check: prefill-only, tail=1 -> expected fires = 0 at alpha=0
        # (an all-zero diff never satisfies _layer_is_steered), else
        # L_band * n_samples * tail_len (tail_len=1, not exposed on this
        # driver's regenerate() call so it is fixed at 1 here).
        tail_len = 1
        expected_fires = 0 if alpha == 0 else n_layers_band * n_samples * tail_len
        print(f"steering_fires: {fires}  (expected {expected_fires})")
        if fires != expected_fires:
            raise RuntimeError(
                f"steering_fires mismatch at alpha={alpha}, layers={st}-{en}: "
                f"got {fires}, expected {expected_fires} "
                f"(L_band={n_layers_band} x n_samples={n_samples} x tail_len={tail_len}). "
                "This means the injection did not fire on the expected sites -- "
                "stop and inspect before trusting this cell's accuracy."
            )

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
                    "n_layers_band": n_layers_band,
                    "mask_path": mask_path,
                    "mask_sha256": mask_sha256,
                    "prompt_template": prompt_template,
                    "prompt_template_sha256": sha256_text(prompt_template),
                    "cot": False,
                    "role": "neutral",
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "batch_size": args.batch_size,
                    "prefill_only": True,
                    "prefill_tail_len": tail_len,
                    "steering_fires": fires,
                    "steering_fires_expected": expected_fires,
                    "n_samples": len(updated),
                    "sample_ids": [s.get("sample_id") for s in updated],
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
                "no_marker", "marker_unparsed", "no_answer", "multi_marker",
                "first_last_disagree", "loop", "truncated",
            ])
            writer.writeheader()
            writer.writerow({
                "model": args.model, "size": args.size, "gsm_config": args.gsm_config,
                "alpha": alpha, "start": st, "end": en,
                "correct": accuracy["correct"], "total": accuracy["total"],
                "accuracy_percentage": accuracy["accuracy_percentage"],
                "steering_fires": fires,
                "no_marker": sum(1 for s in updated if s["no_marker"]),
                "marker_unparsed": sum(1 for s in updated if s["marker_unparsed"]),
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
