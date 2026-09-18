#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSM8K role-abstention prompt v2 -- behavioral pilot, NO STEERING.

Standalone script. Does NOT import get_answer_regenerate_gsm8k.py, does NOT
call vc.regenerate(), does NOT touch template.py. Uses VicundaModel.generate()
only (no diff_matrices), matching the "no-hook" contract of that method.

Two prompts, identical except one word (expert / non expert):

    Solve the following math problem.
    Question: {question}
    Now you are an honest expert.
    If you can solve the problem, provide your final numeric answer after '####'.
    Otherwise, answer exactly "I am not sure" and do not provide a numeric answer.
    Answer:

Bare-string, No-CoT, greedy, no chat template, no few-shot, no stop_strings, no
output truncation/post-processing. We want to see whether the model itself
stops after abstaining.

Data: the SAME fixed 300-question GSM8K benchmark file used by the frozen
GSM8K pipeline (benchmark/gsm8k_test_sample.json), same order. The pairing
identity is checked against the manifold split's frozen question-text digest
(sha256 over "\n".join(question texts), same convention as
manifold/split_manifest.py::question_digest) -- NOT computed here, only
checked, since that file lives server-side.

@author: paveenhuang
"""

import os
import re
import gc
import csv
import json
import hashlib
import argparse
from typing import List

import torch
from tqdm import tqdm

from llms import VicundaModel
import utils


# Frozen pairing digest this pilot must reproduce (300 GSM8K questions, same
# order as every other GSM8K experiment in this repo). Computed the same way
# as manifold/split_manifest.py::question_digest: sha256("\n".join(question
# texts)). If this does not match, the benchmark file or its order has moved
# and this pilot is NOT comparable to any other GSM8K result in the repo.
EXPECTED_QUESTION_DIGEST = "64af9b38dac72f35ee1efa97bea8026e0c4d97b35730e76361eea9581a23cf35"


ROLE_EXPERT = "an honest expert"
ROLE_NON_EXPERT = "an honest non expert"

PROMPT_TEMPLATE = (
    "Solve the following math problem.\n"
    "Question: {question}\n"
    "Now you are {role}.\n"
    "If you can solve the problem, provide your final numeric answer after '####'.\n"
    "Otherwise, answer exactly \"I am not sure\" and do not provide a numeric answer.\n"
    "Answer: "
)


def build_prompt(question: str, role: str) -> str:
    return PROMPT_TEMPLATE.format(question=question, role=role)


def question_text_digest(samples: List[dict]) -> str:
    """sha256 over ordered question texts -- identical convention to
    manifold/split_manifest.py::question_digest."""
    qs = [s["question"] for s in samples]
    return hashlib.sha256("\n".join(qs).encode()).hexdigest()


def run_check(samples: List[dict]) -> None:
    digest = question_text_digest(samples)
    print(f"[check] n_questions = {len(samples)}")
    print(f"[check] question_text_sha256 = {digest}")
    expected = EXPECTED_QUESTION_DIGEST
    if digest == expected:
        print("[check] MATCHES the frozen manifold split digest -> same 300 "
              "questions, same order as the rest of the repo's GSM8K work.")
    else:
        print("[check] DOES NOT MATCH the digest supplied at freeze time "
              f"({expected}). Do NOT proceed until this is resolved -- either "
              "the benchmark file or its order differs from every other GSM8K "
              "experiment in this repo, which would make this pilot "
              "incomparable.")
        raise SystemExit(1)
    assert len(samples) == 300, f"expected exactly 300 samples, got {len(samples)}"

    p_expert = build_prompt(samples[0]["question"], ROLE_EXPERT)
    p_non = build_prompt(samples[0]["question"], ROLE_NON_EXPERT)
    # Both prompts come from the SAME PROMPT_TEMPLATE.format() call, so the
    # only way they can differ is the `role` substitution -- verify that
    # directly (a prefix/suffix scan is unreliable here since ROLE_NON_EXPERT
    # contains ROLE_EXPERT's tail as a substring, "...non expert" vs
    # "...expert", which corrupts a naive backward suffix match).
    assert p_expert == PROMPT_TEMPLATE.format(question=samples[0]["question"], role=ROLE_EXPERT)
    assert p_non == PROMPT_TEMPLATE.format(question=samples[0]["question"], role=ROLE_NON_EXPERT)
    assert p_expert.replace(ROLE_EXPERT, ROLE_NON_EXPERT, 1) == p_non, \
        "substituting the role phrase does not turn one prompt into the other"
    assert p_expert.count(ROLE_EXPERT) == 1 and p_non.count(ROLE_NON_EXPERT) == 1
    print("[check] expert/non-expert prompts are both instances of one shared "
          f"PROMPT_TEMPLATE, differing only by the role substitution "
          f"({ROLE_EXPERT!r} vs {ROLE_NON_EXPERT!r}).")
    print("[check] sample prompt (expert, first question):")
    print("-" * 60)
    print(p_expert)
    print("-" * 60)


# ───────────────────── Marker / abstention parsing (offline-safe, no fallback) ─────

HASH_RE = re.compile(r"####\s*([+-]?[\d,]+\.?\d*)")
ABSTAIN_EXACT = "I am not sure"


def all_hash_markers(text: str) -> List[str]:
    """All '#### <number>' matches, in order. No fallback to 'last number in
    text' -- that fallback (utils.extract_gsm8k_answer's tail) would silently
    score reasoning-without-a-marker as a valid submission, which is exactly
    the failure mode this pilot must avoid."""
    return [m.replace(",", "") for m in HASH_RE.findall(text)]


def is_strict_abstention(generated: str) -> bool:
    """Strict: the ENTIRE generated reply, stripped, is exactly the phrase."""
    return generated.strip() == ABSTAIN_EXACT


def contains_abstain_phrase(generated: str) -> bool:
    """Diagnostic only (substring hit) -- NOT counted as abstention."""
    return ABSTAIN_EXACT in generated


def classify_sample(generated: str, gold: str) -> dict:
    markers = all_hash_markers(generated)
    has_marker = len(markers) > 0
    strict_abstain = is_strict_abstention(generated)
    contains_phrase = contains_abstain_phrase(generated)
    # "abstained then kept going and gave a number": phrase present but is NOT
    # a strict (whole-reply) abstention, AND at least one #### marker exists.
    abstain_then_answered = contains_phrase and (not strict_abstain) and has_marker
    # A #### marker co-occurring with the abstain phrase anywhere in the text
    # can never be counted as a clean abstention, regardless of strictness.
    marker_and_phrase_cooccur = contains_phrase and has_marker

    pred_first = markers[0] if markers else None
    correct_first = (
        utils.is_correct_gsm8k(pred_first, gold) if pred_first is not None else False
    )

    return {
        "n_markers": len(markers),
        "has_marker": has_marker,
        "pred_first_hash": pred_first,
        "correct_first_hash": correct_first,
        "strict_abstain": strict_abstain,
        "contains_abstain_phrase": contains_phrase,
        "abstain_then_answered": abstain_then_answered,
        "marker_and_phrase_cooccur": marker_and_phrase_cooccur,
    }


# ───────────────────── Generation ─────────────────────

def run_role(
    vc: VicundaModel,
    samples: List[dict],
    role_label: str,
    role_phrase: str,
    batch_size: int,
    max_new_tokens: int,
) -> List[dict]:
    prompts = [build_prompt(s["question"], role_phrase) for s in samples]
    outputs: List[str] = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(prompts), batch_size), total=n_batches,
                  desc=f"gsm8k-abstention-v2 [{role_label}]"):
        batch = prompts[i:i + batch_size]
        out = vc.generate(
            batch,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=0.9,
            batch_size=batch_size,
        )
        outputs.extend(out)

    records = []
    for sample, prompt, generated in zip(samples, prompts, outputs):
        rec = {
            "question_idx": sample.get("question_idx", None),
            "question": sample["question"],
            "gold_answer": sample["answer"],
            "role": role_label,
            "prompt": prompt,
            "generated": generated,
        }
        rec.update(classify_sample(generated, sample["answer"]))
        records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser(
        description="GSM8K role-abstention prompt v2 pilot (no steering)")
    ap.add_argument("--model", type=str, required=True,
                     help="Short model tag, e.g. llama3 / qwen2.5 (output path only)")
    ap.add_argument("--model_dir", type=str, required=True,
                     help="HF repo id or local snapshot path")
    ap.add_argument("--size", type=str, required=True)
    ap.add_argument("--test_file", type=str, required=True,
                     help="Path (relative to --base_dir) to gsm8k_test_sample.json")
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--ans_file", type=str, default="answer_gsm8k_role_abstention_v2")
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--check", action="store_true",
                     help="Verify pairing digest + prompt diff, then exit. No model load.")
    global args
    args = ap.parse_args()

    data_path = os.path.join(args.base_dir, args.test_file)
    samples = utils.load_json(data_path)
    print(f"Loaded {len(samples)} GSM8K samples from {data_path}")

    if args.check:
        run_check(samples)
        return

    # Always verify identity before generating, even outside --check.
    digest = question_text_digest(samples)
    expected = EXPECTED_QUESTION_DIGEST
    if digest != expected:
        raise SystemExit(
            f"[FATAL] question_text_sha256={digest} does not match the frozen "
            f"pairing digest {expected}. Refusing to generate -- this benchmark "
            "file/order is not the one every other GSM8K experiment uses."
        )
    assert len(samples) == 300, f"expected 300 samples, got {len(samples)}"

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    out_root = os.path.join(args.base_dir, args.model, args.ans_file)
    os.makedirs(out_root, exist_ok=True)

    all_records = {}
    stats = {}
    with torch.no_grad():
        for role_label, role_phrase in (("expert", ROLE_EXPERT),
                                         ("non_expert", ROLE_NON_EXPERT)):
            recs = run_role(vc, samples, role_label, role_phrase,
                             batch_size=args.batch_size,
                             max_new_tokens=args.max_new_tokens)
            all_records[role_label] = recs

            n = len(recs)
            n_strict = sum(r["strict_abstain"] for r in recs)
            n_marker = sum(r["has_marker"] for r in recs)
            n_no_marker = n - n_marker
            n_correct_first = sum(r["correct_first_hash"] for r in recs)
            n_abstain_then_answered = sum(r["abstain_then_answered"] for r in recs)
            n_phrase = sum(r["contains_abstain_phrase"] for r in recs)
            n_cooccur = sum(r["marker_and_phrase_cooccur"] for r in recs)
            stats[role_label] = {
                "n": n,
                "strict_abstain_rate": round(n_strict / n * 100, 2),
                "first_hash_acc_pct": round(n_correct_first / n * 100, 2),
                "no_marker_rate_pct": round(n_no_marker / n * 100, 2),
                "abstain_then_answered_rate_pct": round(n_abstain_then_answered / n * 100, 2),
                "contains_abstain_phrase_rate_pct": round(n_phrase / n * 100, 2),
                "marker_and_phrase_cooccur_rate_pct": round(n_cooccur / n * 100, 2),
            }
            print(f"[{role_label}] n={n} strict_abstain={n_strict} "
                  f"first#### acc={stats[role_label]['first_hash_acc_pct']}% "
                  f"no_marker={n_no_marker} "
                  f"abstain_then_answered={n_abstain_then_answered}")

    out_path = os.path.join(out_root, f"gsm8k_role_abstention_v2_{args.size}.json")
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump(
            {
                "meta": {
                    "protocol": "gsm8k-role-abstention-v2",
                    "model": args.model,
                    "model_dir": args.model_dir,
                    "size": args.size,
                    "n_samples": len(samples),
                    "question_text_sha256": digest,
                    "max_new_tokens": args.max_new_tokens,
                    "batch_size": args.batch_size,
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "use_chat": False,
                    "cot": False,
                    "generation_call": "VicundaModel.generate (no diff_matrices, no steering)",
                    "prompt_template": PROMPT_TEMPLATE,
                    "role_expert_phrase": ROLE_EXPERT,
                    "role_non_expert_phrase": ROLE_NON_EXPERT,
                },
                "stats": stats,
                "data": all_records,
            },
            fw, ensure_ascii=False, indent=2,
        )
    print("Saved ->", out_path)

    csv_path = os.path.join(out_root, f"summary_gsm8k_role_abstention_v2_{args.model}_{args.size}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "size", "role"] + list(next(iter(stats.values())).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for role_label, s in stats.items():
            writer.writerow({"model": args.model, "size": args.size, "role": role_label, **s})
    print("Saved ->", csv_path)

    del all_records
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
