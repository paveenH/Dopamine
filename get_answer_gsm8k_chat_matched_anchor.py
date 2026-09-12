#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_answer_gsm8k_chat_matched_anchor.py -- GSM8K No-CoT, chat-template
MATCHED-ANCHOR control condition. SECOND phase of this experiment -- run only
after get_answer_math_chat_matched_anchor.py has completed and its results
have been reviewed (run_gsm8k_chat_matched_anchor.sh enforces this: it
refuses to launch unless all four MATH matched-anchor cells already exist AND
an explicit CONFIRMED=1 is set).

INDEPENDENT of every existing GSM8K tree: the frozen bare-string main line
(run_gsm8k.sh / get_answer_regenerate_gsm8k.py) and the native chat sweep
(run_gsm8k_chat_sweep.sh / get_answer_gsm8k_chat_sweep.py, protocol
"gsm8k-chat-sweep-v1"). None of those is touched, read, or overwritten by
this script.

QUESTION THIS CONTROLS FOR (identical logic to the MATH sibling; see
get_answer_math_chat_matched_anchor.py's docstring for the full rationale).
get_answer_gsm8k_chat_sweep.py's own docstring records the SAME measured
confound on GSM8K: apply_chat_template's Jinja `| trim` strips the "Answer: "
anchor's trailing space, moving the last prefill token from id 220 ' ' (bare)
to id 271 '\\n\\n' (native chat, the assistant header). This script keeps the
full chat template but re-creates the bare injection site under it, isolating
"does the interface matter" from "does the injection site matter".

THIS IS NOT A WORKPOINT SEARCH AND NOT THE NINE-POINT SWEEP. alpha in
{-6, 0, 6, 8} is the SAME fixed four-point family as the MATH sibling (chosen
for cross-task comparability of this NEW family, not re-derived per task).

WHAT IS HELD IDENTICAL to both the frozen bare and native-chat GSM8K cells:
  - the SAME 300-question fixed sample (benchmark/gsm8k_test_sample.json).
  - the SAME neutral, No-CoT, plain-wording prompt BODY
    (template.build_gsm8k_default_suite(cot=False, wording="plain")["neutral"]),
    MODULO the trailing "Answer: " anchor, stripped and re-attached after the
    chat template's assistant header (see chat_matched_anchor_lib.py).
  - the SAME mask file and band: mask/llama3_non_logits/nmd_0.5_11_20_8B.npy,
    layers [11,20).
  - the SAME generation params: max_new_tokens=768, temperature=0.0 (greedy),
    batch_size=24, prefill_only=True, prefill_tail_len=1. GSM8K's OWN budget
    -- NOT MATH's 2048/8.
  - alpha=0 uses the SAME generation code path as every other alpha (an
    all-zero diff matrix via vc.regenerate(diff_matrices=raw_mask*0)), and is
    RE-RUN under the matched-anchor wrapper as this family's OWN baseline.
  - the SAME answer extractor (utils.extract_gsm8k_answer /
    utils.is_correct_gsm8k) for the inline process-state fields; the
    AUTHORITATIVE first_acc/last_acc reading is computed offline by
    RoleAnswer/analyze_chat_matched_anchor.py --task gsm8k.

THE ONLY EXPERIMENTAL VARIABLE, held constant across all four alpha, is the
matched-anchor construction in chat_matched_anchor_lib.py (shared verbatim
with the MATH generator, precisely so the two tasks cannot silently apply
different anchor logic): strip the trailing "Answer: " from the rendered
body, wrap the remainder with apply_chat_template(...,
add_generation_prompt=True), strip a duplicated leading BOS, then re-append
"Answer: " directly after the assistant header. assert_matched_anchor_tail()
FAILS CLOSED before any generation if the last prefill token is not id 220,
a single ASCII space.

Output tree (independent from both the bare and native-chat GSM8K trees):
    components/llama3/answer_mdf_gsm8k_chat_matched_anchor_v1/mdf_<alpha>/
        gsm8k_chat_matched_anchor_8B_answers_11_20.json

Existing output files are NEVER overwritten -- fail closed (die()). There is
no --allow_overwrite escape hatch. All FOUR paths are checked BEFORE the
first cell runs.

All four alpha are driven by ONE launcher invocation, ONE model load, so the
same-machine/same-GPU pairing holds structurally rather than by convention.

@author: GSM8K chat-template matched-anchor control (2026-09-12)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np
from tqdm import tqdm

import utils
from llms import VicundaModel
from template import build_gsm8k_default_suite
from utils import extract_gsm8k_answer, is_correct_gsm8k

import chat_matched_anchor_lib as CMA

PROTOCOL = "gsm8k-chat-matched-anchor-v1"
CONDITION = "Chat Matched-Anchor"
PROMPT_WRAPPER_ID = "llama3-chat-template-matched-anchor-v1"

# Same four-point family as the MATH sibling -- NOT the frozen bare/
# native-chat nine-point set.
EXPECTED_ALPHAS = {-6, 0, 6, 8}
BAND = (11, 20)

# Descriptive-metadata-only note: GSM8K's frozen bare tree carries ALL nine
# doses (GSM8K_DIRS mdf_-8..mdf_8), unlike MATH's five-dose bare tree -- so
# all four of this family's alpha DO have a bare counterpart here.
BARE_ALPHAS = (-8, -6, -4, -2, 0, 2, 4, 6, 8)
NATIVE_CHAT_ALPHAS = (-8, -6, -4, -2, 0, 2, 4, 6, 8)


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser(
        description="GSM8K No-CoT chat-template MATCHED-ANCHOR control "
                    "(independent of both the frozen bare GSM8K main line "
                    "and the native chat sweep)")
    p.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="Hugging Face repo id, per the repo-wide convention.")
    p.add_argument("--size", default="8B")
    p.add_argument("--test_file", required=True,
                   help="benchmark/gsm8k_test_sample.json -- the SAME "
                        "300-question fixed sample the bare/native-chat "
                        "cells use")
    p.add_argument("--mask_path", required=True,
                   help="SAME mask as the bare/native-chat cells: "
                        "mask/llama3_non_logits/nmd_0.5_11_20_8B.npy")
    p.add_argument("--configs", required=True, nargs="+",
                   help="0-11-20 neg6-11-20 6-11-20 8-11-20 (exactly the "
                        "frozen four-point matched-anchor dose set, GSM8K's "
                        "own signed directory-naming convention)")
    p.add_argument("--out_dir", required=True,
                   help="components/llama3/answer_mdf_gsm8k_chat_matched_anchor_v1")
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--max_new_tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--fmt_wording", default="plain", choices=["plain"],
                   help="Fixed to 'plain' -- 'pushy' is out of scope.")
    return p.parse_args()


def main():
    args = parse_args()

    cfgs = utils.parse_configs(args.configs)
    got_alphas = sorted(al for al, _ in cfgs)
    want_alphas = sorted(EXPECTED_ALPHAS)
    if got_alphas != want_alphas:
        die(f"--configs alphas {got_alphas} != this protocol's frozen "
            f"four-point dose set {want_alphas}. Exact match required.")

    samples = utils.load_json(args.test_file)
    n = len(samples)
    print(f"Loaded {n} GSM8K samples from {args.test_file}")

    templates = build_gsm8k_default_suite(cot=False, wording=args.fmt_wording)
    neutral_tmpl = templates["neutral"]
    prompt_body_sha256 = hashlib.sha256(
        neutral_tmpl.encode("utf-8")).hexdigest()

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    if vc.tokenizer.padding_side != "left":
        die(f"tokenizer.padding_side is {vc.tokenizer.padding_side!r}, "
            "expected 'left'.")

    chat_template_str = getattr(vc.tokenizer, "chat_template", None) or ""
    chat_template_hash = hashlib.sha256(
        chat_template_str.encode("utf-8")).hexdigest()

    raw_mask = np.load(args.mask_path)
    mask_sha256 = hashlib.sha256(open(args.mask_path, "rb").read()).hexdigest()
    os.makedirs(args.out_dir, exist_ok=True)

    n_decoder = len(vc._find_decoder_layers())
    if raw_mask.shape[0] != n_decoder:
        die(f"mask has {raw_mask.shape[0]} rows but the model has {n_decoder} "
            "decoder layers; regenerate() needs one row per layer (zero "
            "rows outside the band). Do NOT slice the mask to the band.")
    nz_rows = sorted(int(i) for i in np.nonzero(np.any(raw_mask != 0, axis=1))[0])
    want_rows = sorted(utils.decoder_layer_range(*BAND))
    if nz_rows != want_rows:
        die(f"mask non-zero rows {nz_rows} != decoder_layer_range{BAND} "
            f"{want_rows}; the mask does not match this protocol's band.")

    try:
        devs = sorted({str(p.device) for p in vc.model.parameters()})
    except Exception:
        devs = []
    device_note = {"host": platform.node(),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "param_devices": devs,
                   "sharded": len(devs) > 1}

    # ---- prompt construction: the matched-anchor mechanism -----------------
    bare_prompts_with_anchor = [
        neutral_tmpl.format(context=s["question"]) for s in samples]
    matched_prompts = []
    for i, full_body in enumerate(bare_prompts_with_anchor):
        body_no_anchor = CMA.strip_trailing_answer_anchor(
            full_body, f"sample_idx={i}")
        wrapped = CMA.build_matched_anchor_prompt(vc, body_no_anchor)
        if i == 0:
            CMA.assert_chat_template_effective(vc, body_no_anchor, wrapped)
        CMA.assert_no_double_bos(vc, wrapped, f"sample_idx={i}")
        final_text = wrapped + CMA.ANSWER_ANCHOR
        CMA.assert_matched_anchor_tail(vc, final_text, f"sample_idx={i}")
        matched_prompts.append(final_text)

    inject_tail_matched = CMA.tail_ids(vc, matched_prompts[0])
    inject_tail_bare = CMA.tail_ids(vc, bare_prompts_with_anchor[0])
    print(f"injection tail (matched-anchor) = {inject_tail_matched}")
    print(f"injection tail (bare reference) = {inject_tail_bare}")
    if inject_tail_matched[-1]["id"] != inject_tail_bare[-1]["id"]:
        die("matched-anchor construction did not reproduce the bare "
            "injection token even though assert_matched_anchor_tail passed "
            "per-sample -- this should be impossible; refusing to continue.")
    print("[ok] matched-anchor last prefill token == bare last prefill token "
          f"(id={inject_tail_matched[-1]['id']!r}, "
          f"text={inject_tail_matched[-1]['text']!r})")

    prompt_sha256 = hashlib.sha256(
        "\n".join(matched_prompts).encode("utf-8")).hexdigest()
    bare_prompt_sha256 = hashlib.sha256(
        "\n".join(bare_prompts_with_anchor).encode("utf-8")).hexdigest()
    test_file_sha256 = hashlib.sha256(
        open(args.test_file, "rb").read()).hexdigest()
    sample_order_sha256 = hashlib.sha256(
        "\n".join(s["question"] for s in samples).encode("utf-8")).hexdigest()

    print(f"prompt_body_sha256={prompt_body_sha256}")
    print(f"chat_template_hash={chat_template_hash}")
    print(f"matched_anchor_prompt_sha256={prompt_sha256}")
    print(f"bare_prompt_sha256={bare_prompt_sha256}")
    print(f"test_file_sha256={test_file_sha256}")
    print(f"sample_order_sha256={sample_order_sha256}")
    print("First matched-anchor prompt (repr, truncated to 400 chars):")
    print(repr(matched_prompts[0][:400]))
    print("... (last 60 chars) ...")
    print(repr(matched_prompts[0][-60:]))

    out_paths = {}
    for alpha, (ls, le) in cfgs:
        if (ls, le) != BAND:
            die(f"layer band {(ls, le)} != {BAND} -- this protocol is "
                "frozen at the bare/native-chat cells' own band.")
        out_paths[alpha] = os.path.join(
            args.out_dir, f"mdf_{alpha}",
            f"gsm8k_chat_matched_anchor_{args.size}_answers_{ls}_{le}.json")

    existing = [op for op in out_paths.values() if os.path.exists(op)]
    if existing:
        die("the following matched-anchor output path(s) already exist -- "
            "refusing to run ANY cell of this four-point family: "
            + ", ".join(existing) +
            ". Delete them deliberately first if a re-run is truly intended.")

    n_layers = len(utils.decoder_layer_range(*BAND))

    for alpha, (ls, le) in cfgs:
        out_path = out_paths[alpha]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        diff = raw_mask * alpha
        vc.steering_fire_count(reset=True)

        gen = []
        for i in tqdm(range(0, len(matched_prompts), args.batch_size),
                      desc=f"gsm8k-chat-matched-anchor a={alpha}"):
            batch = matched_prompts[i: i + args.batch_size]
            gen.extend(vc.regenerate(
                batch,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                diff_matrices=list(diff),
                batch_size=args.batch_size,
                return_metadata=True,
            ))

        if len(gen) != n:
            die(f"generation returned {len(gen)} rows for {n} prompts at "
                f"alpha={alpha}; zip() would silently drop "
                f"{abs(len(gen) - n)} sample(s).")

        fires = vc.steering_fire_count()
        expect = 0 if alpha == 0 else n_layers * n
        if fires != expect:
            die(f"steering_fires {fires} != {expect} (L={n_layers}, n={n}, "
                f"alpha={alpha}); the intervention is unverified, so the "
                "cell is not usable.")

        rows = []
        for s, g in zip(samples, gen):
            text = g["text"]
            pred_answer = extract_gsm8k_answer(text)
            correct = is_correct_gsm8k(pred_answer, s["answer"])
            rows.append({
                "question": s["question"],
                "answer": s["answer"],
                "generated_neutral": text,
                "pred_answer_neutral": pred_answer,
                "correct_neutral": correct,
                "generated_token_count": g["generated_token_count"],
                "stop_reason": g["stop_reason"],
            })

        n_correct = sum(r["correct_neutral"] for r in rows)
        accuracy_pct = round(n_correct / n * 100, 2) if n else 0.0

        json.dump({
            "meta": {
                "protocol": PROTOCOL,
                "condition": CONDITION,
                "model": "llama3", "size": args.size,
                "alpha": alpha, "layer_start": ls, "layer_end": le, "L": n_layers,
                "mask_path": args.mask_path, "mask_sha256": mask_sha256,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature, "top_p": args.top_p,
                "batch_size": args.batch_size,
                "suite": "default", "cot": False, "fmt_wording": args.fmt_wording,
                "role": "neutral",
                "prefill_only": True, "prefill_tail_len": 1,
                "prompt_wrapper_id": PROMPT_WRAPPER_ID,
                "chat_template_applied": True,
                "chat_template_hash": chat_template_hash,
                "prompt_body_sha256": prompt_body_sha256,
                "prompt_sha256": prompt_sha256,
                "bare_prompt_sha256": bare_prompt_sha256,
                "test_file_sha256": test_file_sha256,
                "sample_order_sha256": sample_order_sha256,
                "steering_fires": fires,
                "padding_side": vc.tokenizer.padding_side,
                "injection_anchor_text": CMA.ANSWER_ANCHOR,
                "injection_tail_token_id": inject_tail_matched[-1]["id"],
                "injection_tail_token_text": inject_tail_matched[-1]["text"],
                "injection_tail_tokens_matched_anchor": inject_tail_matched,
                "injection_tail_tokens_bare_reference": inject_tail_bare,
                "injection_position": (
                    "The literal 'Answer: ' anchor is appended directly "
                    "after the chat template's assistant generation header "
                    "(<|start_header_id|>assistant<|end_header_id|>\\n\\n), "
                    "NOT left inside the (Jinja-trimmed) user turn. "
                    "Prefill-only tail=1 steering therefore injects into the "
                    "anchor's trailing space -- token id "
                    f"{inject_tail_matched[-1]['id']} -- reproducing the "
                    "bare condition's injection site under the full chat "
                    "template."),
                "injection_site_matches_bare": (
                    inject_tail_matched[-1]["id"] == inject_tail_bare[-1]["id"]
                    and inject_tail_matched[-1]["text"]
                    == inject_tail_bare[-1]["text"]),
                "n": n,
                "accuracy_inline_pct": accuracy_pct,
                "provenance": device_note,
                "not_a_workpoint_search": True,
                "g_prefill_measured": False,
                "g_prefill_omitted_reason": (
                    "signal hook is bs=1-only; measuring it would change the "
                    "generation path this cell exists to hold fixed"),
                "bare_counterpart": (
                    f"components/llama3/answer_mdf_gsm8k/mdf_{alpha}/"
                    f"gsm8k_{args.size}_answers_20_{ls}_{le}.json"),
                "bare_counterpart_exists": alpha in BARE_ALPHAS,
                "native_chat_counterpart": (
                    f"components/llama3/answer_mdf_gsm8k_chat_v1/mdf_{alpha}/"
                    f"gsm8k_chat_{args.size}_answers_{ls}_{le}.json"),
                "native_chat_counterpart_exists": alpha in NATIVE_CHAT_ALPHAS,
                "comparison_note": (
                    "Comparisons against the bare and native-chat trees are "
                    "DESCRIPTIVE ONLY and are never pooled into this "
                    "family's own Holm m=3 statistics (three non-zero alpha "
                    "vs this family's OWN alpha=0)."),
            },
            "data": rows,
        }, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  wrote {out_path}  steering_fires={fires}  "
              f"inline_acc={accuracy_pct}%")

    print("\nAll GSM8K matched-anchor cells finished.")
    print("Next: sync this OUT DIR to the offline analysis workspace "
          "~/Documents/RSNResult/RoleAnswer/, then, FROM THAT BOX:")
    print("      python3.10 analyze_chat_matched_anchor.py --task gsm8k")


if __name__ == "__main__":
    main()
