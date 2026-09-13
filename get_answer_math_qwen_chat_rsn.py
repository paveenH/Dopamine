#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
get_answer_math_qwen_chat_rsn.py -- MATH No-CoT, Qwen2.5-7B-Instruct, Native
Chat RSN four-point verification (alpha in {-8, 0, +6, +8}). INDEPENDENT of
the frozen bare Qwen MATH line (run_math_qwen25.sh /
get_answer_regenerate_math.py) and of the Llama MATH chat sweep
(get_answer_math_chat_sweep.py) -- none is touched, read, or overwritten.

Structural sibling of get_answer_gsm8k_qwen_chat_rsn.py, following its
tokenizer/injection-site/fail-closed conventions but with MATH's own template,
extractor, generation budget, batch size, and output tree -- reusing GSM8K's
budget here would truncate MATH solutions and manufacture an extraction floor
(the same caveat run_math_qwen25.sh records for the bare Qwen MATH line).

THIS IS NOT A NINE-POINT SWEEP AND NOT A WORKPOINT SEARCH. alpha in
{-8, 0, +6, +8} is a FIXED four-point verification family, per the task
brief.

WHAT IS HELD IDENTICAL to the frozen bare Qwen MATH cells (run_math_qwen25.sh)
so that only the chat wrapper differs:
  - the SAME 300-problem fixed sample: benchmark/math_test_sample.json,
    truncated to the first n_samples=300 EXACTLY as
    get_answer_regenerate_math.py does (all_samples[:n])
  - the SAME neutral, No-CoT prompt BODY (template.build_math_suite(cot=False)
    ["neutral"])
  - the SAME mask file and band: mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy,
    layers [16,22), L=6
  - the SAME generation params: max_new_tokens=2048, temperature=0.0 (greedy),
    batch_size=8, prefill_only=True, prefill_tail_len=1
  - alpha=0 uses the SAME generation code path as every other alpha (an
    all-zero diff matrix via vc.regenerate(diff_matrices=raw_mask*0)), but is
    RE-RUN under the chat wrapper rather than reused from the bare baseline

ANSWER EXTRACTION IS THE CENTRALIZED utils FUNCTION, NOT A LOCAL COPY.
utils.extract_boxed / utils.extract_math_answer / utils.is_correct_math are
imported and used verbatim. Inline fields are process-state ONLY; the
AUTHORITATIVE reading is the offline first_acc (first \boxed{}) via the
frozen all_boxed / norm_math / fallback_math chain in
RoleAnswer/analyze_first_last_acc.py.

THE ONLY EXPERIMENTAL VARIABLE, held constant across all four alpha, is that
the identical rendered prompt STRING is wrapped with

    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )

with a duplicated leading BOS stripped before vc.regenerate() tokenizes it.
NO assistant-side "Answer: " anchor is added and NO matched-anchor
construction is used -- plain native chat, per the task brief.

INJECTION TOKEN IS READ OUT AT RUNTIME, NEVER ASSUMED -- measured on the real
Qwen tokenizer for this exact prompt, not assumed to equal any figure
recorded elsewhere in this repo for a different prompt or model.

Output tree (independent from the bare answer_math tree):
    components/qwen2.5/answer_math_chat_v1/mdf_<alpha>/
        math_chat_7B_16_22.json

Existing output files are NEVER overwritten -- fail closed (die()). There is
no --allow_overwrite escape hatch. All four paths are checked BEFORE the
first cell runs.

@author: Qwen2.5-7B Native Chat RSN four-point verification (2026-09-13)
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
from template import build_math_suite
from utils import extract_boxed, extract_math_answer, is_correct_math

PROTOCOL = "math-qwen-chat-rsn-v1"
PROMPT_WRAPPER_ID = "qwen2.5-chat-template-v1"

EXPECTED_ALPHAS = {-8, 0, 6, 8}
BAND = (16, 22)


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser(
        description="MATH No-CoT Qwen2.5 Native Chat RSN four-point "
                    "verification (independent of the frozen bare Qwen line)")
    p.add_argument("--model_dir", default="Qwen/Qwen2.5-7B-Instruct",
                   help="Hugging Face repo id, per the repo-wide convention.")
    p.add_argument("--size", default="7B")
    p.add_argument("--test_file", required=True,
                   help="benchmark/math_test_sample.json -- the SAME sample "
                        "file the bare Qwen MATH cells use")
    p.add_argument("--mask_path", required=True,
                   help="SAME mask as the bare Qwen cells: "
                        "mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy")
    p.add_argument("--configs", required=True, nargs="+",
                   help="e.g. neg8-16-22 0-16-22 6-16-22 8-16-22 (alpha "
                        "restricted to the frozen four-point set)")
    p.add_argument("--out_dir", required=True,
                   help="components/qwen2.5/answer_math_chat_v1")
    p.add_argument("--n_samples", type=int, default=300,
                   help="Truncation applied EXACTLY as "
                        "get_answer_regenerate_math.py does (all_samples[:n]).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.9,
                   help="Matches get_answer_regenerate_math.py's own default. "
                        "Inert under greedy; recorded for metadata parity.")
    return p.parse_args()


def strip_leading_bos(vc, text: str) -> str:
    bos = getattr(vc.tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        return text[len(bos):]
    return text


def assert_no_double_bos(vc, text: str, label: str) -> None:
    bos_id = getattr(vc.tokenizer, "bos_token_id", None)
    ids = vc.tokenizer(text, add_special_tokens=True)["input_ids"]
    if bos_id is not None and len(ids) >= 2 and ids[:2] == [bos_id, bos_id]:
        die(f"double BOS in the {label} chat prompt (head={ids[:4]}) -- "
            "strip_leading_bos did not remove the chat template's own "
            "serialized BOS before this add_special_tokens=True call.")


def assert_chat_template_effective(vc, bare_prompt: str, wrapped: str) -> None:
    if wrapped == bare_prompt:
        die("chat template had NO effect on the first sample (wrapped text is "
            "byte-identical to the bare prompt) -- apply_chat_template did not "
            "actually run, or the tokenizer has no chat_template.")
    if not getattr(vc.tokenizer, "chat_template", None):
        die("tokenizer.chat_template is empty/None -- cannot apply a chat "
            "template that does not exist.")
    if "<|im_start|>assistant" not in wrapped:
        die("expected Qwen assistant turn marker '<|im_start|>assistant' not "
            "found in the first wrapped prompt -- add_generation_prompt=True "
            "did not produce the expected Qwen chat header.")


def main():
    args = parse_args()

    cfgs = utils.parse_configs(args.configs)
    got_alphas = sorted(al for al, _ in cfgs)
    want_alphas = sorted(EXPECTED_ALPHAS)
    if got_alphas != want_alphas:
        die(f"--configs alphas {got_alphas} != this protocol's frozen "
            f"four-point verification set {want_alphas}. Exact match "
            "required: a non-integer dose, a missing dose, a duplicate, or "
            "an extra dose all land here.")

    all_samples = utils.load_json(args.test_file)
    n = min(len(all_samples), args.n_samples)
    samples = all_samples[:n]
    if n != args.n_samples:
        die(f"test file holds only {len(all_samples)} problems but "
            f"--n_samples={args.n_samples}; the bare cells were built on "
            f"{args.n_samples}, so a shorter sample would not be pairable.")
    print(f"Loaded {n} MATH problems from {args.test_file}")

    templates = build_math_suite(cot=False)
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
            "decoder layers; regenerate() needs one row per layer (zero rows "
            "outside the band). Do NOT slice the mask to the band.")
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

    bare_prompts = [neutral_tmpl.format(context=s["question"]) for s in samples]
    chat_prompts = []
    for i, p in enumerate(bare_prompts):
        wrapped = vc.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        wrapped = strip_leading_bos(vc, wrapped)
        if i == 0:
            assert_chat_template_effective(vc, p, wrapped)
        assert_no_double_bos(vc, wrapped, f"sample_idx={i}")
        chat_prompts.append(wrapped)

    def _tail_ids(text, k=4):
        ids = vc.tokenizer(text, add_special_tokens=True)["input_ids"][-k:]
        return [{"id": int(i), "text": vc.tokenizer.decode([i])} for i in ids]

    inject_tail_chat = _tail_ids(chat_prompts[0])
    inject_tail_bare = _tail_ids(bare_prompts[0])
    print(f"injection tail (chat) = {inject_tail_chat}")
    print(f"injection tail (bare) = {inject_tail_bare}")
    if inject_tail_chat[-1]["id"] == inject_tail_bare[-1]["id"]:
        print("[note] chat and bare end on the SAME token id -- the anchor "
              "survived the chat template's trim.")
    else:
        print("[note] chat and bare end on DIFFERENT token ids -- the chat "
              "template trimmed the 'Answer: ' anchor, so the injection site "
              "moved to the assistant turn marker. Recorded in meta.")

    terminators = list(vc.terminators)
    terminator_texts = [vc.tokenizer.decode([t]) for t in terminators]
    print(f"terminators (ids) = {terminators}")
    print(f"terminators (text) = {terminator_texts}")

    prompt_sha256 = hashlib.sha256(
        "\n".join(chat_prompts).encode("utf-8")).hexdigest()
    bare_prompt_sha256 = hashlib.sha256(
        "\n".join(bare_prompts).encode("utf-8")).hexdigest()
    test_file_sha256 = hashlib.sha256(
        open(args.test_file, "rb").read()).hexdigest()
    sample_order_sha256 = hashlib.sha256(
        "\n".join(s["question"] for s in samples).encode("utf-8")).hexdigest()

    print(f"prompt_body_sha256={prompt_body_sha256}")
    print(f"chat_template_hash={chat_template_hash}")
    print(f"chat_prompt_sha256={prompt_sha256}")
    print(f"bare_prompt_sha256={bare_prompt_sha256}")
    print(f"test_file_sha256={test_file_sha256}")
    print(f"sample_order_sha256={sample_order_sha256}")
    print("First wrapped prompt (repr, truncated to 400 chars):")
    print(repr(chat_prompts[0][:400]))

    out_paths = {}
    for alpha, (ls, le) in cfgs:
        if (ls, le) != BAND:
            die(f"layer band {(ls, le)} != {BAND} -- this protocol is frozen at "
                "the bare Qwen cells' own band; do not pass a different band "
                "under this protocol name.")
        alpha_tag = f"neg{abs(alpha)}" if alpha < 0 else str(alpha)
        out_paths[alpha] = os.path.join(
            args.out_dir, f"mdf_{alpha_tag}",
            f"math_chat_{args.size}_{ls}_{le}.json")
    for alpha, op in out_paths.items():
        if os.path.exists(op):
            die(f"{op} already exists -- refusing to overwrite a frozen "
                "chat-RSN cell. Delete it deliberately first if a re-run is "
                "truly intended.")

    n_layers = len(utils.decoder_layer_range(*BAND))

    for alpha, (ls, le) in cfgs:
        out_path = out_paths[alpha]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        diff = raw_mask * alpha
        vc.steering_fire_count(reset=True)

        gen = []
        for i in tqdm(range(0, len(chat_prompts), args.batch_size),
                      desc=f"qwen-math-chat-rsn a={alpha}"):
            batch = chat_prompts[i: i + args.batch_size]
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
                f"alpha={alpha}); the intervention is unverified, so the cell "
                "is not usable.")

        rows = []
        n_correct = 0
        for s, g in zip(samples, gen):
            text = g["text"]
            gold_boxed = extract_boxed(s["answer"])
            pred = extract_math_answer(text)
            ok = is_correct_math(pred, gold_boxed) if gold_boxed else False
            n_correct += int(ok)
            rows.append({
                "question": s["question"],
                "gold_solution": s["answer"],
                "gold_answer": gold_boxed,
                "level": s.get("level", ""),
                "type": s.get("type", ""),
                "generated": text,
                "pred_answer": pred,
                "correct": ok,
                "generated_token_count": g["generated_token_count"],
                "stop_reason": g["stop_reason"],
            })

        accuracy_pct = round(n_correct / n * 100, 2) if n else 0.0

        json.dump({
            "meta": {
                "protocol": PROTOCOL,
                "task": "math",
                "model": "qwen2.5", "size": args.size,
                "alpha": alpha, "layer_start": ls, "layer_end": le,
                "L": n_layers,
                "mask_path": args.mask_path, "mask_sha256": mask_sha256,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature, "top_p": args.top_p,
                "batch_size": args.batch_size,
                "n_samples": args.n_samples,
                "suite": "math", "cot": False,
                "role": "neutral",
                "prefill_only": True, "prefill_tail_len": 1,
                "prompt_wrapper_id": PROMPT_WRAPPER_ID,
                "chat_template_applied": True,
                "chat_template_hash": chat_template_hash,
                "prompt_template": neutral_tmpl,
                "prompt_body_sha256": prompt_body_sha256,
                "prompt_sha256": prompt_sha256,
                "bare_prompt_sha256": bare_prompt_sha256,
                "test_file_sha256": test_file_sha256,
                "sample_order_sha256": sample_order_sha256,
                "steering_fires": fires,
                "padding_side": vc.tokenizer.padding_side,
                "injection_tail_tokens_chat": inject_tail_chat,
                "injection_tail_tokens_bare_reference": inject_tail_bare,
                "injection_site_moved_vs_bare": (
                    inject_tail_chat[-1]["id"] != inject_tail_bare[-1]["id"]),
                "terminators_ids": terminators,
                "terminators_text": terminator_texts,
                "n": n,
                "accuracy_inline_pct": accuracy_pct,
                "accuracy_inline_note": (
                    "process-state only; offline first_acc is authoritative"),
                "provenance": device_note,
                "not_a_workpoint_search": True,
                "verification_family": "four-point {-8,0,+6,+8}, NOT the "
                    "full nine-point bare/Llama-chat dose set",
                "g_prefill_measured": False,
                "g_prefill_omitted_reason": (
                    "signal hook is bs=1-only; measuring it would change the "
                    "generation path this cell exists to hold fixed"),
                "bare_counterpart": (
                    f"components/qwen2.5/answer_math/"
                    f"mdf_{'neg' + str(abs(alpha)) if alpha < 0 else alpha}/"
                    f"math_{args.size}_{ls}_{le}.json"),
                "bare_counterpart_exists": True,
                "bare_coverage_note": (
                    "Bare-vs-chat is DESCRIPTIVE only and is never pooled "
                    "with the chat family's Holm m=3 statistics."),
            },
            "data": rows,
        }, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  wrote {out_path}  steering_fires={fires}  "
              f"inline_acc={accuracy_pct}%")

    print("\nAll MATH Qwen Native Chat RSN cells finished.")
    print("Next: sync this OUT DIR to the offline analysis workspace "
          "~/Documents/RSNResult/RoleAnswer/ then run the unified offline "
          "analyzer from there.")


if __name__ == "__main__":
    main()
