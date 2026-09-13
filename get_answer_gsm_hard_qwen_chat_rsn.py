#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
get_answer_gsm_hard_qwen_chat_rsn.py -- GSM-Hard No-CoT, Qwen2.5-7B-Instruct,
Native Chat RSN four-point verification (alpha in {-8, 0, +6, +8}).
INDEPENDENT of the frozen bare Qwen GSM-Hard P3 line
(run_wps_gsm_hard.sh / get_answer_gsm_hard_blind.py) and of the Llama GSM-Hard
chat sweep (get_answer_gsm_hard_chat_sweep.py) -- none is touched, read, or
overwritten by this script.

Structural sibling of get_answer_gsm8k_qwen_chat_rsn.py / get_answer_gsm_hard
_chat_sweep.py, following their tokenizer/injection-site/fail-closed
conventions with GSM-Hard's own label-free questions file, sample_id keying,
and output tree. A shared script would invite exactly the cross-task
parameter mixing this repo has been bitten by.

THIS IS NOT A NINE-POINT SWEEP AND NOT A WORKPOINT SEARCH. alpha in
{-8, 0, +6, +8} is a FIXED four-point verification family, per the task
brief.

  ** SCOPE NOTE, LOAD-BEARING FOR THE BARE COMPARISON **
  The frozen bare Qwen GSM-Hard tree (get_answer_gsm_hard_blind.py via
  run_wps_gsm_hard.sh) carries alpha in {0, +6, +8, +10} (No-CoT) -- it does
  NOT carry alpha=-8. So this family's alpha=-8 cell has NO bare counterpart
  and cannot be paired against one; it is recorded and reported as
  UNPAIRED, never silently dropped. The other three doses (0, +6, +8) DO have
  a bare counterpart.

WHAT IS HELD IDENTICAL to the frozen bare Qwen GSM-Hard cells so that only
the chat wrapper differs:
  - the SAME 300-question label-free sample file
    (components/benchmark/gsm_hard_p3_questions.json, questions_sha256
    48cc7635..., dataset revision pinned in docs/PREREG_P3.md), in the SAME
    ORDER, carrying the SAME sample_id values
  - the SAME neutral, No-CoT, plain-wording prompt BODY
    (template.select_templates_gsm8k(suite="default", cot=False,
    wording="plain")["neutral"]) -- the GSM8K body, which is what the frozen
    bare GSM-Hard cells also use
  - the SAME mask file and band: mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy,
    layers [16,22), L=6
  - the SAME generation params: max_new_tokens=768, temperature=0.0 (greedy),
    top_p=1.0, batch_size=24, prefill_only=True, prefill_tail_len=1
  - alpha=0 uses the SAME generation code path as every other alpha (an
    all-zero diff matrix via vc.regenerate(diff_matrices=raw_mask*0)), but is
    RE-RUN under the chat wrapper rather than reused from the bare baseline

LABEL-FREE, AND STRUCTURALLY SO. GSM-Hard gold lives in a SEPARATE sealed
file; this script reads only the label-free questions file, refuses any
input not declaring contains_labels=false, and scans every row for a leaked
label key -- the same firewall get_answer_gsm_hard_blind.py and
get_answer_gsm_hard_chat_sweep.py have. It therefore writes NO correctness
field and prints NO accuracy.

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
Qwen tokenizer for this exact prompt.

Output tree (independent from the bare gsm_hard_p3 tree):
    components/qwen2.5/gsm_hard_chat_v1/mdf_<alpha>/
        gsm_hard_chat_7B_16_22.json

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
from template import select_templates_gsm8k

PROTOCOL = "gsm-hard-qwen-chat-rsn-v1"
PROMPT_WRAPPER_ID = "qwen2.5-chat-template-v1"

EXPECTED_ALPHAS = {-8, 0, 6, 8}
BAND = (16, 22)

# FROZEN generation budget, matching the bare Qwen GSM-Hard line
# (get_answer_gsm_hard_blind.py via run_wps_gsm_hard.sh) exactly. These CLI
# flags exist so the launcher can pass them explicitly; main() hard-fails on
# any other value, closing the gap where bypassing the launcher could still
# write into the same protocol name/output tree under a different,
# unpairable budget.
EXPECTED_N = 300
EXPECTED_MAX_NEW_TOKENS = 768
EXPECTED_BATCH_SIZE = 24
EXPECTED_TEMPERATURE = 0.0

# The frozen GSM-Hard questions digest (docs/PREREG_P3.md, p3-v1). Checked so
# a swapped or regenerated sample file names itself instead of producing a
# plausible-looking curve over different questions.
EXPECTED_QUESTIONS_SHA256 = (
    "48cc763545d2ee23835833f5165456b90db194863420a10b6741a57cff781d02")

# The frozen bare Qwen GSM-Hard tree's No-CoT dose coverage
# (get_answer_gsm_hard_blind.py via run_wps_gsm_hard.sh): {0, +6, +8, +10}.
# alpha=-8 of THIS family has NO bare counterpart.
BARE_ALPHAS = (0, 6, 8, 10)

FORBIDDEN_KEYS = ("answer", "gold", "gold_answer", "correct", "accuracy",
                  "target")


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser(
        description="GSM-Hard No-CoT Qwen2.5 Native Chat RSN four-point "
                    "verification (independent of the frozen bare P3 line)")
    p.add_argument("--model_dir", default="Qwen/Qwen2.5-7B-Instruct",
                   help="Hugging Face repo id, per the repo-wide convention.")
    p.add_argument("--size", default="7B")
    p.add_argument("--questions", required=True,
                   help="components/benchmark/gsm_hard_p3_questions.json -- "
                        "the SAME label-free 300-question file the bare P3 "
                        "cells use")
    p.add_argument("--mask_path", required=True,
                   help="SAME mask as the bare Qwen cells: "
                        "mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy")
    p.add_argument("--configs", required=True, nargs="+",
                   help="e.g. neg8-16-22 0-16-22 6-16-22 8-16-22 (alpha "
                        "restricted to the frozen four-point set)")
    p.add_argument("--out_dir", required=True,
                   help="components/qwen2.5/gsm_hard_chat_v1")
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--max_new_tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--expect_questions_sha256",
                   default=EXPECTED_QUESTIONS_SHA256,
                   help="Frozen digest of the P3 question set. Pass '' only "
                        "to deliberately bypass (not recommended).")
    return p.parse_args()


def load_questions(path, expect_sha):
    """Label-free loader, byte-equivalent in intent to
    get_answer_gsm_hard_blind.py's / get_answer_gsm_hard_chat_sweep.py's:
    refuse anything that does not declare itself label-free, and scan every
    row for a leaked label key."""
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d["meta"], d["data"]
    if meta.get("contains_labels") is not False:
        die("questions file does not declare contains_labels=false; "
            "generation must be label-free.")
    for i, s in enumerate(data):
        bad = [k for k in s if k.lower() in FORBIDDEN_KEYS]
        if bad:
            die(f"label field {bad} present at row {i} of the questions file; "
                "generation must be label-free.")
    got = meta.get("questions_sha256")
    if expect_sha and got != expect_sha:
        die(f"questions_sha256 {got} != the frozen P3 digest {expect_sha}. "
            "This sweep must run on the SAME 300 questions as the bare "
            "cells; a different sample would make every bare-vs-chat "
            "pairing invalid.")
    ids = [s["sample_id"] for s in data]
    if len(set(ids)) != len(ids):
        die("duplicate sample_id in the questions file; per-item pairing "
            "against the bare cells would be ambiguous.")
    return meta, data


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

    # Budget/batch/temperature are FROZEN at the bare Qwen GSM-Hard line's
    # own values -- not a free choice under this protocol name.
    if args.max_new_tokens != EXPECTED_MAX_NEW_TOKENS:
        die(f"--max_new_tokens={args.max_new_tokens}, expected exactly "
            f"{EXPECTED_MAX_NEW_TOKENS} -- this protocol's cells must share "
            "one generation budget with each other and with the bare Qwen "
            "GSM-Hard line they are compared against.")
    if args.batch_size != EXPECTED_BATCH_SIZE:
        die(f"--batch_size={args.batch_size}, expected exactly "
            f"{EXPECTED_BATCH_SIZE} -- batch size affects padding and is "
            "part of this protocol's frozen generation path.")
    if args.temperature != EXPECTED_TEMPERATURE:
        die(f"--temperature={args.temperature}, expected exactly "
            f"{EXPECTED_TEMPERATURE} -- this protocol is greedy-only; a "
            "non-zero temperature would make the cell non-reproducible.")

    qmeta, samples = load_questions(args.questions, args.expect_questions_sha256)
    n = len(samples)
    # FROZEN at n=300, matching the bare Qwen GSM-Hard line this family is
    # designed to sit beside.
    if n != EXPECTED_N:
        die(f"questions file holds {n} GSM-Hard questions, expected exactly "
            f"{EXPECTED_N} -- the bare Qwen GSM-Hard cells this family is "
            "designed to sit beside are all built on 300 fixed questions, "
            "so a different count would not be pairable.")
    print(f"Loaded {n} GSM-Hard questions from {args.questions} "
          f"(digest {qmeta['questions_sha256'][:16]}, label-free check OK)")

    templates = select_templates_gsm8k(suite="default", cot=False,
                                       wording="plain")
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
    questions_file_sha256 = hashlib.sha256(
        open(args.questions, "rb").read()).hexdigest()
    sample_order_sha256 = hashlib.sha256(
        "\n".join(str(s["sample_id"]) for s in samples).encode("utf-8")
    ).hexdigest()

    print(f"prompt_body_sha256={prompt_body_sha256}")
    print(f"chat_template_hash={chat_template_hash}")
    print(f"chat_prompt_sha256={prompt_sha256}")
    print(f"bare_prompt_sha256={bare_prompt_sha256}")
    print(f"sample_order_sha256={sample_order_sha256}")
    print("First wrapped prompt (repr, truncated to 400 chars):")
    print(repr(chat_prompts[0][:400]))

    out_paths = {}
    for alpha, (ls, le) in cfgs:
        if (ls, le) != BAND:
            die(f"layer band {(ls, le)} != {BAND} -- this protocol is frozen at "
                "the bare cells' own band; do not pass a different band under "
                "this protocol name.")
        tag = f"mdf_{alpha}".replace("-", "neg")
        out_paths[alpha] = os.path.join(
            args.out_dir, tag, f"gsm_hard_chat_{args.size}_{ls}_{le}.json")
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
                      desc=f"qwen-gsm-hard-chat-rsn a={alpha}"):
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

        # NO correctness field, NO accuracy print -- the gold is not reachable
        # from this script by construction.
        rows = [{"sample_id": s["sample_id"], "question": s["question"],
                 "generated": g["text"],
                 "generated_token_count": g["generated_token_count"],
                 "stop_reason": g["stop_reason"]}
                for s, g in zip(samples, gen)]

        json.dump({
            "meta": {
                "protocol": PROTOCOL,
                "task": "gsm_hard",
                "model": "qwen2.5", "size": args.size,
                "alpha": alpha, "layer_start": ls, "layer_end": le,
                "L": n_layers,
                "mask_path": args.mask_path, "mask_sha256": mask_sha256,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature, "top_p": args.top_p,
                "batch_size": args.batch_size,
                "suite": "default", "cot": False, "fmt_wording": "plain",
                "role": "neutral",
                "prefill_only": True, "prefill_tail_len": 1,
                "prompt_wrapper_id": PROMPT_WRAPPER_ID,
                "chat_template_applied": True,
                "chat_template_hash": chat_template_hash,
                "prompt_template": neutral_tmpl,
                "prompt_body_sha256": prompt_body_sha256,
                "prompt_sha256": prompt_sha256,
                "bare_prompt_sha256": bare_prompt_sha256,
                "questions_sha256": qmeta["questions_sha256"],
                "questions_file_sha256": questions_file_sha256,
                "sample_order_sha256": sample_order_sha256,
                "steering_fires": fires,
                "padding_side": vc.tokenizer.padding_side,
                "injection_tail_tokens_chat": inject_tail_chat,
                "injection_tail_tokens_bare_reference": inject_tail_bare,
                "injection_site_moved_vs_bare": (
                    inject_tail_chat[-1]["id"] != inject_tail_bare[-1]["id"]),
                "terminators_ids": terminators,
                "terminators_text": terminator_texts,
                "contains_labels": False,
                "n": n,
                "provenance": device_note,
                "not_a_workpoint_search": True,
                "verification_family": "four-point {-8,0,+6,+8}, NOT the "
                    "full nine-point bare/Llama-chat dose set",
                "g_prefill_measured": False,
                "g_prefill_omitted_reason": (
                    "signal hook is bs=1-only; measuring it would change the "
                    "generation path this cell exists to hold fixed"),
                "bare_counterpart_exists": alpha in BARE_ALPHAS,
                "bare_coverage_note": (
                    "The frozen bare Qwen GSM-Hard tree carries alpha in "
                    "{0,+6,+8,+10}. alpha=-8 of THIS family has NO bare "
                    "counterpart and MUST be reported as unpaired, not "
                    "silently dropped. Bare-vs-chat is DESCRIPTIVE only and "
                    "is never pooled with the chat family's Holm m=3 "
                    "statistics."),
            },
            "data": rows,
        }, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  wrote {out_path}  steering_fires={fires}  n={n}")

    print("\nAll GSM-Hard Qwen Native Chat RSN cells finished. "
          "NO accuracy was computed -- by construction.")
    print("Next: sync this OUT DIR to the offline analysis workspace "
          "~/Documents/RSNResult/RoleAnswer/ then score against the sealed "
          "gold with the frozen offline extractor.")


if __name__ == "__main__":
    main()
