#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_answer_gsm8k_qwen_chat_rsn.py -- GSM8K No-CoT, Qwen2.5-7B-Instruct, Native
Chat RSN four-point verification (alpha in {-8, 0, +6, +8}). INDEPENDENT of the
frozen bare Qwen GSM8K line (run_gsm8k_qwen25.sh / get_answer_regenerate_gsm8k.py)
and of the Llama chat sweep (get_answer_gsm8k_chat_sweep.py) -- none is touched,
read, or overwritten by this script.

STRUCTURAL SIBLING of get_answer_gsm8k_chat_sweep.py, which this follows rather
than reuses: a shared script would invite the cross-model parameter mixing this
repo has repeatedly been bitten by (Qwen's band, mask, size, and injection token
are all model-specific facts, not knobs). The Llama script is left byte-unchanged.

WHY THIS IS A NEW SCRIPT, NOT `--use_chat` ON THE BARE RUNNER. Per CLAUDE.md,
get_answer_regenerate_gsm8k.py declares --use_chat but never branches on it in
prompt construction -- that flag is a DEAD PATH. Using it would silently run a
non-chat generation while claiming to have run Chat. This script is a wholly
separate generator/launcher pair so the frozen bare pipeline is never at risk
of a "looks enabled but isn't" regression.

THIS IS NOT A NINE-POINT SWEEP AND NOT A WORKPOINT SEARCH. alpha in
{-8, 0, +6, +8} is a FIXED four-point verification family, read from the task
brief -- not re-selected, not extended to the full bare/Llama nine-point set.

WHAT IS HELD IDENTICAL to the frozen bare Qwen GSM8K cells (run_gsm8k_qwen25.sh)
so that only the chat wrapper differs:
  - the SAME 300-question fixed sample (benchmark/gsm8k_test_sample.json)
  - the SAME neutral, No-CoT, plain-wording prompt BODY
    (template.build_gsm8k_default_suite(cot=False, wording="plain")["neutral"])
  - the SAME mask file and band: mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy,
    layers [16,22), L=6 (Llama's [11,20)/L=9 does NOT transfer)
  - the SAME generation params: max_new_tokens=768, temperature=0.0 (greedy),
    batch_size=24, prefill_only=True, prefill_tail_len=1
  - alpha=0 uses the SAME generation code path as every other alpha (an
    all-zero diff matrix via vc.regenerate(diff_matrices=raw_mask*0)), but is
    RE-RUN under the chat wrapper rather than reused from the bare baseline
  - the SAME answer extractor (utils.extract_gsm8k_answer / is_correct_gsm8k)
    for the inline process-state fields; the AUTHORITATIVE first_acc/last_acc
    reading is computed offline by the frozen extractors in
    RoleAnswer/analyze_first_last_acc.py, never redefined here

THE ONLY EXPERIMENTAL VARIABLE, held constant across all four alpha, is how the
identical prompt STRING is wrapped before tokenization:

    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )

with a duplicated leading BOS stripped before vc.regenerate() tokenizes it
(add_special_tokens=True internally). NO assistant-side "Answer: " anchor is
added and NO matched-anchor construction is used -- this is plain native chat,
per the task brief.

INJECTION TOKEN IS READ OUT AT RUNTIME, NEVER ASSUMED. Per CLAUDE.md's
cross-model pre-flight rule, the injection site is model-specific (Llama's
chat header token is id 271 '\n\n'; Qwen's is a DIFFERENT id/text on its own
tokenizer -- check_igt_qwen.py records Qwen's assistant-header last token as
id 198 '\n' in a different context, but this script measures its OWN prompt's
actual tail rather than assuming that figure transfers). Both the chat tail and
the bare-reference tail are tokenized and decoded here, on the real Qwen
tokenizer, and recorded per cell.

Output tree (independent from the bare answer_mdf_gsm8k tree):
    components/qwen2.5/answer_mdf_gsm8k_chat_v1/mdf_<alpha>/
        gsm8k_chat_7B_answers_16_22.json

Existing output files are NEVER overwritten -- fail closed (die()). There is no
--allow_overwrite escape hatch. All four paths are checked BEFORE the first
cell runs, so a partial failure cannot leave a half-written family.

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
from template import build_gsm8k_default_suite
from utils import extract_gsm8k_answer, is_correct_gsm8k

PROTOCOL = "gsm8k-qwen-chat-rsn-v1"
PROMPT_WRAPPER_ID = "qwen2.5-chat-template-v1"

# Fixed, frozen four-point verification family -- per the task brief, NOT the
# full nine-point bare/Llama-chat dose set.
EXPECTED_ALPHAS = {-8, 0, 6, 8}
BAND = (16, 22)

# FROZEN generation budget, matching the bare Qwen GSM8K line
# (run_gsm8k_qwen25.sh) exactly. These CLI flags exist so the launcher can
# pass them explicitly (readable in the process listing / logs), NOT so a
# caller can silently drift them -- main() hard-fails on any other value,
# closing the gap where bypassing the launcher and invoking this script
# directly could still write into the same protocol name/output tree under a
# different, unpairable budget.
EXPECTED_N = 300
EXPECTED_MAX_NEW_TOKENS = 768
EXPECTED_BATCH_SIZE = 24
EXPECTED_TEMPERATURE = 0.0


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser(
        description="GSM8K No-CoT Qwen2.5 Native Chat RSN four-point "
                    "verification (independent of the frozen bare Qwen line)")
    p.add_argument("--model_dir", default="Qwen/Qwen2.5-7B-Instruct",
                   help="Hugging Face repo id, per the repo-wide convention.")
    p.add_argument("--size", default="7B")
    p.add_argument("--test_file", required=True,
                   help="benchmark/gsm8k_test_sample.json -- the SAME 300-"
                        "question fixed sample the bare Qwen sweep uses")
    p.add_argument("--mask_path", required=True,
                   help="SAME mask as the bare Qwen sweep: "
                        "mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy")
    p.add_argument("--configs", required=True, nargs="+",
                   help="e.g. neg8-16-22 0-16-22 6-16-22 8-16-22 (alpha "
                        "restricted to the frozen four-point set)")
    p.add_argument("--out_dir", required=True,
                   help="components/qwen2.5/answer_mdf_gsm8k_chat_v1")
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--max_new_tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--fmt_wording", default="plain", choices=["plain"],
                   help="Fixed to 'plain' (the main-line wording).")
    p.add_argument("--expect_test_file_sha256", default="",
                   help="Optional: the frozen SHA256 of gsm8k_test_sample.json "
                        "(matching GSM-Hard's own --expect_questions_sha256 "
                        "convention). Empty (default) = no check -- there is "
                        "currently no frozen digest constant for this file "
                        "anywhere in the repo (unlike GSM-Hard's sealed "
                        "questions file). Pass it once one is established so "
                        "a swapped/regenerated sample file names itself "
                        "before a multi-hour cell runs, instead of only "
                        "being caught by a later sample-alignment check.")
    return p.parse_args()


def strip_leading_bos(vc, text: str) -> str:
    """vc.regenerate's _regenerate_prefill_only tokenizes with
    add_special_tokens=True, so an un-stripped chat-templated string that
    already serialized a BOS text yields two leading BOS ids. Qwen2.5's chat
    template does not prepend a BOS string by default (bos_token_id is often
    None for Qwen), but this check is kept model-agnostic rather than assumed
    away -- see assert_no_double_bos below, which is the actual gate."""
    bos = getattr(vc.tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        return text[len(bos):]
    return text


def assert_no_double_bos(vc, text: str, label: str) -> None:
    """Hard invariant: tokenize with add_special_tokens=True (the same call
    vc.regenerate makes internally) and refuse if the first two ids are both
    BOS. A no-op check if the tokenizer has no bos_token_id (Qwen2.5's is
    commonly None), which is correct -- there is nothing to double."""
    bos_id = getattr(vc.tokenizer, "bos_token_id", None)
    ids = vc.tokenizer(text, add_special_tokens=True)["input_ids"]
    if bos_id is not None and len(ids) >= 2 and ids[:2] == [bos_id, bos_id]:
        die(f"double BOS in the {label} chat prompt (head={ids[:4]}) -- "
            "strip_leading_bos did not remove the chat template's own "
            "serialized BOS text before this add_special_tokens=True call.")


def assert_chat_template_effective(vc, bare_prompt: str, wrapped: str) -> None:
    """The chat template actually fired (wrapped != bare-string would be a
    no-op template). Does NOT assume a Llama-specific header string -- Qwen's
    chat template markers (e.g. <|im_start|>assistant) differ."""
    if wrapped == bare_prompt:
        die("chat template had NO effect on the first sample (wrapped text "
            "is byte-identical to the bare prompt) -- apply_chat_template "
            "did not actually run, or the tokenizer has no chat_template.")
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
    # EXACT match on the frozen four-point family, as a SORTED LIST -- not a
    # set-subset, and with NO int() coercion (utils.parse_configs accepts
    # float alpha tokens, so int(6.5) would silently read as 6 and pass a
    # naive subset check).
    got_alphas = sorted(al for al, _ in cfgs)
    want_alphas = sorted(EXPECTED_ALPHAS)
    if got_alphas != want_alphas:
        die(f"--configs alphas {got_alphas} != this protocol's frozen "
            f"four-point verification set {want_alphas}. Exact match "
            "required: a non-integer dose, a missing dose, a duplicate, or "
            "an extra dose all land here.")

    # Budget/batch/temperature are FROZEN at the bare Qwen GSM8K line's own
    # values -- not a free choice under this protocol name. A caller who
    # bypasses the launcher and invokes this script directly with a
    # different budget would otherwise still write a syntactically valid
    # cell into the same output tree/protocol, silently unpairable with its
    # siblings.
    if args.max_new_tokens != EXPECTED_MAX_NEW_TOKENS:
        die(f"--max_new_tokens={args.max_new_tokens}, expected exactly "
            f"{EXPECTED_MAX_NEW_TOKENS} -- this protocol's cells must share "
            "one generation budget with each other and with the bare Qwen "
            "GSM8K line they are compared against.")
    if args.batch_size != EXPECTED_BATCH_SIZE:
        die(f"--batch_size={args.batch_size}, expected exactly "
            f"{EXPECTED_BATCH_SIZE} -- batch size affects padding and is "
            "part of this protocol's frozen generation path.")
    if args.temperature != EXPECTED_TEMPERATURE:
        die(f"--temperature={args.temperature}, expected exactly "
            f"{EXPECTED_TEMPERATURE} -- this protocol is greedy-only; a "
            "non-zero temperature would make the cell non-reproducible and "
            "not comparable to the rest of the family.")

    samples = utils.load_json(args.test_file)
    n = len(samples)
    # FROZEN at n=300, matching the bare Qwen GSM8K line this family is
    # designed to sit beside -- an accidentally truncated or duplicated test
    # file would still produce a syntactically valid protocol file (every
    # other check here is agnostic to n) and could silently drift out of
    # pairability without this check.
    if n != EXPECTED_N:
        die(f"test file holds {n} GSM8K samples, expected exactly "
            f"{EXPECTED_N} -- the bare Qwen GSM8K cells this family is "
            "designed to sit beside are all built on 300 fixed samples, so "
            "a different count would not be pairable. Refusing to write a "
            "protocol file whose sample count silently differs.")
    print(f"Loaded {n} GSM8K samples from {args.test_file}")

    test_file_sha256_pre = hashlib.sha256(
        open(args.test_file, "rb").read()).hexdigest()
    if args.expect_test_file_sha256 and test_file_sha256_pre != args.expect_test_file_sha256:
        die(f"test_file SHA256 {test_file_sha256_pre} != expected "
            f"{args.expect_test_file_sha256}. A different sample file would "
            "make every downstream bare-vs-chat pairing invalid; refusing "
            "before the model load rather than after a multi-hour cell.")

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

    # vc.regenerate requires ONE diff row per decoder layer in the WHOLE model
    # (28 for Qwen2.5-7B) -- raw_mask is already full-length with zero rows
    # outside the band. Slicing it to the band is a known failure mode
    # (ProofWriter's "diff_matrices length != layers" bug). Checked HERE so it
    # fails before the first multi-minute cell rather than during it.
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

    # Provenance, per CLAUDE.md's repo-wide rule: record the device rather
    # than constrain it. All four cells share this record because they share
    # ONE model load in ONE invocation.
    try:
        devs = sorted({str(p.device) for p in vc.model.parameters()})
    except Exception:
        devs = []
    device_note = {"host": platform.node(),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "param_devices": devs,
                   "sharded": len(devs) > 1}

    # Render every bare prompt ONCE, wrap it ONCE -- shared byte-identically
    # by every alpha of this sweep, so the digests below attest "same prompt
    # across the whole curve".
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

    # INJECTION SITE -- READ OUT, NEVER ASSUMED. Qwen's chat template trims
    # the user-turn content just as Llama's does, so the bare condition's
    # trailing "Answer: " anchor is likely stripped here too -- but this is
    # MEASURED on the real Qwen tokenizer rather than assumed to equal any
    # figure recorded elsewhere in this repo for a different prompt.
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

    # Qwen terminators: <|im_end|> plus generation_config's own EOS list, both
    # unioned by llms.VicundaModel._build_terminators(). Read out and
    # recorded, never assumed.
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

    print(f"prompt_body_sha256={prompt_body_sha256}")
    print(f"chat_template_hash={chat_template_hash}")
    print(f"chat_prompt_sha256={prompt_sha256}")
    print(f"bare_prompt_sha256={bare_prompt_sha256}")
    print(f"test_file_sha256={test_file_sha256}")
    print("First wrapped prompt (repr, truncated to 400 chars):")
    print(repr(chat_prompts[0][:400]))

    # Band + overwrite are checked for EVERY cell BEFORE the first cell runs.
    out_paths = {}
    for alpha, (ls, le) in cfgs:
        if (ls, le) != BAND:
            die(f"layer band {(ls, le)} != {BAND} -- this protocol is "
                "frozen at the bare Qwen sweep's own band; do not pass a "
                "different band under this protocol name.")
        out_paths[alpha] = os.path.join(
            args.out_dir, f"mdf_{alpha}",
            f"gsm8k_chat_{args.size}_answers_{ls}_{le}.json")
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
                      desc=f"qwen-gsm8k-chat-rsn a={alpha}"):
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
                "task": "gsm8k",
                "model": "qwen2.5", "size": args.size,
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
                "provenance": device_note,
                "not_a_workpoint_search": True,
                "verification_family": "four-point {-8,0,+6,+8}, NOT the "
                    "full nine-point bare/Llama-chat dose set",
                "g_prefill_measured": False,
                "g_prefill_omitted_reason": (
                    "signal hook is bs=1-only; measuring it would change the "
                    "generation path this cell exists to hold fixed"),
                "compares_against": (
                    f"components/qwen2.5/answer_mdf_gsm8k/mdf_{alpha}/"
                    f"gsm8k_{args.size}_answers_20_{ls}_{le}.json "
                    "(frozen bare Qwen sweep, neutral role, same alpha/band; "
                    "DESCRIPTIVE paired contrast only -- bare and chat are "
                    "never pooled into one statistical family)"),
            },
            "data": rows,
        }, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  wrote {out_path}  steering_fires={fires}  "
              f"inline_acc={accuracy_pct}%")

    print("\nAll GSM8K Qwen Native Chat RSN cells finished.")
    print("Next: sync this OUT DIR to the offline analysis workspace "
          "~/Documents/RSNResult/RoleAnswer/ then run the unified offline "
          "analyzer from there.")


if __name__ == "__main__":
    main()
