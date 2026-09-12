#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
get_answer_math_chat_matched_anchor.py -- MATH No-CoT, chat-template
MATCHED-ANCHOR control condition. PRIMARY task of this experiment; run before
get_answer_gsm8k_chat_matched_anchor.py.

INDEPENDENT of every existing MATH tree: the frozen bare-string line
(run_math.sh / get_answer_regenerate_math.py) and the native chat-sweep
(run_math_chat_sweep.sh / get_answer_math_chat_sweep.py, protocol
"math-chat-sweep-v1"). None of those is touched, read, or overwritten by this
script -- it writes to its own new output tree under a new protocol name.

QUESTION THIS CONTROLS FOR. get_answer_math_chat_sweep.py's own docstring
records a MEASURED confound: apply_chat_template's Jinja `| trim` strips the
trailing space of the "Answer: " anchor from the user-turn content, so the
native-chat condition's last prefill token -- the ONLY position prefill-only
tail=1 steering injects into -- moves from the bare condition's anchor token
(id 220, ' ') to the assistant generation header token (id 271, '\n\n').
Native chat therefore changes TWO things at once relative to bare: the
interface (full chat wrapping) AND the injection site. This script keeps the
full chat template but manually re-creates the bare condition's injection
site UNDER it, so that only the interface changes and the injection site is
held matched to bare. If MATH's bare alpha=-6 gain reappears here but not
under native chat, the workpoint's failure under native chat is attributable
to the injection-site shift rather than to the chat interface itself; if it
does not reappear here either, the interface (not the injection site) is
implicated.

THIS IS NOT A WORKPOINT SEARCH AND NOT THE NINE-POINT SWEEP. alpha in
{-6, 0, 6, 8} is a FIXED four-point family declared up front, matching the
two most informative flat/collapse cells from the native chat sweep (0 as
baseline, -6 as the frozen bare workpoint, +6/+8 as the region where the
native chat sweep showed early-candidate/generation-compression onset) plus
the shared baseline. This is a smaller, targeted family, not a re-search.

WHAT IS HELD IDENTICAL to both the frozen bare and native-chat MATH cells:
  - the SAME 300-problem fixed sample: benchmark/math_test_sample.json,
    truncated to the first n_samples=300 EXACTLY as
    get_answer_regenerate_math.py / get_answer_math_chat_sweep.py do
    (all_samples[:n]) -- same problems, same order.
  - the SAME neutral, No-CoT prompt BODY
    (template.build_math_suite(cot=False)["neutral"]), MODULO the trailing
    "Answer: " anchor, which this script strips from the user-turn content
    and re-attaches after the chat template's assistant header instead (see
    chat_matched_anchor_lib.py for the full mechanism).
  - the SAME mask file and band: mask/llama3_non_logits/nmd_0.5_11_20_8B.npy,
    layers [11,20), L=9.
  - the SAME generation params as the frozen MATH launchers: 2048 /
    temperature 0.0 (greedy) / batch_size 8 / prefill_only / tail=1. MATH's
    OWN budget -- NOT GSM8K's 768/24. Reusing GSM8K's budget would truncate
    MATH solutions and manufacture an extraction floor (per CLAUDE.md).
  - alpha=0 uses the SAME generation code path as every other alpha (an
    all-zero diff matrix via vc.regenerate(diff_matrices=raw_mask*0)), and is
    RE-RUN under the matched-anchor wrapper (it is this family's OWN
    baseline, not the bare or native-chat alpha=0 cell -- those used
    different prompt strings).

THE ONLY EXPERIMENTAL VARIABLE, held constant across all four alpha, is the
matched-anchor construction in chat_matched_anchor_lib.py: strip the trailing
"Answer: " from the rendered body, wrap the remainder with
apply_chat_template(..., add_generation_prompt=True), strip a duplicated
leading BOS, then re-append "Answer: " directly after the assistant header.
assert_matched_anchor_tail() FAILS CLOSED before any generation if the last
prefill token is not id 220, a single ASCII space.

ANSWER EXTRACTION IS THE CENTRALIZED utils FUNCTION, NOT A LOCAL COPY.
utils.extract_boxed / utils.extract_math_answer / utils.is_correct_math are
imported and used verbatim, exactly as get_answer_regenerate_math.py and
get_answer_math_chat_sweep.py do. These inline fields are process-state ONLY;
the AUTHORITATIVE reading is the offline first_acc (first \boxed{}) computed
by RoleAnswer/analyze_chat_matched_anchor.py through the frozen all_boxed /
norm_math / fallback_math extractors, reused from analyze_first_last_acc.py
via chat_sweep_common.py -- never reimplemented.

Output tree (independent from both the bare and native-chat MATH trees):
    components/llama3/answer_math_chat_matched_anchor_v1/mdf_<alpha_tag>/
        math_chat_matched_anchor_8B_11_20.json

Existing output files are NEVER overwritten -- fail closed (die()). There is
no --allow_overwrite escape hatch. All FOUR paths are checked BEFORE the
first cell runs, so a long run cannot die on cell 4 after three are written.

@author: MATH chat-template matched-anchor control (2026-09-12)
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
# Centralized MATH extraction -- imported, never reimplemented here.
from utils import extract_boxed, extract_math_answer, is_correct_math

import chat_matched_anchor_lib as CMA

PROTOCOL = "math-chat-matched-anchor-v1"
CONDITION = "Chat Matched-Anchor"
PROMPT_WRAPPER_ID = "llama3-chat-template-matched-anchor-v1"

EXPECTED_ALPHAS = {-6, 0, 6, 8}
BAND = (11, 20)

# For descriptive metadata only: which of these four alpha have a counterpart
# in the frozen bare tree (MATH_DIRS' No-CoT subset, -8/-6/-4/0/+4) and in the
# native chat sweep (all nine of -8..+8).
BARE_ALPHAS = (-8, -6, -4, 0, 4)
NATIVE_CHAT_ALPHAS = (-8, -6, -4, -2, 0, 2, 4, 6, 8)


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser(
        description="MATH No-CoT chat-template MATCHED-ANCHOR control "
                    "(independent of both the frozen bare MATH line and the "
                    "native chat sweep)")
    p.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="Hugging Face repo id, per the repo-wide convention.")
    p.add_argument("--size", default="8B")
    p.add_argument("--test_file", required=True,
                   help="benchmark/math_test_sample.json -- the SAME sample "
                        "file the bare and native-chat MATH cells use")
    p.add_argument("--mask_path", required=True,
                   help="SAME mask as the bare/native-chat cells: "
                        "mask/llama3_non_logits/nmd_0.5_11_20_8B.npy")
    p.add_argument("--configs", required=True, nargs="+",
                   help="0-11-20 neg6-11-20 6-11-20 8-11-20 (exactly the "
                        "frozen four-point matched-anchor dose set)")
    p.add_argument("--out_dir", required=True,
                   help="components/llama3/answer_math_chat_matched_anchor_v1")
    p.add_argument("--n_samples", type=int, default=300,
                   help="Truncation applied EXACTLY as "
                        "get_answer_regenerate_math.py / "
                        "get_answer_math_chat_sweep.py do (all_samples[:n]).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.9,
                   help="Matches get_answer_regenerate_math.py's / "
                        "get_answer_math_chat_sweep.py's own default. Inert "
                        "under greedy; recorded for metadata parity.")
    return p.parse_args()


def main():
    args = parse_args()

    cfgs = utils.parse_configs(args.configs)
    # EXACT match on the frozen four-point family, as a SORTED LIST -- not a
    # set-subset, and with NO int() coercion (parse_configs accepts floats, so
    # int(2.5) would silently read as 2 and pass a subset check).
    got_alphas = sorted(al for al, _ in cfgs)
    want_alphas = sorted(EXPECTED_ALPHAS)
    if got_alphas != want_alphas:
        die(f"--configs alphas {got_alphas} != this protocol's frozen "
            f"four-point dose set {want_alphas}. Exact match required: a "
            "non-integer dose, a missing dose, a duplicate, or an extra dose "
            "all land here. A partial family must not be written under this "
            "protocol name.")

    all_samples = utils.load_json(args.test_file)
    # Truncate EXACTLY as get_answer_regenerate_math.py / the native chat
    # sweep do, so the sample set is the same problems in the same order.
    n = min(len(all_samples), args.n_samples)
    samples = all_samples[:n]
    if n != args.n_samples:
        die(f"test file holds only {len(all_samples)} problems but "
            f"--n_samples={args.n_samples}; the bare/native-chat cells were "
            f"built on {args.n_samples}, so a shorter sample would not be "
            "pairable.")
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
            die(f"layer band {(ls, le)} != {BAND} -- this protocol is frozen "
                "at the bare/native-chat cells' own band; do not pass a "
                "different band under this protocol name.")
        alpha_tag = f"neg{abs(alpha)}" if alpha < 0 else str(alpha)
        out_paths[alpha] = os.path.join(
            args.out_dir, f"mdf_{alpha_tag}",
            f"math_chat_matched_anchor_{args.size}_{ls}_{le}.json")

    # Spec: "启动前一次性检查四个输出路径；任一已存在则整次拒绝运行." All four
    # checked BEFORE any cell runs.
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
                      desc=f"math-chat-matched-anchor a={alpha}"):
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
                "condition": CONDITION,
                "task": "math",
                "model": "llama3", "size": args.size,
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
                # Injection identity -- recorded per spec item 4/metadata list.
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
                    "template, rather than the native-chat condition's "
                    "assistant-header token."),
                "injection_site_matches_bare": (
                    inject_tail_matched[-1]["id"] == inject_tail_bare[-1]["id"]
                    and inject_tail_matched[-1]["text"]
                    == inject_tail_bare[-1]["text"]),
                "n": n,
                "accuracy_inline_pct": accuracy_pct,
                "accuracy_inline_note": (
                    "process-state only; offline first_acc is authoritative "
                    "and reads 1-2 items lower per cell due to two known "
                    "unpatched extractor gaps, matching the native-chat MATH "
                    "sweep's own documented behaviour"),
                "provenance": device_note,
                "not_a_workpoint_search": True,
                "g_prefill_measured": False,
                "g_prefill_omitted_reason": (
                    "signal hook is bs=1-only; measuring it would change the "
                    "generation path this cell exists to hold fixed"),
                "bare_counterpart": (
                    f"components/llama3/answer_math/"
                    f"mdf_{'neg' + str(abs(alpha)) if alpha < 0 else alpha}/"
                    f"math_{args.size}_{ls}_{le}.json"
                    if alpha in BARE_ALPHAS else None),
                "bare_counterpart_exists": alpha in BARE_ALPHAS,
                "native_chat_counterpart": (
                    f"components/llama3/answer_math_chat_v1/"
                    f"mdf_{'neg' + str(abs(alpha)) if alpha < 0 else alpha}/"
                    f"math_chat_{args.size}_{ls}_{le}.json"),
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

    print("\nAll MATH matched-anchor cells finished.")
    print("Next: sync this OUT DIR to the offline analysis workspace "
          "~/Documents/RSNResult/RoleAnswer/ -- which is NOT part of this "
          "repo and is NOT present on the server -- then, FROM THAT BOX:")
    print("      python3.10 analyze_chat_matched_anchor.py --task math")
    print("\nOnce MATH is complete and reviewed, proceed to phase 2:")
    print("      bash run_gsm8k_chat_matched_anchor.sh")


if __name__ == "__main__":
    main()
