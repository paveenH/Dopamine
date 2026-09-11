#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
get_answer_math_chat_sweep.py -- MATH No-CoT, chat-template INTERFACE
condition, full 9-point alpha sweep. INDEPENDENT of the frozen bare-string
MATH line (run_math.sh / run_math_llama3_wp.sh /
get_answer_regenerate_math.py) -- none is touched, read, or overwritten.

Sibling of get_answer_gsm8k_chat_sweep.py / get_answer_gsm_hard_chat_sweep.py,
which this follows structurally rather than reuses: MATH differs in template
(\boxed{} rather than ####), generation budget (2048 vs 768), batch size
(8 vs 24), sample file and output tree. A shared script would invite exactly
the cross-task parameter mixing this repo has been bitten by; in particular
reusing GSM8K's 768-token budget here would truncate MATH solutions and
manufacture an extraction floor.

THIS IS NOT A WORKPOINT SEARCH. alpha in {-8,-6,-4,-2,0,2,4,6,8} is a FIXED
nine-point family declared up front, not re-selected from any result.

  ** SCOPE NOTE, LOAD-BEARING FOR THE BARE COMPARISON **
  The frozen bare MATH tree carries only FIVE No-CoT doses (-8/-6/-4/0/+4;
  MATH_DIRS in RoleAnswer/analyze_first_last_acc.py). So four of this sweep's
  nine cells (-2/+2/+6/+8) have NO bare counterpart and CANNOT be paired
  against one. That is a property of the frozen bare tree, not a defect here:
  this sweep is a complete NINE-POINT CHAT curve read against its OWN chat
  alpha=0 baseline, and the bare-vs-chat table is a five-row DESCRIPTIVE
  subset. The analyzer states this rather than silently dropping the four
  unpaired rows.

WHAT IS HELD IDENTICAL to the frozen bare MATH cells (so samples, prompt body
and decoding are the same and only the wrapper differs):
  - the SAME 300-problem fixed sample: benchmark/math_test_sample.json,
    truncated to the first n_samples=300 EXACTLY as
    get_answer_regenerate_math.py does (all_samples[:n]) -- same problems,
    same order
  - the SAME neutral, No-CoT prompt BODY
    (template.build_math_suite(cot=False)["neutral"])
  - the SAME mask file and band: mask/llama3_non_logits/nmd_0.5_11_20_8B.npy,
    layers [11,20), L=9
  - the SAME generation params as the frozen MATH launchers: 2048 /
    temperature 0.0 (greedy) / batch_size 8 / prefill_only / tail=1.
    top_p defaults to 0.9 here to match get_answer_regenerate_math.py's own
    default (GSM8K's runner defaults to 1.0) -- under greedy this is inert
    (regenerate ignores top_p when temperature<=0), but it is recorded so the
    two trees' metadata agree rather than differing on a dead field.
  - alpha=0 uses the SAME generation code path as every other alpha (an
    all-zero diff matrix via vc.regenerate(diff_matrices=raw_mask*0)), but is
    RE-RUN under the chat wrapper rather than reused from the bare baseline
    (the bare alpha=0 cell used a different prompt string).

THE ONLY PLANNED INTERFACE CHANGE, held constant across all nine alpha, is
that the identical rendered prompt STRING is wrapped with

    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )

with a duplicated leading BOS stripped before vc.regenerate() tokenizes it.

ANSWER EXTRACTION IS THE CENTRALIZED utils FUNCTION, NOT A LOCAL COPY.
utils.extract_boxed / utils.extract_math_answer / utils.is_correct_math are
imported and used verbatim, exactly as get_answer_regenerate_math.py does, so
the inline process-state fields mean the same thing in both trees. These
inline fields are process-state ONLY; the AUTHORITATIVE reading is the offline
first_acc (first \boxed{}) computed by RoleAnswer/analyze_math_chat_sweep.py
through the frozen all_boxed / norm_math / fallback_math extractors. Per
CLAUDE.md (corrected 2026-08-21) MATH's MAIN caliber is FIRST \boxed{} --
utils.extract_math_answer takes the FIRST because tail loops pollute the last;
last_acc is a tail-pollution SENSITIVITY readout only. Gold is the LAST
\boxed{} of the dataset solution (extract_boxed default which="last"), which
is the opposite caliber ON PURPOSE: first is correct for model PREDICTIONS,
last for the dataset GOLD.

KNOWN CONFOUND -- MEASURED, NOT ASSUMED, AND IT MUST TRAVEL WITH EVERY RESULT.
apply_chat_template applies Jinja `| trim` to the user-turn content, which
STRIPS the trailing space of the prompt body's "Answer: " anchor. So the last
prompt token -- the ONLY position prefill-only tail=1 steering injects into --
differs between the two conditions. Both tails are READ OUT at run time (never
assumed to be 271) and stored per cell. This experiment therefore changes the
interface AND the injection site TOGETHER; that is intrinsic to chat wrapping
and cannot be removed while still testing a chat interface, but a chat-vs-bare
difference may NOT be attributed to "the wrapper" alone.

Output tree (independent from the bare answer_math tree):
    components/llama3/answer_math_chat_v1/mdf_<alpha>/
        math_chat_8B_11_20.json

Existing output files are NEVER overwritten -- fail closed (die()). There is no
--allow_overwrite escape hatch. All nine paths are checked BEFORE the first
cell runs, so a long sweep cannot die on cell 9 after eight are written.

@author: MATH chat-template interface sweep (2026-09-11)
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

PROTOCOL = "math-chat-sweep-v1"
PROMPT_WRAPPER_ID = "llama3-chat-template-v1"

EXPECTED_ALPHAS = {-8, -6, -4, -2, 0, 2, 4, 6, 8}
BAND = (11, 20)

# The frozen bare MATH tree's No-CoT dose coverage (MATH_DIRS in
# RoleAnswer/analyze_first_last_acc.py). Used only to record, per cell, whether
# a bare counterpart exists for that alpha.
BARE_ALPHAS = (-8, -6, -4, 0, 4)


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser(
        description="MATH No-CoT chat-template interface sweep (independent of "
                    "the frozen bare MATH line)")
    p.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="Hugging Face repo id, per the repo-wide convention.")
    p.add_argument("--size", default="8B")
    p.add_argument("--test_file", required=True,
                   help="benchmark/math_test_sample.json -- the SAME sample "
                        "file the bare MATH cells use")
    p.add_argument("--mask_path", required=True,
                   help="SAME mask as the bare cells: "
                        "mask/llama3_non_logits/nmd_0.5_11_20_8B.npy")
    p.add_argument("--configs", required=True, nargs="+",
                   help="0-11-20 neg8-11-20 ... 8-11-20 (exactly the frozen "
                        "nine-point dose set)")
    p.add_argument("--out_dir", required=True,
                   help="components/llama3/answer_math_chat_v1")
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
    header = "<|start_header_id|>assistant<|end_header_id|>"
    if header not in wrapped:
        die(f"expected assistant generation header {header!r} not found in the "
            "first wrapped prompt -- add_generation_prompt=True did not produce "
            "the expected Llama-3 chat header.")


def main():
    args = parse_args()

    cfgs = utils.parse_configs(args.configs)
    # EXACT match on the frozen nine-point family, as a SORTED LIST -- not a
    # set-subset, and with NO int() coercion (parse_configs accepts floats, so
    # int(2.5) would silently read as 2 and pass a subset check; a subset check
    # also admits a partial curve or a repeated dose).
    got_alphas = sorted(al for al, _ in cfgs)
    want_alphas = sorted(EXPECTED_ALPHAS)
    if got_alphas != want_alphas:
        die(f"--configs alphas {got_alphas} != this protocol's frozen "
            f"nine-point dose set {want_alphas}. Exact match required: a "
            "non-integer dose, a missing dose, a duplicate, or an extra dose "
            "all land here. A partial curve must not be written under this "
            "protocol name.")

    all_samples = utils.load_json(args.test_file)
    # Truncate EXACTLY as get_answer_regenerate_math.py does, so the sample set
    # is the same problems in the same order as the frozen bare cells.
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
              "moved to the assistant header. Recorded in meta.")

    prompt_sha256 = hashlib.sha256(
        "\n".join(chat_prompts).encode("utf-8")).hexdigest()
    bare_prompt_sha256 = hashlib.sha256(
        "\n".join(bare_prompts).encode("utf-8")).hexdigest()
    test_file_sha256 = hashlib.sha256(
        open(args.test_file, "rb").read()).hexdigest()
    # The bare MATH cells carry NO meta block at all (no digest, no template),
    # so a digest of the QUESTION LIST is the only thing that can attest the
    # two trees share a sample. The analyzer recomputes it from the bare cells
    # and compares -- which is why it is stored here.
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
                "the bare cells' own band; do not pass a different band under "
                "this protocol name.")
        # Tag matches the bare MATH tree's convention (mdf_neg8, not mdf_-8).
        alpha_tag = f"neg{abs(alpha)}" if alpha < 0 else str(alpha)
        out_paths[alpha] = os.path.join(
            args.out_dir, f"mdf_{alpha_tag}",
            f"math_chat_{args.size}_{ls}_{le}.json")
    for alpha, op in out_paths.items():
        if os.path.exists(op):
            die(f"{op} already exists -- refusing to overwrite a frozen "
                "chat-sweep cell. Delete it deliberately first if a re-run is "
                "truly intended.")

    n_layers = len(utils.decoder_layer_range(*BAND))

    for alpha, (ls, le) in cfgs:
        out_path = out_paths[alpha]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        diff = raw_mask * alpha
        vc.steering_fire_count(reset=True)

        gen = []
        for i in tqdm(range(0, len(chat_prompts), args.batch_size),
                      desc=f"math-chat a={alpha}"):
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
            # Gold = LAST \boxed{} of the dataset solution; prediction = FIRST
            # \boxed{} of the generation (utils.extract_math_answer). Opposite
            # calibers ON PURPOSE -- see the module docstring.
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
                "injection_tail_tokens_chat": inject_tail_chat,
                "injection_tail_tokens_bare_reference": inject_tail_bare,
                "injection_site_moved_vs_bare": (
                    inject_tail_chat[-1]["id"] != inject_tail_bare[-1]["id"]),
                "n": n,
                # Process-state only. The AUTHORITATIVE reading is the offline
                # first_acc (first \boxed{}) via the frozen all_boxed /
                # norm_math / fallback_math chain in
                # RoleAnswer/analyze_math_chat_sweep.py. CLAUDE.md records that
                # offline first_acc reads 1-2 items BELOW inline in every MATH
                # cell (two known, deliberately unpatched extractor gaps: a
                # leading-zero normalisation gap, and an empty first \boxed{}
                # occupying the first-marker slot). Uniform across cells, so it
                # cannot move an ordering -- but it means inline and offline
                # will not agree exactly, and that is expected, not a defect.
                "accuracy_inline_pct": accuracy_pct,
                "accuracy_inline_note": (
                    "process-state only; offline first_acc is authoritative and "
                    "reads 1-2 items lower per cell due to two known unpatched "
                    "extractor gaps"),
                "provenance": device_note,
                "not_a_workpoint_search": True,
                # NOT MEASURED, and deliberately not faked. track_dopamine_signal.py's
                # hook reads hs[0] and stores ONE scalar per forward, i.e. it is
                # bs=1-only; this sweep runs bs=8 to stay identical to the bare
                # MATH curve's generation path. Reusing it would mean either bs=1
                # (a DIFFERENT padding regime than the cell it is meant to attest)
                # or rewriting the hook -- both change the generation path the
                # protocol exists to hold fixed. The closed form
                # alpha*mean_l||m_l||^2 is NOT a substitute: that identity holds
                # ONLY at the FIRST steered layer, because each later layer also
                # carries the propagated residue of the layers below it.
                # steering_fires attests the intervention FIRED; it does not
                # measure its magnitude.
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
                "bare_coverage_note": (
                    "The frozen bare MATH tree carries only five No-CoT doses "
                    "(-8/-6/-4/0/+4). alpha in (-2,+2,+6,+8) has NO bare "
                    "counterpart, so those cells are chat-only and cannot enter "
                    "a paired bare-vs-chat contrast. Bare-vs-chat is DESCRIPTIVE "
                    "only and is never pooled with the chat family's statistics."),
            },
            "data": rows,
        }, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  wrote {out_path}  steering_fires={fires}  "
              f"inline_acc={accuracy_pct}%")

    print("\nAll MATH chat-sweep cells finished.")
    print("Next: python3.10 RoleAnswer/analyze_math_chat_sweep.py "
          "(offline, reuses the frozen extractors; first_acc is MAIN)")


if __name__ == "__main__":
    main()
