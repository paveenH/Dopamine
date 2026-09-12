#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRUXEval-O CoT chat-template interface-condition diagnostic, llama3 ONLY.
Protocol `cruxeval-o-cot-chat-v1`.

WHY. `get_answer_cruxeval_cot.py` (protocol `cot-transfer-followup-v0`) runs
the CoT follow-up bare-string, matching the P4c No-CoT convention. Llama's CoT
No-CoT and CoT cells both carry a high degenerate-tail / loop rate (see the
`ReasoningBare.md` P4c and ProofWriter-OWA precedent), which raises the
question this script exists to answer: is that loop/truncation behaviour, and
the accompanying accuracy null, an artifact of the bare-string prompt
interface, or does it persist under the model's native chat interface?

WHAT IS HELD IDENTICAL to `get_answer_cruxeval_cot.py`
-------------------------------------------------------
  - the SAME 300 items, SAME sample_id order, SAME frozen digest
    (FORMAL_DIGEST, imported)
  - the SAME CoT prompt BODY, verbatim, including the frozen
    "Let's think step by step." cue and the "#### <Python literal>" marker
    convention (PROMPT_COT / PROMPT_COT_SHA256, imported unchanged)
  - the SAME FIRST-marker parser and scoring convention (this script computes
    no accuracy; scoring is eval_cruxeval.py, imported convention only)
  - the SAME mask file, band [11,20), full 32-row mask (never pre-sliced to
    9 layers -- injection range is controlled by the layer band exactly as in
    every other launcher)
  - max_new_tokens=768, batch_size=24, temperature=0.0 (greedy), top_p=1.0
    (BUDGET / BATCH_SIZE, imported)
  - prefill_only=True, prefill_tail_len=1
  - alpha=0 uses the SAME generation path as every other alpha (an all-zero
    diff matrix -- there is no separate no-op branch)

THE ONLY EXPERIMENTAL VARIABLE is how the identical CoT prompt STRING is
wrapped before tokenization: the full bare prompt (unchanged from
`build_prompt()`) is placed as the sole user-turn content in
`tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
tokenize=False, add_generation_prompt=True)`, with a duplicated leading BOS
stripped before `vc.regenerate()` tokenizes it (which calls
`add_special_tokens=True` internally) -- reusing
`bandit_pv6_episode._strip_leading_bos` rather than reinventing it, the same
pattern `proofwriter_owa/get_answer_proofwriter_owa_chat.py` and
`proofwriter_owa/diag_chat_template.py` already use and that this task was
told to reuse.

Model is ALWAYS the Hugging Face id `meta-llama/Llama-3.1-8B-Instruct` -- no
local model path, matching every other launcher's convention.

Native EOS/EOT termination: `VicundaModel._build_terminators()` already
registers `<|eot_id|>` (and `<|end_of_turn|>` if present) as additional
`eos_token_id`s for every generation path, including `regenerate` -- this
script does not touch that, it is a repo-wide fixed behaviour.

DOSES: alpha in {0, -6} ONLY -- llama3's own frozen GSM8K workpoint. No
neighbour, no reverse cell exists here; this is a single planned contrast
(`-6 vs 0` under the chat wrapper), not a new dose search and not an extension
of the diagnostic matrix `cruxeval-p4c-v0`/`cot-transfer-followup-v0` define.

OUTPUT is written to its own tag under the SAME `cruxeval_cot` parent tree the
bare CoT follow-up uses, isolated by tag rather than by top-level directory:
    components/llama3/cruxeval_cot_followup/formal_chat_v1_mdf_0/
    components/llama3/cruxeval_cot_followup/formal_chat_v1_mdf_neg6/
meta.protocol = "cruxeval-o-cot-chat-v1" (NOT "cot-transfer-followup-v0"),
so it can never be mistaken for a bare CoT cell by any consumer that checks
meta.protocol. Score with:
    python eval_cruxeval.py --protocol cruxeval-o-cot-chat-v1 ...
(eval_cruxeval.py's --protocol flag, added alongside this script; the default
`cruxeval-p4c-v0` bare-string path and its Holm m=2 family are untouched.)

Three additional meta fields record the wrapping condition explicitly:
    meta.prompt_wrapper_id      = "llama3-chat-template-v1"
    meta.chat_template_applied  = true
    meta.chat_template_hash     = sha256 of tokenizer.chat_template's own
                                   string (attests WHICH template was used)
    meta.bare_prompt_sha256_prefix = the frozen CoT template's own hash
                                      (PROMPT_COT_SHA256), for cross-reference
    meta.rendered_prompt_sha256    = sha256 over ALL 300 rendered (post-chat-
                                      wrap, post-BOS-strip) prompt strings for
                                      this cell -- attests the actual strings
                                      fed to the tokenizer, not just the body

No Qwen. This script does not touch, import as a runtime dependency, or
modify `get_answer_cruxeval_cot.py`, `get_answer_cruxeval.py`,
`run_cruxeval.sh`, `run_cruxeval_cot.sh`, `eval_cot_transfer_followup.py`, or
any GSM8K/other-task runner.

@author: paveenhuang
"""

import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llms import VicundaModel                                        # noqa: E402
import utils                                                          # noqa: E402
from bandit_pv6_episode import _strip_leading_bos                     # noqa: E402
from get_answer_cruxeval_cot import (                                 # noqa: E402
    PROMPT_COT, PROMPT_COT_SHA256, ANCHOR, N_FORMAL, N_PREFLIGHT,
    FORMAL_DIGEST, BUDGET, BATCH_SIZE, LABEL_FIELDS, build_prompt,
    is_loop, descriptive, die,
)

PROTOCOL = "cruxeval-o-cot-chat-v1"
BASE_PROTOCOL = "cot-transfer-followup-v0"
PROMPT_WRAPPER_ID = "llama3-chat-template-v1"

# This script's OWN frozen matrix. Deliberately NOT EXPECTED_CELLS from
# get_answer_cruxeval_cot.py (which also lists -4/+4 and qwen2.5) -- that
# would let a diagnostic cell for a comparison this protocol does not run
# pass validation by accident.
EXPECTED_CELLS = {(0, 11, 20), (-6, 11, 20)}
WORKPOINT_ALPHA = -6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--size", default="8B")
    p.add_argument("--questions", required=True,
                   help="the LABEL-FREE *_formal_blind.json (SAME file the "
                        "bare CoT follow-up uses)")
    p.add_argument("--mask_path", required=True,
                   help="the FULL 32-row mask "
                        "(mask/llama3_non_logits/nmd_0.5_11_20_8B.npy) -- "
                        "never pre-sliced to the 9-layer band")
    p.add_argument("--configs", required=True, nargs="+",
                   help="e.g. 0-11-20 neg6-11-20")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_new_tokens", type=int, default=BUDGET)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--tag", default="formal_chat_v1",
                   help="output subdir prefix -- keeps this run in its own "
                        "formal_chat_v1_mdf_<alpha> tag, isolated from the "
                        "bare mdf_<alpha> tag under the same cruxeval_cot "
                        "parent tree")
    p.add_argument("--preflight", action="store_true",
                   help="FORMAT ONLY: first N_PREFLIGHT items, output tagged "
                        "preflight so the scorer refuses it.")
    return p.parse_args()


def load_questions(path, preflight=False):
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d["meta"], d["data"]
    if meta.get("contains_labels") is not False:
        die("questions file does not declare contains_labels=false. Point "
            "--questions at the *_formal_blind.json, not the gold copy.")
    if meta.get("questions_sha256") != FORMAL_DIGEST:
        die(f"questions_sha256 {meta.get('questions_sha256')!r} != the frozen "
            f"{FORMAL_DIGEST!r}; this is a different sample")
    if len(data) != N_FORMAL:
        die(f"{len(data)} items, expected {N_FORMAL}")
    if sorted(s["sample_id"] for s in data) != list(range(N_FORMAL)):
        die(f"sample_ids do not cover 0..{N_FORMAL - 1}")
    if preflight:
        data = data[:N_PREFLIGHT]
    for s in data:
        bad = [k for k in s if k.lower() in LABEL_FIELDS]
        if bad:
            die(f"label field {bad} present in the questions file; "
                "generation must be label-free")
    return meta, data


def assert_no_double_bos(vc, text: str, label: str) -> None:
    """Same hard invariant as diag_chat_template.py / check_cgt_seq_qwen.py:
    tokenize with add_special_tokens=True (the same call vc.regenerate makes
    internally) and refuse if the first two ids are both BOS."""
    bos_id = getattr(vc.tokenizer, "bos_token_id", None)
    ids = vc.tokenizer(text, add_special_tokens=True)["input_ids"]
    if bos_id is not None and len(ids) >= 2 and ids[:2] == [bos_id, bos_id]:
        die(f"double BOS in the {label} chat prompt (head={ids[:4]}) -- "
            "strip_leading_bos did not remove the chat template's own "
            "serialized BOS text before this add_special_tokens=True call.")


def main():
    args = parse_args()

    got = hashlib.sha256(PROMPT_COT.encode()).hexdigest()[:16]
    if got != PROMPT_COT_SHA256:
        die(f"CoT prompt template sha256 {got} != the frozen "
            f"{PROMPT_COT_SHA256}. The template is imported from "
            "get_answer_cruxeval_cot.py and is frozen there; it may not be "
            "edited.")

    if args.max_new_tokens != BUDGET or args.batch_size != BATCH_SIZE:
        die(f"budget/batch are inherited frozen at {BUDGET}/{BATCH_SIZE}; got "
            f"{args.max_new_tokens}/{args.batch_size}")
    if args.temperature != 0.0:
        die("temperature is frozen at 0.0 (greedy)")

    parsed = utils.parse_configs(args.configs)
    for alpha, (ls, le) in parsed:
        if (alpha, ls, le) not in EXPECTED_CELLS:
            die(f"cell (alpha={alpha}, band={ls}-{le}) is not in this "
                f"script's frozen matrix {sorted(EXPECTED_CELLS)} -- this is "
                "a single pre-specified {-6 vs 0} chat-interface comparison, "
                "not a dose search.")

    meta, samples = load_questions(args.questions, args.preflight)

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    if vc.tokenizer.padding_side != "left":
        die(f"tokenizer.padding_side is {vc.tokenizer.padding_side!r}, "
            "expected 'left'.")

    raw_mask = np.load(args.mask_path)
    if raw_mask.shape[0] < 32:
        die(f"mask has {raw_mask.shape[0]} rows; expected the FULL 32-row "
            "mask, not a band-sliced one -- injection range is controlled "
            "by the layer band, not by pre-slicing the mask.")
    mask_sha = hashlib.sha256(open(args.mask_path, "rb").read()).hexdigest()
    os.makedirs(args.out_dir, exist_ok=True)

    chat_template_str = getattr(vc.tokenizer, "chat_template", None) or ""
    chat_template_hash = hashlib.sha256(
        chat_template_str.encode("utf-8")).hexdigest()

    device_note = {"host": platform.node(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    # Rendered ONCE, shared by every alpha of this run: the bare prompt body
    # is byte-identical to build_prompt() (imported, unchanged), only the
    # chat wrapping is added here.
    bare_prompts = [build_prompt(s) for s in samples]
    chat_prompts = []
    for i, p in enumerate(bare_prompts):
        wrapped = vc.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        wrapped = _strip_leading_bos(vc, wrapped)
        assert_no_double_bos(vc, wrapped, f"sample_id={samples[i]['sample_id']}")
        chat_prompts.append(wrapped)

    bare_prompt_sha256 = hashlib.sha256(
        "\n".join(bare_prompts).encode("utf-8")).hexdigest()
    rendered_prompt_sha256 = hashlib.sha256(
        "\n".join(chat_prompts).encode("utf-8")).hexdigest()

    print(f"[chat] chat_template_hash={chat_template_hash[:16]} "
          f"bare_prompt_sha256={bare_prompt_sha256[:16]} "
          f"rendered_prompt_sha256={rendered_prompt_sha256[:16]}")
    if not args.preflight:
        head_ids = vc.tokenizer(chat_prompts[0], add_special_tokens=True)["input_ids"]
        tail_id = head_ids[-1]
        print(f"[chat] sample_id=0 head_ids[:4]={head_ids[:4]} "
              f"final(injection)_token_id={tail_id} "
              f"decoded={vc.tokenizer.decode([tail_id])!r}")

    for alpha, (ls, le) in parsed:
        tag = f"mdf_{alpha}".replace("-", "neg")
        stem = "cruxeval_o_cot_chat_preflight" if args.preflight \
            else "cruxeval_o_cot_chat"
        subdir = f"{args.tag}_{tag}"
        out = os.path.join(args.out_dir, subdir,
                           f"{stem}_{args.size}_{ls}_{le}.json")
        if os.path.exists(out):
            print(f"skip existing {out}")
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)

        diff = raw_mask * alpha
        vc.steering_fire_count(reset=True)

        gen = []
        for i in tqdm(range(0, len(chat_prompts), args.batch_size),
                      desc=f"cruxeval-o-cot-chat a={alpha}"):
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

        if len(gen) != len(samples):
            die(f"generation returned {len(gen)} rows for {len(samples)} "
                f"prompts at alpha={alpha}; zip() would silently drop "
                f"{abs(len(gen) - len(samples))} sample(s).")

        fires = vc.steering_fire_count()
        n_layers = len(utils.decoder_layer_range(ls, le))
        expect = 0 if alpha == 0 else n_layers * len(samples)
        if fires != expect:
            die(f"steering_fires {fires} != {expect} (L={n_layers}, "
                f"n={len(samples)}, alpha={alpha}); the intervention is "
                "unverified, so the cell is not usable.")

        rows = []
        n_truncated = 0
        for s, g in zip(samples, gen):
            text = g["text"]
            truncated = (g["stop_reason"] == "budget_exhausted")
            n_truncated += int(truncated)
            rows.append({"sample_id": s["sample_id"], "source_id": s["source_id"],
                         "generated": text,
                         "generated_token_count": g["generated_token_count"],
                         "stop_reason": g["stop_reason"],
                         "truncated": truncated,
                         **descriptive(text)})

        n_marker = sum(1 for r in rows if r["n_markers"] > 0)
        if n_marker == 0 and not args.preflight:
            die(f"cell {subdir} produced NO '####' marker in any of "
                f"{len(rows)} generations. Prompt and parser are frozen and "
                "may NOT be redesigned in response.")

        json.dump({"meta": {"protocol": PROTOCOL, "base_protocol": BASE_PROTOCOL,
                            "task": "cruxeval_o",
                            "model": "llama3", "size": args.size,
                            "alpha": alpha, "layer_start": ls, "layer_end": le,
                            "L": n_layers,
                            "role": ("workpoint" if alpha == WORKPOINT_ALPHA
                                     else "baseline"),
                            "mask_path": args.mask_path, "mask_sha256": mask_sha,
                            "max_new_tokens": args.max_new_tokens,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "batch_size": args.batch_size,
                            "cot": True, "few_shot": False,
                            "stop_strings": None,
                            "prefill_only": True, "prefill_tail_len": 1,
                            "chat_template_applied": True,
                            "prompt_wrapper_id": PROMPT_WRAPPER_ID,
                            "chat_template_hash": chat_template_hash,
                            "bare_prompt_sha256_prefix": PROMPT_COT_SHA256,
                            "bare_prompt_sha256": bare_prompt_sha256,
                            "rendered_prompt_sha256": rendered_prompt_sha256,
                            "steering_fires": fires,
                            "questions_sha256": meta["questions_sha256"],
                            "revision": meta["revision"],
                            "prompt_template": PROMPT_COT,
                            "prompt_sha256_prefix": PROMPT_COT_SHA256,
                            "anchor": ANCHOR,
                            "padding_side": vc.tokenizer.padding_side,
                            "provenance": device_note,
                            "contains_labels": False,
                            "accuracy_computed": False,
                            "n_truncated": n_truncated,
                            "truncation_rate": (n_truncated / len(rows)
                                               if rows else None),
                            "preflight": args.preflight,
                            "exploratory_followup": True,
                            "interface_condition": {
                                "purpose": ("chat-template interface-condition "
                                           "diagnostic for the bare CoT "
                                           "follow-up (cot-transfer-followup-"
                                           "v0): does apply_chat_template "
                                           "wrapping reduce llama3's "
                                           "loop/truncation and change "
                                           "accuracy at its own workpoint "
                                           "alpha=-6? Independent interface "
                                           "condition, NOT a redefinition of "
                                           "any workpoint and NEVER a "
                                           "replacement for the bare "
                                           "cruxeval_o_cot cells."),
                                "compares_against": (
                                    "components/llama3/cruxeval_cot_followup/"
                                    "mdf_<alpha>/cruxeval_o_cot_8B_11_20.json "
                                    "(the bare CoT follow-up)"),
                                "same_300_items": True, "same_order": True,
                                "same_cot_prompt_body": True,
                                "same_mask": True,
                                "only_variable": ("prompt wrapping "
                                                  "(bare-string vs chat "
                                                  "template)"),
                            },
                            "n": len(rows)},
                   "data": rows}, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  wrote {out}  steering_fires={fires}  marker_rate="
              f"{n_marker / len(rows):.3f}  truncation_rate="
              f"{n_truncated / len(rows):.3f}")

    print("\nGeneration complete. NO accuracy was computed -- by construction.")
    print("This is a chat-template INTERFACE-CONDITION diagnostic; it does "
          "not replace the bare cruxeval-p4c-v0 or cot-transfer-followup-v0 "
          "results.")
    print("Score with: python eval_cruxeval.py --protocol "
          f"{PROTOCOL} --generations <mdf_0 file> <mdf_neg6 file> "
          "--gold_file <gold> --out <out.json>")


if __name__ == "__main__":
    main()
