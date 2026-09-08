#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRUXEval-O chat-template four-point dose sweep, llama3 ONLY.
Two protocols, selected by --mode:
    --mode cot    -> protocol "cruxeval-o-cot-chat-sweep-v1"
    --mode nocot  -> protocol "cruxeval-o-nocot-chat-sweep-v1"

WHY A SWEEP. The existing chat diagnostic (get_answer_cruxeval_cot_chat.py,
protocol cruxeval-o-cot-chat-v1) ran a SINGLE pre-specified comparison
(llama3's frozen GSM8K workpoint alpha=-6 vs 0) under the chat template and
found chat sharply reduces loop/truncation/no-marker rates but showed no
detectable accuracy effect at that one dose (m=1, p=0.88). This script asks a
DIFFERENT, EXPLORATORY question: is there ANY task-specific workpoint among
{-6,-4,0,+4} under the chat interface, and does CoT change that dose-response?
It does NOT replace, redefine, or supersede the m=1 cruxeval-o-cot-chat-v1
result, which stays untouched on disk under its own protocol string.

WHAT IS HELD IDENTICAL to the existing bare/chat CRUXEval-O runners
--------------------------------------------------------------------
  - the SAME 300 items, SAME sample_id order, SAME frozen digest
    (FORMAL_DIGEST, imported)
  - --mode cot:   the SAME CoT prompt BODY, verbatim, imported from
                  get_answer_cruxeval_cot.py (PROMPT_COT / PROMPT_COT_SHA256)
  - --mode nocot: the SAME No-CoT prompt BODY, verbatim, imported from
                  get_answer_cruxeval.py (PROMPT / PROMPT_SHA256)
  - the SAME FIRST-marker "#### <Python literal>" convention in both modes
  - the SAME mask file, band [11,20), FULL 32-row mask (never pre-sliced --
    injection range is controlled by the layer band alone, exactly as every
    other launcher in this repo)
  - max_new_tokens/batch_size/temperature/top_p are INHERITED, mode-specific:
    --mode cot   uses get_answer_cruxeval_cot.py's BUDGET/BATCH_SIZE (768/24)
    --mode nocot uses get_answer_cruxeval.py's    BUDGET/BATCH_SIZE (768/24)
    (both currently 768/24; imported, never a separately hardcoded literal,
    so a future change to either frozen runner is inherited automatically
    rather than silently drifting)
  - prefill_only=True, prefill_tail_len=1, greedy (temperature=0.0)
  - alpha=0 uses the SAME generation path as every other alpha (an all-zero
    diff matrix -- no separate no-op branch)
  - return_metadata=True: every row carries generated_token_count,
    stop_reason, truncated -- same convention as
    get_answer_cruxeval_cot_chat.py

THE ONLY EXPERIMENTAL VARIABLE relative to the bare runner of the same mode is
the prompt wrapper: the identical bare prompt STRING (unchanged from either
frozen build_prompt()) is placed as the sole user-turn content in
tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
tokenize=False, add_generation_prompt=True), with a duplicated leading BOS
stripped via bandit_pv6_episode._strip_leading_bos before vc.regenerate()
tokenizes it (add_special_tokens=True internally) -- the same pattern
get_answer_cruxeval_cot_chat.py and proofwriter_owa/get_answer_proofwriter_owa
_chat.py already use.

DOSES: alpha in {-6, -4, 0, +4} in BOTH modes -- llama3's own frozen
No-CoT/CoT diagnostic matrix (EXPECTED_CELLS in get_answer_cruxeval.py /
get_answer_cruxeval_cot.py), never a new search. This is a NEW, INDEPENDENT
generation of all four points under the chat wrapper -- it does not read,
reuse, or overwrite any existing {-6,0} cruxeval-o-cot-chat-v1 file, nor any
bare cruxeval-p4c-v0 / cot-transfer-followup-v0 file.

Model is ALWAYS the Hugging Face id meta-llama/Llama-3.1-8B-Instruct.

OUTPUT (isolated by protocol-specific tag under the mode's own out_dir; never
overwrites the {-6,0} single-comparison chat cells or any bare cell):
    --mode cot:
      components/llama3/cruxeval_cot_followup/chat_sweep_v1_mdf_{0,neg6,neg4,4}/
          cruxeval_o_cot_chat_sweep_8B_11_20.json
      meta.protocol = "cruxeval-o-cot-chat-sweep-v1"
    --mode nocot:
      components/llama3/cruxeval/chat_sweep_v1_mdf_{0,neg6,neg4,4}/
          cruxeval_o_nocot_chat_sweep_8B_11_20.json
      meta.protocol = "cruxeval-o-nocot-chat-sweep-v1"

Score with eval_cruxeval_chat_sweep.py (a NEW, separate scorer -- it does not
modify eval_cruxeval.py, whose default cruxeval-p4c-v0 behaviour and
--protocol cruxeval-o-cot-chat-v1 m=1 behaviour are both untouched).

No Qwen. This script does not modify get_answer_cruxeval.py,
get_answer_cruxeval_cot.py, get_answer_cruxeval_cot_chat.py, eval_cruxeval.py,
run_cruxeval.sh, run_cruxeval_cot.sh, run_cruxeval_cot_chat.sh, or any
GSM8K/other-task runner.

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

# CoT-mode ingredients, imported verbatim -- never redefined here.
from get_answer_cruxeval_cot import (                                 # noqa: E402
    PROMPT_COT, PROMPT_COT_SHA256,
    N_FORMAL as N_FORMAL_COT, N_PREFLIGHT as N_PREFLIGHT_COT,
    FORMAL_DIGEST as FORMAL_DIGEST_COT,
    BUDGET as BUDGET_COT, BATCH_SIZE as BATCH_SIZE_COT,
    LABEL_FIELDS as LABEL_FIELDS_COT,
    ANCHOR as ANCHOR_COT,
    build_prompt as build_prompt_cot,
    descriptive as descriptive_cot,
    die,
)
# No-CoT-mode ingredients, imported verbatim -- never redefined here.
from get_answer_cruxeval import (                                     # noqa: E402
    PROMPT as PROMPT_NOCOT, PROMPT_SHA256 as PROMPT_NOCOT_SHA256,
    N_FORMAL as N_FORMAL_NOCOT, N_PREFLIGHT as N_PREFLIGHT_NOCOT,
    FORMAL_DIGEST as FORMAL_DIGEST_NOCOT,
    BUDGET as BUDGET_NOCOT, BATCH_SIZE as BATCH_SIZE_NOCOT,
    LABEL_FIELDS as LABEL_FIELDS_NOCOT,
    ANCHOR as ANCHOR_NOCOT,
    build_prompt as build_prompt_nocot,
    descriptive as descriptive_nocot,
)

# Both frozen digests must agree -- both runners were built from the SAME
# frozen 300-item sample (data_cruxeval.py). If they ever diverged, "the same
# 300 items" would be false for one of the two modes without anything else
# noticing.
if FORMAL_DIGEST_COT != FORMAL_DIGEST_NOCOT:
    die(f"CoT digest {FORMAL_DIGEST_COT!r} != No-CoT digest "
        f"{FORMAL_DIGEST_NOCOT!r}; the two frozen runners disagree on the "
        "sample, which should be structurally impossible")
FORMAL_DIGEST = FORMAL_DIGEST_COT
N_FORMAL = N_FORMAL_COT

MODE_CONFIG = {
    "cot": dict(
        protocol="cruxeval-o-cot-chat-sweep-v1",
        base_protocol="cot-transfer-followup-v0",
        prompt_template=PROMPT_COT, prompt_sha256=PROMPT_COT_SHA256,
        anchor=ANCHOR_COT, n_preflight=N_PREFLIGHT_COT,
        budget=BUDGET_COT, batch_size=BATCH_SIZE_COT,
        label_fields=LABEL_FIELDS_COT,
        build_prompt=build_prompt_cot, descriptive=descriptive_cot,
        cot_flag=True, stem="cruxeval_o_cot_chat_sweep",
        default_out_subdir="cruxeval_cot_followup", tag_prefix="chat_sweep_v1",
    ),
    "nocot": dict(
        protocol="cruxeval-o-nocot-chat-sweep-v1",
        base_protocol="cruxeval-p4c-v0",
        prompt_template=PROMPT_NOCOT, prompt_sha256=PROMPT_NOCOT_SHA256,
        anchor=ANCHOR_NOCOT, n_preflight=N_PREFLIGHT_NOCOT,
        budget=BUDGET_NOCOT, batch_size=BATCH_SIZE_NOCOT,
        label_fields=LABEL_FIELDS_NOCOT,
        build_prompt=build_prompt_nocot, descriptive=descriptive_nocot,
        cot_flag=False, stem="cruxeval_o_nocot_chat_sweep",
        default_out_subdir="cruxeval", tag_prefix="chat_sweep_v1",
    ),
}

PROMPT_WRAPPER_ID = "llama3-chat-template-v1"
EXPECTED_ALPHAS = {0, -6, -4, 4}   # llama3's own frozen four-point matrix


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=sorted(MODE_CONFIG))
    p.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--size", default="8B")
    p.add_argument("--questions", required=True,
                   help="the LABEL-FREE *_formal_blind.json -- SAME file "
                        "every CRUXEval-O runner uses")
    p.add_argument("--mask_path", required=True,
                   help="the FULL 32-row mask "
                        "(mask/llama3_non_logits/nmd_0.5_11_20_8B.npy) -- "
                        "never pre-sliced to the 9-layer band")
    p.add_argument("--configs", required=True, nargs="+",
                   help="e.g. 0-11-20 neg6-11-20 neg4-11-20 4-11-20")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="default: the mode's own frozen budget")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=None,
                   help="default: the mode's own frozen batch size")
    p.add_argument("--tag", default=None,
                   help="output subdir prefix; default is the mode's own "
                        "chat_sweep_v1, isolated from every other cell tag "
                        "under the same parent tree")
    p.add_argument("--preflight", action="store_true",
                   help="FORMAT ONLY: first N_PREFLIGHT items, output tagged "
                        "preflight so the scorer refuses it.")
    return p.parse_args()


def load_questions(cfg, path, preflight=False):
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
        data = data[:cfg["n_preflight"]]
    for s in data:
        bad = [k for k in s if k.lower() in cfg["label_fields"]]
        if bad:
            die(f"label field {bad} present in the questions file; "
                "generation must be label-free")
    return meta, data


def assert_no_double_bos(vc, text: str, label: str) -> None:
    bos_id = getattr(vc.tokenizer, "bos_token_id", None)
    ids = vc.tokenizer(text, add_special_tokens=True)["input_ids"]
    if bos_id is not None and len(ids) >= 2 and ids[:2] == [bos_id, bos_id]:
        die(f"double BOS in the {label} chat prompt (head={ids[:4]}) -- "
            "strip_leading_bos did not remove the chat template's own "
            "serialized BOS text before this add_special_tokens=True call.")


def main():
    args = parse_args()
    cfg = MODE_CONFIG[args.mode]

    got = hashlib.sha256(cfg["prompt_template"].encode()).hexdigest()[:16]
    if got != cfg["prompt_sha256"]:
        die(f"[{args.mode}] prompt template sha256 {got} != the frozen "
            f"{cfg['prompt_sha256']}. The template is imported from the "
            "frozen runner and is frozen there; it may not be edited.")

    budget = args.max_new_tokens if args.max_new_tokens is not None else cfg["budget"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["batch_size"]
    if budget != cfg["budget"] or batch_size != cfg["batch_size"]:
        die(f"[{args.mode}] budget/batch are inherited frozen at "
            f"{cfg['budget']}/{cfg['batch_size']}; got {budget}/{batch_size}")
    if args.temperature != 0.0:
        die("temperature is frozen at 0.0 (greedy)")
    if args.top_p != 1.0:
        die("top_p is frozen at 1.0 (greedy decoding must not be diluted by "
            "nucleus sampling)")

    tag_prefix = args.tag if args.tag is not None else cfg["tag_prefix"]

    parsed = utils.parse_configs(args.configs)
    for alpha, (ls, le) in parsed:
        if (ls, le) != (11, 20):
            die(f"band {ls}-{le} is not llama3's frozen band 11-20")
        if alpha not in EXPECTED_ALPHAS:
            die(f"alpha={alpha} is not in the frozen four-point matrix "
                f"{sorted(EXPECTED_ALPHAS)} -- this sweep does not search "
                "new doses.")

    meta, samples = load_questions(cfg, args.questions, args.preflight)

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

    bare_prompts = [cfg["build_prompt"](s) for s in samples]
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

    print(f"[{args.mode}] chat_template_hash={chat_template_hash[:16]} "
          f"bare_prompt_sha256={bare_prompt_sha256[:16]} "
          f"rendered_prompt_sha256={rendered_prompt_sha256[:16]}")

    for alpha, (ls, le) in parsed:
        tag = f"mdf_{alpha}".replace("-", "neg")
        stem = f"{cfg['stem']}_preflight" if args.preflight else cfg["stem"]
        subdir = f"{tag_prefix}_{tag}"
        out = os.path.join(args.out_dir, subdir,
                           f"{stem}_{args.size}_{ls}_{le}.json")
        if os.path.exists(out):
            print(f"skip existing {out}")
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)

        diff = raw_mask * alpha
        vc.steering_fire_count(reset=True)

        gen = []
        for i in tqdm(range(0, len(chat_prompts), batch_size),
                      desc=f"{args.mode}-chat-sweep a={alpha}"):
            batch = chat_prompts[i: i + batch_size]
            gen.extend(vc.regenerate(
                batch,
                max_new_tokens=budget,
                temperature=args.temperature,
                top_p=args.top_p,
                diff_matrices=list(diff),
                batch_size=batch_size,
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
                         **cfg["descriptive"](text)})

        n_marker = sum(1 for r in rows if r["n_markers"] > 0)
        if n_marker == 0 and not args.preflight:
            die(f"cell {subdir} produced NO '####' marker in any of "
                f"{len(rows)} generations. Prompt and parser are frozen and "
                "may NOT be redesigned in response.")

        json.dump({"meta": {"protocol": cfg["protocol"],
                            "base_protocol": cfg["base_protocol"],
                            "task": "cruxeval_o",
                            "model": "llama3", "size": args.size,
                            "alpha": alpha, "layer_start": ls, "layer_end": le,
                            "L": n_layers,
                            "role": "sweep_point",
                            "mask_path": args.mask_path, "mask_sha256": mask_sha,
                            "max_new_tokens": budget,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "batch_size": batch_size,
                            "cot": cfg["cot_flag"], "few_shot": False,
                            "stop_strings": None,
                            "prefill_only": True, "prefill_tail_len": 1,
                            "chat_template_applied": True,
                            "prompt_wrapper_id": PROMPT_WRAPPER_ID,
                            "chat_template_hash": chat_template_hash,
                            "bare_prompt_sha256_prefix": cfg["prompt_sha256"],
                            "bare_prompt_sha256": bare_prompt_sha256,
                            "rendered_prompt_sha256": rendered_prompt_sha256,
                            "steering_fires": fires,
                            "questions_sha256": meta["questions_sha256"],
                            "revision": meta["revision"],
                            "prompt_template": cfg["prompt_template"],
                            "prompt_sha256_prefix": cfg["prompt_sha256"],
                            "anchor": cfg["anchor"],
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
                                "purpose": (
                                    "chat-template four-point dose sweep, "
                                    f"mode={args.mode}: is there ANY "
                                    "task-specific workpoint among "
                                    "{-6,-4,0,+4} under the chat interface? "
                                    "EXPLORATORY, own Holm m=3 family "
                                    "(3 non-zero doses vs this cell's own "
                                    "alpha=0), separate from and NOT "
                                    "superseding the m=1 single-comparison "
                                    "cruxeval-o-cot-chat-v1 result."),
                                "same_300_items": True, "same_order": True,
                                "same_prompt_body": True, "same_mask": True,
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
    print(f"[{args.mode}] chat four-point dose sweep. Score with "
          "eval_cruxeval_chat_sweep.py (separate from eval_cruxeval.py).")


if __name__ == "__main__":
    main()
