#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LogiQA 2.0 explicit-CoT follow-up runner (`cot-transfer-followup-v0`).

Fork of `get_answer_logiqa2.py` with exactly ONE planned change: the project's
existing GSM8K/MATH CoT cue ("Let's think step by step.\\n") is inserted before
the final-answer instruction. Everything else -- items, order, digest, blind/
gold firewall, mask, band, budget (1024, the P4 stage-1 frozen value), batch
size (8), parser (LAST=MAIN / FIRST=sensitivity), marker, anchor -- is
inherited verbatim from `logiqa2-p4-v0`. See `docs/PREREG_COT_TRANSFER_FOLLOWUP.md`.

THIS IS A POST-HOC EXPLORATORY FOLLOW-UP, NOT A REPLICATION. It does not
touch, rescale, or supersede the frozen No-CoT P4 result.

FROZEN, ASSERTED AT RUNTIME
---------------------------
* CoT prompt template sha256 275ea7768b2f94ec -- editing it is fatal
* anchor "Response: ", prefill_only, tail_len=1, greedy, bs=8
* max_new_tokens=1024 (matches the P4 stage-1 frozen budget; not re-derived)
* steering_fires must equal 0 at alpha=0 and L*B*1 otherwise
* the 300 items must carry the frozen P4 digest 4d4b25e071a2a6dd
* the input file must declare contains_labels: false
* the CoT template's final injection token (the tokenized tail of
  "Response: ") must equal the No-CoT template's -- verified on the actual
  tokenizer before any generation; a mismatch is a hard stop
"""

import argparse
import hashlib
import json
import os
import platform
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llms import VicundaModel                                  # noqa: E402
from utils import parse_configs, decoder_layer_range           # noqa: E402

PROTOCOL = "cot-transfer-followup-v0"
BASE_PROTOCOL = "logiqa2-p4-v0"
COT_CUE = "Let's think step by step.\n"

# The frozen No-CoT template, reused only to build the token-injection check --
# not used to generate anything here.
PROMPT_NOCOT = (
    "{passage}\n"
    "\n"
    "{question}\n"
    "A) {opt_a}\n"
    "B) {opt_b}\n"
    "C) {opt_c}\n"
    "D) {opt_d}\n"
    "\n"
    "End your response with 'Final answer: X' where X is A, B, C, or D.\n"
    "\n"
    "Response: "
)
PROMPT_NOCOT_SHA256 = "c42dc9c81f117a6c"

PROMPT_COT = (
    "{passage}\n"
    "\n"
    "{question}\n"
    "A) {opt_a}\n"
    "B) {opt_b}\n"
    "C) {opt_c}\n"
    "D) {opt_d}\n"
    "\n"
    f"{COT_CUE}"
    "End your response with 'Final answer: X' where X is A, B, C, or D.\n"
    "\n"
    "Response: "
)
PROMPT_COT_SHA256 = "275ea7768b2f94ec"
ANCHOR = "Response: "
ANSWER_RE = re.compile(r"Final answer:\s*([A-D])\b")

FORMAL_BUDGET = 1024
FORMAL_DIGEST = "4d4b25e071a2a6dd"
N_FORMAL = 300
FORMAL_BATCH_SIZE = 8

# CoT matrix extends the No-CoT 2-cell matrix to the 4-point shape shared by
# all three tasks in this follow-up (see PREREG_COT_TRANSFER_FOLLOWUP.md S1).
EXPECTED_CELLS = {
    "llama3":  {(0, 11, 20), (-6, 11, 20), (-4, 11, 20), (4, 11, 20)},
    "qwen2.5": {(0, 16, 22), (8, 16, 22), (6, 16, 22), (-6, 16, 22)},
}

LABEL_FIELDS = {"answer", "answer_letter", "label", "gold", "gold_answer",
                "correct", "target", "solution", "type"}


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def build_prompt(item):
    got = hashlib.sha256(PROMPT_COT.encode()).hexdigest()[:16]
    if got != PROMPT_COT_SHA256:
        die(f"CoT prompt template sha256 {got} != frozen {PROMPT_COT_SHA256}; "
            f"the template is frozen by PREREG_COT_TRANSFER_FOLLOWUP.md and may "
            f"not be edited")
    o = item["options"]
    if len(o) != 4:
        die(f"item {item['sample_id']} has {len(o)} options")
    p = PROMPT_COT.format(passage=item["passage"].strip(),
                          question=item["question"].strip(),
                          opt_a=o[0], opt_b=o[1], opt_c=o[2], opt_d=o[3])
    if not p.endswith(ANCHOR):
        die(f"CoT prompt does not end at the frozen anchor {ANCHOR!r}")
    return p


def verify_injection_token_equal(vc, sample_item):
    """Fail-closed check that the CoT and No-CoT templates share the same
    final prompt token (the anchor tail), so prefill-only tail_len=1 steering
    lands on the same site in both conditions. This must be VERIFIED on the
    real tokenizer, not assumed from matching anchor strings."""
    got_nocot = hashlib.sha256(PROMPT_NOCOT.encode()).hexdigest()[:16]
    if got_nocot != PROMPT_NOCOT_SHA256:
        die(f"No-CoT reference template sha256 {got_nocot} != frozen "
            f"{PROMPT_NOCOT_SHA256}; cannot verify injection-token equality "
            f"against a drifted reference")
    o = sample_item["options"]
    p_nocot = PROMPT_NOCOT.format(passage=sample_item["passage"].strip(),
                                  question=sample_item["question"].strip(),
                                  opt_a=o[0], opt_b=o[1], opt_c=o[2], opt_d=o[3])
    p_cot = build_prompt(sample_item)

    tok = vc.tokenizer
    ids_nocot = tok(p_nocot, add_special_tokens=True)["input_ids"]
    ids_cot = tok(p_cot, add_special_tokens=True)["input_ids"]
    last_nocot, last_cot = ids_nocot[-1], ids_cot[-1]
    dec_nocot = tok.decode([last_nocot])
    dec_cot = tok.decode([last_cot])
    print(f"[cot-followup] injection-token check: No-CoT last token id="
          f"{last_nocot} ({dec_nocot!r}), CoT last token id={last_cot} "
          f"({dec_cot!r})")
    if last_nocot != last_cot:
        die(f"injection token differs between CoT ({last_cot}, {dec_cot!r}) "
            f"and No-CoT ({last_nocot}, {dec_nocot!r}) templates. Per "
            f"PREREG_COT_TRANSFER_FOLLOWUP.md S3.4 this must be identical -- "
            f"the prompts are frozen wrong if it is not, and this is a hard "
            f"stop rather than something to patch around.")
    print("[cot-followup] injection-token equality VERIFIED")


def is_loop(text):
    """Shared is_loop convention (final 40-char block recurs >=4x), unchanged
    from get_answer_logiqa2.py so degeneracy stays comparable across the
    CoT/No-CoT and cross-task boundary."""
    b = text[-40:]
    return len(b) == 40 and text.count(b) >= 4


def descriptive(text, matches):
    """Morphological ONLY -- unchanged definition from get_answer_logiqa2.py.
    Text before the marker is NOT assumed to be reasoning."""
    stripped = text.lstrip()
    m = ANSWER_RE.search(text)
    return {
        "n_markers": len(matches),
        "answer_first": stripped.startswith("Final answer"),
        "pre_marker_chars": m.start() if m else None,
        "first_marker_pos": (m.start() / len(text)) if (m and text) else None,
        "degenerate": is_loop(text),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama3", "qwen2.5"])
    ap.add_argument("--size", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--configs", required=True, nargs="+")
    ap.add_argument("--formal_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=FORMAL_BATCH_SIZE)
    ap.add_argument("--max_new_tokens", type=int, default=FORMAL_BUDGET)
    args = ap.parse_args()

    if args.batch_size != FORMAL_BATCH_SIZE:
        die(f"--batch_size {args.batch_size} != the inherited P4 value "
            f"{FORMAL_BATCH_SIZE}; this follow-up changes only the prompt")
    if args.max_new_tokens != FORMAL_BUDGET:
        die(f"--max_new_tokens {args.max_new_tokens} != the inherited P4 "
            f"stage-1 budget {FORMAL_BUDGET}; this follow-up changes only "
            f"the prompt")
    if os.path.exists(args.out):
        die(f"{args.out} exists; refusing to overwrite")

    with open(args.formal_file, encoding="utf-8") as f:
        blob = json.load(f)
    meta, items = blob.get("meta", {}), blob.get("data", [])
    if meta.get("contains_labels") is not False:
        die(f"{args.formal_file} is not the BLIND formal file "
            f"(contains_labels={meta.get('contains_labels')!r})")
    for it in items:
        leaked = LABEL_FIELDS & set(it)
        if leaked:
            die(f"item {it.get('sample_id')} carries label field(s) "
                f"{sorted(leaked)}; gold must be unreachable during generation")
    if len(items) != N_FORMAL:
        die(f"formal file has {len(items)} items, expected {N_FORMAL}")
    if not str(meta.get("digest", "")).startswith(FORMAL_DIGEST):
        die(f"formal digest {str(meta.get('digest'))[:16]} != frozen "
            f"{FORMAL_DIGEST}; this is not the P4 sample")
    ids = [it["sample_id"] for it in items]
    if sorted(ids) != list(range(N_FORMAL)):
        die("sample_ids are not a full 0..299 cover")

    print(f"[cot-followup] {PROTOCOL} (base {BASE_PROTOCOL})")
    print(f"[cot-followup] {len(items)} items, digest {str(meta.get('digest'))[:16]}")
    print(f"[cot-followup] budget {FORMAL_BUDGET} (inherited from P4 stage-1), "
          f"bs={args.batch_size}")
    print("[cot-followup] this is a POST-HOC EXPLORATORY follow-up, not a "
          "replication of the No-CoT P4 result")

    cfgs = parse_configs(args.configs)
    got = {(al, ls, le) for al, (ls, le) in cfgs}
    want = EXPECTED_CELLS[args.model]
    if not got.issubset(want):
        die(f"{args.model} cells {sorted(got)} contain doses outside the "
            f"frozen CoT matrix {sorted(want)}; alpha is read from the "
            f"frozen GSM8K record, never re-searched")

    prompts = [build_prompt(it) for it in items]
    print("[cot-followup] prompts built; all end at the frozen anchor")

    if not os.path.isfile(args.mask):
        die(f"mask not found: {args.mask}")
    raw_mask = np.load(args.mask)
    if raw_mask.ndim != 2:
        die(f"mask has shape {raw_mask.shape}; expected 2-D (layers, hidden)")

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    n_layers = len(vc._find_decoder_layers())
    if raw_mask.shape[0] != n_layers:
        die(f"mask has {raw_mask.shape[0]} rows but the model has {n_layers} "
            f"decoder layers")

    verify_injection_token_equal(vc, items[0])

    # Provenance, not a constraint: cells are NOT required to share a GPU
    # (project-wide convention). Because bf16 greedy may vary across
    # hardware, a contrast between cells on different devices is reported as
    # a CROSS-RUN pairing; pairing means alignment by sample_id, never
    # hardware identity.
    device_note = {
        "host": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    results = {}
    for alpha, (ls, le) in cfgs:
        band = decoder_layer_range(ls, le)
        L = len(band)
        tag = f"a{alpha}".replace("-", "neg")
        print(f"\n[cot-followup] cell alpha={alpha} band=[{ls},{le}) L={L}")

        diff = list(raw_mask * alpha)
        vc.steering_fire_count(reset=True)
        out = vc.regenerate(prompts, diff_matrices=diff,
                            max_new_tokens=FORMAL_BUDGET, temperature=0.0,
                            prefill_only=True, batch_size=args.batch_size,
                            prefill_tail_len=1, return_metadata=True)
        fires = vc.steering_fire_count(reset=False)
        expect = 0 if alpha == 0 else L * len(prompts) * 1
        print(f"[cot-followup] steering_fires={fires} expected={expect}")
        if fires != expect:
            die(f"steering_fires {fires} != expected {expect}; the "
                f"intervention is unverified and no output is readable")

        if len(out) != N_FORMAL:
            die(f"generation returned {len(out)} outputs for {N_FORMAL} "
                f"prompts; zip() would silently drop items")

        rows = []
        for it, r in zip(items, out):
            m = ANSWER_RE.findall(r["text"])
            rows.append({
                "sample_id": it["sample_id"], "key": it["key"],
                "official_id": it["official_id"],
                "generated": r["text"], "raw_text": r["raw_text"],
                "generated_token_count": r["generated_token_count"],
                "stop_reason": r["stop_reason"],
                "n_matches": len(m),
                "last_match": m[-1] if m else None,
                "first_match": m[0] if m else None,
                "agree_first_last": (len(m) > 0 and m[0] == m[-1]),
                **descriptive(r["text"], m),
            })
        results[tag] = {"alpha": alpha, "layer_start": ls, "layer_end": le,
                        "L": L, "steering_fires": fires, "rows": rows}

        n_ok = sum(1 for x in rows if x["n_matches"] >= 1)
        n_tr = sum(1 for x in rows if x["stop_reason"] == "budget_exhausted")
        n_af = sum(1 for x in rows if x["answer_first"])
        n_dg = sum(1 for x in rows if x["degenerate"])
        lens = [x["generated_token_count"] for x in rows]
        print(f"[cot-followup] scorable {n_ok}/{len(rows)} | exhausted {n_tr} | "
              f"answer_first {n_af} | degenerate {n_dg} | "
              f"len med {sorted(lens)[len(lens)//2]}")
        if n_ok == 0:
            die(f"cell {tag} produced NO valid marker in any of {len(rows)} "
                f"outputs. Prompt and parser are frozen and may NOT be "
                f"redesigned in response.")

    payload = {
        "meta": {"protocol": PROTOCOL, "base_protocol": BASE_PROTOCOL,
                 "cot": True, "cot_cue": COT_CUE,
                 "model": args.model, "size": args.size,
                 "max_new_tokens": FORMAL_BUDGET,
                 "batch_size": args.batch_size,
                 "formal_digest": meta.get("digest"),
                 "prompt_sha256_prefix": PROMPT_COT_SHA256,
                 "accuracy_computed": False,
                 "exploratory_followup": True,
                 "provenance": device_note,
                 "parsing": {"main": "LAST", "sensitivity": "FIRST",
                             "rescue_generation": False, "denominator": 300}},
        "cells": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[cot-followup] wrote {args.out}")
    print("[cot-followup] NO accuracy computed. This is exploratory: it does "
          "not replace the frozen No-CoT P4 result.")


if __name__ == "__main__":
    main()
