#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LogiQA 2.0 P4 FORMAL RUNNER (`logiqa2-p4-v0`, amendments 02/03/04/05/06).

Generates the 300 formal items through one model's two cells and writes the raw
generations plus the frozen descriptive fields. It does NOT score anything.

WHY GENERATION AND SCORING ARE SEPARATE
---------------------------------------
Accuracy is computed by `eval_logiqa2.py`, which runs afterwards and reads the
gold. Keeping them apart means a generation run cannot quietly become an
accuracy run, and it mirrors the P3 split that made the ordering guarantee
checkable rather than merely intended. This file never opens the gold file.

FROZEN, ASSERTED AT RUNTIME
---------------------------
* prompt template sha256 c42dc9c81f117a6c (p4-amend-02) -- editing it is fatal
* anchor "Response: ", prefill_only, tail_len=1, greedy, bs=8
* max_new_tokens=1024 (p4-amend-04, stage-1 freeze; overridable ONLY to re-run
  the frozen value, never silently)
* steering_fires must equal 0 at alpha=0 and L*B*1 otherwise
* the 300 items must carry the frozen digest 4d4b25e071a2a6dd
"""

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llms import VicundaModel                                  # noqa: E402
from utils import parse_configs, decoder_layer_range           # noqa: E402

PROTOCOL = "logiqa2-p4-v0"
AMENDMENTS = "p4-amend-02,03,04,05,06"

PROMPT = (
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
PROMPT_SHA256 = "c42dc9c81f117a6c"
ANSWER_RE = re.compile(r"Final answer:\s*([A-D])\b")

FORMAL_BUDGET = 1024              # p4-amend-04
FORMAL_DIGEST = "4d4b25e071a2a6dd"
N_FORMAL = 300


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def build_prompt(item):
    if hashlib.sha256(PROMPT.encode()).hexdigest()[:16] != PROMPT_SHA256:
        die(f"prompt template does not match the frozen sha256; it is frozen "
            f"by p4-amend-02 and may not be edited")
    o = item["options"]
    if len(o) != 4:
        die(f"item {item['sample_id']} has {len(o)} options")
    p = PROMPT.format(passage=item["passage"].strip(),
                      question=item["question"].strip(),
                      opt_a=o[0], opt_b=o[1], opt_c=o[2], opt_d=o[3])
    if not p.endswith("Response: "):
        die("prompt does not end at the frozen anchor 'Response: '")
    return p


def is_loop(text):
    """The repo's is_loop convention, shared with GSM8K so the two tasks stay
    comparable: the final 40-character block recurs >= 4x. A permissive n-gram
    proxy read 80-86% on GSM8K and was entirely false positives."""
    b = text[-40:]
    return len(b) == 40 and text.count(b) >= 4


def descriptive(text, matches):
    """Frozen by p4-amend-05. Morphological ONLY -- text before the marker is
    NOT assumed to be reasoning, and no content judgement is made."""
    stripped = text.lstrip()
    m = ANSWER_RE.search(text)
    return {
        "n_markers": len(matches),
        "answer_first": stripped.startswith("Final answer"),
        "pre_marker_chars": m.start() if m else None,
        # auxiliary: confounded by tail length -- a longer repetition tail
        # lowers this independently of anything preceding the marker
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
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=FORMAL_BUDGET)
    args = ap.parse_args()

    if args.max_new_tokens != FORMAL_BUDGET:
        die(f"--max_new_tokens {args.max_new_tokens} != the stage-1 frozen "
            f"budget {FORMAL_BUDGET} (p4-amend-04). Changing it needs a new "
            f"amendment, not a flag.")
    if os.path.exists(args.out):
        die(f"{args.out} exists; refusing to overwrite")

    with open(args.formal_file, encoding="utf-8") as f:
        blob = json.load(f)
    meta, items = blob.get("meta", {}), blob.get("data", [])
    if len(items) != N_FORMAL:
        die(f"formal file has {len(items)} items, expected {N_FORMAL}")
    if not str(meta.get("digest", "")).startswith(FORMAL_DIGEST):
        die(f"formal digest {str(meta.get('digest'))[:16]} != frozen "
            f"{FORMAL_DIGEST}; this is not the sample stage 0 froze")
    ids = [it["sample_id"] for it in items]
    if sorted(ids) != list(range(N_FORMAL)):
        die("sample_ids are not a full 0..299 cover")

    print(f"[p4] {PROTOCOL} / {AMENDMENTS}")
    print(f"[p4] {len(items)} items, digest {str(meta.get('digest'))[:16]}")
    print(f"[p4] budget {FORMAL_BUDGET} (stage-1 frozen), bs={args.batch_size}")

    cfgs = parse_configs(args.configs)
    if len(cfgs) != 2:
        die(f"expected exactly 2 cells, got {len(cfgs)}")

    prompts = [build_prompt(it) for it in items]
    print("[p4] prompts built; all end at the frozen anchor")

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    raw_mask = np.load(args.mask)
    n_layers = len(vc._find_decoder_layers())
    if raw_mask.shape[0] != n_layers:
        die(f"mask has {raw_mask.shape[0]} rows but the model has {n_layers} "
            f"decoder layers")

    results = {}
    for alpha, (ls, le) in cfgs:
        band = decoder_layer_range(ls, le)
        L = len(band)
        tag = f"a{alpha}".replace("-", "neg")
        print(f"\n[p4] cell alpha={alpha} band=[{ls},{le}) L={L}")

        diff = list(raw_mask * alpha)
        vc.steering_fire_count(reset=True)
        out = vc.regenerate(prompts, diff_matrices=diff,
                            max_new_tokens=FORMAL_BUDGET, temperature=0.0,
                            prefill_only=True, batch_size=args.batch_size,
                            prefill_tail_len=1, return_metadata=True)
        fires = vc.steering_fire_count(reset=False)
        expect = 0 if alpha == 0 else L * len(prompts) * 1
        print(f"[p4] steering_fires={fires} expected={expect}")
        if fires != expect:
            die(f"steering_fires {fires} != expected {expect}; the "
                f"intervention is unverified and no output is readable")

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
                # MAIN = LAST, sensitivity = FIRST (frozen, p4-amend-02)
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
        print(f"[p4] scorable {n_ok}/{len(rows)} | exhausted {n_tr} | "
              f"answer_first {n_af} | degenerate {n_dg} | "
              f"len med {sorted(lens)[len(lens)//2]}")
        # p4-amend-05: only a cell with NO marker anywhere is a hard stop
        if n_ok == 0:
            die(f"cell {tag} produced NO valid marker in any of {len(rows)} "
                f"outputs. PREREG v0 section 4: HARD STOP. The prompt and "
                f"parser are frozen and may NOT be redesigned.")

    payload = {
        "meta": {"protocol": PROTOCOL, "amendments": AMENDMENTS,
                 "model": args.model, "size": args.size,
                 "max_new_tokens": FORMAL_BUDGET,
                 "batch_size": args.batch_size,
                 "formal_digest": meta.get("digest"),
                 "prompt_sha256_prefix": PROMPT_SHA256,
                 "accuracy_computed": False,
                 "parsing": {"main": "LAST", "sensitivity": "FIRST",
                             "rescue_generation": False, "denominator": 300}},
        "cells": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[p4] wrote {args.out}")
    print("[p4] NO accuracy computed. Score with eval_logiqa2.py once BOTH "
          "models are done.")


if __name__ == "__main__":
    main()
