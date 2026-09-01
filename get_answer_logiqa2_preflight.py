#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LogiQA 2.0 P4 FORMAT-ONLY PREFLIGHT (`logiqa2-p4-v0` + `p4-amend-02`).

Runs the 20 no-gold preflight items through all four cells at
`max_new_tokens=512` and reports FORMAT ONLY: does the output carry a unique
`Final answer: [A-D]`, how long is it, and did it stop at EOS or exhaust the
budget.

WHAT THIS DELIBERATELY CANNOT DO
--------------------------------
* It cannot compute accuracy. The preflight input file carries no gold field
  (`data_logiqa2.py` builds it from a whitelist), and this script additionally
  refuses to run if a gold field appears. Protocol section 4 forbids accuracy
  here, and the P2 label firewall established that "unused" is far weaker than
  "unreachable".
* It cannot run the formal 300. `--n` is not exposed and the input path is
  asserted to be the preflight file with `contains_labels: false`. The formal
  runner is a separate entry point that does not exist until stage 1 freezes
  the budget.
* It cannot redesign anything. If the format is wrong, the protocol's response
  is a HARD STOP, not a new prompt or a looser parser.

The only decision this feeds is the frozen budget rule:

    if ANY of the 20 outputs in ANY of the 4 cells has generated length
    >= 511, the formal budget becomes 1024; otherwise it stays 512.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llms import VicundaModel                                  # noqa: E402
from utils import parse_configs, decoder_layer_range           # noqa: E402

PROTOCOL = "logiqa2-p4-v0"
AMENDMENT = "p4-amend-02"

# Frozen verbatim by p4-amend-02, which REPLACED amend-01's prompt before any
# generation existed. amend-01 said "Think step by step." with a "Reasoning: "
# anchor -- an explicit CoT elicitation, which confounds the question P4 asks
# (whether a fixed workpoint changes SPONTANEOUS reasoning formation). The
# instruction here is output-format only and the anchor is neutral, so whether
# the model reasons before answering is an OBSERVABLE rather than a given.
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
PROMPT_SHA256 = "c42dc9c81f117a6c"          # first 16 hex, asserted at runtime
ANSWER_RE = re.compile(r"Final answer:\s*([A-D])\b")

PREFLIGHT_BUDGET = 512
UPGRADE_THRESHOLD = PREFLIGHT_BUDGET - 1    # >= 511, never == 512
UPGRADED_BUDGET = 1024

LABEL_FIELDS = {"answer", "label", "gold", "gold_answer", "correct",
                "target", "solution"}


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def build_prompt(item):
    import hashlib
    if hashlib.sha256(PROMPT.encode()).hexdigest()[:16] != PROMPT_SHA256:
        die(f"prompt template does not match the frozen sha256; the prompt is "
            f"frozen by {AMENDMENT} and may not be edited")
    o = item["options"]
    if len(o) != 4:
        die(f"item {item['sample_id']} has {len(o)} options")
    p = PROMPT.format(passage=item["passage"].strip(),
                      question=item["question"].strip(),
                      opt_a=o[0], opt_b=o[1], opt_c=o[2], opt_d=o[3])
    if not p.endswith("Response: "):
        die("prompt does not end at the frozen anchor 'Response: '")
    return p


def load_preflight(path):
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    meta, data = blob.get("meta", {}), blob.get("data", [])
    if meta.get("contains_labels") is not False:
        die(f"{path} is not the no-gold preflight file "
            f"(contains_labels={meta.get('contains_labels')!r}). This entry "
            f"point refuses anything else.")
    if len(data) != 20:
        die(f"preflight must be 20 items, got {len(data)}")
    for it in data:
        leaked = LABEL_FIELDS & set(it)
        if leaked:
            die(f"preflight item {it.get('sample_id')} carries label field(s) "
                f"{sorted(leaked)}; accuracy must be unreachable here")
    raw = json.dumps(data)
    for f_ in LABEL_FIELDS:
        if f'"{f_}"' in raw:
            die(f"preflight payload mentions label field {f_!r}")
    return meta, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama3", "qwen2.5"])
    ap.add_argument("--size", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--configs", required=True, nargs="+",
                    help="exactly two cells: the alpha=0 baseline and the "
                         "transferred workpoint")
    ap.add_argument("--preflight_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=8)   # frozen by amend-01
    args = ap.parse_args()

    if os.path.exists(args.out):
        die(f"{args.out} exists; refusing to overwrite")

    meta, items = load_preflight(args.preflight_file)
    print(f"[pre] {PROTOCOL} / {AMENDMENT}")
    print(f"[pre] {len(items)} items, NO gold (verified)")
    print(f"[pre] budget {PREFLIGHT_BUDGET}, upgrade if any length >= "
          f"{UPGRADE_THRESHOLD}")

    cfgs = parse_configs(args.configs)
    if len(cfgs) != 2:
        die(f"expected exactly 2 cells, got {len(cfgs)}")

    prompts = [build_prompt(it) for it in items]
    print(f"[pre] prompts built; all end at the frozen anchor")

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
        print(f"\n[pre] cell alpha={alpha} band=[{ls},{le}) L={L}")

        diff = list(raw_mask * alpha)
        vc.steering_fire_count(reset=True)
        out = vc.regenerate(prompts, diff_matrices=diff, max_new_tokens=PREFLIGHT_BUDGET,
                            temperature=0.0, prefill_only=True,
                            batch_size=args.batch_size, prefill_tail_len=1,
                            return_metadata=True)
        fires = vc.steering_fire_count(reset=False)
        expect = 0 if alpha == 0 else L * len(prompts) * 1
        print(f"[pre] steering_fires={fires} expected={expect}")
        if fires != expect:
            die(f"steering_fires {fires} != expected {expect}; the intervention "
                f"is unverified and no output from this cell is readable")

        rows = []
        for it, r in zip(items, out):
            m = ANSWER_RE.findall(r["text"])
            rows.append({
                "sample_id": it["sample_id"], "key": it["key"],
                "generated": r["text"],
                # frozen field: untruncated, special tokens PRESERVED, not
                # stripped -- `generated` is the cleaned text the parser reads
                "raw_text": r["raw_text"],
                "generated_token_count": r["generated_token_count"],
                "stop_reason": r["stop_reason"],
                "n_matches": len(m),
                "last_match": m[-1] if m else None,
                "first_match": m[0] if m else None,
                "agree_first_last": (len(m) > 0 and m[0] == m[-1]),
            })
        results[tag] = {"alpha": alpha, "layer_start": ls, "layer_end": le,
                        "L": L, "steering_fires": fires, "rows": rows}

        n_ok = sum(1 for x in rows if x["n_matches"] >= 1)
        n_trunc = sum(1 for x in rows if x["stop_reason"] == "budget_exhausted")
        lens = [x["generated_token_count"] for x in rows]
        print(f"[pre] format {n_ok}/{len(rows)} | budget_exhausted {n_trunc} "
              f"| len min/med/max {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")

    payload = {
        "meta": {"protocol": PROTOCOL, "amendment": AMENDMENT,
                 "model": args.model, "size": args.size,
                 "preflight_budget": PREFLIGHT_BUDGET,
                 "upgrade_threshold": UPGRADE_THRESHOLD,
                 "upgraded_budget": UPGRADED_BUDGET,
                 "preflight_digest": meta.get("digest"),
                 "accuracy_computed": False,
                 "prompt_sha256_prefix": PROMPT_SHA256},
        "cells": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[pre] wrote {args.out}")
    print("[pre] NOTE: the 512/1024 decision is made across ALL FOUR cells "
          "together, by decide_p4_budget.py -- not per model.")


if __name__ == "__main__":
    main()
