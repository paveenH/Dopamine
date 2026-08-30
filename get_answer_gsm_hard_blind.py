#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSM-Hard generation for the P3 BLIND cross-task validation. LABEL-FREE.

Protocol: p3-v1 (`docs/PREREG_P3.md`, tag `p3-prereg-v1`).

WHY THIS EXISTS INSTEAD OF REUSING get_answer_regenerate_gsm8k.py.
That script calls `is_correct_gsm8k` per sample, writes `correct_*` fields and
PRINTS accuracy (lines 83-96). Running it would compute and display GSM-Hard
accuracy during generation, which is exactly what P3 must not do -- the blind
property, once lost, cannot be restored by any later re-freeze. So the
generation path is forked, not reused.

WHAT IS IDENTICAL TO THE FROZEN GSM8K PATH (deliberately, so the commitment
features mean the same thing they did in P2):
  - prompt construction: neutral -> templates["neutral"], else templates["neg"],
    bypassing construct_prompt's default(+honest) branch, matching
    track_dopamine_signal.py
  - vc.regenerate(...) with the same generation arguments
  - prefill-only steering via diff_matrices

WHAT IS REMOVED: extract_gsm8k_answer, is_correct_gsm8k, `answer`, `correct_*`,
`pred_answer_*`, accuracy printing, the summary CSV's accuracy columns. The
input file physically has no gold field, so a label is unreachable, not merely
unused.

Output per cell: sample_id / question / generated / alpha / provenance.

@author: paveenhuang
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
from tqdm import tqdm

from llms import VicundaModel
from template import select_templates
import utils

FORBIDDEN_KEYS = ("answer", "gold", "gold_answer", "correct", "accuracy", "target")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--size", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--questions", required=True,
                   help="the LABEL-FREE questions JSON from data_gsm_hard.py")
    p.add_argument("--mask_path", required=True)
    # nargs="+" like every other runner here (get_answer_regenerate_gsm8k.py
    # uses nargs="*"). Without it argparse takes only the first dose and reports
    # the rest as unrecognized.
    p.add_argument("--configs", required=True, nargs="+",
                   help="e.g. neg8-11-20 neg6-11-20 neg4-11-20 0-11-20 4-11-20")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_new_tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--limit", type=int, default=0, help="format preflight only; 0 = all")
    return p.parse_args()


def load_questions(path):
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d["meta"], d["data"]
    if meta.get("contains_labels") is not False:
        sys.exit("FAIL: questions file does not declare contains_labels=false")
    for s in data:                       # fail closed on any leaked label
        bad = [k for k in s if k.lower() in FORBIDDEN_KEYS]
        if bad:
            sys.exit(f"FAIL: label field {bad} present in the questions file; "
                     "generation must be label-free")
    return meta, data


def main():
    args = parse_args()
    meta, samples = load_questions(args.questions)
    if args.limit:
        samples = samples[: args.limit]
        print(f"[PREFLIGHT] limited to {len(samples)} samples -- format check only")

    templates = select_templates(suite="default", task="gsm8k", cot=False)
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    raw_mask = np.load(args.mask_path)
    mask_sha = hashlib.sha256(open(args.mask_path, "rb").read()).hexdigest()
    os.makedirs(args.out_dir, exist_ok=True)

    for alpha, (ls, le) in utils.parse_configs(args.configs):
        tag = f"mdf_{alpha}".replace("-", "neg")
        out = os.path.join(args.out_dir, tag,
                           f"gsm_hard_{args.size}_{ls}_{le}.json")
        if os.path.exists(out):
            print(f"skip existing {out}")
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)

        diff = raw_mask * alpha
        vc.steering_fire_count(reset=True)

        prompts = [templates["neutral"].format(context=s["question"]) for s in samples]
        gen = []
        for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"gsm8k-hard a={alpha}"):
            gen.extend(vc.regenerate(
                prompts[i: i + args.batch_size],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                diff_matrices=list(diff),
                batch_size=args.batch_size,
            ))

        fires = vc.steering_fire_count()
        rows = [{"sample_id": s["sample_id"], "question": s["question"],
                 "generated": g} for s, g in zip(samples, gen)]
        json.dump({"meta": {"protocol": "p3-v1", "task": "gsm_hard",
                            "model": args.model, "size": args.size,
                            "alpha": alpha, "layer_start": ls, "layer_end": le,
                            "mask_path": args.mask_path, "mask_sha256": mask_sha,
                            "max_new_tokens": args.max_new_tokens,
                            "temperature": args.temperature,
                            "batch_size": args.batch_size,
                            "steering_fires": fires,
                            "questions_sha256": meta["questions_sha256"],
                            "prompt_template": templates["neutral"],
                            "contains_labels": False,
                            "n": len(rows)},
                   "data": rows}, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  wrote {out}  steering_fires={fires}")

    print("\nGeneration complete. NO accuracy was computed -- by construction.")
    print("Next: extract commitment features, freeze p3_predictions.json, "
          "and only then unseal gold.")


if __name__ == "__main__":
    main()
