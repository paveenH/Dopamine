#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BBH numeric-task generation for the P5 fixed-workpoint transfer. LABEL-FREE.

Protocol: `bbh-p5-v0`. Tasks: object_counting / multistep_arithmetic_two.

WHY A FORK RATHER THAN get_answer_regenerate_gsm8k.py
-----------------------------------------------------
That script scores each sample with `is_correct_gsm8k`, writes `correct_*` and
PRINTS accuracy. This is the same reason `get_answer_gsm_hard_blind.py` exists.
BBH gold is public, so there is no seal to break -- but the stage-0 gate is
judged on the alpha=0 cell alone, and the workpoint cells must be generated
without any accuracy in the loop, so that the gate decision and the transfer
test stay separable. Generation and scoring are separate scripts, as in P3/P4.

WHAT IS HELD IDENTICAL TO THE FROZEN GSM8K PATH (so the transfer test varies
only the reasoning content):
    prompt   templates["neutral"] from select_templates_gsm8k(suite="default",
             cot=False, wording="plain") -- the neutral #### directive. The
             "pushy" wording is a KNOWN early-#### inducer and is unavailable
             here (there is no --wording flag) so it cannot be reached.
    budget   max_new_tokens=768, temperature=0.0, batch_size=24
    steering vc.regenerate(..., diff_matrices=...), prefill-only, tail=1
    parser   utils.extract_gsm8k_answer -- IMPORTED by the scorer, not rewritten

NO FEW-SHOT AND NO CoT. The BBH release ships 3-shot CoT prompts; inheriting
them would add exemplars and explicit reasoning, re-mixing the variables LogiQA
was meant to separate. This runner has no --cot and no exemplar path at all.

alpha=0 PASSES A REAL ALL-ZERO MATRIX, exactly as the GSM8K driver does
(`diff = raw_mask * alpha`, unconditional). Hooks register and the zero add
executes; `steering_fires` reads 0 only because a zero row is not counted as
steered. This is NOT the pv6/PV10 "no hook at all" path -- do not "optimize"
it, or alpha=0 would run through a different execution path than the steered
cells and stop being their baseline.

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
from template import select_templates_gsm8k
import utils

TASKS = ("object_counting", "multistep_arithmetic_two")
FORBIDDEN_KEYS = ("target", "answer", "gold", "gold_answer", "correct",
                  "accuracy", "solution")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--size", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--questions", required=True,
                   help="the LABEL-FREE *_blind.json from data_bbh_numeric.py")
    p.add_argument("--mask_path", required=True)
    p.add_argument("--configs", required=True, nargs="+",
                   help="e.g. 0-11-20   (stage 0 is alpha=0 ONLY)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_new_tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=24)
    return p.parse_args()


def load_questions(path, task):
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d["meta"], d["data"]
    if meta.get("contains_labels") is not False:
        sys.exit("FAIL: questions file does not declare contains_labels=false. "
                 "Point --questions at the *_blind.json, not the gold copy.")
    if meta.get("task") != task:
        sys.exit(f"FAIL: --task {task!r} but the file declares "
                 f"{meta.get('task')!r}; a wrong pairing would silently score "
                 "one task's generations against another's manifest.")
    for s in data:                      # fail closed on any leaked label
        bad = [k for k in s if k.lower() in FORBIDDEN_KEYS]
        if bad:
            sys.exit(f"FAIL: label field {bad} present in the questions file; "
                     "generation must be label-free")
    return meta, data


def main():
    args = parse_args()
    meta, samples = load_questions(args.questions, args.task)

    # cot=False and wording="plain" are hardcoded, not flags: the CoT and pushy
    # variants each change what the commitment features mean, and this protocol
    # has no cell that uses them.
    templates = select_templates_gsm8k(suite="default", cot=False, wording="plain")
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    raw_mask = np.load(args.mask_path)
    mask_sha = hashlib.sha256(open(args.mask_path, "rb").read()).hexdigest()
    os.makedirs(args.out_dir, exist_ok=True)

    for alpha, (ls, le) in utils.parse_configs(args.configs):
        tag = f"mdf_{alpha}".replace("-", "neg")
        out = os.path.join(args.out_dir, tag,
                           f"bbh_{args.task}_{args.size}_{ls}_{le}.json")
        if os.path.exists(out):
            print(f"skip existing {out}")
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)

        diff = raw_mask * alpha
        vc.steering_fire_count(reset=True)

        prompts = [templates["neutral"].format(context=s["question"]) for s in samples]
        gen = []
        for i in tqdm(range(0, len(prompts), args.batch_size),
                      desc=f"bbh-{args.task} a={alpha}"):
            gen.extend(vc.regenerate(
                prompts[i: i + args.batch_size],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                diff_matrices=list(diff),
                batch_size=args.batch_size,
            ))

        fires = vc.steering_fire_count()
        # L = number of steered decoder layers; alpha=0 must read 0 fires
        # because every mask row is zero, not because no hook ran.
        n_layers = len(utils.decoder_layer_range(ls, le))
        expect = 0 if alpha == 0 else n_layers * len(samples)
        if fires != expect:
            sys.exit(f"FAIL: steering_fires {fires} != {expect} "
                     f"(L={n_layers}, n={len(samples)}, alpha={alpha}); "
                     "the intervention is unverified, so the cell is not usable.")

        rows = [{"sample_id": s["sample_id"], "question": s["question"],
                 "generated": g} for s, g in zip(samples, gen)]
        json.dump({"meta": {"protocol": "bbh-p5-v0", "task": args.task,
                            "model": args.model, "size": args.size,
                            "alpha": alpha, "layer_start": ls, "layer_end": le,
                            "L": n_layers,
                            "mask_path": args.mask_path, "mask_sha256": mask_sha,
                            "max_new_tokens": args.max_new_tokens,
                            "temperature": args.temperature,
                            "batch_size": args.batch_size,
                            "cot": False, "few_shot": False,
                            "steering_fires": fires,
                            "questions_sha256": meta["questions_sha256"],
                            "revision": meta["revision"],
                            "prompt_template": templates["neutral"],
                            "contains_labels": False,
                            "accuracy_computed": False,
                            "n": len(rows)},
                   "data": rows}, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  wrote {out}  steering_fires={fires}")

    print("\nGeneration complete. NO accuracy was computed -- by construction.")
    print("Next: python eval_bbh_numeric.py   (the only script that reads gold)")


if __name__ == "__main__":
    main()
