#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BBH numeric-task explicit-CoT follow-up runner (`cot-transfer-followup-v0`).

Fork of `get_answer_bbh_numeric.py` with exactly ONE planned change: the
prompt is built with `cot=True` instead of `cot=False`, i.e.
`select_templates_gsm8k(suite="default", cot=True, wording="plain")["neutral"]`
instead of the `cot=False` variant. This is NOT a hand-written template -- it
is the project's existing GSM8K/MATH CoT template, already implemented in
`template.py` and used unmodified. The two templates differ by exactly the
line `Let's think step by step.` inserted before the `####` directive; the
final marker instruction and the `Answer: ` anchor are byte-identical.

Everything else -- questions, order, digest, blind/gold firewall, mask, band,
budget (768), batch size (24), parser (utils.extract_gsm8k_answer, FIRST=MAIN)
-- is inherited verbatim from `bbh-p4b-v0`. See
`docs/PREREG_COT_TRANSFER_FOLLOWUP.md`.

THIS IS A POST-HOC EXPLORATORY FOLLOW-UP, NOT A REPLICATION. It does not
touch, rescale, or supersede the frozen No-CoT P4b `object_counting` result.

TASK IS RESTRICTED TO object_counting: it is the only BBH task in this
follow-up's frozen 6-comparison Holm family (PREREG_COT_TRANSFER_FOLLOWUP.md
S4.1). `multistep_arithmetic_two` has no No-CoT P4b workpoint cell and no CoT
cell authorised here -- accepting it would silently run an unregistered task.

alpha=0 PASSES A REAL ALL-ZERO MATRIX, exactly as every other runner in this
family. Hooks register and the zero add executes; steering_fires reads 0 only
because a zero row is not counted as steered.

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

from llms import VicundaModel
from template import select_templates_gsm8k
import utils

PROTOCOL = "cot-transfer-followup-v0"
BASE_PROTOCOL = "bbh-p4b-v0"
# Restricted to the one task this follow-up's Holm family covers. See the
# module docstring: multistep_arithmetic_two is not authorised here.
TASKS = ("object_counting",)
FORBIDDEN_KEYS = ("target", "answer", "gold", "gold_answer", "correct",
                  "accuracy", "solution")

# CoT matrix mirrors the frozen GSM8K workpoint plus the neighbour/reverse
# doses already used in P4c, so all three follow-up tasks share one shape.
EXPECTED_CELLS = {
    "llama3":  {(0, 11, 20), (-6, 11, 20), (-4, 11, 20), (4, 11, 20)},
    "qwen2.5": {(0, 16, 22), (8, 16, 22), (6, 16, 22), (-6, 16, 22)},
}


def die(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--size", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--questions", required=True,
                   help="the LABEL-FREE *_blind.json from data_bbh_numeric.py")
    p.add_argument("--mask_path", required=True)
    p.add_argument("--configs", required=True, nargs="+")
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
        die("questions file does not declare contains_labels=false. Point "
            "--questions at the *_blind.json, not the gold copy.")
    if meta.get("task") != task:
        die(f"--task {task!r} but the file declares {meta.get('task')!r}")
    for s in data:
        bad = [k for k in s if k.lower() in FORBIDDEN_KEYS]
        if bad:
            die(f"label field {bad} present in the questions file; "
                "generation must be label-free")
    return meta, data


def verify_injection_token_equal(vc, templates, sample_question):
    """The CoT and No-CoT GSM8K templates are byte-identical from the anchor
    'Answer: ' backward, so their final prompt token must match. Verified on
    the real tokenizer rather than assumed, per
    PREREG_COT_TRANSFER_FOLLOWUP.md S3.4."""
    nocot_templates = select_templates_gsm8k(suite="default", cot=False,
                                             wording="plain")
    p_cot = templates["neutral"].format(context=sample_question)
    p_nocot = nocot_templates["neutral"].format(context=sample_question)
    tok = vc.tokenizer
    ids_cot = tok(p_cot, add_special_tokens=True)["input_ids"]
    ids_nocot = tok(p_nocot, add_special_tokens=True)["input_ids"]
    last_cot, last_nocot = ids_cot[-1], ids_nocot[-1]
    dec_cot, dec_nocot = tok.decode([last_cot]), tok.decode([last_nocot])
    print(f"[cot-followup] injection-token check: No-CoT last token id="
          f"{last_nocot} ({dec_nocot!r}), CoT last token id={last_cot} "
          f"({dec_cot!r})")
    if last_cot != last_nocot:
        die(f"injection token differs between CoT ({last_cot}, {dec_cot!r}) "
            f"and No-CoT ({last_nocot}, {dec_nocot!r}) templates -- hard stop "
            f"per PREREG_COT_TRANSFER_FOLLOWUP.md S3.4")
    print("[cot-followup] injection-token equality VERIFIED")


def main():
    args = parse_args()
    meta, samples = load_questions(args.questions, args.task)

    cfgs = utils.parse_configs(args.configs)
    got = {(al, ls, le) for al, (ls, le) in cfgs}
    want = EXPECTED_CELLS[args.model]
    if not got.issubset(want):
        die(f"{args.model} cells {sorted(got)} contain doses outside the "
            f"frozen CoT matrix {sorted(want)}; alpha is read from the "
            f"frozen GSM8K record, never re-searched")

    # cot=True is the ONLY planned change from get_answer_bbh_numeric.py.
    templates = select_templates_gsm8k(suite="default", cot=True, wording="plain")
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    raw_mask = np.load(args.mask_path)
    mask_sha = hashlib.sha256(open(args.mask_path, "rb").read()).hexdigest()
    os.makedirs(args.out_dir, exist_ok=True)

    verify_injection_token_equal(vc, templates, samples[0]["question"])

    # Provenance, not a constraint: cells are NOT required to share a GPU
    # (project-wide convention). Because bf16 greedy may vary across
    # hardware, a contrast between cells on different devices is reported as
    # a CROSS-RUN pairing; pairing means alignment by sample_id, never
    # hardware identity.
    device_note = {
        "host": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    for alpha, (ls, le) in cfgs:
        tag = f"mdf_{alpha}".replace("-", "neg")
        out = os.path.join(args.out_dir, tag,
                           f"bbh_{args.task}_cot_{args.size}_{ls}_{le}.json")
        if os.path.exists(out):
            print(f"skip existing {out}")
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)

        diff = raw_mask * alpha
        vc.steering_fire_count(reset=True)

        prompts = [templates["neutral"].format(context=s["question"]) for s in samples]
        gen = []
        for i in tqdm(range(0, len(prompts), args.batch_size),
                      desc=f"bbh-{args.task}-cot a={alpha}"):
            gen.extend(vc.regenerate(
                prompts[i: i + args.batch_size],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                diff_matrices=list(diff),
                batch_size=args.batch_size,
            ))

        fires = vc.steering_fire_count()
        n_layers = len(utils.decoder_layer_range(ls, le))
        expect = 0 if alpha == 0 else n_layers * len(samples)
        if fires != expect:
            die(f"steering_fires {fires} != {expect} "
               f"(L={n_layers}, n={len(samples)}, alpha={alpha}); "
               "the intervention is unverified, so the cell is not usable.")

        rows = [{"sample_id": s["sample_id"], "question": s["question"],
                 "generated": g} for s, g in zip(samples, gen)]
        json.dump({"meta": {"protocol": PROTOCOL, "base_protocol": BASE_PROTOCOL,
                            "task": args.task,
                            "model": args.model, "size": args.size,
                            "alpha": alpha, "layer_start": ls, "layer_end": le,
                            "L": n_layers,
                            "mask_path": args.mask_path, "mask_sha256": mask_sha,
                            "max_new_tokens": args.max_new_tokens,
                            "temperature": args.temperature,
                            "batch_size": args.batch_size,
                            "cot": True, "few_shot": False,
                            "steering_fires": fires,
                            "questions_sha256": meta["questions_sha256"],
                            "revision": meta["revision"],
                            "prompt_template": templates["neutral"],
                            "contains_labels": False,
                            "accuracy_computed": False,
                            "exploratory_followup": True,
                            "provenance": device_note,
                            "n": len(rows)},
                   "data": rows}, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  wrote {out}  steering_fires={fires}")

    print("\nGeneration complete. NO accuracy was computed -- by construction.")
    print("This is a POST-HOC EXPLORATORY follow-up; it does not replace the "
          "frozen No-CoT bbh-p4b-v0 result.")


if __name__ == "__main__":
    main()
