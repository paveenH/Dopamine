#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinQA generation, protocol `finqa-v0`. Task-specific steering exploration --
NOT a GSM8K fixed-workpoint transfer test (alpha is swept fresh here) and NOT
a blind validation (FinQA test gold is public). Generation and scoring are
still kept as separate scripts, matching the repo's usual discipline: this
script never computes accuracy and never imports finqa_scoring's gold-facing
helpers beyond the label-free diagnostic bundle.

WHAT IS HELD FIXED, matching the frozen GSM8K generation path:
    steering  vc.regenerate(..., diff_matrices=..., prefill_only=True,
              prefill_tail_len=1) -- prefill-only, tail=1
    sampling  temperature=0.0 (greedy), bare-string (no chat template)
    role      neutral only (no persona)
alpha=0 passes a REAL all-zero matrix (`raw_mask * alpha`, unconditional), so
hooks register and the zero add executes; `steering_fires` reads 0 only
because a zero row is not counted as steered (same convention as every other
regenerate-based launcher in this repo -- do not special-case it away).

INPUT: the sample's `question`, `pre_text`, `post_text`, `table_linear` only.
`gold_raw` / `gold_value` / `id` / `program_re` are present in the loader's
JSON but this script never reads them for anything other than pass-through
`sample_id` bookkeeping (needed to join with gold at scoring time).

@author: paveenhuang
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llms import VicundaModel  # noqa: E402
import utils  # noqa: E402
from prompt_finqa import build_finqa_prompt  # noqa: E402
from finqa_scoring import diagnostics_for_text  # noqa: E402

PROTOCOL = "finqa-v0"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["llama3", "qwen2.5"])
    p.add_argument("--size", required=True)
    p.add_argument("--model_dir", required=True,
                    help="HF model id, e.g. meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--questions", required=True,
                    help="finqa_formal.json from data_finqa.py")
    p.add_argument("--mask_path", required=True)
    p.add_argument("--configs", required=True, nargs="+",
                    help="e.g. 0-11-20 neg6-11-20 4-11-20")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--indices", type=int, nargs="*", default=None,
                    help="restrict to these sample_ids (preflight use)")
    return p.parse_args()


def load_samples(path, indices=None):
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d["meta"], d["data"]
    if indices is not None:
        idx_set = set(indices)
        data = [s for s in data if s["sample_id"] in idx_set]
        if len(data) != len(idx_set):
            sys.exit(f"[FATAL] requested {len(idx_set)} indices, found {len(data)} "
                      f"in {path}")
    return meta, data


def main():
    args = parse_args()
    meta, samples = load_samples(args.questions, args.indices)
    q_sha = hashlib.sha256(
        json.dumps([s["question"] for s in samples], ensure_ascii=False).encode()
    ).hexdigest()[:16]

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    raw_mask = np.load(args.mask_path)
    os.makedirs(args.out_dir, exist_ok=True)

    prompts = [build_finqa_prompt(s) for s in samples]

    for alpha, (ls, le) in utils.parse_configs(args.configs):
        tag = f"mdf_{alpha}".replace("-", "neg")
        out_path = os.path.join(args.out_dir, tag, f"finqa_{args.size}_{ls}_{le}.json")
        if os.path.exists(out_path):
            print(f"skip existing {out_path}")
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        diff = raw_mask * alpha
        vc.steering_fire_count(reset=True)

        rows = []
        for i in tqdm(range(0, len(prompts), args.batch_size),
                      desc=f"finqa a={alpha}"):
            batch_prompts = prompts[i: i + args.batch_size]
            batch_samples = samples[i: i + args.batch_size]
            outs = vc.regenerate(
                batch_prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                diff_matrices=list(diff),
                prefill_only=True,
                prefill_tail_len=1,
                batch_size=args.batch_size,
                return_metadata=True,
            )
            for s, o in zip(batch_samples, outs):
                diag = diagnostics_for_text(
                    o["text"], args.max_new_tokens, o["generated_token_count"])
                rows.append({
                    "sample_id": s["sample_id"],
                    "id": s["id"],
                    "question": s["question"],
                    "generated": o["text"],
                    "generated_token_count": o["generated_token_count"],
                    "stop_reason": o["stop_reason"],
                    **diag,
                })

        fires = vc.steering_fire_count()
        n_layers = len(utils.decoder_layer_range(ls, le))
        expect = 0 if alpha == 0 else n_layers * len(samples)
        if fires != expect:
            sys.exit(f"[FATAL] steering_fires {fires} != {expect} "
                      f"(L={n_layers}, n={len(samples)}, alpha={alpha}); "
                      "the intervention is unverified.")

        json.dump({
            "meta": {
                "protocol": PROTOCOL, "model": args.model, "size": args.size,
                "alpha": alpha, "layer_start": ls, "layer_end": le, "L": n_layers,
                "n_samples": len(samples), "questions_sha256_16": q_sha,
                "max_new_tokens": args.max_new_tokens, "temperature": 0.0,
                "batch_size": args.batch_size, "role": "neutral", "cot": True,
                "steering_fires": fires, "accuracy_computed": False,
                "prompt_template_module": "prompt_finqa.FINQA_TEMPLATE",
                "restricted_indices": args.indices,
            },
            "data": rows,
        }, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"wrote {out_path}  (fires={fires}, n={len(samples)})")


if __name__ == "__main__":
    main()
