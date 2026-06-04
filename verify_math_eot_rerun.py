#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify whether the stored MATH results were produced by the CURRENT (eot-fixed)
code. Re-generates the FIRST N MATH samples (neutral, alpha=0, No-CoT) through
the EXACT same path get_answer_regenerate_math.py uses (vc.regenerate,
prefill-only, build_math_suite), then byte-compares each `generated` string
against what is stored in the reference JSON.

If the stored file was produced by the eot-fixed code with identical args, the
re-generated text must match exactly (greedy / temperature=0 is deterministic).
A MISMATCH means the stored file was produced by DIFFERENT code (e.g. pre-eot,
no <|eot_id|> terminator) — i.e. it was NOT actually re-run with this code.

Usage (on the server, with GPU + model):
    python verify_math_eot_rerun.py \
        --model_dir meta-llama/Llama-3.1-8B-Instruct \
        --ref /data1/paveen/Dopamine/components/llama3/answer_math/mdf_0/math_8B_11_20.json \
        --test_file benchmark/math_test_sample.json \
        --n 3
    # (or point --ref at the local RoleAnswer copy if running off-server)
"""

import os
import json
import copy
import argparse

import numpy as np
import torch

from llms import VicundaModel
from template import build_math_suite
import utils
# Reuse the production runner so the generation path is byte-for-byte identical.
from get_answer_regenerate_math import run_math


def short(s, n=160):
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"...(+{len(s)-n} chars)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--ref", required=True,
                    help="reference JSON to compare against (the stored math_eot mdf_0 neutral file)")
    ap.add_argument("--test_file", default="benchmark/math_test_sample.json")
    ap.add_argument("--base_dir", default="/data1/paveen/Dopamine/components")
    ap.add_argument("--mask_type", default="nmd")
    ap.add_argument("--percentage", type=float, default=0.5)
    ap.add_argument("--size", default="8B")
    ap.add_argument("--hs", default="llama3")
    ap.add_argument("--type", default="non")
    ap.add_argument("--layer_start", type=int, default=11)
    ap.add_argument("--layer_end", type=int, default=20)
    ap.add_argument("--n", type=int, default=3, help="number of leading samples to re-run")
    ap.add_argument("--max_new_tokens", type=int, default=2048)  # match run_math.sh
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--batch_size", type=int, default=8)
    a = ap.parse_args()

    # run_math() reads a few fields off the global `args` of the imported module;
    # mirror them here so the call behaves exactly like the production script.
    import get_answer_regenerate_math as gm
    gm.args = a

    # ---- terminator sanity: prove the eot fix is live in THIS process ----
    vc = VicundaModel(model_path=a.model_dir)
    vc.model.eval()
    eot_id = vc.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    print(f"[terminators] {vc.terminators}")
    print(f"[terminators] <|eot_id|>={eot_id}  in_terminators={eot_id in vc.terminators}")
    if eot_id not in vc.terminators:
        print("[WARN] <|eot_id|> NOT in terminators — this process is running PRE-EOT code.")

    # ---- load reference + leading samples ----
    ref = json.load(open(a.ref))
    ref_data = ref["data"][:a.n]
    print(f"[ref] {a.ref}  role={ref.get('role')}  using first {len(ref_data)} samples")

    data_path = os.path.join(a.base_dir, a.test_file)
    all_samples = utils.load_json(data_path)[:a.n]

    template = build_math_suite(cot=False)

    # alpha=0 no-op mask (same as production baseline)
    mask_name = f"{a.mask_type}_{a.percentage}_{a.layer_start}_{a.layer_end}_{a.size}.npy"
    mask_path = os.path.join(a.base_dir, "mask", f"{a.hs}_{a.type}_logits", mask_name)
    diff_mtx = np.load(mask_path) * 0  # alpha=0

    samples = copy.deepcopy(all_samples)
    with torch.no_grad():
        results, acc = run_math(vc, samples, diff_mtx, template,
                                alpha=0, st=a.layer_start, en=a.layer_end, role="neutral")

    # ---- compare ----
    print("\n" + "=" * 80)
    print(f"{'idx':>3} {'match':>7} {'new_len':>8} {'ref_len':>8}  first divergence")
    print("-" * 80)
    n_match = 0
    for i, (new, r) in enumerate(zip(results, ref_data)):
        ng, rg = new["generated"], r["generated"]
        same = ng == rg
        n_match += int(same)
        div = ""
        if not same:
            j = 0
            while j < min(len(ng), len(rg)) and ng[j] == rg[j]:
                j += 1
            div = f"@char {j}"
        print(f"{i:>3} {('YES' if same else 'NO'):>7} {len(ng):>8} {len(rg):>8}  {div}")

    print("=" * 80)
    print(f"RESULT: {n_match}/{len(ref_data)} identical")
    if n_match == len(ref_data):
        print(">>> MATCH: stored file IS reproducible with current (eot-fixed) code.")
    else:
        print(">>> MISMATCH: stored file was produced by DIFFERENT code (likely pre-eot).")
        # show the first mismatch in detail
        for i, (new, r) in enumerate(zip(results, ref_data)):
            if new["generated"] != r["generated"]:
                print(f"\n--- sample {i} detail ---")
                print(f"[new tail] ...{short(new['generated'][-200:])}")
                print(f"[ref tail] ...{short(r['generated'][-200:])}")
                break


if __name__ == "__main__":
    main()
