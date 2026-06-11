#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke test for get_answer_cgt.py — does NOT touch the main run.

Loads the model once, then runs a handful of single rounds with timing +
raw model output, so we can tell apart three failure modes:
  1. CPU-offload crawl  -> first generation takes tens of seconds / minutes
  2. parsing failure     -> fast but invalid_rate high / garbage <choice>
  3. all fine            -> ~1-3 s/round, valid <choice>

Run on the server (same env as run_cgt.sh):
  CUDA_VISIBLE_DEVICES=<gpu> python smoke_cgt.py \
      --model_dir meta-llama/Llama-3.1-8B-Instruct \
      --base_dir /data1/paveen/Dopamine/components \
      --use_chat --max_new_tokens 256 --n_rounds 3
"""
import os
import time
import argparse
import numpy as np
import torch

from llms import VicundaModel
import utils
import get_answer_cgt as cgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--hs", default="llama3")
    ap.add_argument("--size", default="8B")
    ap.add_argument("--type", default="non")
    ap.add_argument("--mask_type", default="nmd")
    ap.add_argument("--percentage", type=float, default=0.5)
    ap.add_argument("--config", default="0-11-20", help="single alpha-st-en")
    ap.add_argument("--base_dir", default="/data1/paveen/Dopamine/components")
    ap.add_argument("--use_chat", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--n_rounds", type=int, default=3)
    args = ap.parse_args()

    (alpha, (st, en)), = utils.parse_configs([args.config])
    mask_dir = os.path.join(args.base_dir, "mask", f"{args.hs}_{args.type}_logits")
    mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}.npy"
    mask_path = os.path.join(mask_dir, mask_name)
    print(f"[mask] {mask_path}  exists={os.path.exists(mask_path)}")
    raw_mask = np.load(mask_path)
    diff_mtx = list(raw_mask * alpha)
    print(f"[mask] shape={raw_mask.shape}  alpha={alpha}  len(diff_mtx)={len(diff_mtx)}")

    t0 = time.time()
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    print(f"[load] model ready in {time.time() - t0:.1f}s")
    print(f"[load] #decoder_layers={len(vc._find_decoder_layers())}  "
          f"(must equal len(diff_mtx)={len(diff_mtx)})")
    # report device map to expose CPU offload
    devs = {str(p.device) for p in vc.model.parameters()}
    print(f"[load] param devices = {devs}  <-- if 'cpu' appears, layers are offloaded (SLOW)")

    seed = 0
    box_seq = cgt.make_box_sequence(seed)
    choice_order = cgt.rotate_choice_order(seed)
    system_prompt = cgt.build_system_prompt(choice_order)
    remain = cgt.INIT_MONEY
    history = []

    for r in range(args.n_rounds):
        blue, red = box_seq[r]
        user_prompt = cgt.build_user_prompt(r + 1, remain, blue, red, history)
        prompt = cgt.to_chat(vc, system_prompt, user_prompt, args.use_chat)

        t0 = time.time()
        with torch.no_grad():
            out = vc.regenerate(
                inputs=[prompt], diff_matrices=diff_mtx,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
            )
        dt = time.time() - t0
        raw = out[0] if isinstance(out, list) else out
        slot, valid = cgt.parse_choice(raw)
        ct = choice_order[slot]
        print(f"\n=== round {r+1}  blue={blue} red={red}  took {dt:.1f}s ===")
        print(f"  parsed slot={slot} valid={valid} -> choice_true={ct} "
              f"color={cgt.choice_true_to_color(ct)} bet={cgt.choice_true_to_bet(ct)}")
        print(f"  --- raw output (first 400 chars) ---\n{raw[:400]}")
        # advance a tiny bit so history path is exercised
        history.append({
            "round": r + 1, "choice_color": cgt.choice_true_to_color(ct),
            "choice_percent": cgt.decimal_to_percentage(cgt.choice_true_to_bet(ct)),
            "token_color": "blue" if blue >= red else "red", "payoff": 10,
        })

    print("\n[smoke] done. If each round took ~1-3s and valid=True, the full "
          "run_cgt.sh is fine (it's just 30x64x9 generations = long, not hung).")


if __name__ == "__main__":
    main()
