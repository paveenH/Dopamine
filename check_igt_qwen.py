#!/usr/bin/env python
"""check_igt_qwen.py — pre-flight for porting IGT to a new model (Qwen2.5-7B).

Same role as check_cgt_seq_qwen.py, adapted to IGT's own prompt surface. It
verifies the things that silently differ between Llama3 and Qwen2.5 and would
otherwise surface only after a multi-hour sweep produced uninterpretable data:

  1. TOKENIZER / CHAT TEMPLATE — renders, and (unlike Llama) whether a BOS is
     serialized that would double up.
  2. INJECTION SITE — IGT's default anchor is EMPTY (`anchor="default"` ->
     choice_anchor = ""), deliberately: an anchor here suppresses the reasoning
     span that both the learning readout and prefill steering need (the
     CGT-simple4 failure). So injection lands on whatever the chat template's
     assistant header ends with, which is MODEL-SPECIFIC:
        Llama-3.1 -> id 271 '\\n\\n'      Qwen2.5 -> id 198 '\\n'
     Read it out; never assume Llama's.
  3. MASK — exists, row count matches the model's decoder layers, and non-zero
     rows sit exactly on the band that will be hooked. Qwen uses layers 16-21
     (written 16-22, exclusive end) -> L=6, NOT Llama's 11-20 -> L=9.
  4. STEERING FIRES — a real generation reports L*B*t injection SITES.

  5. ALPHA=0 PATH — IGT builds `diff_mtx = list(raw_mask * alpha)`
     UNCONDITIONALLY and `run_episode`'s single `gen` closure always calls
     `vc.regenerate(diff_matrices=diff_mtx)`. So at alpha=0 it passes a REAL
     all-zero matrix, not None (regenerate rejects None, llms.py:880). Hooks DO
     register and the zero add DOES execute; fires read 0 only because
     `_layer_is_steered` is False on an all-zero row. Same convention as
     CGT-seq, NOT the pv6/PV10 "no hook at all" case. Checking `vc.generate`
     would verify a path the experiment never takes.

Read-only: registers hooks and generates a few tokens, writes nothing.

Usage (server, roleplaying env):
  python check_igt_qwen.py --model_dir Qwen/Qwen2.5-7B-Instruct --size 7B \
      --hs qwen2.5 --layer_start 16 --layer_end 22 --alpha 4
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llms import VicundaModel  # noqa: E402
from utils import decoder_layer_range, mask_slice_for  # noqa: E402
import get_answer_igt as ig  # noqa: E402
from get_answer_cgt import build_chat_messages2  # noqa: E402


def hr(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--size", default="7B")
    ap.add_argument("--hs", default="qwen2.5", help="mask dir prefix")
    ap.add_argument("--type", default="non")
    ap.add_argument("--mask_type", default="nmd")
    ap.add_argument("--percentage", type=float, default=0.5)
    ap.add_argument("--layer_start", type=int, default=16)
    ap.add_argument("--layer_end", type=int, default=22)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--base_dir", default="/data1/paveen/Dopamine/components")
    ap.add_argument("--prompt_ver", default="v6b")
    ap.add_argument("--anchor", default="default")
    args = ap.parse_args()

    failures = []

    # ---------------------------------------------------------------- 1. tokenizer
    hr("1. TOKENIZER / CHAT TEMPLATE")
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    tok = vc.tokenizer
    print(f"tokenizer      : {type(tok).__name__}")
    print(f"vocab size     : {len(tok)}")
    print(f"bos / eos      : {tok.bos_token!r} ({tok.bos_token_id}) / "
          f"{tok.eos_token!r} ({tok.eos_token_id})")
    if tok.chat_template is None:
        failures.append("tokenizer has NO chat_template (IGT is run with --use_chat)")
        print("!! no chat_template")

    system_prompt = ig.build_igt_system_prompt(args.prompt_ver)
    # round 1 (no feedback yet) and a mid-game round WITH feedback: the feedback
    # block is what grows the prompt, so both are worth reading out.
    t1 = ig.build_igt_user_turn(1, ig.INIT_MONEY, "", prompt_ver=args.prompt_ver)
    t50 = ig.build_igt_user_turn(50, 1850, "You opened Chest 2 and won 100 points, "
                                 "but lost 1250 points.", prompt_ver=args.prompt_ver)

    # anchor exactly as run_episode picks it
    choice_anchor = "Chest: " if args.anchor == "chest" else ""

    # ------------------------------------------------------------- 2. anchor site
    hr("2. INJECTION SITE (prefill-only steers the LAST prompt token)")
    print(f"anchor mode = {args.anchor!r} -> choice_anchor = {choice_anchor!r}")
    if args.anchor == "default":
        print("NOTE: IGT deliberately uses NO anchor; injection sits on the chat")
        print("      template's assistant header. This is by design (an anchor")
        print("      kills the reasoning span), so read the token, don't 'fix' it.")
    prompts = {}
    for name, turns in [
        ("round 1", [{"role": "user", "content": t1}]),
        ("round 50 (with feedback + history)",
         [{"role": "user", "content": t1},
          {"role": "assistant", "content": "I will start by exploring. Chest: 1"},
          {"role": "user", "content": t50}]),
    ]:
        prompt = build_chat_messages2(vc, system_prompt, turns,
                                      answer_anchor=choice_anchor)
        prompts[name] = prompt
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        tail = ids[-6:]
        print(f"\n--- {name} ---")
        print(f"  n_tokens        : {len(ids)}")
        print(f"  last 6 ids      : {tail}")
        print(f"  last 6 decoded  : {[tok.decode([i]) for i in tail]}")
        print(f"  INJECTION TOKEN : id={ids[-1]} -> {tok.decode([ids[-1]])!r}")
        with_special = tok(prompt, add_special_tokens=True)["input_ids"]
        if (tok.bos_token_id is not None
                and with_special[:2] == [tok.bos_token_id, tok.bos_token_id]):
            failures.append(f"{name}: double BOS when add_special_tokens=True")
            print("  !! DOUBLE BOS with add_special_tokens=True")
        else:
            print("  double-BOS check: ok")

    # the injection token must be STABLE across rounds, or different rounds are
    # steering different things within one episode
    ids1 = tok(prompts["round 1"], add_special_tokens=False)["input_ids"][-1]
    ids50 = tok(prompts["round 50 (with feedback + history)"],
                add_special_tokens=False)["input_ids"][-1]
    if ids1 != ids50:
        failures.append(f"injection token differs by round: {ids1} vs {ids50}")
        print(f"\n!! injection token CHANGES between rounds ({ids1} vs {ids50})")
    else:
        print(f"\ninjection token stable across rounds: id={ids1}")

    # ---------------------------------------------------------------------- 3. mask
    hr("3. MASK")
    mask_dir = os.path.join(args.base_dir, "mask", f"{args.hs}_{args.type}_logits")
    mask_name = (f"{args.mask_type}_{args.percentage}_{args.layer_start}_"
                 f"{args.layer_end}_{args.size}.npy")
    mask_path = os.path.join(mask_dir, mask_name)
    print(f"path : {mask_path}")
    if not os.path.exists(mask_path):
        failures.append(f"mask not found: {mask_path}")
        print("!! NOT FOUND — cannot check layers or fires")
        raw_mask = None
    else:
        raw_mask = np.load(mask_path)
        n_layers = len(vc._find_decoder_layers())
        nz = [i for i, row in enumerate(raw_mask) if np.count_nonzero(row)]
        band = list(decoder_layer_range(args.layer_start, args.layer_end))
        print(f"shape            : {raw_mask.shape}")
        print(f"model layers     : {n_layers}")
        print(f"non-zero rows    : {nz}")
        print(f"hooked band      : {band}   (decoder_layer_range, half-open)")
        print(f"expected L       : {len(band)}")
        if raw_mask.shape[0] != n_layers:
            failures.append(f"mask rows {raw_mask.shape[0]} != decoder layers {n_layers}")
            print("!! mask row count does not match the model's decoder layers")
        if sorted(nz) != sorted(band):
            failures.append(f"mask non-zero rows {nz} != hooked band {band}")
            print("!! non-zero rows do NOT coincide with the band that gets hooked")
        else:
            print("layer alignment  : ok")
        sub = mask_slice_for(raw_mask, args.layer_start, args.layer_end)
        print(f"mask_slice_for   : {np.asarray(sub).shape}")

    # ------------------------------------------------------------- 4. steering fires
    hr("4. STEERING FIRES (observed injection sites)")
    if raw_mask is None:
        print("skipped (no mask)")
    else:
        L = len([i for i, row in enumerate(raw_mask) if np.count_nonzero(row)])
        diff_mtx = list(raw_mask * args.alpha)
        for name in prompts:
            vc.steering_fire_count(reset=True)
            out = vc.regenerate([prompts[name]], diff_matrices=diff_mtx,
                                max_new_tokens=16, prefill_only=True,
                                prefill_tail_len=1)
            fires = vc.steering_fire_count(reset=False)
            expect = L * 1 * 1
            ok = (fires == expect)
            print(f"\n--- {name} ---")
            print(f"  alpha={args.alpha}  expected fires = L*B*t = {L}*1*1 = {expect}")
            print(f"  OBSERVED fires  = {fires}   {'OK' if ok else '<-- MISMATCH'}")
            print(f"  generation      : {out[0][:160]!r}")
            if not ok:
                failures.append(f"{name}: fires {fires} != expected {expect}")
                if fires == 0:
                    print("  hint: hook never fired")
                elif fires == raw_mask.shape[0]:
                    print("  hint: zero rows are being counted (all layers)")

        # ------------------------------------------------------- 5. alpha=0 path
        hr("5. ALPHA=0 PATH (regenerate + all-zero diff — the REAL driver path)")
        zero_mtx = list(raw_mask * 0.0)
        vc.steering_fire_count(reset=True)
        out0 = vc.regenerate([prompts["round 1"]], diff_matrices=zero_mtx,
                             max_new_tokens=16, prefill_only=True,
                             prefill_tail_len=1)
        z = vc.steering_fire_count(reset=False)
        print(f"  fires at alpha=0 : {z}   {'OK (expected 0)' if z == 0 else '<-- expected 0'}")
        print(f"  generation       : {out0[0][:160]!r}")
        if z != 0:
            failures.append(f"alpha=0 fires {z} != 0")
        # the parser must accept what alpha=0 actually produces
        import random as _r
        choice, valid = ig.parse_choice(out0[0], _r.Random(0))
        print(f"  parse_choice     : choice={choice} valid={valid}"
              f"   {'OK' if valid else '<-- alpha=0 generation does not parse'}")
        if not valid:
            failures.append("alpha=0 generation failed to parse "
                            "(baseline would be pure fallback noise)")

    # -------------------------------------------------------------------- verdict
    hr("VERDICT")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASSED — interface is sound. Next: alpha=0 baseline "
          "(bash run_igt_qwen25.sh --baseline).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
