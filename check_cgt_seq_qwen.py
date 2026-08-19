#!/usr/bin/env python
"""check_cgt_seq_qwen.py — pre-flight for porting CGT-Sequential to a new model.

Verifies the four things that silently differ between Llama3 and Qwen2.5 and
that would otherwise be discovered only AFTER a multi-hour sweep produced
uninterpretable numbers:

  1. TOKENIZER / CHAT TEMPLATE — the prompt renders, and (unlike Llama) does not
     serialize a BOS that would double up. Reports the final token ids.
  2. ANCHOR / INJECTION SITE — prefill-only steering lands on the LAST prompt
     token, so what that token IS decides what is being steered. CGT-seq uses
     `Color: ` at the colour step and (deliberately) NO anchor at the bet step,
     so the bet-step injection sits on the chat template's assistant header.
     This is a Llama-validated choice; on a new tokenizer it must be re-read,
     not assumed.
  3. MASK — the file exists, its layer count matches the configured band, and
     its non-zero rows sit exactly on the decoder layers that will be hooked.
  4. STEERING FIRES — an actual generation with a non-zero alpha reports the
     arithmetically expected number of injection SITES. `steering_fire_count`
     counts (steered layer, sequence, token position) triples that received a
     non-zero add, so with L steered layers, B prompts and tail t the expected
     value is L*B*t. A count of 0 means the hook never fired; a count equal to
     the full 32-layer band means zero rows are being counted.

Read-only: registers hooks and generates a few tokens, writes nothing.

Usage (server, roleplaying env):
  python check_cgt_seq_qwen.py \
      --model_dir Qwen/Qwen2.5-7B-Instruct --size 7B \
      --hs qwen2.5 --layer_start 16 --layer_end 22 --alpha 4
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llms import VicundaModel  # noqa: E402
from utils import decoder_layer_range, mask_slice_for  # noqa: E402
import get_answer_cgt_seq as cs  # noqa: E402
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
    ap.add_argument("--prompt_ver", default="v4")
    ap.add_argument("--anchor", default="default")
    ap.add_argument("--presentation", default="desc")
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
        failures.append("tokenizer has NO chat_template (CGT-seq requires --use_chat)")
        print("!! no chat_template — CGT-seq needs chat (bare gives qdm~0.50)")

    system_prompt = cs.build_seq_system_prompt(args.presentation, args.prompt_ver)
    color_turn = cs.build_color_user_turn(1, cs.INIT_MONEY, 7, 3, False,
                                          prompt_ver=args.prompt_ver)
    # same expression run_episode uses, so the tier shown here is the real one
    tiers = cs.BET_PCTS if args.presentation == "asc" else list(reversed(cs.BET_PCTS))
    bet_turn = cs.build_bet_user_turn("Blue", tiers[0], 1, len(tiers),
                                      next_pct=tiers[1], prompt_ver=args.prompt_ver)

    # anchors exactly as run_episode picks them
    if args.anchor == "answer":
        color_anchor, bet_anchor = "Answer: ", "Answer: "
    elif args.anchor == "none":
        color_anchor, bet_anchor = "", ""
    else:
        color_anchor, bet_anchor = "Color: ", ""

    # ------------------------------------------------------------- 2. anchor site
    hr("2. ANCHOR / INJECTION SITE (prefill-only steers the LAST prompt token)")
    prompts = {}
    for name, turns, anchor in [
        ("colour step", [{"role": "user", "content": color_turn}], color_anchor),
        ("bet step", [{"role": "user", "content": color_turn},
                      {"role": "assistant", "content": "Color: Blue"},
                      {"role": "user", "content": bet_turn}], bet_anchor),
    ]:
        prompt = build_chat_messages2(vc, system_prompt, turns, answer_anchor=anchor)
        prompts[name] = prompt
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        tail = ids[-6:]
        print(f"\n--- {name} (anchor={anchor!r}) ---")
        print(f"  n_tokens        : {len(ids)}")
        print(f"  last 6 ids      : {tail}")
        print(f"  last 6 decoded  : {[tok.decode([i]) for i in tail]}")
        print(f"  INJECTION TOKEN : id={ids[-1]} -> {tok.decode([ids[-1]])!r}")
        # double-BOS: the template may serialize a BOS while the tokenizer adds one
        with_special = tok(prompt, add_special_tokens=True)["input_ids"]
        if (tok.bos_token_id is not None
                and with_special[:2] == [tok.bos_token_id, tok.bos_token_id]):
            failures.append(f"{name}: double BOS when add_special_tokens=True")
            print("  !! DOUBLE BOS with add_special_tokens=True")
        else:
            print("  double-BOS check: ok")

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
        for name in ("colour step", "bet step"):
            vc.steering_fire_count(reset=True)
            out = vc.regenerate([prompts[name]], diff_matrices=diff_mtx,
                                max_new_tokens=8, prefill_only=True,
                                prefill_tail_len=1)
            fires = vc.steering_fire_count(reset=False)
            expect = L * 1 * 1          # L steered layers x 1 prompt x tail 1
            ok = (fires == expect)
            print(f"\n--- {name} ---")
            print(f"  alpha={args.alpha}  expected fires = L*B*t = {L}*1*1 = {expect}")
            print(f"  OBSERVED fires  = {fires}   {'OK' if ok else '<-- MISMATCH'}")
            print(f"  generation      : {out[0][:120]!r}")
            if not ok:
                failures.append(f"{name}: fires {fires} != expected {expect}")
                if fires == 0:
                    print("  hint: hook never fired")
                elif fires == raw_mask.shape[0]:
                    print("  hint: zero rows are being counted (all 32 layers)")

        # alpha=0 must register NO hook at all (house convention: unsteered is a
        # different code path from steered-by-zero)
        vc.steering_fire_count(reset=True)
        vc.generate([prompts["bet step"]], max_new_tokens=4)
        z = vc.steering_fire_count(reset=False)
        print(f"\nalpha=0 path (generate, no diff): fires={z}  "
              f"{'OK' if z == 0 else '<-- should be 0'}")
        if z != 0:
            failures.append(f"alpha=0 path fired {z} times")

    # -------------------------------------------------------------------- verdict
    hr("VERDICT")
    if failures:
        print(f"FAIL ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS — tokenizer, anchor site, mask alignment and steering fires all check out.")
    print("Next: run the validity pilot (alpha 0/+-4 x asc/desc) before the full sweep.")


if __name__ == "__main__":
    main()
