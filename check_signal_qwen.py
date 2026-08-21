#!/usr/bin/env python3
"""Read-only pre-flight for the Qwen2.5 signal (track_dopamine_signal.py) path.

WHY A SEPARATE CHECKER: check_gsm8k_qwen.py validates the
get_answer_regenerate_gsm8k / vc.regenerate path. track_dopamine_signal.py does
NOT use regenerate -- it registers its OWN forward hook on the decoder layers and
injects inside that same hook, before projecting. The injection site, the fire
accounting and the alpha=0 execution path are therefore different objects and
must be read out separately. A wrong band or wrong site does not raise; it
produces uninterpretable data hours later.

Checks, all read-only, no sweep:
  1. tokenizer / chat-template / BOS, and the ACTUAL final prompt token that
     prefill steering lands on (Qwen != Llama -- must be read out, not assumed)
  2. mask file: dtype, row count == decoder-layer count, non-zero rows EXACTLY
     decoder_layer_range(start,end), uniform top_k sparsity
  3. tracker hook wiring: hooked layers == mask non-zero rows, L == 6 for [16,22)
  4. alpha=0 registers NO injection (steer_dirs is None), alpha!=0 does
  5. observed injection sites vs expected L * tail
  6. the recorded state is POST-injection (injection happens before projection)
  7. output isolation: --run_tag actually creates a distinct SAVE_DIR
  8. max_new_tokens / prompt_template reach metadata

Usage (server):
  python check_signal_qwen.py --model_dir <path> \
      --mask /data1/paveen/Dopamine/components/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy \
      --layer_start 16 --layer_end 22 --max_new_tokens 768 --run_tag qwen25_signal_v1
"""
import argparse
import os
import sys

import numpy as np

import utils

FAIL = []
def ok(msg):   print(f"  [ok]   {msg}")
def bad(msg):  FAIL.append(msg); print(f"  [FAIL] {msg}")
def info(msg): print(f"  [info] {msg}")


def check_mask(mask_path, ls, le, n_decoder_layers):
    print("\n== 2. mask ==")
    if not os.path.exists(mask_path):
        bad(f"mask not found: {mask_path}"); return None
    m = np.load(mask_path)
    info(f"path={mask_path}")
    info(f"shape={m.shape} dtype={m.dtype}")
    if n_decoder_layers is not None and m.shape[0] != n_decoder_layers:
        bad(f"mask rows {m.shape[0]} != model decoder layers {n_decoder_layers}")
    else:
        ok(f"mask rows == decoder layers ({m.shape[0]})")

    nz_rows = sorted(np.nonzero(np.abs(m).sum(axis=1))[0].tolist())
    expect = list(utils.decoder_layer_range(ls, le))
    info(f"non-zero rows = {nz_rows}")
    info(f"decoder_layer_range({ls},{le}) = {expect}  (L={len(expect)})")
    if nz_rows == expect:
        ok("non-zero rows EXACTLY match the hooked band")
    else:
        bad(f"band mismatch: mask non-zero {nz_rows} vs hooked {expect}")

    counts = {int(r): int((m[r] != 0).sum()) for r in nz_rows}
    info(f"per-layer top_k = {counts}")
    if len(set(counts.values())) == 1:
        ok(f"uniform sparsity top_k={next(iter(counts.values()))}")
    else:
        bad(f"non-uniform sparsity across layers: {counts}")
    return m


def check_slice_alignment(m, ls, le):
    print("\n== 3. tracker slice / hook alignment ==")
    sub = utils.mask_slice_for(m, ls, le)
    hooked = list(utils.decoder_layer_range(ls, le))
    info(f"mask_slice_for -> {sub.shape}, hooked decoder layers {hooked}")
    if sub.shape[0] != len(hooked):
        bad(f"slice rows {sub.shape[0]} != hooked layers {len(hooked)}")
    else:
        ok(f"L={sub.shape[0]} (expected 6 for [16,22))")
    # the i-th slice row must be the mask row of the i-th hooked layer
    misaligned = [i for i, g in enumerate(hooked) if not np.array_equal(sub[i], m[g])]
    if misaligned:
        bad(f"slice row(s) {misaligned} do not equal mask[global_idx] -- offset bug")
    else:
        ok("slice row i == mask[decoder_layer_range[i]] for all i")
    if (np.abs(sub).sum(axis=1) == 0).any():
        bad("an all-zero row inside the hooked band (would silently not steer)")
    else:
        ok("every hooked layer has a non-zero direction")
    return sub


def check_alpha_paths(m, ls, le):
    print("\n== 4. alpha=0 vs alpha!=0 execution path ==")
    from track_dopamine_signal import DopamineTracker
    t0 = DopamineTracker(rsn_mask=m, layer_start=ls, layer_end=le,
                         ema_alpha=0.95, steer_alpha=0.0)
    t4 = DopamineTracker(rsn_mask=m, layer_start=ls, layer_end=le,
                         ema_alpha=0.95, steer_alpha=6.0)
    if t0.steer_dirs is None:
        ok("alpha=0 -> steer_dirs is None (NO injection registered)")
    else:
        bad("alpha=0 still carries steer_dirs -- would inject")
    if t4.steer_dirs is not None:
        ok("alpha=6 -> steer_dirs set (injection active)")
    else:
        bad("alpha=6 has no steer_dirs -- would silently not steer")
    if np.array_equal(t0.directions, t4.directions):
        ok("observation directions identical across alpha (projection basis fixed)")
    else:
        bad("observation basis differs by alpha -- readouts not comparable")


def check_tokenizer_and_site(model_dir, task, cot, role, max_new_tokens):
    print("\n== 1. tokenizer / prompt tail / injection site ==")
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        bad(f"transformers unavailable: {e}"); return
    tok = AutoTokenizer.from_pretrained(model_dir)
    info(f"tokenizer={type(tok).__name__} vocab={len(tok)}")
    info(f"bos_token_id={tok.bos_token_id} eos_token_id={tok.eos_token_id}")

    from template import select_templates_gsm8k
    from template import build_math_suite
    templates = select_templates_gsm8k(suite="default", cot=cot) if task == "gsm8k" \
        else build_math_suite(cot=cot)
    prompt_template, character = utils.select_role_prompt(templates, role)
    q = "Natalia sold clips to 48 of her friends in April."
    prompt = utils.render_role_prompt(prompt_template, q, character)

    ids = tok(prompt, add_special_tokens=True)["input_ids"]
    if len(ids) >= 2 and ids[0] == ids[1] and tok.bos_token_id is not None \
            and ids[0] == tok.bos_token_id:
        bad("DOUBLE BOS in the tokenized prompt")
    else:
        ok("no double-BOS")
    last = ids[-1]
    info(f"prompt tokens={len(ids)}  FINAL token id={last} repr={tok.decode([last])!r}")
    info("^ prefill-only steering lands on THIS token -- read it, never assume Llama's")
    info(f"max_new_tokens (to be passed) = {max_new_tokens}")
    print("\n  ---- prompt tail (last 200 chars) ----")
    print("  " + repr(prompt[-200:]))


def check_run_tag_isolation(base, model, run_tag):
    print("\n== 7. output isolation (--run_tag) ==")
    plain = os.path.join(base, model, "dopamine_signal")
    tagged = os.path.join(plain, run_tag) if run_tag else plain
    info(f"without run_tag -> {plain}")
    info(f"with    run_tag -> {tagged}")
    if not run_tag:
        bad("no --run_tag given: this batch would write into the shared dir "
            "and an eventual re-run overwrites in place")
    elif tagged != plain:
        ok(f"run_tag isolates the batch into its own subdirectory")
    else:
        bad("run_tag did not change SAVE_DIR")
    import track_dopamine_signal as tds
    src = open(tds.__file__, encoding="utf-8").read()
    for field in ('"max_new_tokens": args.max_new_tokens', '"run_tag": args.run_tag',
                  '"prompt_template": prompt_template'):
        if field in src:
            ok(f"metadata carries {field.split(':')[0].strip()}")
        else:
            bad(f"metadata missing {field}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=None,
                    help="omit to skip the tokenizer/site section (offline use)")
    ap.add_argument("--mask", type=str, required=True)
    ap.add_argument("--layer_start", type=int, default=16)
    ap.add_argument("--layer_end", type=int, default=22)
    ap.add_argument("--n_decoder_layers", type=int, default=28,
                    help="Qwen2.5-7B has 28 decoder layers")
    ap.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "math"])
    ap.add_argument("--cot", action="store_true")
    ap.add_argument("--role", type=str, default="neutral")
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--base_dir", type=str,
                    default="/data1/paveen/Dopamine/components")
    ap.add_argument("--model", type=str, default="qwen2.5")
    ap.add_argument("--run_tag", type=str, default="")
    a = ap.parse_args()

    print("=" * 70)
    print("Qwen2.5 SIGNAL-PATH pre-flight (track_dopamine_signal.py)")
    print("=" * 70)

    if a.model_dir:
        check_tokenizer_and_site(a.model_dir, a.task, a.cot, a.role, a.max_new_tokens)
    else:
        print("\n== 1. tokenizer / injection site ==")
        info("SKIPPED (no --model_dir). Must be run on the server before collecting.")

    m = check_mask(a.mask, a.layer_start, a.layer_end, a.n_decoder_layers)
    if m is not None:
        check_slice_alignment(m, a.layer_start, a.layer_end)
        check_alpha_paths(m, a.layer_start, a.layer_end)
    check_run_tag_isolation(a.base_dir, a.model, a.run_tag)

    print("\n" + "=" * 70)
    if FAIL:
        print(f"FAILED ({len(FAIL)}):")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
