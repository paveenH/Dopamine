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
  5. REAL-HOOK toy forward on CPU: alpha=0 is bit-identical, alpha!=0 hits
     only the last prefill token, decode is never injected, exactly L layers
  6. the recorded state is POST-injection, verified numerically against the
     co-design identity delta = alpha * mean_l ||mask_l||^2
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


def check_toy_forward(m, ls, le):
    """Drive the REAL DopamineTracker hooks with a real nn.Module stack on CPU.

    WHY NOT A FAKE: a fake that models INTENT cannot test ARITHMETIC -- it will
    happily agree with a wrong implementation. The repo has already been bitten
    by exactly this (the site-counter bug passed a green FakeVC suite). So this
    registers the actual hooks on real torch modules and reads the numbers back.

    Verifies, by measurement rather than by reading source:
      - alpha=0 leaves the hidden state bit-identical (pure observer)
      - alpha!=0 modifies ONLY the prefill call, and only its LAST token
      - decode steps are never injected
      - exactly L layers are touched (L=6 for [16,22)), none outside the band
      - the recorded projection is POST-injection, and its increment equals
        alpha * ||mask_l||^2 (the co-design identity)
    """
    print("\n== 5/6. real-hook toy forward (CPU) ==")
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        bad(f"torch unavailable, cannot run the toy forward: {e}")
        return
    from track_dopamine_signal import DopamineTracker

    H = m.shape[1]
    n_layers = m.shape[0]
    hooked = list(utils.decoder_layer_range(ls, le))

    class Layer(nn.Module):
        """Returns a tuple like a real decoder layer, and records what it emitted
        AFTER the forward hook has had its chance to mutate the tensor."""
        def __init__(self, idx):
            super().__init__()
            self.idx = idx
        def forward(self, x):
            return (x,)

    def build(seed):
        torch.manual_seed(seed)
        return [Layer(i) for i in range(n_layers)]

    def run(alpha, seq_len, seed=0):
        """One forward per layer, in order, threading the tensor through --
        mirroring how a real stack propagates an in-place edit downstream."""
        layers = build(seed)
        t = DopamineTracker(rsn_mask=m, layer_start=ls, layer_end=le,
                            ema_alpha=0.95, steer_alpha=alpha)
        t.attach(layers)
        torch.manual_seed(seed)
        x = torch.randn(1, seq_len, H)
        before = x.clone()
        try:
            for lyr in layers:
                x = lyr(x)[0]
        finally:
            t.detach()
        return t, before, x

    # --- alpha = 0 : pure observer, tensor must be untouched ---
    t0, before0, after0 = run(0.0, seq_len=7)
    if torch.equal(before0, after0):
        ok("alpha=0 leaves the hidden state BIT-IDENTICAL (pure observer)")
    else:
        bad("alpha=0 modified the hidden state -- not a pure observer")
    if t0._prefill_proj is None:
        bad("alpha=0 recorded no prefill projection (hook never completed)")
    else:
        ok(f"alpha=0 recorded a prefill projection ({t0._prefill_proj:.4f})")

    # --- alpha != 0 : prefill only, last token only, exactly L layers ---
    ALPHA = 6.0
    t6, before6, after6 = run(ALPHA, seq_len=7)
    delta = (after6 - before6)[0]              # (seq_len, H)
    touched_tokens = sorted(torch.nonzero(delta.abs().sum(dim=1)).flatten().tolist())
    if touched_tokens == [delta.shape[0] - 1]:
        ok(f"alpha={ALPHA:g} modified ONLY the last prompt token (idx {touched_tokens[0]})")
    else:
        bad(f"alpha={ALPHA:g} touched token idx {touched_tokens}, expected only the last")

    # the accumulated edit on the last token must equal alpha * sum(mask rows)
    expect_vec = torch.as_tensor(
        ALPHA * utils.mask_slice_for(m, ls, le).sum(axis=0), dtype=delta.dtype)
    got_vec = delta[-1]
    if torch.allclose(got_vec, expect_vec, atol=1e-3, rtol=1e-3):
        ok(f"accumulated edit == alpha * sum(mask[{hooked[0]}..{hooked[-1]}]) "
           f"-> exactly {len(hooked)} layers injected")
    else:
        bad(f"accumulated edit != alpha*sum(mask rows); "
            f"max|diff|={float((got_vec-expect_vec).abs().max()):.4g} "
            f"(wrong layer count or wrong rows)")

    # no dimension outside the mask support may move
    support = np.nonzero(np.abs(utils.mask_slice_for(m, ls, le)).sum(axis=0))[0]
    off = np.setdiff1d(torch.nonzero(got_vec.abs()).flatten().numpy(), support)
    if off.size == 0:
        ok("no hidden dimension outside the mask support was modified")
    else:
        bad(f"{off.size} dimension(s) outside the mask support were modified")

    # --- decode step (seq_len == 1) must never be injected ---
    td, befored, afterd = run(ALPHA, seq_len=1)
    if torch.equal(befored, afterd):
        ok(f"alpha={ALPHA:g} does NOT inject on a decode step (seq_len=1)")
    else:
        bad(f"alpha={ALPHA:g} injected during decode -- steering is not prefill-only")

    # --- the recorded projection is POST-injection: co-design identity ---
    # NOTE ON THE TOY'S GEOMETRY: the identity delta == alpha*mean_l||mask_l||^2
    # holds only if each layer OBSERVES its own injection alone. Threading one
    # tensor through identity layers makes layer l also see layers <l's edits
    # (a real stack's non-linearities do not accumulate this way), which inflates
    # the delta by the cross-layer mask overlap. So this sub-check feeds every
    # layer an INDEPENDENT clean input -- isolating exactly the quantity the
    # identity is about. The propagation behaviour is already covered above by
    # the accumulated-edit check.
    def run_isolated(alpha, seed=0):
        layers = build(seed)
        t = DopamineTracker(rsn_mask=m, layer_start=ls, layer_end=le,
                            ema_alpha=0.95, steer_alpha=alpha)
        t.attach(layers)
        torch.manual_seed(seed)
        base = torch.randn(1, 7, H)
        try:
            for lyr in layers:
                lyr(base.clone())          # each layer sees the SAME clean input
        finally:
            t.detach()
        return t

    sub = utils.mask_slice_for(m, ls, le).astype(np.float32)
    norm_sq = float((sub * sub).sum(axis=1).mean())   # mean_l ||mask_l||^2
    ti0 = run_isolated(0.0)
    ti6 = run_isolated(ALPHA)
    got = ti6._prefill_proj - ti0._prefill_proj
    want = ALPHA * norm_sq
    if abs(got - want) <= max(1e-3, 2e-3 * abs(want)):
        ok(f"recorded prefill projection is POST-injection: "
           f"delta={got:.4f} == alpha*mean||mask_l||^2={want:.4f}")
    else:
        bad(f"projection delta {got:.4f} != alpha*mean||mask_l||^2 {want:.4f} "
            f"-- recorded state is NOT post-injection (or the basis differs)")


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
                    help="REQUIRED unless --skip_tokenizer is passed explicitly.")
    ap.add_argument("--skip_tokenizer", action="store_true",
                    help="Explicitly skip the tokenizer/injection-site section. "
                         "Offline/local use ONLY -- the server pre-flight MUST "
                         "read out the real final prompt token.")
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
    elif a.skip_tokenizer:
        print("\n== 1. tokenizer / injection site ==")
        info("SKIPPED via explicit --skip_tokenizer (local use).")
        info("The server pre-flight MUST run this section before collecting.")
        print("\n  *** PARTIAL CHECK -- not sufficient to authorize collection ***")
    else:
        print("\n== 1. tokenizer / injection site ==")
        bad("no --model_dir given. The injection token is model-specific and "
            "must be READ OUT, never assumed. Pass --model_dir, or "
            "--skip_tokenizer to acknowledge a partial local check.")

    m = check_mask(a.mask, a.layer_start, a.layer_end, a.n_decoder_layers)
    if m is not None:
        check_slice_alignment(m, a.layer_start, a.layer_end)
        check_alpha_paths(m, a.layer_start, a.layer_end)
        check_toy_forward(m, a.layer_start, a.layer_end)
    check_run_tag_isolation(a.base_dir, a.model, a.run_tag)

    print("\n" + "=" * 70)
    if FAIL:
        print(f"FAILED ({len(FAIL)}):")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    if a.skip_tokenizer and not a.model_dir:
        print("PARTIAL CHECKS PASSED (tokenizer section skipped)")
        print("NOT sufficient to start collection -- re-run with --model_dir.")
    else:
        print("ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
