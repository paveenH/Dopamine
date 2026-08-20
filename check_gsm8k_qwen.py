#!/usr/bin/env python
"""check_gsm8k_qwen.py — read-only technical pre-flight for the Qwen2.5 GSM8K port.

This checks the WIRING ONLY: model, tokenizer, mask, layer band, benchmark file
and output path. It does NOT look at accuracy, commit rate, or any behavioural
quantity, and it has no pass/fail gate on results -- alpha=0 is run afterwards
and its output is read by a human.

Five things it reads out, each of which differs between Llama3 and Qwen2.5 and
each of which is silent when wrong:

  1. TOKENIZER   -- which tokenizer class, vocab size, bos/eos, and the
                    terminator list VicundaModel actually built. Qwen ends its
                    turn with <|im_end|> and ships a LIST eos in
                    generation_config; llms._build_terminators() already unions
                    both, so this is a verification, not a fix.
  2. PROMPT      -- the real rendered prompt for one benchmark question, plus
                    its last 6 token ids/pieces. GSM8K templates end in
                    "Answer: ", and prefill-only steering lands on that final
                    token, so which token that is must be recorded, not assumed.
  3. MASK        -- shape, decoder-layer count, and that the non-zero rows are
                    EXACTLY decoder_layer_range(start, end). This is the
                    2026-05-30 offset bug's check.
  4. FIRES       -- observed steering sites from VicundaModel.steering_fire_count()
                    against L * B * tail_len, and that alpha=0 fires 0.
                    NOTE: unlike CGT-seq, get_answer_regenerate_gsm8k.py builds
                    diff = mask * alpha unconditionally, so alpha=0 passes a REAL
                    all-zero matrix and hooks DO register; the count is 0 only
                    because _layer_is_steered is False on an all-zero row.
  5. PATHS       -- benchmark file and the output dir the sweep will write to.

Usage:
  python check_gsm8k_qwen.py                       # full check (loads the model)
  python check_gsm8k_qwen.py --no_model            # paths/mask/benchmark only
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils  # noqa: E402
from template import select_templates_gsm8k  # noqa: E402
from utils import decoder_layer_range  # noqa: E402


def hr(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="/data1/paveen/Dopamine/components")
    ap.add_argument("--model", default="qwen2.5")
    ap.add_argument("--model_dir", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--hs", default="qwen2.5")
    ap.add_argument("--type", default="non")
    ap.add_argument("--size", default="7B")
    ap.add_argument("--mask_type", default="nmd")
    ap.add_argument("--percentage", type=float, default=0.5)
    ap.add_argument("--layer_start", type=int, default=16)
    ap.add_argument("--layer_end", type=int, default=22)
    ap.add_argument("--test_file", default="benchmark/gsm8k_test_sample.json")
    ap.add_argument("--ans_file", default="answer_mdf_gsm8k")
    ap.add_argument("--ans_file_cot", default="answer_mdf_gsm8k_cot")
    ap.add_argument("--fmt_wording", default="plain")
    ap.add_argument("--alpha", type=float, default=4.0,
                    help="non-zero alpha used for the fires read-out")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--no_model", action="store_true",
                    help="skip everything needing the weights (checks 1,2,4)")
    a = ap.parse_args()

    # ---------------- 5. PATHS + benchmark ----------------
    hr("5. PATHS / BENCHMARK")
    data_path = os.path.join(a.base_dir, a.test_file)
    mask_dir = os.path.join(a.base_dir, "mask", f"{a.hs}_{a.type}_logits")
    mask_name = f"{a.mask_type}_{a.percentage}_{a.layer_start}_{a.layer_end}_{a.size}.npy"
    mask_path = os.path.join(mask_dir, mask_name)
    out_nocot = os.path.join(a.base_dir, a.model, a.ans_file)
    out_cot = os.path.join(a.base_dir, a.model, a.ans_file_cot)

    print(f"benchmark      : {data_path}")
    print(f"mask           : {mask_path}")
    print(f"out (No-CoT)   : {out_nocot}/mdf_<alpha>/")
    print(f"out (CoT)      : {out_cot}/mdf_<alpha>/")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")

    if not os.path.exists(data_path):
        print(f"[FAIL] benchmark file missing: {data_path}")
        return 1
    samples = utils.load_json(data_path)
    print(f"[ok] benchmark loaded: {len(samples)} samples")
    print(f"     first question   : {samples[0]['question'][:80]}...")
    print(f"     first gold answer: {samples[0]['answer'][-40:]!r}")

    # Warn (do not fail) if an output dir already holds results -- overwriting is
    # the caller's decision, but it must not be silent.
    for label, d in (("No-CoT", out_nocot), ("CoT", out_cot)):
        if os.path.isdir(d):
            existing = sorted(x for x in os.listdir(d) if x.startswith("mdf_"))
            if existing:
                print(f"[warn] {label} output dir ALREADY has cells: {existing}")

    # ---------------- 3. MASK ----------------
    hr("3. MASK / LAYER ALIGNMENT")
    if not os.path.exists(mask_path):
        print(f"[FAIL] mask missing: {mask_path}")
        print("       (get_answer_regenerate_gsm8k.py loads the mask even at")
        print("        alpha=0, so this blocks the baseline cell too.)")
        return 1
    mask = np.load(mask_path)
    print(f"[ok] mask shape {mask.shape}  dtype={mask.dtype}  finite={np.isfinite(mask).all()}")

    nz_rows = sorted(int(i) for i in np.nonzero(np.any(mask != 0, axis=1))[0])
    want = list(decoder_layer_range(a.layer_start, a.layer_end))
    print(f"     non-zero rows      : {nz_rows}")
    print(f"     decoder_layer_range: {want}   (L = {len(want)})")
    if nz_rows != want:
        print("[FAIL] mask band != hooked band -- this is the silent offset bug.")
        return 1
    print("[ok] mask band == hooked band")

    counts = {int(np.count_nonzero(mask[r])) for r in nz_rows}
    top_k = int(mask.shape[1] * a.percentage / 100)
    print(f"     nonzeros/steered row: {counts}   (expected top_k={top_k})")

    if a.no_model:
        print("\n[--no_model] skipping tokenizer / prompt / fires checks.")
        return 0

    # ---------------- 1. TOKENIZER ----------------
    hr("1. TOKENIZER / TERMINATORS")
    from llms import VicundaModel  # noqa: E402  (heavy import, only when needed)
    vc = VicundaModel(model_path=a.model_dir)
    vc.model.eval()
    tk = vc.tokenizer
    print(f"[ok] tokenizer      : {type(tk).__name__}")
    print(f"     vocab_size     : {len(tk)}")
    print(f"     bos_token_id   : {tk.bos_token_id}")
    print(f"     eos_token_id   : {tk.eos_token_id}  ({tk.eos_token!r})")
    print(f"     padding_side   : {tk.padding_side}")
    gen_cfg = getattr(vc.model, "generation_config", None)
    print(f"     gen_cfg.eos    : {getattr(gen_cfg, 'eos_token_id', None)}")
    print(f"     terminators    : {vc.terminators}")
    print(f"       decoded      : {[tk.decode([t]) for t in vc.terminators]}")
    im_end = tk.convert_tokens_to_ids("<|im_end|>")
    print(f"     <|im_end|> id  : {im_end}  in terminators: {im_end in vc.terminators}")

    n_layers = len(vc._find_decoder_layers())
    print(f"     decoder layers : {n_layers}   (mask rows {mask.shape[0]})")
    if n_layers != mask.shape[0]:
        print("[FAIL] decoder-layer count != mask rows")
        return 1

    # ---------------- 2. PROMPT / INJECTION SITE ----------------
    hr("2. PROMPT / INJECTION TOKEN")
    for cot in (False, True):
        tmpl = select_templates_gsm8k(suite="default", cot=cot, wording=a.fmt_wording)
        prompt = tmpl["neutral"].format(context=samples[0]["question"])
        ids = tk(prompt, return_tensors=None)["input_ids"]
        tail = ids[-6:]
        tag = "CoT   " if cot else "No-CoT"
        print(f"\n--- {tag} (wording={a.fmt_wording}) ---")
        print(repr(prompt))
        print(f"  n_tokens   : {len(ids)}")
        print(f"  last 6 ids : {tail}")
        print(f"  last 6 str : {[tk.decode([t]) for t in tail]}")
        print(f"  INJECTION SITE (last token) = id {ids[-1]} -> {tk.decode([ids[-1]])!r}")
        has_cot_line = "step by step" in prompt
        print(f"  'Let's think step by step' present: {has_cot_line}  (expected {cot})")
        print(f"  '####' directive count: {prompt.count('####')}  (expected 1)")

    # ---------------- 4. FIRES ----------------
    hr("4. STEERING FIRES")
    tmpl = select_templates_gsm8k(suite="default", cot=False, wording=a.fmt_wording)
    probe = [tmpl["neutral"].format(context=s["question"]) for s in samples[: a.batch_size]]
    L, B, tail_len = len(want), len(probe), 1

    for alpha in (0.0, a.alpha):
        diff = list(mask * alpha)
        vc.steering_fire_count(reset=True)
        out = vc.regenerate(
            probe, max_new_tokens=24, temperature=0.0,
            diff_matrices=diff, batch_size=a.batch_size,
        )
        fires = vc.steering_fire_count(reset=True)
        expect = 0 if alpha == 0 else L * B * tail_len
        ok = "ok" if fires == expect else "FAIL"
        print(f"[{ok}] alpha={alpha:+.1f}  fires={fires}  expected={expect}"
              f"   (L={L} x B={B} x tail={tail_len})")
        print(f"      sample gen: {out[0][:100]!r}")
        if fires != expect:
            return 1

    print("\n[ok] alpha=0 fires 0 via a REAL all-zero diff (hooks register; the")
    print("     zero add executes; _layer_is_steered is False on a zero row).")

    hr("TECHNICAL CHECK COMPLETE")
    print("Wiring verified. No behavioural judgement was made and none is implied.")
    print("Next: run --baseline (alpha=0), then READ the raw output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
