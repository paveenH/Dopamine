#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProofWriter OWA generation. LABEL-FREE. Protocol `proofwriter-owa-v0`.

This is ProofWriter's OWN task-specific workpoint exploration -- it does NOT
read or reuse the frozen GSM8K workpoint (llama -6 / qwen +8) from the
P3/P4/P4b/P4c fixed-workpoint transfer line. Its own dose set is:
    llama3   layers [11,20)  alpha in {-6,-4,0,+4}
    qwen2.5  layers [16,22)  alpha in {-6,0,+6,+8}
enforced via EXPECTED_CELLS below, exactly as get_answer_bbh_numeric_cot.py
enforces its own frozen matrix.

WHY A STANDALONE SCRIPT rather than reusing get_answer_regenerate_gsm8k.py or
any other existing runner: ProofWriter's prompt, gold schema (True/False/
Unknown, not a number/letter/boxed literal) and label firewall are all new;
reusing a numeric/MCQ runner would either silently break its own frozen
behaviour or require bending this task into a shape it does not have.

WHAT IS HELD, matching every other runner in this repo:
  - vc.regenerate(..., diff_matrices=..., prefill_only=True, prefill_tail_len=1)
  - bare-string (no chat template)
  - greedy (temperature=0.0)
  - batch_size=8 (fixed per spec; NOT swept)
  - ONE model's whole alpha curve uses the SAME batched vc.regenerate() path
    for every alpha including 0 -- alpha=0 passes a REAL all-zero diff matrix
    (`raw_mask * 0`), exactly like every other runner in this family. Hooks
    register and the zero add executes; steering_fires reads 0 only because a
    zero row is never counted as steered. This is deliberate and matches the
    project-wide convention -- do not "optimize" alpha=0 onto vc.generate().

OOM POLICY: this script does NOT catch torch.cuda.OutOfMemoryError and fall
back to a smaller batch size. If the process OOMs, it stops and the traceback
is the report -- per the task brief, this must never silently become bs=1.

TRUNCATION is measured with return_metadata=True (llms.py's
stop_reason=="budget_exhausted"), so the per-cell JSON records an exact
truncation flag per sample rather than inferring it from text length.

@author: proofwriter_owa task
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt import build_prompt, PROMPT_TEMPLATE_ID  # noqa: E402
from commitment import first_strict_marker_start  # noqa: E402

PROTOCOL = "proofwriter-owa-v0"
FORBIDDEN_KEYS = ("answer", "label", "gold", "gold_answer", "correct",
                  "accuracy", "proof", "proofs", "target")

EXPECTED_CELLS = {
    "llama3":  {(-6, 11, 20), (-4, 11, 20), (0, 11, 20), (4, 11, 20)},
    "qwen2.5": {(-6, 16, 22), (0, 16, 22), (6, 16, 22), (8, 16, 22)},
}

# FROZEN 2026-09-04 (PREREG_PROOFWRITER_OWA.md S6). See the --max_new_tokens
# argparse help text and the hard check in main() for why this is enforced,
# not merely defaulted.
MAX_NEW_TOKENS_FROZEN = 1024


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(EXPECTED_CELLS))
    p.add_argument("--size", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--manifest", required=True,
                   help="the LABEL-FREE manifest_blind.json (or a preflight/"
                        "pilot subset built from it)")
    p.add_argument("--mask_path", required=True)
    p.add_argument("--configs", required=True, nargs="+",
                   help="e.g. 0-11-20 neg6-11-20 neg4-11-20 4-11-20")
    p.add_argument("--out_dir", required=True)
    # FROZEN, not a tunable CLI knob (review finding, 2026-09-04): raised
    # from the original default of 768 to 1024 after the llama3 alpha=0
    # preflight (30 items) showed 30/30 truncation at 768. Manual inspection
    # found the truncation is NOT uniform in cause -- some genuine multi-hop
    # reasoning that simply needed more room, some stable degenerate
    # repetition loops (llama3-specific, known/expected, not a bug this
    # budget change is meant to cure). Per the user's explicit decision:
    # raise ONCE to 1024, do not chase it further upward. A comment-only
    # constraint left the value changeable from the command line with no
    # enforcement, so this is now a hard CLI-level rejection of any other
    # value (see MAX_NEW_TOKENS_FROZEN below and the check in main()) rather
    # than a default that could silently be overridden.
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS_FROZEN,
                   help=f"FROZEN at {MAX_NEW_TOKENS_FROZEN} "
                        "(PREREG_PROOFWRITER_OWA.md S6); any other value is "
                        "rejected, not silently accepted.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_shot", type=int, default=0,
                   help="frozen TRAIN-split exemplars; default 0 (zero-shot)")
    p.add_argument("--exemplar_file", default=None,
                   help="required if --n_shot > 0; frozen train-split "
                        "exemplar pool built offline, never from test")
    p.add_argument("--tag", default=None,
                   help="optional suffix for out_dir subdirs, e.g. "
                        "'preflight' or 'pilot' -- keeps those runs from "
                        "colliding with the formal sweep's file names")
    return p.parse_args()


def load_manifest(path):
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d["meta"], d["data"]
    if meta.get("contains_labels") is not False:
        die("manifest file does not declare contains_labels=false. Point "
            "--manifest at manifest_blind.json (or a blind subset of it), "
            "never at manifest_gold.json.")
    for s in data:
        bad = [k for k in s if k.lower() in FORBIDDEN_KEYS]
        if bad:
            die(f"label field {bad} present in the manifest file; "
                "generation must be label-free")
    return meta, data


def load_exemplars(path, n_shot):
    if not path:
        die("--n_shot > 0 requires --exemplar_file")
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d.get("meta", {}), d["data"]
    if meta.get("split") != "train":
        die(f"exemplar file declares split={meta.get('split')!r}, must be "
            "'train' -- exemplars must never come from the test split")
    if len(data) < n_shot:
        die(f"exemplar file has {len(data)} rows, need >= {n_shot}")
    return data[:n_shot]


def main():
    args = parse_args()
    # HARD REJECTION, not just a default (review finding, 2026-09-04): a
    # comment-only constraint on the argparse default left --max_new_tokens
    # freely overridable from the command line with no enforcement, so the
    # "raise once to 1024 and never again" decision was not actually
    # binding. Any value other than the frozen one is refused outright.
    if args.max_new_tokens != MAX_NEW_TOKENS_FROZEN:
        die(f"--max_new_tokens={args.max_new_tokens} != the frozen value "
            f"{MAX_NEW_TOKENS_FROZEN} (PREREG_PROOFWRITER_OWA.md S6). The "
            "budget was raised from 768 to 1024 exactly once, by explicit "
            "human decision, after the llama3 alpha=0 preflight; it is not "
            "raised further without a new, equally explicit decision that "
            "changes MAX_NEW_TOKENS_FROZEN in this file (and the matching "
            "value in PREREG_PROOFWRITER_OWA.md S6), not a one-off CLI flag.")
    meta, samples = load_manifest(args.manifest)

    cfgs_raw = args.configs
    import utils  # noqa: E402  (repo root on sys.path via the insert above)
    cfgs = utils.parse_configs(cfgs_raw)
    got = {(al, ls, le) for al, (ls, le) in cfgs}
    want = EXPECTED_CELLS[args.model]
    if not got.issubset(want):
        die(f"{args.model} cells {sorted(got)} are outside this task's OWN "
            f"frozen dose set {sorted(want)}. This is ProofWriter's own "
            "workpoint exploration; it does not read or search the GSM8K "
            "fixed-workpoint transfer doses.")

    exemplars = load_exemplars(args.exemplar_file, args.n_shot) if args.n_shot > 0 else []

    from llms import VicundaModel  # noqa: E402

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    if vc.tokenizer.padding_side != "left":
        die(f"tokenizer.padding_side is {vc.tokenizer.padding_side!r}, "
            "expected 'left'. This must be true unconditionally from "
            "VicundaModel.__init__; something has overridden it.")
    raw_mask = np.load(args.mask_path)
    mask_sha = hashlib.sha256(open(args.mask_path, "rb").read()).hexdigest()
    os.makedirs(args.out_dir, exist_ok=True)

    device_note = {"host": platform.node(),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    prompts = [build_prompt(s["theory_text"], s["question_text"], exemplars)
              for s in samples]
    # Rendered once, shared by every alpha of this model/manifest: proves the
    # prompts are byte-identical across the whole curve so the analyzer can
    # cross-check it per cell (matches zebralogic/get_answer_zebralogic.py's
    # prompt_sha256 convention -- ProofWriter's generator was missing this
    # entirely, so no consumer could ever detect a prompt drift between doses).
    prompt_sha256 = hashlib.sha256(
        "\n".join(prompts).encode("utf-8")).hexdigest()

    for alpha, (ls, le) in cfgs:
        tag = f"mdf_{alpha}".replace("-", "neg")
        subdir = tag if not args.tag else f"{args.tag}_{tag}"
        out = os.path.join(args.out_dir, subdir,
                           f"proofwriter_owa_{args.size}_{ls}_{le}.json")
        if os.path.exists(out):
            print(f"skip existing {out}")
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)

        diff = raw_mask * alpha
        vc.steering_fire_count(reset=True)

        gen = []
        for i in tqdm(range(0, len(prompts), args.batch_size),
                      desc=f"proofwriter-owa a={alpha}"):
            batch = prompts[i: i + args.batch_size]
            gen.extend(vc.regenerate(
                batch,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                diff_matrices=list(diff),
                batch_size=args.batch_size,
                return_metadata=True,
            ))

        # HARD LENGTH CHECK (review finding #1, 2026-09-04): zip(samples, gen)
        # below silently truncates to the shorter sequence if a batch call
        # ever returned fewer rows than it was given (a partial-batch bug, an
        # OOM-recovery path returning a short list, etc.) -- the cell would
        # then write fewer rows than samples with NO error, and every
        # downstream consumer (steering_fires count, the evaluator's row-count
        # check) would only catch the row-count half of this, not WHICH
        # samples got dropped. Assert 1:1 length before doing anything else
        # with `gen`.
        if len(gen) != len(samples):
            die(f"generation returned {len(gen)} rows for {len(samples)} "
                f"prompts at alpha={alpha}; zip() would silently drop "
                f"{abs(len(gen) - len(samples))} sample(s). This must never "
                "happen under vc.regenerate's documented contract -- treat "
                "as a hard failure, not a partial result.")

        fires = vc.steering_fire_count()
        n_layers = len(utils.decoder_layer_range(ls, le))
        expect = 0 if alpha == 0 else n_layers * len(samples)
        if fires != expect:
            die(f"steering_fires {fires} != {expect} "
                f"(L={n_layers}, n={len(samples)}, alpha={alpha}); "
                "the intervention is unverified, so the cell is not usable.")

        rows = []
        n_truncated = 0
        for s, g in zip(samples, gen):
            truncated = (g["stop_reason"] == "budget_exhausted")
            n_truncated += int(truncated)
            text = g["text"]
            # pre_answer_reasoning_tokens is computed HERE, not in the
            # evaluator: commitment.per_sample_commitment() only fills this
            # field when handed a real tokenizer, and the evaluator runs
            # offline with no model/tokenizer loaded (eval_proofwriter_owa.py
            # was calling per_sample_commitment(text) with no tokenizer at
            # all, so the field was unconditionally None in every real run).
            # The tokenizer is already loaded here; reuse
            # first_strict_marker_start (the SAME function
            # per_sample_commitment uses internally) so the pre-marker slice
            # tokenized here is defined identically to what the evaluator
            # would derive char-wise.
            marker_start = first_strict_marker_start(text)
            pre_answer_tokens = None
            if marker_start is not None:
                try:
                    pre_answer_tokens = len(vc.tokenizer(
                        text[:marker_start], add_special_tokens=False)["input_ids"])
                except Exception:
                    pre_answer_tokens = None
            rows.append({
                "sample_id": s["sample_id"], "key": s["key"],
                "dataset": s["dataset"],
                "official_theory_id": s["official_theory_id"],
                "official_qid": s["official_qid"],
                "generated": text,
                "generated_token_count": g["generated_token_count"],
                "pre_answer_reasoning_tokens": pre_answer_tokens,
                "stop_reason": g["stop_reason"],
                "truncated": truncated,
            })

        json.dump({"meta": {
            "protocol": PROTOCOL, "prompt_template_id": PROMPT_TEMPLATE_ID,
            "model": args.model, "size": args.size,
            "alpha": alpha, "layer_start": ls, "layer_end": le, "L": n_layers,
            "mask_path": args.mask_path, "mask_sha256": mask_sha,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature, "top_p": args.top_p,
            "batch_size": args.batch_size,
            "n_shot": args.n_shot,
            "cot": True, "cot_note": ("own construction, NOT an official "
                                      "ProofWriter LLM prompt"),
            "chat_template": False, "prefill_only": True, "prefill_tail_len": 1,
            "steering_fires": fires,
            "prompt_sha256": prompt_sha256,
            "padding_side": vc.tokenizer.padding_side,
            "manifest_sha256_16": meta.get("manifest_sha256_16"),
            "n": len(rows),
            "n_truncated": n_truncated,
            "truncation_rate": n_truncated / len(rows) if rows else None,
            "contains_labels": False, "accuracy_computed": False,
            "provenance": device_note,
            "not_a_transfer_test": True,
        }, "data": rows}, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  wrote {out}  steering_fires={fires}  "
              f"truncation_rate={n_truncated}/{len(rows)}")

    print("\nGeneration complete. NO accuracy was computed -- by construction.")
    print("Next: python proofwriter_owa/eval_proofwriter_owa.py "
          "(the only script that reads gold)")


if __name__ == "__main__":
    main()
