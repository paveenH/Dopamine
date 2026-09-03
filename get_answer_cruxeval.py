#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRUXEval-O generation for the P4c fixed-workpoint transfer. LABEL-FREE.

Protocol `cruxeval-p4c-v0`. Third task of P4's one question (after LogiQA 2.0
and BBH numeric), so P4c, not P5.

WHY A FORK RATHER THAN get_answer_regenerate_gsm8k.py
-----------------------------------------------------
That script scores each sample with `is_correct_gsm8k`, writes `correct_*` and
PRINTS accuracy. Here, generation must produce no correctness field at all, so
that the preflight cannot be read for accuracy and the transfer test stays
separable from the generation step. Generation and scoring are separate
scripts, as in P3/P4/P4b.

The separation is STRUCTURAL. This runner reads the whitelist-built
`*_formal_blind.json` and REFUSES a file whose meta does not say
`contains_labels: false`. "The code does not access gold" is much weaker than
"gold is not reachable" -- the P2 lesson.

FROZEN, ASSERTED AT RUNTIME
---------------------------
* prompt template sha256 (see PROMPT_SHA256) -- editing it is fatal
* anchor "Response: ", prefill_only, tail=1, greedy, 768 tokens, bs=24
* alpha must be one of that model's four frozen doses; nothing else is
  expressible
* steering_fires must be 0 at alpha=0 and L*n*1 otherwise
* the 300 items must carry the frozen digest

NO CoT, NO FEW-SHOT, NO STOP STRING. `Think step by step` is absent so that
whether reasoning is externalised stays the model's own behaviour. A stop
string is impossible here anyway: the prompt literally contains `####` and HF
`stop_strings` matches anywhere in the output -- the CGT failure, where
invalid_rate jumped 0.02 to 0.11.

alpha=0 PASSES A REAL ALL-ZERO MATRIX, exactly as the GSM8K driver does
(`diff = raw_mask * alpha`, unconditional). Hooks register and the zero add
executes; `steering_fires` reads 0 only because a zero row is not counted as
steered. Do not "optimize" this into a no-hook path, or alpha=0 would run
through a different execution path than the steered cells and stop being their
baseline.

@author: paveenhuang
"""

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llms import VicundaModel                                   # noqa: E402
import utils                                                    # noqa: E402

PROTOCOL = "cruxeval-p4c-v0"

# Frozen verbatim by the pre-registration. Neutral, No-CoT, no few-shot.
PROMPT = (
    "[PYTHON]\n"
    "{code}\n"
    "\n"
    "assert f({input}) == ??\n"
    "[/PYTHON]\n"
    "\n"
    "Complete the assertion by predicting the output of executing the function.\n"
    "End your response with exactly one line in the following format:\n"
    "#### <Python literal>\n"
    "\n"
    "Response: "
)
PROMPT_SHA256 = "0d94415eb6a62a78"   # frozen 2026-09-03; editing PROMPT is fatal
ANCHOR = "Response: "

N_FORMAL = 300
N_PREFLIGHT = 8
FORMAL_DIGEST = "4580b7a9a9ef6054"
BUDGET = 768
BATCH_SIZE = 24

# The complete frozen configuration per model -- alpha AND band, not just
# "four cells". MAIN is the workpoint read from the frozen GSM8K record; the
# other three are diagnostics that are reported but never redefine it.
EXPECTED_CELLS = {
    "llama3":  {(0, 11, 20), (-6, 11, 20), (-4, 11, 20), (4, 11, 20)},
    "qwen2.5": {(0, 16, 22), (8, 16, 22), (6, 16, 22), (-6, 16, 22)},
}
WORKPOINT = {"llama3": -6, "qwen2.5": 8}

LABEL_FIELDS = {"output", "target", "answer", "label", "gold", "gold_answer",
                "correct", "accuracy", "solution"}

MARKER_RE = re.compile(r"####[ \t]*(.*)")


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def build_prompt(item):
    p = PROMPT.format(code=item["code"].rstrip(), input=item["input"])
    if not p.endswith(ANCHOR):
        die(f"prompt does not end at the frozen anchor {ANCHOR!r}")
    return p


def is_loop(text):
    """The repo's shared is_loop convention, so degeneracy stays comparable
    with GSM8K/LogiQA: the final 40-character block recurs >= 4x. A permissive
    n-gram proxy read 80-86% on GSM8K and was entirely false positives."""
    b = text[-40:]
    return len(b) == 40 and text.count(b) >= 4


def descriptive(text):
    """Morphological ONLY. Text before the marker is NOT assumed to be
    reasoning, and no content judgement is made. These fields are provenance
    for later exploratory reading; none of them is a gate, and each is an
    OUTCOME of alpha, so stratifying accuracy on one is post-treatment
    stratification -- consistent-with evidence, never mediation."""
    ms = list(MARKER_RE.finditer(text))
    first = ms[0] if ms else None
    stripped = text.lstrip()
    return {
        "n_markers": len(ms),
        "answer_first": stripped.startswith("####"),
        "pre_marker_chars": first.start() if first else None,
        # AUXILIARY: a longer repetition tail lowers this independently of
        # anything preceding the marker, so it cannot carry a "more reasoning"
        # reading on its own.
        "first_marker_pos": (first.start() / len(text)) if (first and text) else None,
        "degenerate_tail": is_loop(text),
        "gen_chars": len(text),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(EXPECTED_CELLS))
    p.add_argument("--size", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--questions", required=True,
                   help="the LABEL-FREE *_formal_blind.json from data_cruxeval.py")
    p.add_argument("--mask_path", required=True)
    p.add_argument("--configs", required=True, nargs="+",
                   help="e.g. 0-11-20 neg6-11-20")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_new_tokens", type=int, default=BUDGET)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--preflight", action="store_true",
                   help="FORMAT ONLY: first N_PREFLIGHT items, output tagged "
                        "preflight so the scorer refuses it. Computes no "
                        "correctness field -- there is no accuracy to read.")
    return p.parse_args()


def load_questions(path, preflight=False):
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d["meta"], d["data"]
    if meta.get("contains_labels") is not False:
        die("questions file does not declare contains_labels=false. Point "
            "--questions at the *_formal_blind.json, not the gold copy.")
    if meta.get("protocol") != PROTOCOL:
        die(f"questions file protocol {meta.get('protocol')!r} != {PROTOCOL!r}")
    if meta.get("questions_sha256") != FORMAL_DIGEST:
        die(f"questions_sha256 {meta.get('questions_sha256')!r} != the frozen "
            f"{FORMAL_DIGEST!r}; this is a different sample")
    if len(data) != N_FORMAL:
        die(f"{len(data)} items, expected {N_FORMAL}")
    if sorted(s["sample_id"] for s in data) != list(range(N_FORMAL)):
        die(f"sample_ids do not cover 0..{N_FORMAL - 1}")
    # The digest and the count above are checked on the FULL file first, so a
    # preflight still proves it is reading the frozen sample; only then is the
    # prefix taken. Truncating before the check would let a preflight run on
    # anything at all.
    if preflight:
        data = data[:N_PREFLIGHT]
    for s in data:                      # fail closed on any leaked label
        bad = [k for k in s if k.lower() in LABEL_FIELDS]
        if bad:
            die(f"label field {bad} present in the questions file; generation "
                "must be label-free")
    return meta, data


def main():
    args = parse_args()

    # The template is frozen by the pre-registration; an edit changes the
    # injection site and what every descriptive field means.
    got = hashlib.sha256(PROMPT.encode()).hexdigest()[:16]
    if got != PROMPT_SHA256:
        die(f"prompt template sha256 {got} != the frozen {PROMPT_SHA256}. The "
            "template is frozen by the pre-registration and may not be edited.")

    if args.max_new_tokens != BUDGET or args.batch_size != BATCH_SIZE:
        die(f"budget/batch are frozen at {BUDGET}/{BATCH_SIZE}; got "
            f"{args.max_new_tokens}/{args.batch_size}")
    if args.temperature != 0.0:
        die("temperature is frozen at 0.0 (greedy)")

    parsed = utils.parse_configs(args.configs)
    allowed = EXPECTED_CELLS[args.model]
    for alpha, (ls, le) in parsed:
        if (alpha, ls, le) not in allowed:
            die(f"cell (alpha={alpha}, band={ls}-{le}) is not in {args.model}'s "
                f"frozen matrix {sorted(allowed)}. This protocol never searches "
                "doses on CRUXEval; alpha is read from the frozen GSM8K record.")

    meta, samples = load_questions(args.questions, args.preflight)

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    raw_mask = np.load(args.mask_path)
    mask_sha = hashlib.sha256(open(args.mask_path, "rb").read()).hexdigest()
    os.makedirs(args.out_dir, exist_ok=True)

    # Provenance, not a constraint: cells are NOT required to share a GPU.
    # Because bf16 greedy may vary across hardware, a contrast between cells on
    # different devices is reported as a CROSS-RUN pairing; pairing means
    # alignment by the frozen item order, never hardware identity.
    device_note = {
        "host": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    for alpha, (ls, le) in parsed:
        tag = f"mdf_{alpha}".replace("-", "neg")
        stem = "cruxeval_o_preflight" if args.preflight else "cruxeval_o"
        out = os.path.join(args.out_dir, tag,
                           f"{stem}_{args.size}_{ls}_{le}.json")
        if os.path.exists(out):
            print(f"skip existing {out}")
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)

        diff = raw_mask * alpha
        vc.steering_fire_count(reset=True)

        prompts = [build_prompt(s) for s in samples]
        gen = []
        for i in tqdm(range(0, len(prompts), args.batch_size),
                      desc=f"cruxeval-o a={alpha}"):
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
            die(f"steering_fires {fires} != {expect} (L={n_layers}, "
                f"n={len(samples)}, alpha={alpha}); the intervention is "
                "unverified, so the cell is not usable.")

        rows = [{"sample_id": s["sample_id"], "source_id": s["source_id"],
                 "generated": g, **descriptive(g)}
                for s, g in zip(samples, gen)]

        # HARD STOP 4: systematic parser failure. Only a cell with NO valid
        # marker anywhere qualifies -- a high but non-total failure rate is a
        # result to report, not a licence to redefine the parser.
        n_marker = sum(1 for r in rows if r["n_markers"] > 0)
        if n_marker == 0 and not args.preflight:
            die(f"cell {tag} produced NO '####' marker in any of {len(rows)} "
                "generations. This is a systematic parser failure (hard stop "
                "5 of the pre-registration). It is NOT a licence to redefine "
                "the parser or the prompt.")

        json.dump({"meta": {"protocol": PROTOCOL, "task": "cruxeval_o",
                            "model": args.model, "size": args.size,
                            "alpha": alpha, "layer_start": ls, "layer_end": le,
                            "L": n_layers,
                            "role": ("workpoint" if alpha == WORKPOINT[args.model]
                                     else ("baseline" if alpha == 0
                                           else "diagnostic")),
                            "mask_path": args.mask_path, "mask_sha256": mask_sha,
                            "max_new_tokens": args.max_new_tokens,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "batch_size": args.batch_size,
                            "cot": False, "few_shot": False,
                            "stop_strings": None,
                            "steering_fires": fires,
                            "questions_sha256": meta["questions_sha256"],
                            "revision": meta["revision"],
                            "prompt_template": PROMPT,
                            "prompt_sha256_prefix": PROMPT_SHA256,
                            "anchor": ANCHOR,
                            "provenance": device_note,
                            "contains_labels": False,
                            "accuracy_computed": False,
                            "preflight": args.preflight,
                            "n": len(rows)},
                   "data": rows}, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  wrote {out}  steering_fires={fires}  marker_rate="
              f"{n_marker / len(rows):.3f}")

    print("\nGeneration complete. NO accuracy was computed -- by construction.")
    print("Next: python eval_cruxeval.py   (the only script that reads gold)")


if __name__ == "__main__":
    main()
