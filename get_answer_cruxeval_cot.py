#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRUXEval-O explicit-CoT follow-up runner (`cot-transfer-followup-v0`).

Fork of `get_answer_cruxeval.py` with exactly ONE planned change: the
project's existing GSM8K/MATH CoT cue ("Let's think step by step.\\n") is
inserted before the final-format instruction. Everything else -- items,
order, digest, blind/gold firewall, mask, band, budget (768), batch size
(24), parser (FIRST=MAIN '####'+ast.literal_eval), anchor ("Response: ") --
is inherited verbatim from `cruxeval-p4c-v0`. See
`docs/PREREG_COT_TRANSFER_FOLLOWUP.md`.

THIS IS A POST-HOC EXPLORATORY FOLLOW-UP, NOT A REPLICATION. It does not
touch, rescale, or supersede the frozen No-CoT P4c result.

FROZEN, ASSERTED AT RUNTIME
---------------------------
* CoT prompt template sha256 d3ec2bd57b927b4a -- editing it is fatal
* anchor "Response: ", prefill_only, tail=1, greedy, 768 tokens, bs=24
* alpha must be one of that model's four frozen doses; nothing else is
  expressible
* steering_fires must be 0 at alpha=0 and L*n*1 otherwise
* the 300 items must carry the frozen digest
* the CoT template's final injection token must equal the No-CoT template's --
  verified on the actual tokenizer; a mismatch is a hard stop

NO few-shot, NO role/persona, NO stop string (the prompt still literally
contains '####').

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

PROTOCOL = "cot-transfer-followup-v0"
BASE_PROTOCOL = "cruxeval-p4c-v0"
COT_CUE = "Let's think step by step.\n"

# The frozen No-CoT template, reused only to build the token-injection check.
PROMPT_NOCOT = (
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
PROMPT_NOCOT_SHA256 = "0d94415eb6a62a78"

PROMPT_COT = (
    "[PYTHON]\n"
    "{code}\n"
    "\n"
    "assert f({input}) == ??\n"
    "[/PYTHON]\n"
    "\n"
    f"{COT_CUE}"
    "Complete the assertion by predicting the output of executing the function.\n"
    "End your response with exactly one line in the following format:\n"
    "#### <Python literal>\n"
    "\n"
    "Response: "
)
PROMPT_COT_SHA256 = "d3ec2bd57b927b4a"
ANCHOR = "Response: "

N_FORMAL = 300
N_PREFLIGHT = 8
FORMAL_DIGEST = "4580b7a9a9ef6054"
BUDGET = 768
BATCH_SIZE = 24

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
    p = PROMPT_COT.format(code=item["code"].rstrip(), input=item["input"])
    if not p.endswith(ANCHOR):
        die(f"prompt does not end at the frozen anchor {ANCHOR!r}")
    return p


def is_loop(text):
    """Shared is_loop convention (final 40-char block recurs >=4x), unchanged
    from get_answer_cruxeval.py."""
    b = text[-40:]
    return len(b) == 40 and text.count(b) >= 4


def descriptive(text):
    """Morphological ONLY, unchanged definition from get_answer_cruxeval.py."""
    ms = list(MARKER_RE.finditer(text))
    first = ms[0] if ms else None
    stripped = text.lstrip()
    return {
        "n_markers": len(ms),
        "answer_first": stripped.startswith("####"),
        "pre_marker_chars": first.start() if first else None,
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
                        "preflight so the scorer refuses it.")
    return p.parse_args()


def load_questions(path, preflight=False):
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d["meta"], d["data"]
    if meta.get("contains_labels") is not False:
        die("questions file does not declare contains_labels=false. Point "
            "--questions at the *_formal_blind.json, not the gold copy.")
    if meta.get("questions_sha256") != FORMAL_DIGEST:
        die(f"questions_sha256 {meta.get('questions_sha256')!r} != the frozen "
            f"{FORMAL_DIGEST!r}; this is a different sample")
    if len(data) != N_FORMAL:
        die(f"{len(data)} items, expected {N_FORMAL}")
    if sorted(s["sample_id"] for s in data) != list(range(N_FORMAL)):
        die(f"sample_ids do not cover 0..{N_FORMAL - 1}")
    if preflight:
        data = data[:N_PREFLIGHT]
    for s in data:
        bad = [k for k in s if k.lower() in LABEL_FIELDS]
        if bad:
            die(f"label field {bad} present in the questions file; generation "
                "must be label-free")
    return meta, data


def verify_injection_token_equal(vc, sample_item):
    """The CoT and No-CoT templates are byte-identical from the anchor
    'Response: ' backward, so their final prompt token must match. Verified on
    the real tokenizer, per PREREG_COT_TRANSFER_FOLLOWUP.md S3.4."""
    got_nocot = hashlib.sha256(PROMPT_NOCOT.encode()).hexdigest()[:16]
    if got_nocot != PROMPT_NOCOT_SHA256:
        die(f"No-CoT reference template sha256 {got_nocot} != frozen "
            f"{PROMPT_NOCOT_SHA256}; cannot verify injection-token equality "
            f"against a drifted reference")
    p_nocot = PROMPT_NOCOT.format(code=sample_item["code"].rstrip(),
                                  input=sample_item["input"])
    p_cot = build_prompt(sample_item)
    tok = vc.tokenizer
    ids_nocot = tok(p_nocot, add_special_tokens=True)["input_ids"]
    ids_cot = tok(p_cot, add_special_tokens=True)["input_ids"]
    last_nocot, last_cot = ids_nocot[-1], ids_cot[-1]
    dec_nocot, dec_cot = tok.decode([last_nocot]), tok.decode([last_cot])
    print(f"[cot-followup] injection-token check: No-CoT last token id="
          f"{last_nocot} ({dec_nocot!r}), CoT last token id={last_cot} "
          f"({dec_cot!r})")
    if last_nocot != last_cot:
        die(f"injection token differs between CoT ({last_cot}, {dec_cot!r}) "
            f"and No-CoT ({last_nocot}, {dec_nocot!r}) templates -- hard stop "
            f"per PREREG_COT_TRANSFER_FOLLOWUP.md S3.4")
    print("[cot-followup] injection-token equality VERIFIED")


def main():
    args = parse_args()

    got = hashlib.sha256(PROMPT_COT.encode()).hexdigest()[:16]
    if got != PROMPT_COT_SHA256:
        die(f"CoT prompt template sha256 {got} != the frozen "
            f"{PROMPT_COT_SHA256}. The template is frozen and may not be "
            f"edited.")

    if args.max_new_tokens != BUDGET or args.batch_size != BATCH_SIZE:
        die(f"budget/batch are inherited frozen at {BUDGET}/{BATCH_SIZE}; got "
            f"{args.max_new_tokens}/{args.batch_size}")
    if args.temperature != 0.0:
        die("temperature is frozen at 0.0 (greedy)")

    parsed = utils.parse_configs(args.configs)
    allowed = EXPECTED_CELLS[args.model]
    for alpha, (ls, le) in parsed:
        if (alpha, ls, le) not in allowed:
            die(f"cell (alpha={alpha}, band={ls}-{le}) is not in {args.model}'s "
                f"frozen CoT matrix {sorted(allowed)}; alpha is read from the "
                "frozen GSM8K record, never re-searched here.")

    meta, samples = load_questions(args.questions, args.preflight)

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    raw_mask = np.load(args.mask_path)
    mask_sha = hashlib.sha256(open(args.mask_path, "rb").read()).hexdigest()
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.preflight:
        verify_injection_token_equal(vc, samples[0])

    device_note = {
        "host": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    for alpha, (ls, le) in parsed:
        tag = f"mdf_{alpha}".replace("-", "neg")
        stem = "cruxeval_o_cot_preflight" if args.preflight else "cruxeval_o_cot"
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
                      desc=f"cruxeval-o-cot a={alpha}"):
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

        n_marker = sum(1 for r in rows if r["n_markers"] > 0)
        if n_marker == 0 and not args.preflight:
            die(f"cell {tag} produced NO '####' marker in any of {len(rows)} "
                "generations. Prompt and parser are frozen and may NOT be "
                "redesigned in response.")

        json.dump({"meta": {"protocol": PROTOCOL, "base_protocol": BASE_PROTOCOL,
                            "task": "cruxeval_o",
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
                            "cot": True, "cot_cue": COT_CUE, "few_shot": False,
                            "stop_strings": None,
                            "steering_fires": fires,
                            "questions_sha256": meta["questions_sha256"],
                            "revision": meta["revision"],
                            "prompt_template": PROMPT_COT,
                            "prompt_sha256_prefix": PROMPT_COT_SHA256,
                            "anchor": ANCHOR,
                            "provenance": device_note,
                            "contains_labels": False,
                            "accuracy_computed": False,
                            "preflight": args.preflight,
                            "exploratory_followup": True,
                            "n": len(rows)},
                   "data": rows}, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  wrote {out}  steering_fires={fires}  marker_rate="
              f"{n_marker / len(rows):.3f}")

    print("\nGeneration complete. NO accuracy was computed -- by construction.")
    print("This is a POST-HOC EXPLORATORY follow-up; it does not replace the "
          "frozen No-CoT cruxeval-p4c-v0 result.")


if __name__ == "__main__":
    main()
