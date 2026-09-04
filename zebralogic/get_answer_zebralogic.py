#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZebraLogic-Easy generation for the zebralogic-easy-v0 four-point workpoint
exploration. LABEL-FREE (reads only the *_blind.json shape scaffold, never a
real solution cell). Protocol: docs/PREREG_ZEBRALOGIC_EASY.md.

Mirrors get_answer_bbh_numeric.py's structure and conventions:
  - alpha=0 passes a REAL all-zero diff matrix, unconditional. Hooks register
    and the zero add executes; steering_fires reads 0 only because a zero row
    is not counted as steered (see llms.py steering_fire_count docstring).
    This is NOT the pv6/PV10 "no hook at all" path -- do not "optimize" it.
  - Generation and scoring are separate scripts; this one computes no
    accuracy and reads no gold.
  - Bare-string prompt, no chat template (see prereg section 3).

WHAT DIFFERS FROM BBH: the prompt is the official one-shot ZEBRA_GRID template
(zebralogic/official_zebra_grid_template.py), not a GSM8K-style neutral
directive, and the per-item JSON answer scaffold is built fresh per item from
that item's own solution SHAPE (house count + column headers) -- label-free,
since the shape is identical between the public and private datasets.

MODES:
  --mode preflight   5 fixed items (index 0..4 of the frozen order), alpha=0
                      only, plus a non-zero SMOKE alpha (not one of the four
                      frozen doses) purely to assert steering_fires == L*N.
                      Writes to a separate _preflight/ output tree.
  --mode canary       a small fixed deterministic item subset (frozen indices,
                      see CANARY_INDICES), alpha=0 only, meant to be run on
                      each GPU in use so eval_zebralogic.py --canary_check can
                      compare Puzzle/Cell accuracy, parse status and
                      truncation status across cards before any cross-GPU
                      pooling is trusted. Writes to a separate _canary/{device
                      tag} output tree.
  --mode formal       the full 280-item, frozen-dose sweep. Refuses any alpha
                      outside the model's frozen four-point set (0 plus the
                      three specified in --configs, matching FROZEN_ALPHAS).

@author: paveenhuang
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

# `python zebralogic/get_answer_zebralogic.py` (how run_zebralogic.sh invokes
# this, and the only supported invocation) sets sys.path[0] to this script's
# OWN directory (zebralogic/), not the repo root -- verified: Python's
# sys.path[0] rule is "the script's directory", not cwd. `llms`/`utils` live
# at the repo root, so it must be added explicitly; this script's own
# directory must also stay on the path for the official_* sibling imports.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

import numpy as np
from tqdm import tqdm

from llms import VicundaModel
import utils

from official_zebra_grid_template import apply_lgp_grid_template  # noqa: E402

PROTOCOL = "zebralogic-easy-v0"
N_EASY = 280
N_PER_SIZE = 40
N_SIZES = 7
PREFLIGHT_PER_SIZE = 5
# Prereg S5 (corrected 2026-09-04): 5 items PER SIZE, not 5 items total from
# one size. Frozen indices are the first 5 of each size's 40-item block in
# the frozen sample_id order written by data_zebralogic.py --
# {0..4, 40..44, 80..84, 120..124, 160..164, 200..204, 240..244}, 35 items.
PREFLIGHT_INDICES = tuple(
    size_rank * N_PER_SIZE + j
    for size_rank in range(N_SIZES)
    for j in range(PREFLIGHT_PER_SIZE)
)
PREFLIGHT_N = len(PREFLIGHT_INDICES)  # 35
# Frozen small canary subset: first item of each of the 7 easy sizes (indices
# are 0, 40, 80, 120, 160, 200, 240 in the frozen sample_id order written by
# data_zebralogic.py, i.e. size_rank * N_PER_SIZE) plus the LAST item of the
# largest easy size, for 8 items total -- deterministic, no RNG. The largest
# easy size is size_rank 6 (last of the 7), whose block spans indices
# 240..279, so its last item is index 279 (6*40 + 39), not 239 -- 239 was the
# LAST item of size_rank 5's block, a wrong-block bug fixed 2026-09-04.
CANARY_INDICES = tuple(i * N_PER_SIZE for i in range(N_SIZES)) + (N_EASY - 1,)

# Prereg S3 (2026-09-04): 2048 is the default, 3072 is the ONLY allowed
# escalation step, and 4096 is explicitly out of scope. Enforced structurally
# so this cannot be silently bumped by a launcher typo or an ad-hoc override.
ALLOWED_MAX_NEW_TOKENS = (2048, 3072)

# Frozen four-point dose sets, read from docs/PREREG_ZEBRALOGIC_EASY.md
# section 3 -- this script refuses any --configs alpha outside these sets in
# --mode formal, so a typo cannot silently sweep an unregistered dose.
FROZEN_ALPHAS = {
    "llama3": (-6, -4, 0, 4),
    "qwen2.5": (-6, 0, 6, 8),
}
FORBIDDEN_KEYS = ("solution", "answer", "gold", "gold_answer", "correct",
                  "accuracy")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(FROZEN_ALPHAS))
    p.add_argument("--size", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--items", required=True,
                   help="the LABEL-FREE *_blind.json from data_zebralogic.py")
    p.add_argument("--mask_path", required=True)
    p.add_argument("--configs", required=True, nargs="+",
                   help="e.g. neg6-11-20 0-11-20 neg4-11-20 4-11-20")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mode", required=True, choices=("preflight", "canary", "formal"))
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device_tag", default=None,
                   help="required for --mode canary; a short label identifying "
                        "the physical GPU this run used (e.g. nvidia-smi name "
                        "or a card index), used only in the output path and "
                        "metadata so eval_zebralogic.py --canary_check can "
                        "compare across cards")
    return p.parse_args()


def load_items(path, model):
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d["meta"], d["data"]
    if meta.get("contains_labels") is not False:
        sys.exit("FAIL: items file does not declare contains_labels=false. "
                 "Point --items at the *_blind.json, not any gold-bearing file.")
    if meta.get("protocol") != PROTOCOL:
        sys.exit(f"FAIL: items file protocol {meta.get('protocol')!r} != "
                 f"{PROTOCOL!r}")
    if len(data) != N_EASY:
        sys.exit(f"FAIL: expected {N_EASY} items, got {len(data)}")
    for s in data:
        bad = [k for k in s if k.lower() in FORBIDDEN_KEYS]
        if bad:
            sys.exit(f"FAIL: label field {bad} present in the items file; "
                     "generation must be label-free")
    ids = [s["id"] for s in data]
    sample_ids = [s["sample_id"] for s in data]
    if sample_ids != list(range(N_EASY)):
        sys.exit("FAIL: sample_id is not exactly 0..279 in order; the frozen "
                 "item order is not intact.")
    if len(set(ids)) != N_EASY:
        sys.exit("FAIL: item ids are not unique.")
    return meta, data


def build_prompts(samples):
    """Render the official one-shot prompt per item. The per-item JSON answer
    scaffold is built from each item's own `solution_shape` (header + house
    count), which is label-free by construction -- see solution_shape() in
    data_zebralogic.py."""
    prompts = []
    for s in samples:
        fake_item = {
            "puzzle": s["puzzle"],
            "solution": {
                "header": s["solution_shape"]["header"],
                "rows": [["___"] * len(s["solution_shape"]["header"])] * s["solution_shape"]["n_rows"],
            },
        }
        prompts.append(apply_lgp_grid_template(fake_item))
    return prompts


def gpu_name():
    """Best-effort GPU name for provenance; never fatal if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        names = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        return names[0] if names else None
    except Exception:
        return None


def existing_cell_is_valid(path, expect):
    """Before trusting an existing output file as 'already done' (the
    `skip existing` path), verify it is actually a complete, valid cell
    matching THIS run's configuration -- not just that a file happens to be
    present at that path. A prior version skipped on bare `os.path.exists`,
    so a truncated/corrupt file (an old pre-atomic-write crash artifact) or a
    stale file from a DIFFERENT configuration (different alpha misfiled,
    different mask, different token budget, different prompt) would be
    silently accepted as finished forever.

    `expect` is a dict of {meta_field: expected_value} to check. Returns
    (True, None) if the file is a valid, matching, complete cell; (False,
    reason) otherwise -- never raises, so the caller can log the reason and
    regenerate rather than crash on a bad pre-existing file.
    """
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, f"unreadable/corrupt JSON ({type(e).__name__}: {e})"
    if "meta" not in d or "data" not in d:
        return False, "missing 'meta' or 'data' top-level key"
    m = d["meta"]
    if m.get("accuracy_computed") is not False:
        return False, "meta.accuracy_computed is not False (unexpected schema)"
    n_expect = expect.pop("n", None)
    if n_expect is not None and len(d["data"]) != n_expect:
        return False, f"n={len(d['data'])} != expected {n_expect}"
    mismatches = [(k, m.get(k), v) for k, v in expect.items() if m.get(k) != v]
    if mismatches:
        return False, f"config mismatch: {mismatches}"
    return True, None


def hostname():
    """Best-effort machine identity for provenance; never fatal if
    unavailable. Prereg S4 requires SAME MACHINE (not same physical GPU) for
    all of one model's alpha cells -- cuda_visible_devices alone cannot
    confirm this (two different machines can both report "0"), so this is
    what the eval-time same-machine check actually keys on."""
    try:
        import socket
        return socket.gethostname()
    except Exception:
        return None


def main():
    args = parse_args()
    if args.max_new_tokens not in ALLOWED_MAX_NEW_TOKENS:
        sys.exit(f"FAIL: --max_new_tokens={args.max_new_tokens} not in the "
                 f"prereg-allowed set {ALLOWED_MAX_NEW_TOKENS} (docs/"
                 "PREREG_ZEBRALOGIC_EASY.md section 3). 2048 is default, "
                 "3072 is the ONLY allowed escalation, 4096 is out of scope.")
    meta, all_samples = load_items(args.items, args.model)

    if args.mode == "preflight":
        idx = list(PREFLIGHT_INDICES)
    elif args.mode == "canary":
        if not args.device_tag:
            sys.exit("FAIL: --mode canary requires --device_tag")
        idx = list(CANARY_INDICES)
    else:
        idx = list(range(N_EASY))

    samples = [all_samples[i] for i in idx]
    prompts = build_prompts(samples)
    prompt_hash = hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    if vc.tokenizer.padding_side != "left":
        sys.exit(f"FAIL: tokenizer.padding_side is {vc.tokenizer.padding_side!r}, "
                 "expected 'left'. This must be true from VicundaModel.__init__ "
                 "unconditionally; something has overridden it.")
    raw_mask = np.load(args.mask_path)
    mask_sha = hashlib.sha256(open(args.mask_path, "rb").read()).hexdigest()
    os.makedirs(args.out_dir, exist_ok=True)

    device_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu = gpu_name()
    host = hostname()

    configs = utils.parse_configs(args.configs)
    if args.mode == "formal":
        frozen = set(FROZEN_ALPHAS[args.model])
        bad = [a for a, _ in configs if a not in frozen]
        if bad:
            sys.exit(f"FAIL: --mode formal received alpha(s) {bad} not in the "
                     f"frozen four-point set {sorted(frozen)} for {args.model}. "
                     "This protocol does not search doses.")

    for alpha, (ls, le) in configs:
        tag = f"mdf_{alpha}".replace("-", "neg")
        out = os.path.join(args.out_dir, tag,
                           f"zebralogic_easy_{args.size}_{ls}_{le}.json")
        n_layers = len(utils.decoder_layer_range(ls, le))
        expect_fires = 0 if alpha == 0 else n_layers * len(samples)
        if os.path.exists(out):
            # Verify the existing file is a genuinely complete, matching cell
            # before trusting it as "already done" -- see
            # existing_cell_is_valid()'s docstring for why a bare
            # os.path.exists check is not enough (a truncated pre-atomic-
            # write crash artifact, or a stale file from a different
            # configuration, would otherwise be skipped forever).
            ok, reason = existing_cell_is_valid(out, {
                "n": len(samples),
                "protocol": PROTOCOL, "mode": args.mode,
                "model": args.model, "size": args.size, "alpha": alpha,
                "layer_start": ls, "layer_end": le, "L": n_layers,
                "mask_sha256": mask_sha, "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature, "top_p": args.top_p,
                "batch_size": args.batch_size, "prompt_sha256": prompt_hash,
                "steering_fires": expect_fires,
            })
            if ok:
                print(f"skip existing (validated) {out}")
                continue
            sys.exit(f"FAIL: existing output {out} is present but INVALID "
                     f"({reason}). Refusing to silently regenerate over it "
                     "or silently accept it -- delete it deliberately if it "
                     "is a known-stale/corrupt artifact, then re-run.")
        os.makedirs(os.path.dirname(out), exist_ok=True)

        diff = raw_mask * alpha
        vc.steering_fire_count(reset=True)

        gen = []
        for i in tqdm(range(0, len(prompts), args.batch_size),
                      desc=f"zebralogic-{args.mode} a={alpha}"):
            gen.extend(vc.regenerate(
                prompts[i: i + args.batch_size],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                diff_matrices=list(diff),
                batch_size=args.batch_size,
                return_metadata=True,
            ))

        # HARD LENGTH CHECK (matches get_answer_proofwriter_owa.py's review
        # finding #1 fix): zip(samples, gen) below silently truncates to the
        # shorter sequence if a batch call ever returned fewer rows than it
        # was given (a partial-batch bug, an OOM-recovery path returning a
        # short list, etc.) -- the cell would then write fewer rows than
        # samples with NO error, and the steering_fires check below only
        # catches this by accident (a short `gen` still yields the SAME
        # `len(samples)` used in `expect`, since `expect` is computed from
        # `samples`, not from `gen`). Assert 1:1 length before doing anything
        # else with `gen`.
        if len(gen) != len(samples):
            sys.exit(f"FAIL: generation returned {len(gen)} rows for "
                     f"{len(samples)} prompts at alpha={alpha}; zip() would "
                     f"silently drop {abs(len(gen) - len(samples))} sample(s). "
                     "This must never happen under vc.regenerate's documented "
                     "contract -- treat as a hard failure, not a partial result.")

        fires = vc.steering_fire_count()
        if fires != expect_fires:
            sys.exit(f"FAIL: steering_fires {fires} != {expect_fires} "
                     f"(L={n_layers}, n={len(samples)}, alpha={alpha}); "
                     "the intervention is unverified, so the cell is not usable.")

        rows = []
        for s, g in zip(samples, gen):
            # return_metadata=True always returns a dict here (regenerate
            # raises if requested without prefill_only, which we never do).
            text = g["text"]
            n_tok = g["generated_token_count"]
            stop_reason = g["stop_reason"]
            rows.append({
                "id": s["id"], "sample_id": s["sample_id"], "size": s["size"],
                "puzzle": s["puzzle"], "solution_shape": s["solution_shape"],
                "generated": text, "raw_text": g["raw_text"],
                "generated_token_count": n_tok, "stop_reason": stop_reason,
                # authoritative truncation signal from llms.py, not a token-
                # count heuristic: "budget_exhausted" means no terminator was
                # ever produced within max_new_tokens.
                "truncated": stop_reason == "budget_exhausted",
            })

        # Write to a .tmp path and atomically rename into place only after a
        # full, valid dump. Writing the final path directly means a crash or
        # kill mid-write leaves a half-written file at `out`; the existence
        # check above would then treat it as a finished cell and `skip` it
        # forever on retry, silently losing that cell from the sweep.
        payload = {"meta": {
            "protocol": PROTOCOL, "mode": args.mode,
            "model": args.model, "size": args.size, "alpha": alpha,
            "layer_start": ls, "layer_end": le, "L": n_layers,
            "mask_path": args.mask_path, "mask_sha256": mask_sha,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature, "top_p": args.top_p,
            "batch_size": args.batch_size, "prefill_tail_len": 1,
            "chat_template": False, "role": "neutral",
            "cot": "one_shot_worked_example_in_prompt",
            "steering_fires": fires,
            "prompt_sha256": prompt_hash,
            "item_ids_sha256": hashlib.sha256(
                "\n".join(s["id"] for s in samples).encode("utf-8")).hexdigest(),
            "cuda_visible_devices": device_env, "gpu_name": gpu, "hostname": host,
            "device_tag": args.device_tag,
            "n": len(rows), "n_indices": idx,
            "source_easy_ids_sha256": meta.get("easy_ids_sha256"),
            "source_revision": meta.get("revision"),
            "contains_labels": False, "accuracy_computed": False,
        }, "data": rows}
        tmp_out = out + ".tmp"
        with open(tmp_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_out, out)
        print(f"  wrote {out}  steering_fires={fires}  n={len(rows)}")

    print(f"\nGeneration complete (mode={args.mode}). NO accuracy was computed.")
    print("Next: python zebralogic/eval_zebralogic.py  (the only script that "
          "reads private gold)")


if __name__ == "__main__":
    main()
