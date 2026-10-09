#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Willingness Self-Evaluation v3 (protocol `willingness-v3`), eight tasks, nine doses.

Differences from v2 (get_action_willingness_v2.py, left untouched and imported for its verified helpers):
  * NEW prompt `PROMPT_V3` (frozen verbatim; en dash in "0–9"; ends in ONE ASCII space).
  * Per task a FIXED random subset of 300 items: seed=42, without replacement, drawn from the
    pooled item list in its original order (MMLU = all 57 subjects pooled), with an INDEPENDENT
    RNG per task so grouping / launch order cannot change the draw.  <300 items => hard error.
  * Doses -8,-6,-4,-2,0,+2,+4,+6,+8 on the same 300 items in the same order; alpha=0 re-run.
  * Adds the natural-log entropy of the digit-internal distribution.
  * Results go to {out_root}/{task}/mdf_{alpha}/willingness_v3_{task}_{size}_{st}_{en}.json
    (out_root = components/llama3 for the formal run; NO extra willingness_v3/ level).
  * The pooled item list is cross-checked against the digests recorded in the v2 result
    metadata (same files, same split, same option order).

Scoring is unchanged: next-token logits over "0".."9", softmax INSIDE the ten digits, argmax =
score.  Nothing is generated.  `digit_mass` is only P(next token is a digit).
"""

import os
import sys
import json
import time
import math
import random
import socket
import argparse
import platform
from datetime import datetime, timezone

import numpy as np

import get_action_willingness_v2 as v2  # constants + verified helpers only; v2 itself is not modified

PROTOCOL = "willingness-v3"
PROMPT_VERSION = "willingness-v3-prompt-1"

PROMPT_V3 = (
    "Here is a question:\n"
    "{context}\n"
    "\n"
    "Honestly rate your willingness to work on solving this question.\n"
    "There is no preferred rating.\n"
    "Use 0 (not willing at all) to 9 (very willing).\n"
    "Respond with a single digit only.\n"
    "\n"
    "Your willingness rating (0–9) is: "
)
LABELS = v2.LABELS
TASK_DEFAULTS = v2.TASK_DEFAULTS
ALL_TASKS = v2.ALL_TASKS
DEFAULT_CONFIGS = ["neg8-11-20", "neg6-11-20", "neg4-11-20", "neg2-11-20", "0-11-20",
                   "2-11-20", "4-11-20", "6-11-20", "8-11-20"]
SEED = 42
N_PER_TASK = 300

# Pool digests / file hashes recorded in the v2 result metadata (all 24 v2 cells, host Turing,
# limit=None).  The v3 pool MUST reproduce them: same files, same split, same option order.
V2_REFERENCE = {
    "mmlu":       {"n": 14042, "samples_digest": "e119b977ea5c0507ce3ac809cd17da7bb519af3b4cba59a63e1d1a18a90172bc", "data_file_sha256": None},
    "mmlupro":    {"n": 12032, "samples_digest": "3ecc3b1cbb26dfa4aeaded49133361e68069e9fccefa48cb8c71b865f2741a80", "data_file_sha256": "1be0fafa2f42a9154fb88ea75833ac158124d5f1e5d45c139b074f953a016a38"},
    "gpqa":       {"n": 1192,  "samples_digest": "080e359e1771aaef1f173dfcbe2237dabee5e3012c9800ce783680faad00d71d", "data_file_sha256": "f903ccf645650ba439ac460f1e3ba93570c91b9aa39094a0c8d1e778f614971d"},
    "arlsat":     {"n": 2091,  "samples_digest": "c87a670019cfeda4cf9858993736623f9ee7ccceeb39cbe7c9988efcc9bfda05", "data_file_sha256": "d4b6a8676aaa2ef094cbd5c4747d827600ee7cc306943ba3fe65ab3db9076f3a"},
    "logiqa":     {"n": 1572,  "samples_digest": "19d614dca2a9ffa2f103ef755387f9748bff3b452fb9862d493c8b7455dc2783", "data_file_sha256": "b37bb78ec70b6d99db90f677b6e8ef5182ed0ddd6db8f4807f1e3ecd4fffeac5"},
    "medqa":      {"n": 1273,  "samples_digest": "8a32bbe32f31182e2ff3d7d832d509cb8673ab81da8adebde3a3b5855fe0aa78", "data_file_sha256": "6b6e46a3d78d3b8b9aa51fa3e703de3bde3589df866bdc3e1e005bd9c17b3050"},
    "truthfulqa": {"n": 817,   "samples_digest": "3e639b1aa492e519e16e96eebcec540b54c52d3e8d7140a7098ac87a0378197e", "data_file_sha256": "5c3e5dc2710adcf46b35b1e4e01242771f36a1b371a73a3fd4e9c8c7a489ca5f"},
    "gsm8k":      {"n": 300,   "samples_digest": "bdde0601da6786c9d4a4da43b7c5afb63f01a74a10314178291227add395310e", "data_file_sha256": "6d9efbccb03ec73f9efb2a8e8244c652a771509872e15c154fd524d684052c98"},
}
V2_MASK_SHA256 = "8bd72a0dbad82066b31fba40e3a516168542bc08149579c278695123e2a876c8"


def build_prompt(context):
    p = PROMPT_V3.format(context=context)
    assert p.endswith(": ") and not p.endswith("  "), "prompt must end in exactly one ASCII space"
    assert "(0–9)" in p and "(0-9)" not in p, "last line must use the en dash"
    return p


def entropy_nats(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def select_items(task, pool, seed=SEED, n=N_PER_TASK):
    """Independent RNG per task; sorted ascending so every dose sees the same fixed order."""
    if len(pool) < n:
        sys.exit(f"[FATAL] {task}: pool has {len(pool)} items < {n}; refusing to change the sample size")
    rng = random.Random(seed)
    pick = sorted(rng.sample(range(len(pool)), n))
    out = []
    for i in pick:
        s = dict(pool[i])
        s["pool_idx"] = pool[i]["idx"]
        s["sample_id"] = f"{task}:{pool[i]['idx']}"
        out.append(s)
    return out


def sample_manifest(task, path, pool, sel, file_sha, seed, n):
    return {
        "protocol": PROTOCOL, "task": task, "seed": seed, "n_selected": n,
        "rng": "random.Random(seed).sample(range(pool_n), n), per-task RNG, then sorted ascending",
        "source_path": path, "source_file_sha256": file_sha,
        "pool_n": len(pool), "pool_digest": v2.samples_digest(pool),
        "selection_digest": v2.samples_digest(sel),
        "selected": [{"sample_id": s["sample_id"], "pool_idx": s["pool_idx"], "sub": s["sub"],
                      "text_sha256": s["text_sha256"]} for s in sel],
    }


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def run_cell(vc, samples, prompts, opt_ids, diff_mtx, n_steered, tail_len):
    import torch
    records = []
    vc.steering_fire_count(reset=True)
    for s, prompt in zip(samples, prompts):
        with torch.no_grad():
            raw = vc.regenerate_logits([prompt], diff_mtx, tail_len=tail_len)[0]
        d, p, mass = v2.softmax_digits(raw, opt_ids)
        pred = int(np.argmax(p))
        records.append({
            "sample_id": s["sample_id"], "idx": s["pool_idx"], "sub": s["sub"],
            "text_sha256": s["text_sha256"], "prompt_sha256": v2.sha256_text(prompt),
            "digit_logits": [float(x) for x in d], "digit_probs": [float(x) for x in p],
            "score": pred, "score_prob": float(p[pred]),
            "expected_score": float((p * np.arange(10)).sum()),
            "entropy_nats": entropy_nats(p), "digit_mass": mass,
        })
    fires = vc.steering_fire_count(reset=True)
    expect = n_steered * len(samples) * tail_len
    if fires != expect:
        raise RuntimeError(f"steering_fires {fires} != expected {expect} "
                           f"(L={n_steered} x n={len(samples)} x tail={tail_len})")
    return records, fires


def main():
    ap = argparse.ArgumentParser(description="Willingness v3 (0-9) with NMD steering, eight tasks x nine doses")
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--size", default="8B")
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--mask_type", default="nmd")
    ap.add_argument("--percentage", type=float, default=0.5)
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--task_file", nargs="*", default=[], help="override: NAME=PATH")
    ap.add_argument("--out_root", required=True, help="formal: components/llama3 ; smoke: a /tmp dir")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=N_PER_TASK)
    ap.add_argument("--tail_len", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="smoke only: first N of the FIXED selected items")
    ap.add_argument("--v2_check", choices=["strict", "off"], default="strict",
                    help="'off' exists for fake-data unit tests only")
    args = ap.parse_args()

    import utils
    from llms import VicundaModel

    configs = utils.parse_configs(args.configs)
    doses = [a for a, _ in configs]
    if 0 not in doses or len(set(doses)) != len(doses):
        sys.exit("configs must include alpha=0 exactly once and no duplicate doses")
    if len({tuple(se) for _, se in configs}) != 1:
        sys.exit("all configs must share one layer range (paired design)")
    overrides = dict(kv.split("=", 1) for kv in args.task_file)

    # ---- data, sampling, v2 reconciliation (cheap failures before the model) ----
    task_data = {}
    for t in args.tasks:
        rel, key, kind = TASK_DEFAULTS[t]
        path = overrides.get(t, os.path.normpath(os.path.join(args.data_root, rel)))
        try:
            pool = v2.load_task_samples(t, path, key, kind, 0)
        except (FileNotFoundError, KeyError) as e:
            sys.exit(f"[FATAL] {e}")
        file_sha = v2.sha256_file(path) if kind == "file" else None
        if args.v2_check == "strict":
            ref = V2_REFERENCE[t]
            got = {"n": len(pool), "samples_digest": v2.samples_digest(pool), "data_file_sha256": file_sha}
            bad = [k for k in ref if ref[k] != got[k]]
            if bad:
                sys.exit(f"[FATAL] {t}: pool differs from the v2 run in {bad} "
                         f"(v2 {[ref[k] for k in bad]} vs now {[got[k] for k in bad]}); refusing to guess a data version")
        sel = select_items(t, pool, args.seed, args.n)
        man = sample_manifest(t, path, pool, sel, file_sha, args.seed, args.n)
        man_path = os.path.join(args.out_root, t, f"willingness_v3_sample_{t}.json")
        if os.path.exists(man_path):
            old = json.load(open(man_path, encoding="utf-8"))
            for k in ("seed", "n_selected", "pool_digest", "selection_digest", "source_file_sha256"):
                if old.get(k) != man[k]:
                    sys.exit(f"[FATAL] {man_path} exists with different {k}; refusing to overwrite the sampling manifest")
        else:
            write_json_atomic(man_path, man)
        used = sel[:args.limit] if args.limit else sel
        task_data[t] = {"path": path, "file_sha": file_sha, "selected": used, "man": man}
        print(f"[data] {t}: pool={len(pool)} pool_digest={man['pool_digest'][:16]} selected={len(sel)} "
              f"selection_digest={man['selection_digest'][:16]} using={len(used)} path={path}")

    st, en = configs[0][1]
    mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}.npy"
    mask_path = os.path.join(args.mask_dir, mask_name)
    mask = np.load(mask_path)
    mask_sha = v2.sha256_file(mask_path)
    if args.v2_check == "strict" and mask_sha != V2_MASK_SHA256:
        sys.exit(f"[FATAL] mask sha256 {mask_sha[:16]} != the v2 mask {V2_MASK_SHA256[:16]}")
    nz_rows = [int(i) for i in np.where(np.abs(mask).sum(axis=1) > 0)[0]]
    expect_rows = [int(i) for i in utils.decoder_layer_range(st, en)]
    n_steered = len(nz_rows)
    if nz_rows != expect_rows:
        sys.exit(f"mask non-zero decoder rows {nz_rows} != expected {expect_rows} for layers {st}-{en}")
    print(f"[mask] {mask_path} sha256={mask_sha[:16]} steered_layers={n_steered}")

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    opt_ids = utils.option_token_ids(vc, LABELS)
    if mask.shape[0] != len(vc._find_decoder_layers()):
        sys.exit(f"mask rows {mask.shape[0]} != decoder layers {len(vc._find_decoder_layers())}")

    import torch, transformers
    env = {
        "host": socket.gethostname(), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_visible_gpus": torch.cuda.device_count(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(),
        "hf_device_map_multi": bool(getattr(vc.model, "hf_device_map", None) and
                                    len(set(map(str, vc.model.hf_device_map.values()))) > 1),
    }
    if env["hf_device_map_multi"]:
        sys.exit("model is sharded over several devices; v3 expects one card per worker")

    done_cells, t_start = [], time.time()
    for t, td in task_data.items():
        samples = td["selected"]
        prompts = [build_prompt(s["text"]) for s in samples]
        # real-tokenizer checks over ALL prompts of the task (not a prefix sample)
        v2.tokenizer_checks(vc, prompts, opt_ids)
        last_ids = sorted({vc.tokenizer(p, add_special_tokens=True).input_ids[-1] for p in prompts})
        print(f"[{t}] n={len(samples)} final prompt token ids={last_ids} "
              f"(decoded {[vc.tokenizer.decode([i]) for i in last_ids]}); digit ids={opt_ids}")
        if len(last_ids) != 1:
            sys.exit(f"[{t}] prompts end in different tokens {last_ids}; injection site not uniform")
        prompts_digest = v2.sha256_text("\n".join(v2.sha256_text(p) for p in prompts))
        for alpha, (a_st, a_en) in configs:
            tag = f"neg{abs(alpha)}" if alpha < 0 else str(alpha)
            out_path = os.path.join(args.out_root, t, f"mdf_{tag}",
                                    f"willingness_v3_{t}_{args.size}_{a_st}_{a_en}.json")
            want = {
                "protocol": PROTOCOL, "prompt_version": PROMPT_VERSION,
                "prompt_template_sha256": v2.sha256_text(PROMPT_V3), "prompts_digest": prompts_digest,
                "seed": args.seed, "n_samples": len(samples), "sample_limit": args.limit or None,
                "selection_digest": td["man"]["selection_digest"], "pool_digest": td["man"]["pool_digest"],
                "data_file_sha256": td["file_sha"], "mask_sha256": mask_sha, "alpha": alpha,
                "model": args.model, "model_dir": args.model_dir, "size": args.size,
                "tail_len": args.tail_len, "layer_start": a_st, "layer_end": a_en,
                "host": env["host"], "gpu": env["gpu"], "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
                "torch": env["torch"], "transformers": env["transformers"],
            }
            if os.path.exists(out_path):
                old = json.load(open(out_path, encoding="utf-8"))["meta"]
                diff = [k for k, v in want.items() if old.get(k) != v]
                if diff:
                    sys.exit(f"[FATAL] {out_path} exists with DIFFERENT provenance in {diff}; "
                             f"refusing to resume/overwrite")
                print(f"[skip] complete cell exists: {out_path}")
                done_cells.append(out_path)
                continue
            t0 = time.time()
            recs, fires = run_cell(vc, samples, prompts, opt_ids, list(mask * alpha),
                                   n_steered if alpha != 0 else 0, args.tail_len)
            meta = {
                **want,
                "prompt_template": PROMPT_V3, "example_prompt": prompts[0],
                "task": t, "data_path": td["path"],
                "role": "neutral", "cot": False, "use_chat": False, "bare_string": True,
                "mask_path": mask_path, "mask_type": args.mask_type, "percentage": args.percentage,
                "n_steered_layers": n_steered, "steered_decoder_rows": nz_rows,
                "injection": "output-side, prefill only, last prompt token",
                "steering_fires": fires, "expected_steering_fires": n_steered * len(samples) * args.tail_len if alpha != 0 else 0,
                "digit_token_ids": opt_ids, "final_prompt_token_ids": last_ids,
                "score_rule": "argmax over 10 digit logits; softmax inside the ten digits only; entropy = natural log",
                "v2_reference_checked": args.v2_check == "strict",
                "seconds": round(time.time() - t0, 1),
                "date_utc": datetime.now(timezone.utc).isoformat(), **env,
            }
            write_json_atomic(out_path, {"meta": meta, "records": recs})
            sc = np.array([r["score"] for r in recs])
            ex = np.mean([r["expected_score"] for r in recs])
            print(f"[saved] {t} alpha={alpha:+d}: n={len(recs)} argmax_mean={sc.mean():.2f} E={ex:.2f} "
                  f"counts={np.bincount(sc, minlength=10).tolist()} fires={fires} -> {out_path}")
            done_cells.append(out_path)

    man = {"protocol": PROTOCOL, "tasks": {t: {"n": len(d["selected"]), "selection_digest": d["man"]["selection_digest"],
                                               "pool_digest": d["man"]["pool_digest"], "path": d["path"]}
                                          for t, d in task_data.items()},
           "doses": doses, "seed": args.seed, "mask_sha256": mask_sha, "cells_done": len(done_cells),
           "cells_expected": len(task_data) * len(doses), "seconds": round(time.time() - t_start, 1),
           "pid": os.getpid(), **env}
    write_json_atomic(os.path.join(args.out_root, f"willingness_v3_run_manifest_{int(time.time())}_{os.getpid()}.json"), man)
    print(f"done: {len(done_cells)}/{len(task_data) * len(doses)} cells")


if __name__ == "__main__":
    main()
