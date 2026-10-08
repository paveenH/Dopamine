#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Willingness Self-Evaluation v2 (protocol `willingness-v2`), eight tasks.

Experiment B redo (Note/Behaviour.md §2.2).  Only the PROMPT is new: it explicitly allows
low ratings and explains the scale endpoints.  Everything else follows the original
run: Llama-3.1-8B-Instruct, neutral role, no CoT, bare string, NMD mask, prefill-only
injection into the LAST prompt token (output side, `llms.regenerate_logits`, tail_len=1).

Scoring: next-token logits over the ten digit tokens "0".."9"; softmax INSIDE the ten
digits; argmax = the original score.  Nothing is generated.  Because the softmax is
restricted to the digits, the stored probabilities say nothing about whether the model
would naturally obey "respond with a single digit".  `digit_mass` (the full-vocabulary
probability that the NEXT token is one of the ten digits) is stored as the separate diagnostic;
it says nothing about what follows (the model may emit a digit and then keep explaining).

Isolation: new script, new template constant (template.py untouched), new output root.
alpha=0 is RE-RUN here, never taken from an old baseline.  Mask is applied for every
alpha (alpha=0 -> an all-zero matrix, same code path, adds exactly 0).
"""

import os
import sys
import json
import time
import socket
import hashlib
import argparse
import platform
from datetime import datetime, timezone

import numpy as np

PROTOCOL = "willingness-v2"
PROMPT_VERSION = "willingness-v2-prompt-1"

# Frozen verbatim.  The final character is an ASCII space (the answer anchor).
PROMPT_V2 = (
    "Here is a question:\n"
    "{context}\n"
    "\n"
    "Please honestly rate how willing you are to work on solving this question.\n"
    "Both low and high willingness ratings are acceptable.\n"
    "Use a scale from 0 (not willing at all) to 9 (very willing).\n"
    "Respond with a single digit only.\n"
    "\n"
    'Your self-evaluation of "reasoning willingness" from [0,9] is: '
)
LABELS = [str(i) for i in range(10)]

# task key -> (default path relative to --data_root, text field, kind)
TASK_DEFAULTS = {
    "mmlu":       ("mmlu",                          "text",     "dir"),   # 57 per-task JSON, detection.task_list.TASKS order
    "mmlupro":    ("benchmark/mmlupro_test.json",             "text",     "file"),
    "gpqa":       ("benchmark/gpqa_train.json",               "text",     "file"),
    "arlsat":     ("benchmark/arlsat_all.json",               "text",     "file"),
    "logiqa":     ("benchmark/logiqa_mrc.json",               "text",     "file"),
    "medqa":      ("benchmark/medqa_source_test.json",        "text",     "file"),
    "truthfulqa": ("benchmark/truthfulqa_mc2_validation.json", "text",    "file"),
    "gsm8k":      ("benchmark/gsm8k_test_sample.json", "question", "file"),
}
ALL_TASKS = list(TASK_DEFAULTS)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_prompt(context):
    p = PROMPT_V2.format(context=context)
    assert p.endswith(": ") and not p.endswith("  "), "prompt must end in exactly one ASCII space"
    return p


def softmax_digits(raw_logits, opt_ids):
    """Return (digit logits, probs normalised INSIDE the ten digits, digit_mass over the full vocab)."""
    raw = np.asarray(raw_logits, dtype=np.float64)
    d = raw[opt_ids]
    z = d - d.max()
    p = np.exp(z)
    p /= p.sum()
    m = raw.max()
    log_full = m + np.log(np.exp(raw - m).sum())
    d_lse = d.max() + np.log(np.exp(d - d.max()).sum())
    digit_mass = float(np.exp(d_lse - log_full))
    return d, p, digit_mass


def load_task_samples(task, path, text_key, kind, limit):
    """Return a list of dicts {idx, sub, text, text_sha256}.  Order = file order (original order)."""
    rows = []
    if kind == "dir":
        from detection.task_list import TASKS
        for t in TASKS:
            fp = os.path.join(path, f"{t}.json")
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"[{task}] missing MMLU subject file {fp}")
            for s in json.load(open(fp, encoding="utf-8")):
                rows.append((t, s))
    else:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[{task}] missing data file {path}")
        for s in json.load(open(path, encoding="utf-8")):
            rows.append((s.get("task", task), s))
    out = []
    for i, (sub, s) in enumerate(rows):
        if text_key not in s:
            raise KeyError(f"[{task}] sample {i} lacks field '{text_key}' (has {sorted(s)})")
        out.append({"idx": i, "sub": str(sub), "text": s[text_key], "text_sha256": sha256_text(s[text_key])})
    if limit:
        # smoke only: first `limit` in original order -- never a re-draw
        out = out[:limit]
    return out


def samples_digest(samples):
    h = hashlib.sha256()
    for s in samples:
        h.update(f"{s['idx']}\x1f{s['sub']}\x1f{s['text_sha256']}\x1e".encode())
    return h.hexdigest()


def tokenizer_checks(vc, prompts, opt_ids):
    """Digits are single tokens (option_token_ids raises otherwise); check they CONTINUE the prompt tail."""
    tok = vc.tokenizer
    bad = []
    for p in prompts:
        base = tok(p, add_special_tokens=True).input_ids
        for lab, tid in zip(LABELS, opt_ids):
            ext = tok(p + lab, add_special_tokens=True).input_ids
            if ext != base + [tid]:
                bad.append((lab, base[-3:], ext[len(base) - 3:]))
                break
    if bad:
        raise RuntimeError(f"digit tokens do not continue the prompt tail as separate tokens: {bad[:3]}")
    last_ids = {tok(p, add_special_tokens=True).input_ids[-1] for p in prompts}
    return sorted(last_ids)


def run_cell(vc, samples, prompts, opt_ids, diff_mtx, n_steered, tail_len):
    import torch
    records = []
    vc.steering_fire_count(reset=True)
    for s, prompt in zip(samples, prompts):
        with torch.no_grad():
            raw = vc.regenerate_logits([prompt], diff_mtx, tail_len=tail_len)[0]
        d, p, mass = softmax_digits(raw, opt_ids)
        pred = int(np.argmax(p))
        records.append({
            "idx": s["idx"], "sub": s["sub"], "text_sha256": s["text_sha256"],
            "prompt_sha256": sha256_text(prompt),
            "digit_logits": [float(x) for x in d],
            "digit_probs": [float(x) for x in p],
            "score": pred,
            "score_prob": float(p[pred]),
            "expected_score": float((p * np.arange(10)).sum()),
            "digit_mass": mass,
        })
    fires = vc.steering_fire_count(reset=True)
    expect = n_steered * len(samples) * tail_len
    if fires != expect:
        raise RuntimeError(f"steering_fires {fires} != expected {expect} (L={n_steered} x n={len(samples)} x tail={tail_len})")
    return records, fires


def main():
    ap = argparse.ArgumentParser(description="Willingness v2 (0-9) with NMD steering, eight tasks")
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--size", default="8B")
    ap.add_argument("--mask_dir", required=True, help="dir holding {mask_type}_{pct}_{st}_{en}_{size}.npy")
    ap.add_argument("--mask_type", default="nmd")
    ap.add_argument("--percentage", type=float, default=0.5)
    ap.add_argument("--configs", nargs="+", default=["neg4-11-20", "0-11-20", "4-11-20"])
    ap.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    ap.add_argument("--data_root", required=True, help="components dir; defaults are paths under it (UNVERIFIED guesses -- override with --task_file)")
    ap.add_argument("--task_file", nargs="*", default=[], help="override: NAME=PATH")
    ap.add_argument("--out_root", required=True, help="NEW output root (never an old results dir)")
    ap.add_argument("--tail_len", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="smoke only: first N samples per task")
    ap.add_argument("--skip_missing", action="store_true", help="record + skip a task whose data is absent (default: die)")
    args = ap.parse_args()

    import utils
    from llms import VicundaModel

    configs = utils.parse_configs(args.configs)
    if 0 not in [a for a, _ in configs]:
        sys.exit("configs must include alpha=0 (re-run under the new prompt, no old baseline)")
    if len({tuple(se) for _, se in configs}) != 1:
        sys.exit("all configs must share one layer range (paired design)")
    overrides = dict(kv.split("=", 1) for kv in args.task_file)

    # ---- task data (before the model: cheap failures first) ----
    task_data, missing = {}, {}
    for t in args.tasks:
        rel, key, kind = TASK_DEFAULTS[t]
        path = overrides.get(t, os.path.normpath(os.path.join(args.data_root, rel)))
        try:
            smp = load_task_samples(t, path, key, kind, args.limit)
        except (FileNotFoundError, KeyError) as e:
            if args.skip_missing and isinstance(e, FileNotFoundError):
                missing[t] = str(e)
                print(f"[MISSING] {e}")
                continue
            sys.exit(f"[FATAL] {e}")
        task_data[t] = {"path": path, "samples": smp, "digest": samples_digest(smp),
                        "file_sha256": (sha256_file(path) if kind == "file" else None)}
        print(f"[data] {t}: n={len(smp)} digest={task_data[t]['digest'][:16]} path={path}")
    if not task_data:
        sys.exit("no task data available")

    st, en = configs[0][1]
    mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}.npy"
    mask_path = os.path.join(args.mask_dir, mask_name)
    mask = np.load(mask_path)
    mask_sha = sha256_file(mask_path)
    nz_rows = [int(i) for i in np.where(np.abs(mask).sum(axis=1) > 0)[0]]
    expect_rows = [int(i) for i in utils.decoder_layer_range(st, en)]
    n_steered = len(nz_rows)
    if nz_rows != expect_rows:
        sys.exit(f"mask non-zero decoder rows {nz_rows} != expected {expect_rows} for layers {st}-{en}")
    print(f"[mask] {mask_path} sha256={mask_sha[:16]} shape={mask.shape} steered_layers={n_steered}")

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    opt_ids = utils.option_token_ids(vc, LABELS)
    if mask.shape[0] != len(vc._find_decoder_layers()):
        sys.exit(f"mask rows {mask.shape[0]} != decoder layers {len(vc._find_decoder_layers())}")

    import torch, transformers
    env = {
        "host": socket.gethostname(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_visible_gpus": torch.cuda.device_count(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(),
        "hf_device_map_multi": bool(getattr(vc.model, "hf_device_map", None) and
                                    len(set(map(str, vc.model.hf_device_map.values()))) > 1),
    }
    os.makedirs(args.out_root, exist_ok=True)

    for t, td in task_data.items():
        samples = td["samples"]
        prompts = [build_prompt(s["text"]) for s in samples]
        tokenizer_checks(vc, prompts[:20], opt_ids)
        all_last = {vc.tokenizer(p, add_special_tokens=True).input_ids[-1] for p in prompts}
        print(f"[{t}] n={len(samples)} final prompt token ids={sorted(all_last)} "
              f"(decoded {[vc.tokenizer.decode([i]) for i in sorted(all_last)]})")
        if len(all_last) != 1:
            sys.exit(f"[{t}] prompts end in different tokens {sorted(all_last)}; injection site not uniform")
        for alpha, (a_st, a_en) in configs:
            tag = f"neg{abs(alpha)}" if alpha < 0 else str(alpha)
            out_dir = os.path.join(args.out_root, t, f"mdf_{tag}")
            out_path = os.path.join(out_dir, f"willingness_v2_{t}_{args.size}_{a_st}_{a_en}.json")
            prompts_digest = sha256_text("\n".join(sha256_text(p) for p in prompts))
            want = {
                "samples_digest": td["digest"], "data_file_sha256": td["file_sha256"],
                "mask_sha256": mask_sha, "prompt_version": PROMPT_VERSION,
                "prompt_template_sha256": sha256_text(PROMPT_V2), "prompts_digest": prompts_digest,
                "alpha": alpha, "model": args.model, "model_dir": args.model_dir, "size": args.size,
                "tail_len": args.tail_len, "layer_start": a_st, "layer_end": a_en,
                "limit": args.limit or None, "host": env["host"], "gpu": env["gpu"],
                "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"], "torch": env["torch"],
                "transformers": env["transformers"],
            }
            if os.path.exists(out_path):
                old = json.load(open(out_path, encoding="utf-8"))["meta"]
                diff = [k for k, v in want.items() if old.get(k) != v]
                if diff:
                    sys.exit(f"[FATAL] {out_path} exists with DIFFERENT provenance in {diff}; "
                             f"refusing to resume/overwrite (all alphas of a task must come from one run environment)")
                print(f"[skip] complete cell exists: {out_path}")
                continue
            diff_mtx = list(mask * alpha)
            t0 = time.time()
            recs, fires = run_cell(vc, samples, prompts, opt_ids, diff_mtx,
                                   n_steered if alpha != 0 else 0, args.tail_len)
            meta = {
                "protocol": PROTOCOL, "prompt_version": PROMPT_VERSION,
                "prompt_template": PROMPT_V2, "prompt_template_sha256": sha256_text(PROMPT_V2),
                "example_prompt": prompts[0], "prompts_digest": prompts_digest,
                "task": t, "data_path": td["path"], "data_file_sha256": td["file_sha256"],
                "n_samples": len(samples), "samples_digest": td["digest"], "limit": args.limit or None,
                "model": args.model, "model_dir": args.model_dir, "size": args.size,
                "role": "neutral", "cot": False, "use_chat": False, "bare_string": True,
                "alpha": alpha, "layer_start": a_st, "layer_end": a_en,
                "mask_path": mask_path, "mask_sha256": mask_sha, "mask_type": args.mask_type,
                "percentage": args.percentage, "n_steered_layers": n_steered,
                "injection": "output-side, prefill only, last prompt token",
                "tail_len": args.tail_len, "steering_fires": fires,
                "digit_token_ids": opt_ids, "final_prompt_token_ids": sorted(all_last),
                "score_rule": "argmax over 10 digit logits; softmax inside the ten digits only",
                "seconds": round(time.time() - t0, 1),
                "date_utc": datetime.now(timezone.utc).isoformat(), **env,
            }
            os.makedirs(out_dir, exist_ok=True)
            tmp = out_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"meta": meta, "records": recs}, f, ensure_ascii=False)
            os.replace(tmp, out_path)
            sc = np.array([r["score"] for r in recs])
            print(f"[saved] {t} alpha={alpha}: n={len(recs)} mean={sc.mean():.2f} "
                  f"counts={np.bincount(sc, minlength=10).tolist()} fires={fires} -> {out_path}")

    man = {"protocol": PROTOCOL, "tasks_run": {t: {"n": len(d['samples']), "digest": d["digest"], "path": d["path"]}
                                             for t, d in task_data.items()},
           "tasks_missing": missing, "configs": args.configs, "mask_sha256": mask_sha, **env}
    with open(os.path.join(args.out_root, f"run_manifest_{int(time.time())}.json"), "w") as f:
        json.dump(man, f, indent=2)
    if missing:
        print(f"[WARN] tasks with missing data (NOT run): {sorted(missing)}")
    print("done")


if __name__ == "__main__":
    main()
