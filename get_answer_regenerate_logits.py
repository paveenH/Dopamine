#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for VicundaModel with neuron editing -> logits-based answer selection.
Loads the model once, applies diff matrices to hidden states, and for each prompt
directly reads out the last‐token logits to pick A/B/C/D/E.
"""

import os
import gc
import json
import numpy as np
import torch
from tqdm import tqdm
import argparse
import hashlib
import platform
import sys

from llms import VicundaModel
from detection.task_list import TASKS
from template import select_templates
import utils

# ───────────────────── Helper Functions ─────────────────────────


def run_task(
    vc: VicundaModel,
    task: str,
    diff_mtx: np.ndarray,
):
    """Run one task with a fixed diff_mtx, returning updated data + accuracy."""

    # template
    templates = select_templates(args.suite, args.use_E)  # choose template style
    LABELS = templates["labels"]
    opt_ids = utils.option_token_ids(vc, LABELS)

    # load data
    data_path = os.path.join(MMLU_DIR, f"{task}.json")
    data = utils.load_json(data_path)
    if args.smoke_n:
        data = data[: args.smoke_n]
    prompt_hash = hashlib.sha256()

    # Parse custom roles if provided
    custom_roles = None
    if args.roles:
        custom_roles = [r.strip() for r in args.roles.split(",")]
    roles = utils.make_characters(task, custom_roles)

    tmp_record = utils.record_template(roles, templates)

    # stats accumulator
    stats = {r: {"correct": 0, "E_count": 0, "invalid": 0, "total": 0} for r in roles}

    for sample in tqdm(data, desc=task):
        ctx = sample.get("text", "")
        true_idx = sample.get("label", -1)
        true_lab = LABELS[true_idx]

        for role in roles:
            prompt = utils.construct_prompt(vc, templates, ctx, role, args.use_chat)
            prompt_hash.update(prompt.encode("utf-8"))

            # get raw logits after hooking in diff
            raw_logits = vc.regenerate_logits([prompt], diff_mtx, tail_len=args.tail_len)[0]
            # pick among options A–E
            opt_logits = np.array([raw_logits[i] for i in opt_ids])

            exp = np.exp(opt_logits - opt_logits.max())
            soft = exp / exp.sum()

            pred_idx = int(opt_logits.argmax())
            pred_lab = LABELS[pred_idx]
            pred_prb = float(soft[pred_idx])

            role_key = role.replace(" ", "_")
            sample[f"answer_{role_key}"] = pred_lab
            sample[f"prob_{role_key}"] = pred_prb

            # Add whole prediction
            sample[f"softmax_{role_key}"] = [float(p) for p in soft]
            sample[f"logits_{role_key}"] = [float(l) for l in opt_logits]

            # update stats
            st = stats[role]
            st["total"] += 1
            if pred_lab == true_lab:
                st["correct"] += 1
            elif pred_lab == "E":
                st["E_count"] += 1
            else:
                st["invalid"] += 1

    # accuracy summary
    accuracy = {}
    for role, s in stats.items():
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        accuracy[role] = {**s, "accuracy_percentage": round(pct, 2)}
        print(f"{role:<25} acc={pct:5.2f}%  (correct {s['correct']}/{s['total']}), E={s['E_count']}")

    return data, accuracy, tmp_record, prompt_hash.hexdigest(), list(roles)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_meta(vc):
    import transformers
    dev_map = getattr(vc.model, "hf_device_map", None)
    return {
        "host": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "sharded": bool(dev_map) and len(set(map(str, dev_map.values()))) > 1,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
    }


def _task_inputs(vc, task):
    """Independent re-derivation of a cell's inputs (data, roles, prompt hash)
    from the CURRENT invocation -- used to validate existing cells before skipping them."""
    templates = select_templates(args.suite, args.use_E)
    data = utils.load_json(os.path.join(MMLU_DIR, f"{task}.json"))
    if args.smoke_n:
        data = data[: args.smoke_n]
    custom = [r.strip() for r in args.roles.split(",")] if args.roles else None
    roles = utils.make_characters(task, custom)
    ph = hashlib.sha256()
    for sample in data:
        for role in roles:
            ph.update(utils.construct_prompt(vc, templates, sample.get("text", ""), role, args.use_chat).encode("utf-8"))
    return data, roles, ph.hexdigest()


def _validate_existing(path, expect, vc, task, expected_fires):
    """True iff the saved cell is complete and matches the current inputs; raises on any mismatch."""
    j = utils.load_json(path)
    meta = j.get("meta") or {}
    for k, v in expect.items():
        if k not in meta or meta[k] != v:
            raise SystemExit(f"resume: meta field {k!r} missing or different in {path}")
    data, roles, p_hash = _task_inputs(vc, task)
    if meta.get("rendered_prompt_sha256") != p_hash or meta.get("n_items") != len(data) or meta.get("roles") != roles:
        raise SystemExit(f"resume: prompts / item count / roles differ from current inputs in {path}")
    rows = j.get("data")
    if not isinstance(rows, list) or len(rows) != len(data):
        raise SystemExit(f"resume: result row count differs in {path}")
    for old, cur in zip(rows, data):
        if old.get("text") != cur.get("text") or old.get("label") != cur.get("label"):
            raise SystemExit(f"resume: result items differ from current data in {path}")
        for r in roles:
            rk = r.replace(" ", "_")
            lg = old.get(f"logits_{rk}")
            if old.get(f"answer_{rk}") is None or old.get(f"prob_{rk}") is None or old.get(f"softmax_{rk}") is None \
                    or not isinstance(lg, list) or len(lg) != 5:
                raise SystemExit(f"resume: missing answer/logit fields in {path}")
    if meta.get("steering_fires") != expected_fires or meta.get("steering_fires_expected") != expected_fires:
        raise SystemExit(f"resume: recorded steering fires differ from independent expectation {expected_fires} in {path}")
    return True


def main_mask_set():
    """Off-support controls: several named masks, one model load. Additive mode;
    the default path in main() is untouched. Resume = skip a cell whose saved
    meta matches this invocation exactly; any mismatch is fatal (never overwrite)."""
    pairs = utils.parse_configs(args.configs)
    named = [m.split("=", 1) for m in args.mask_set]
    if any(len(x) != 2 or not x[0] for x in named) or len({x[0] for x in named}) != len(named):
        raise SystemExit("--mask_set needs unique label=path entries")
    for _, mp in named:
        if not os.path.isfile(mp):
            raise SystemExit(f"mask missing: {mp}")
    tasks = TASKS[: args.smoke_tasks] if args.smoke_tasks else TASKS
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    run_env = _run_meta(vc)
    n_roles = len([r for r in args.roles.split(",") if r.strip()]) if args.roles else 1
    jobs = [(label, mp, pairs) for label, mp in named]
    if args.baseline_mask:
        if not os.path.isfile(args.baseline_mask):
            raise SystemExit(f"baseline mask missing: {args.baseline_mask}")
        if any(a == 0 for a, _ in pairs):
            raise SystemExit("--configs must not contain alpha=0 when --baseline_mask is used")
        st0, en0 = pairs[0][1]
        # one baseline per model: RSN mask x 0 (identical execution path, zero injection)
        jobs.insert(0, ("baseline", args.baseline_mask, utils.parse_configs([f"0-{st0}-{en0}"])))
    for label, mp, job_pairs in jobs:
        mask = np.load(mp)
        mask_sha = _sha256_file(mp)
        mask_rows = set(np.flatnonzero(np.any(mask != 0, axis=1)).tolist())
        for alpha, (st, en) in job_pairs:
            diff_mtx = mask * alpha
            cfg_rows = set(utils.decoder_layer_range(st, en))
            if mask_rows != cfg_rows:
                raise SystemExit(f"{mp}: non-zero rows {sorted(mask_rows)} != configured decoder layers {sorted(cfg_rows)}")
            n_layers = len(cfg_rows)  # from the CONFIG, not from the mask
            TOP = max(1, int(args.percentage / 100 * diff_mtx.shape[1]))
            for task in tasks:
                out_dir = os.path.join(SAVE_ROOT, label, f"mdf_{alpha}")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{task}_{args.size}_answers_{TOP}_{st}_{en}.json")
                expect = {"mask_sha256": mask_sha, "alpha": alpha, "layer_start": st, "layer_end": en,
                          "model_dir": args.model_dir, "tail_len": args.tail_len, "roles_arg": args.roles,
                          "suite": args.suite, "use_E": bool(args.use_E), "use_chat": bool(args.use_chat),
                          "smoke_n": args.smoke_n}
                if os.path.exists(out_path):
                    n_it = len(_task_inputs(vc, task)[0])
                    exp_f = n_layers * n_it * n_roles * args.tail_len if alpha != 0 else 0
                    _validate_existing(out_path, expect, vc, task, exp_f)
                    print(f"[skip complete, validated] {out_path}")
                    continue
                print(f"\n=== {label} | {task} | α={alpha} | layers={st}-{en} | steered_rows={n_layers} ===")
                vc.steering_fire_count(reset=True)
                with torch.no_grad():
                    data, accuracy, tmp_record, p_hash, roles = run_task(vc, task, diff_mtx)
                fires = vc.steering_fire_count(reset=True)
                expected = n_layers * len(data) * len(roles) * args.tail_len if alpha != 0 else 0
                if fires != expected:
                    raise SystemExit(f"steering_fires {fires} != expected {expected} for {out_path}")
                meta = {**expect, "protocol": "mmlue-orthogonal-off-v1", "model": args.model, "mask_label": label,
                        "mask_path": mp, "n_items": len(data), "roles": roles, "steered_rows": n_layers,
                        "steering_fires": fires, "steering_fires_expected": expected,
                        "rendered_prompt_sha256": p_hash, "env": run_env}
                with open(out_path, "w", encoding="utf-8") as fw:
                    json.dump({"data": data, "accuracy": accuracy, "template": tmp_record, "meta": meta},
                              fw, ensure_ascii=False, indent=2)
                print("Saved →", out_path)
                del data, accuracy, tmp_record
                gc.collect()
                torch.cuda.empty_cache()
    print("\nAll cells finished.")


# ─────────────────────────── Main ───────────────────────────────


def main():
    if args.mask_set:
        return main_mask_set()

    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("ALPHAS_START_END_PAIRS:", ALPHAS_START_END_PAIRS)

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}.npy"
        mask_path = os.path.join(MASK_DIR, f"{mask_name}")
        diff_mtx = np.load(mask_path) * alpha  # shape: (32, 4096)
        TOP = max(1, int(args.percentage / 100 * diff_mtx.shape[1]))
        for task in TASKS:
            print(f"\n=== {task} | α={alpha} | layers={st}-{en}| TOP={TOP} ===")

            with torch.no_grad():
                updated_data, accuracy, tmp_record, _, _ = run_task(vc, task, diff_mtx)

            # save JSON
            out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{task}_{args.size}_answers_{TOP}_{st}_{en}.json")
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump({"data": updated_data, "accuracy": accuracy, "template": tmp_record}, fw, ensure_ascii=False, indent=2)
            print("Saved →", out_path)

            # Free memory
            del updated_data, accuracy, tmp_record
            gc.collect()
            torch.cuda.empty_cache()

    print("\nAll tasks finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vicunda model with neuron editing and logits output.")
    
    parser.add_argument("--model", type=str, default="qwen2.5_base")
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--hs", type=str, default="qwen2.5")
    parser.add_argument("--size", type=str, default="7B")
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["4-16-22", "1-1-29"], help="List of alpha-start-end triplets, e.g. 4-16-22")
    parser.add_argument("--mask_type", type=str, default="nmd", help="Mask type to load: nmd or random")
    parser.add_argument("--ans_file", type=str, default="answer_mdf")
    parser.add_argument("--E", dest="use_E", action="store_true")
    parser.add_argument("--use_chat", action="store_true", help="Use tokenizer.apply_chat_template for prompts")
    parser.add_argument("--tail_len", type=int, default=1, help="Number of last tokens to apply diff (default: 1)")
    parser.add_argument("--suite", type=str, default="default", choices=["default", "vanilla"], help="Prompt suite for MMLU-Pro")
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base directory for data/output (e.g., /work/<user>/RolePlaying/components). "
                             "If not set, falls back to /{data}/paveen/RolePlaying/components")
    parser.add_argument("--roles", type=str, default=None,
                        help="Comma-separated list of roles. Use {task} as placeholder for task name. "
                             "E.g., 'neutral,{task} expert,non {task} expert'")

    parser.add_argument("--mask_set", nargs="*", default=None,
                        help="label=path entries (explicit masks, one model load). Outputs go to <ans_file>/<label>/mdf_<alpha>/")
    parser.add_argument("--baseline_mask", default=None,
                        help="mask_set mode: add ONE alpha=0 'baseline' job using this (RSN) mask x 0")
    parser.add_argument("--smoke_n", type=int, default=0, help="mask_set mode only: first N items per task")
    parser.add_argument("--smoke_tasks", type=int, default=0, help="mask_set mode only: first N tasks")

    args = parser.parse_args()

    print("Model: ", args.model)
    print("Import model from ", args.model_dir)
    print("HS: ", args.hs)
    print("Mask Type:", args.mask_type)

    # Path setup
    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/RolePlaying/components"

    MASK_DIR = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    MMLU_DIR = os.path.join(BASE, "mmlu")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)

    os.makedirs(SAVE_ROOT, exist_ok=True)
    main()
