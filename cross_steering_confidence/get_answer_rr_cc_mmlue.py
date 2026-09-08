#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Role self-steering (RR) and Confidence self-steering (CC) MMLU-E experiment.

SIMPLIFIED per the current task: only RR (Role mask x Role direction, i.e.
the ORIGINAL RSN mask used as-is) and CC (Confidence mask x Confidence
direction, i.e. the new confidence mask used as-is) are run. NO RC, NO CR, NO
random-support control -- this script does not build or accept any
cross-direction or random vector at all; it only ever loads one of the two
raw NMD mask files below and multiplies by a scalar alpha, exactly like the
existing RSN steering pipeline (get_answer_regenerate_logits.py /
run_mmlue_qwen25.sh) does for its own mask.

Masks used AS-IS (no direction/support decomposition, no norm-matching):
  RR -> mask/llama3_non_logits/nmd_0.5_11_20_8B.npy           (existing Role mask)
  CC -> mask/llama3_non_logits/confidence_0.5_11_20_8B.npy    (new Confidence mask,
                                                                built by
                                                                build_confidence_nmd_mask_for_steering.py)
Both masks store SIGNED per-neuron diff values (detection/nmd.py::get_nmd_mask
convention), not a binary support -- diff_mtx = mask * alpha, identical in
spirit to get_answer_regenerate_logits.py's `np.load(mask_path) * alpha`.

Neither mask is modified or overwritten by this script. Does NOT touch
run_hidden_mmlue_confidence_hs.sh, run_mmlue_qwen25.sh, or
get_answer_regenerate_logits.py.

Roles: EXACTLY confident / unconfident (templates["neg"] via
utils.construct_prompt), matching run_hidden_mmlue_confidence_hs.sh's
protocol -- bare-string, no chat, no CoT, MMLU-E (A-E, "E) I am not sure").

Output layout (fail-closed at the CELL level, resumable at the TASK level):
  {out_root}/{condition}/alpha_{alpha}/{task}_{size}_answers.json
  {out_root}/{condition}/alpha_{alpha}/run_meta_{size}.json
condition in {baseline, RR, CC}. "baseline" is alpha=0 for either condition
(all-zero diff matrix, since the injected vector doesn't matter at alpha=0)
and is written to {out_root}/baseline/alpha_0/ ONCE, shared between RR and CC.

RESUME / COMPLETENESS: identical semantics to the earlier cross-steering
generator -- a cell is COMPLETE only if all 57 tasks are present with the
expected sample count and full confident/unconfident fields, AND run_meta
matches the current invocation's config exactly (a mismatch is a FATAL
refusal, not a silent resume). An incomplete cell has only its missing tasks
completed; complete tasks are never touched.

Usage (on the SERVER, in the project's conda env; interpreter is `python`,
NOT `python3.10`). This script itself is dose-set-agnostic -- --alphas takes
whatever comma-separated list is passed, no hardcoded doses; the REVISED
2026-09-08 round only runs CC at 2,4,6 (see run_rr_cc_mmlue.sh):
  python get_answer_rr_cc_mmlue.py \
      --model_dir meta-llama/Llama-3.1-8B-Instruct --size 8B \
      --mask_dir /data1/paveen/Dopamine/components/mask/llama3_non_logits \
      --mmlu_dir /data1/paveen/Dopamine/components/mmlu \
      --out_root /data1/paveen/Dopamine/components/llama3_confidence/rr_cc_mmlue/results \
      --condition CC --alphas="2,4,6"

Negative alpha lists (if RR or a bidirectional CC check is run later) must
still be passed as --alphas="..." (with '=') so argparse does not treat a
leading '-' as a new flag, e.g. --alphas="-4,-2,2,4".
"""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llms import VicundaModel  # noqa: E402
from detection.task_list import TASKS  # noqa: E402
from template import select_templates  # noqa: E402
import utils  # noqa: E402

ROLES = ["confident", "unconfident"]
N_DECODER_LAYERS = 32
HIDDEN_DIM = 4096
START_LAYER = 11
END_LAYER = 20
TOP_K = 20
CONDITIONS = ["RR", "CC"]
PROTOCOL_VERSION = "rr-cc-mmlue-v1"

MASK_FILENAMES = {
    "RR": "nmd_0.5_11_20_8B.npy",
    "CC": "confidence_0.5_11_20_8B.npy",
}


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing mask: {path}")
    m = np.load(path)
    if m.shape != (N_DECODER_LAYERS, HIDDEN_DIM):
        raise ValueError(f"{path}: shape {m.shape} != expected ({N_DECODER_LAYERS}, {HIDDEN_DIM})")
    if not np.isfinite(m.astype(np.float64)).all():
        raise ValueError(f"{path}: contains NaN/Inf")
    nz_per_row = (m != 0).sum(axis=1)
    for raw_idx in range(N_DECODER_LAYERS):
        decoder_layer = raw_idx + 1
        expected = TOP_K if (START_LAYER <= decoder_layer < END_LAYER) else 0
        if nz_per_row[raw_idx] != expected:
            raise AssertionError(
                f"{path} row {raw_idx} (decoder layer {decoder_layer}): "
                f"nonzero count {nz_per_row[raw_idx]} != expected {expected}"
            )
    return m


def is_task_complete(out_path: Path, mmlu_dir: Path, task: str) -> tuple:
    src_path = mmlu_dir / f"{task}.json"
    try:
        src_data = utils.load_json(src_path)
    except Exception as e:
        return False, f"cannot read source MMLU file: {type(e).__name__}: {e}"
    n_expected = len(src_data)

    if not out_path.exists():
        return False, "output file does not exist"
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        return False, f"cannot read/parse output file: {type(e).__name__}: {e}"

    data = payload.get("data")
    if data is None:
        return False, "output file missing 'data' key"
    if len(data) != n_expected:
        return False, f"sample count mismatch: output has {len(data)}, source has {n_expected}"

    for i, sample in enumerate(data):
        for role in ROLES:
            role_key = role.replace(" ", "_")
            for field_prefix in ("answer_", "prob_", "softmax_", "logits_"):
                key = f"{field_prefix}{role_key}"
                if key not in sample:
                    return False, f"sample {i} missing field '{key}'"
    return True, "complete"


def is_cell_complete(out_dir: Path, mmlu_dir: Path, expected_meta: dict, size: str) -> tuple:
    incomplete_tasks = []
    for task in TASKS:
        out_path = out_dir / f"{task}_{size}_answers.json"
        ok, _ = is_task_complete(out_path, mmlu_dir, task)
        if not ok:
            incomplete_tasks.append(task)
    if incomplete_tasks:
        return False, f"{len(incomplete_tasks)} task(s) incomplete", incomplete_tasks

    meta_path = out_dir / f"run_meta_{size}.json"
    if not meta_path.exists():
        return False, "run_meta file missing", []
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            existing_meta = json.load(f)
    except Exception as e:
        return False, f"cannot read run_meta: {type(e).__name__}: {e}", []

    mismatch_keys = []
    for key, expected_val in expected_meta.items():
        if existing_meta.get(key) != expected_val:
            mismatch_keys.append((key, existing_meta.get(key), expected_val))
    if mismatch_keys:
        detail = "; ".join(f"{k}: stored={s!r} vs current={c!r}" for k, s, c in mismatch_keys)
        return False, f"run_meta metadata mismatch (FATAL, not auto-resumable): {detail}", []

    return True, "complete", []


def run_task(vc: VicundaModel, task: str, diff_mtx: list, mmlu_dir: Path,
             use_chat: bool, tail_len: int) -> tuple:
    templates = select_templates("default", use_E=True)
    labels = templates["labels"]
    opt_ids = utils.option_token_ids(vc, labels)

    data_path = mmlu_dir / f"{task}.json"
    data = utils.load_json(data_path)
    roles = utils.make_characters(task, ROLES)

    # wrong_non_E: a parsed, valid, non-E, WRONG answer -- NOT a format
    # failure. option_token_ids/argmax over exactly the A-E option logits
    # always returns one of the 5 valid labels.
    stats = {r: {"correct": 0, "E_count": 0, "wrong_non_E": 0, "total": 0} for r in roles}

    for sample in tqdm(data, desc=task, leave=False):
        ctx = sample.get("text", "")
        true_idx = sample.get("label", -1)
        true_lab = labels[true_idx] if 0 <= true_idx < len(labels) else None

        for role in roles:
            prompt = utils.construct_prompt(vc, templates, ctx, role, use_chat)
            raw_logits = vc.regenerate_logits([prompt], diff_mtx, tail_len=tail_len)[0]
            opt_logits = np.array([raw_logits[i] for i in opt_ids])

            exp = np.exp(opt_logits - opt_logits.max())
            soft = exp / exp.sum()

            pred_idx = int(opt_logits.argmax())
            pred_lab = labels[pred_idx]
            pred_prb = float(soft[pred_idx])

            role_key = role.replace(" ", "_")
            sample[f"answer_{role_key}"] = pred_lab
            sample[f"prob_{role_key}"] = pred_prb
            sample[f"softmax_{role_key}"] = [float(p) for p in soft]
            sample[f"logits_{role_key}"] = [float(l) for l in opt_logits]

            st = stats[role]
            st["total"] += 1
            if true_lab is not None and pred_lab == true_lab:
                st["correct"] += 1
            elif pred_lab == "E":
                st["E_count"] += 1
            else:
                st["wrong_non_E"] += 1

    accuracy = {}
    for role, s in stats.items():
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        accuracy[role] = {**s, "accuracy_percentage": round(pct, 2)}

    return data, accuracy


def build_layer_norms(vec: np.ndarray) -> dict:
    return {str(l): float(np.linalg.norm(vec[l - 1])) for l in range(1, N_DECODER_LAYERS + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--size", default="8B")
    ap.add_argument("--mask_dir", required=True,
                     help="Dir containing nmd_0.5_11_20_8B.npy (RR) and "
                          "confidence_0.5_11_20_8B.npy (CC)")
    ap.add_argument("--mmlu_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--condition", required=True, choices=CONDITIONS + ["baseline"])
    ap.add_argument("--alphas", required=True,
                     help="Comma-separated alpha values, e.g. '-4,-2,2,4'. Pass as "
                          "--alphas=\"-4,-2,2,4\" so argparse does not treat a leading "
                          "'-' as a new flag. alpha=0 is the shared all-zero baseline.")
    ap.add_argument("--tail_len", type=int, default=1)
    ap.add_argument("--use_chat", action="store_true", default=False)
    args = ap.parse_args()

    mmlu_dir = Path(args.mmlu_dir)
    out_root = Path(args.out_root)
    mask_dir = Path(args.mask_dir)

    if not mmlu_dir.exists():
        print(f"[REFUSE] mmlu_dir not found: {mmlu_dir}")
        sys.exit(1)
    missing = [t for t in TASKS if not (mmlu_dir / f"{t}.json").exists()]
    if missing:
        print(f"[REFUSE] {len(missing)} MMLU task JSON files missing under {mmlu_dir}: {missing[:5]}...")
        sys.exit(1)

    alphas = [float(a) for a in args.alphas.split(",")]

    if args.condition == "baseline":
        if alphas != [0.0]:
            print("[REFUSE] --condition baseline only accepts --alphas 0")
            sys.exit(1)
        mask_path = None
        mask = np.zeros((N_DECODER_LAYERS, HIDDEN_DIM), dtype=np.float16)
        mask_sha = None
    else:
        mask_path = mask_dir / MASK_FILENAMES[args.condition]
        mask = load_mask(mask_path)
        mask_sha = sha256_of_file(mask_path)

    layer_norms_unscaled = build_layer_norms(mask.astype(np.float64))

    plan = []
    for alpha in alphas:
        cond_for_dir = "baseline" if alpha == 0.0 else args.condition
        out_dir = out_root / cond_for_dir / f"alpha_{alpha:g}"

        expected_meta = {
            "protocol_version": PROTOCOL_VERSION,
            "condition": cond_for_dir,
            "alpha": alpha,
            "model_dir": args.model_dir,
            "size": args.size,
            "mask_filename": MASK_FILENAMES.get(cond_for_dir) if cond_for_dir != "baseline" else None,
            "mask_sha256": mask_sha,
            "roles": ROLES,
            "suite": "default",
            "use_E": True,
            "use_chat": args.use_chat,
            "tail_len": args.tail_len,
            "n_tasks": len(TASKS),
        }

        if out_dir.exists():
            complete, reason, incomplete_tasks = is_cell_complete(out_dir, mmlu_dir, expected_meta, args.size)
            if complete:
                print(f"[skip] {cond_for_dir}/alpha_{alpha:g}: already COMPLETE -- not re-run.")
                continue
            if "FATAL" in reason:
                print(f"[REFUSE] {cond_for_dir}/alpha_{alpha:g}: {reason}")
                sys.exit(1)
            print(f"[resume] {cond_for_dir}/alpha_{alpha:g}: {reason}. "
                  f"Completing only: {incomplete_tasks}")
            tasks_to_run = incomplete_tasks
        else:
            tasks_to_run = list(TASKS)

        plan.append((alpha, cond_for_dir, out_dir, tasks_to_run, expected_meta))

    if not plan:
        print("Nothing to do -- every requested cell is already complete.")
        return

    print(f"Condition: {args.condition}")
    print(f"Alphas requested: {alphas}")
    print(f"Mask: {mask_path if mask_path else '(all-zero baseline)'}")
    if mask_sha:
        print(f"Mask sha256: {mask_sha}")
    for alpha, cond_for_dir, out_dir, tasks_to_run, _ in plan:
        print(f"  -> {cond_for_dir}/alpha_{alpha:g}: {len(tasks_to_run)} task(s) to run, out={out_dir}")

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    tokenizer_info = {"tokenizer_class": type(vc.tokenizer).__name__,
                       "vocab_size": getattr(vc.tokenizer, "vocab_size", None)}

    for alpha, cond_for_dir, out_dir, tasks_to_run, expected_meta in plan:
        diff_mtx_full = (mask.astype(np.float32) * alpha).astype(np.float32)
        diff_matrices_list = [diff_mtx_full[i] for i in range(N_DECODER_LAYERS)]
        layer_norms_scaled = build_layer_norms(diff_mtx_full)

        out_dir.mkdir(parents=True, exist_ok=True)

        for task in tasks_to_run:
            print(f"\n=== condition={cond_for_dir} alpha={alpha} task={task} ===")
            with torch.no_grad():
                updated_data, accuracy = run_task(
                    vc, task, diff_matrices_list, mmlu_dir, args.use_chat, args.tail_len
                )
            out_path = out_dir / f"{task}_{args.size}_answers.json"
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump({"data": updated_data, "accuracy": accuracy}, fw, ensure_ascii=False, indent=2)
            for role, s in accuracy.items():
                print(f"  {role:<15} acc={s['accuracy_percentage']:5.2f}%  "
                      f"(correct {s['correct']}/{s['total']}), E={s['E_count']}, "
                      f"wrong_non_E={s['wrong_non_E']}")
            del updated_data, accuracy
            gc.collect()
            torch.cuda.empty_cache()

        task_summary = {}
        for task in TASKS:
            out_path = out_dir / f"{task}_{args.size}_answers.json"
            with open(out_path, "r", encoding="utf-8") as f:
                task_summary[task] = json.load(f)["accuracy"]

        meta = {
            **expected_meta,
            "note": "Simplified RR/CC-only experiment. RR = existing Role NMD mask "
                    "(nmd_0.5_11_20_8B.npy) used AS-IS. CC = new Confidence NMD mask "
                    "(confidence_0.5_11_20_8B.npy) used AS-IS. Both masks store SIGNED "
                    "per-neuron diff values (detection/nmd.py::get_nmd_mask), not a "
                    "binary support. NO direction/support decomposition, NO RC/CR, NO "
                    "random control, NO norm-matching in this stage.",
            "mask_path": str(mask_path) if mask_path else None,
            "mask_layer_l2norms_unscaled": layer_norms_unscaled,
            "mask_layer_l2norms_at_this_alpha": layer_norms_scaled,
            "injection_band_decoder_layers": f"{START_LAYER}-{END_LAYER - 1}",
            "model_and_tokenizer": {"model_dir": args.model_dir, **tokenizer_info},
            "prompt_template": {
                "suite": "default", "use_E": True, "use_chat": args.use_chat,
                "note": "templates['neg'] via utils.construct_prompt for confident/unconfident.",
            },
            "decoding_config": {
                "method": "argmax over A-E option-token logits (regenerate_logits), no sampling",
                "hook_mechanism": "VicundaModel.regenerate_logits -> _apply_diff_hooks, "
                                  "prefill-only, last tail_len prompt token(s)",
            },
            "task_summary": task_summary,
            "tasks_completed_this_invocation": tasks_to_run,
        }
        with open(out_dir / f"run_meta_{args.size}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"\nSaved run meta -> {out_dir / f'run_meta_{args.size}.json'}")

        complete, reason, incomplete_tasks = is_cell_complete(out_dir, mmlu_dir, expected_meta, args.size)
        if not complete:
            print(f"[WARNING] {cond_for_dir}/alpha_{alpha:g}: still incomplete ({reason}). "
                  f"Re-run to complete: {incomplete_tasks}")
        else:
            print(f"[OK] {cond_for_dir}/alpha_{alpha:g}: verified complete (57/57 tasks).")

    print("\nAll requested alphas finished for condition:", args.condition)


if __name__ == "__main__":
    main()
