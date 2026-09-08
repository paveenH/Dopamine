#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Role-Confidence cross-steering MMLU-E experiment (RR / RC / CR / CC + random
controls + alpha=0 baseline).

STANDALONE server-side script -- does NOT modify or import from
get_answer_regenerate_logits.py (that script can only load a mask-file-named
diff matrix and multiply it by an integer alpha; it has no way to accept one
of our six precomputed cross-steering .npy vectors). This script reuses the
SAME underlying mechanism instead:
  - detection.task_list.TASKS            (57 MMLU tasks)
  - template.select_templates(suite="default", use_E=True)  (MMLU-E templates)
  - utils.construct_prompt / utils.make_characters / utils.option_token_ids
  - VicundaModel.regenerate_logits(prompts, diff_matrices, tail_len=1)
    (bare-string, prefill-only, last-token injection -- identical mechanism
    to get_answer_regenerate_logits.py / run_mmlue_qwen25.sh)

diff_matrices actually applied = one of the six precomputed (32, 4096)
float32 vectors from build_cross_steering_vectors.py (RR/RC/CR/CC/RRand/CRand)
multiplied by a scalar alpha (alpha=0 => all-zero => baseline, alpha=+/-1 =>
the vector as norm-matched, i.e. dose is expressed via the vector's OWN
norm-matched magnitude at alpha=+-1, not by re-scaling with a separate large
integer alpha as in the original NMD launchers). This keeps alpha as a
SYMMETRIC DOSE MULTIPLIER on an already norm-matched vector, matching the
task's "对称非零剂量" pilot design directly: alpha in {-1, 0, +1} by default.

Roles: EXACTLY confident / unconfident, matching
run_hidden_mmlue_confidence_hs.sh's protocol (templates["neg"] via
utils.construct_prompt, which special-cases any role containing "confident").
No Expert/Non-Expert roles are run here -- this experiment steers the MODEL's
neurons, not the PROMPT's persona, so the prompt role is held fixed at
confident/unconfident throughout (the two conditions whose HS collection
already exists and whose accuracy/E-rate this experiment is meant to move).

Output layout (fail-closed, one condition per directory, no cross-writes):
  {out_root}/{condition}/alpha_{alpha}/{task}_{size}_answers.json
  {out_root}/{condition}/alpha_{alpha}/summary_{task}_{size}.json
condition in {baseline, RR, RC, CR, CC, RRand, CRand}. "baseline" is
alpha=0 for ANY condition (all-zero diff matrix) and is written ONCE,
shared across all six vectors (since alpha=0 * any vector = the same
all-zero injection) -- never re-run per condition.

Usage (on the SERVER, in the project's conda env; the server interpreter is
`python`, NOT `python3.10`):
  python get_answer_cross_steering_mmlue.py \
      --model_dir meta-llama/Llama-3.1-8B-Instruct \
      --size 8B \
      --vector_dir /data1/paveen/Dopamine/components/llama3_confidence/cross_steering_mmlue/vectors \
      --mmlu_dir /data1/paveen/Dopamine/components/mmlu \
      --out_root /data1/paveen/Dopamine/components/llama3_confidence/cross_steering_mmlue/results \
      --condition RR --alphas -1,0,1
"""
import argparse
import gc
import hashlib
import json
import os
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
CONDITIONS = ["RR", "RC", "CR", "CC", "RRand", "CRand"]


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_vector(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing steering vector: {path}")
    v = np.load(path)
    if v.shape != (N_DECODER_LAYERS, HIDDEN_DIM):
        raise ValueError(f"{path}: shape {v.shape} != expected ({N_DECODER_LAYERS}, {HIDDEN_DIM})")
    if not np.isfinite(v).all():
        raise ValueError(f"{path}: contains NaN/Inf")
    return v.astype(np.float32)


def run_task(vc: VicundaModel, task: str, diff_mtx: np.ndarray, mmlu_dir: Path,
             use_chat: bool, tail_len: int) -> tuple:
    templates = select_templates("default", use_E=True)
    LABELS = templates["labels"]
    opt_ids = utils.option_token_ids(vc, LABELS)

    data_path = mmlu_dir / f"{task}.json"
    data = utils.load_json(data_path)
    roles = utils.make_characters(task, ROLES)

    stats = {r: {"correct": 0, "E_count": 0, "invalid": 0, "total": 0} for r in roles}

    for sample in tqdm(data, desc=task, leave=False):
        ctx = sample.get("text", "")
        true_idx = sample.get("label", -1)
        true_lab = LABELS[true_idx] if 0 <= true_idx < len(LABELS) else None

        for role in roles:
            prompt = utils.construct_prompt(vc, templates, ctx, role, use_chat)
            raw_logits = vc.regenerate_logits([prompt], diff_mtx, tail_len=tail_len)[0]
            opt_logits = np.array([raw_logits[i] for i in opt_ids])

            exp = np.exp(opt_logits - opt_logits.max())
            soft = exp / exp.sum()

            pred_idx = int(opt_logits.argmax())
            pred_lab = LABELS[pred_idx]
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
                st["invalid"] += 1

    accuracy = {}
    for role, s in stats.items():
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        accuracy[role] = {**s, "accuracy_percentage": round(pct, 2)}

    return data, accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--size", default="8B")
    ap.add_argument("--vector_dir", required=True,
                     help="Dir containing steering_{RR,RC,CR,CC,RRand,CRand}_{size}.npy "
                          "from build_cross_steering_vectors.py")
    ap.add_argument("--mmlu_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--condition", required=True, choices=CONDITIONS + ["baseline"])
    ap.add_argument("--alphas", required=True,
                     help="Comma-separated alpha values, e.g. '-1,0,1'. alpha=0 is "
                          "the all-zero baseline and is written to {out_root}/baseline/alpha_0/ "
                          "regardless of --condition.")
    ap.add_argument("--tail_len", type=int, default=1)
    ap.add_argument("--use_chat", action="store_true", default=False,
                     help="Default False: bare-string, matching the confidence HS "
                          "collection and the mask-extraction convention.")
    ap.add_argument("--allow_overwrite", action="store_true", default=False)
    args = ap.parse_args()

    mmlu_dir = Path(args.mmlu_dir)
    out_root = Path(args.out_root)
    vector_dir = Path(args.vector_dir)

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
        vec_path = None
        vec = np.zeros((N_DECODER_LAYERS, HIDDEN_DIM), dtype=np.float32)
        vec_sha = None
    else:
        vec_path = vector_dir / f"steering_{args.condition}_{args.size}.npy"
        vec = load_vector(vec_path)
        vec_sha = sha256_of_file(vec_path)

    # ---- Fail-closed: pre-check ALL output dirs before loading the model ----
    plan = []
    for alpha in alphas:
        cond_for_dir = "baseline" if alpha == 0.0 else args.condition
        out_dir = out_root / cond_for_dir / f"alpha_{alpha:g}"
        if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
            print(f"[REFUSE] Output directory already exists and is non-empty: {out_dir}")
            print("Pass --allow_overwrite to deliberately overwrite, or choose a different --out_root.")
            sys.exit(1)
        plan.append((alpha, cond_for_dir, out_dir))

    print(f"Condition: {args.condition}")
    print(f"Alphas: {alphas}")
    print(f"Vector: {vec_path if vec_path else '(all-zero baseline)'}")
    if vec_sha:
        print(f"Vector sha256: {vec_sha}")
    print(f"Plan: {[(a, str(d)) for a, _, d in plan]}")

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    for alpha, cond_for_dir, out_dir in plan:
        diff_mtx_full = (vec * alpha).astype(np.float32)
        diff_matrices_list = [diff_mtx_full[i] for i in range(N_DECODER_LAYERS)]

        out_dir.mkdir(parents=True, exist_ok=True)
        task_results = {}
        for task in TASKS:
            print(f"\n=== condition={cond_for_dir} alpha={alpha} task={task} ===")
            with torch.no_grad():
                updated_data, accuracy = run_task(
                    vc, task, diff_matrices_list, mmlu_dir, args.use_chat, args.tail_len
                )
            out_path = out_dir / f"{task}_{args.size}_answers.json"
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump({"data": updated_data, "accuracy": accuracy}, fw, ensure_ascii=False, indent=2)
            task_results[task] = accuracy
            for role, s in accuracy.items():
                print(f"  {role:<15} acc={s['accuracy_percentage']:5.2f}%  "
                      f"(correct {s['correct']}/{s['total']}), E={s['E_count']}, invalid={s['invalid']}")

            del updated_data, accuracy
            gc.collect()
            torch.cuda.empty_cache()

        meta = {
            "condition": cond_for_dir,
            "alpha": alpha,
            "model_dir": args.model_dir,
            "size": args.size,
            "steering_vector_path": str(vec_path) if vec_path else None,
            "steering_vector_sha256": vec_sha,
            "roles": ROLES,
            "suite": "default",
            "use_E": True,
            "use_chat": args.use_chat,
            "tail_len": args.tail_len,
            "n_tasks": len(TASKS),
            "task_summary": task_results,
        }
        with open(out_dir / f"run_meta_{args.size}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"\nSaved run meta -> {out_dir / f'run_meta_{args.size}.json'}")

    print("\nAll requested alphas finished for condition:", args.condition)


if __name__ == "__main__":
    main()
