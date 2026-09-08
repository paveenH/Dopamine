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
multiplied by a scalar alpha. Doses are alpha in {-4, -2, 0, +2, +4}: alpha=0
is the shared all-zero baseline; the vectors are already norm-matched to RR
per layer, so alpha=+-2/+-4 scale that norm-matched magnitude by a further
factor of 2/4 (a SYMMETRIC dose multiplier on an already norm-matched vector,
NOT a re-scaling against a separate large integer alpha as in the original
NMD launchers).

PROVENANCE NOTE ON RR (do not delete or reword away from this): an RR-like
condition (Role direction x Role NMD support, at some alpha, on some MMLU-E
run) was reported in an older results table under this project. That older
table's data was inspected before this experiment was built and found to be
at a DIFFERENT, incompatible protocol -- specifically, the historical
`logits_mdf_4/*_20_11_20.json` files carry NO E option in the rendered prompt
text at all (0/57 task files contain "E)"), a null `template` field (so the
exact prompt cannot be re-verified), and implausibly high invalid rates
(up to 71%) consistent with the model never having been offered an E choice
in the first place. So the correct statement is: **RR's historical effect was
reported, but no protocol-complete, sample-level-reproducible source for it
could be located** -- NOT "RR was never tested". This experiment therefore
RE-RUNS RR from scratch under the current, verified MMLU-E protocol, on equal
footing with RC/CR/CC/RRand/CRand.

Roles: EXACTLY confident / unconfident, matching
run_hidden_mmlue_confidence_hs.sh's protocol (templates["neg"] via
utils.construct_prompt, which special-cases any role containing "confident").
No Expert/Non-Expert roles are run here -- this experiment steers the MODEL's
neurons, not the PROMPT's persona, so the prompt role is held fixed at
confident/unconfident throughout.

Output layout (fail-closed at the CELL level, resumable at the TASK level):
  {out_root}/{condition}/alpha_{alpha}/{task}_{size}_answers.json
  {out_root}/{condition}/alpha_{alpha}/run_meta_{size}.json
condition in {baseline, RR, RC, CR, CC, RRand, CRand}. "baseline" is alpha=0
for ANY condition (all-zero diff matrix) and is written to
{out_root}/baseline/alpha_0/ ONCE, shared across all six vectors.

RESUME / COMPLETENESS SEMANTICS (see is_task_complete() / is_cell_complete()):
  - A per-task JSON is considered COMPLETE only if it exists, has valid JSON,
    contains exactly the expected number of samples (matched against the
    live MMLU source JSON's sample count), and every sample carries BOTH
    confident and unconfident answer/prob/softmax/logits fields.
  - A cell (condition, alpha) is considered COMPLETE only if ALL 57 tasks are
    complete AND run_meta_{size}.json exists AND its recorded
    protocol/model/roles/tail_len/alpha/steering_vector_sha256 match the
    CURRENT run's configuration exactly.
  - On start, for each requested (condition, alpha): if the cell is already
    COMPLETE, it is skipped entirely (never re-run, never overwritten). If
    the cell directory exists but is NOT complete (partial from an
    interrupted run, or a metadata mismatch), the script completes ONLY the
    missing/incomplete tasks -- an already-complete task's JSON file is never
    touched or regenerated. A metadata mismatch (e.g. a different tail_len or
    a stale vector sha256) is FATAL and REFUSES the whole cell rather than
    silently mixing two protocols in one output directory -- fix the
    mismatch (or point at a fresh --out_root) before retrying.

Usage (on the SERVER, in the project's conda env; the server interpreter is
`python`, NOT `python3.10`). Pass negative-number lists as a SINGLE
`--alphas="..."` argument (with the `=`) so argparse does not mistake a
leading '-' for a new flag:
  python get_answer_cross_steering_mmlue.py \
      --model_dir meta-llama/Llama-3.1-8B-Instruct \
      --size 8B \
      --vector_dir /data1/paveen/Dopamine/components/llama3_confidence/cross_steering_mmlue/vectors \
      --mmlu_dir /data1/paveen/Dopamine/components/mmlu \
      --out_root /data1/paveen/Dopamine/components/llama3_confidence/cross_steering_mmlue/results \
      --condition RR --alphas="-4,-2,2,4"
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
CONDITIONS = ["RR", "RC", "CR", "CC", "RRand", "CRand"]
LABELS = ["A", "B", "C", "D", "E"]
PROTOCOL_VERSION = "cross-steering-mmlue-v1"


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


def is_task_complete(out_path: Path, mmlu_dir: Path, task: str) -> tuple:
    """Returns (is_complete: bool, reason: str, n_expected: int|None)."""
    src_path = mmlu_dir / f"{task}.json"
    try:
        src_data = utils.load_json(src_path)
    except Exception as e:
        return False, f"cannot read source MMLU file: {type(e).__name__}: {e}", None
    n_expected = len(src_data)

    if not out_path.exists():
        return False, "output file does not exist", n_expected

    try:
        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        return False, f"cannot read/parse output file: {type(e).__name__}: {e}", n_expected

    data = payload.get("data")
    if data is None:
        return False, "output file missing 'data' key", n_expected
    if len(data) != n_expected:
        return False, f"sample count mismatch: output has {len(data)}, source has {n_expected}", n_expected

    for i, sample in enumerate(data):
        for role in ROLES:
            role_key = role.replace(" ", "_")
            for field_prefix in ("answer_", "prob_", "softmax_", "logits_"):
                key = f"{field_prefix}{role_key}"
                if key not in sample:
                    return False, f"sample {i} missing field '{key}'", n_expected

    return True, "complete", n_expected


def is_cell_complete(out_dir: Path, mmlu_dir: Path, expected_meta: dict, size: str) -> tuple:
    """Returns (is_complete: bool, reason: str, incomplete_tasks: list[str]).
    Checks all 57 tasks + run_meta_{size}.json + metadata agreement."""
    incomplete_tasks = []
    for task in TASKS:
        out_path = out_dir / f"{task}_{size}_answers.json"
        ok, reason, _ = is_task_complete(out_path, mmlu_dir, task)
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

    # correct / E_count / wrong_non_E: wrong_non_E counts a PARSED, VALID
    # A-D answer that is simply wrong (renamed from "invalid" -- the old name
    # implied a format failure, but option_token_ids/argmax over exactly the
    # A-E option logits always returns one of the 5 valid labels; there is no
    # separate "prediction outside A-E" failure mode in this logits-argmax
    # extraction path, so what was called "invalid" is really "a wrong,
    # non-E, in-format answer").
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
    ap.add_argument("--vector_dir", required=True,
                     help="Dir containing steering_{RR,RC,CR,CC,RRand,CRand}_{size}.npy "
                          "from build_cross_steering_vectors.py")
    ap.add_argument("--mmlu_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--condition", required=True, choices=CONDITIONS + ["baseline"])
    ap.add_argument("--alphas", required=True,
                     help="Comma-separated alpha values, e.g. '-4,-2,2,4'. Pass as "
                          "--alphas=\"-4,-2,2,4\" (with '=') so argparse does not treat "
                          "a leading '-' as a new flag. alpha=0 is the all-zero baseline "
                          "and is written to {out_root}/baseline/alpha_0/ regardless of "
                          "--condition.")
    ap.add_argument("--tail_len", type=int, default=1)
    ap.add_argument("--use_chat", action="store_true", default=False,
                     help="Default False: bare-string, matching the confidence HS "
                          "collection and the mask-extraction convention.")
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
        layer_norms_unscaled = build_layer_norms(vec)
    else:
        vec_path = vector_dir / f"steering_{args.condition}_{args.size}.npy"
        vec = load_vector(vec_path)
        vec_sha = sha256_of_file(vec_path)
        layer_norms_unscaled = build_layer_norms(vec)

    tokenizer_info = {}

    # ---- Determine per-cell status BEFORE loading the model ----
    plan = []  # list of (alpha, cond_for_dir, out_dir, tasks_to_run, expected_meta)
    for alpha in alphas:
        cond_for_dir = "baseline" if alpha == 0.0 else args.condition
        out_dir = out_root / cond_for_dir / f"alpha_{alpha:g}"

        expected_meta = {
            "protocol_version": PROTOCOL_VERSION,
            "condition": cond_for_dir,
            "alpha": alpha,
            "model_dir": args.model_dir,
            "size": args.size,
            "steering_vector_sha256": vec_sha,
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
                print(f"[skip] {cond_for_dir}/alpha_{alpha:g}: already COMPLETE, "
                      f"passed full-task + sample-count + per-sample-field + metadata check -- not re-run.")
                continue
            if "FATAL" in reason:
                print(f"[REFUSE] {cond_for_dir}/alpha_{alpha:g}: {reason}")
                print("Refusing to mix two protocol configurations in one output directory. "
                      "Fix the mismatch or point --out_root at a fresh directory.")
                sys.exit(1)
            print(f"[resume] {cond_for_dir}/alpha_{alpha:g}: {reason}. "
                  f"Will complete ONLY the missing/incomplete tasks (already-complete tasks untouched): "
                  f"{incomplete_tasks}")
            tasks_to_run = incomplete_tasks
        else:
            tasks_to_run = list(TASKS)

        plan.append((alpha, cond_for_dir, out_dir, tasks_to_run, expected_meta))

    if not plan:
        print("Nothing to do -- every requested (condition, alpha) cell is already complete.")
        return

    print(f"Condition: {args.condition}")
    print(f"Alphas requested: {alphas}")
    print(f"Vector: {vec_path if vec_path else '(all-zero baseline)'}")
    if vec_sha:
        print(f"Vector sha256: {vec_sha}")
    for alpha, cond_for_dir, out_dir, tasks_to_run, _ in plan:
        print(f"  -> {cond_for_dir}/alpha_{alpha:g}: {len(tasks_to_run)} task(s) to run, out={out_dir}")

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    tokenizer_info = {
        "tokenizer_class": type(vc.tokenizer).__name__,
        "vocab_size": getattr(vc.tokenizer, "vocab_size", None),
    }

    for alpha, cond_for_dir, out_dir, tasks_to_run, expected_meta in plan:
        diff_mtx_full = (vec * alpha).astype(np.float32)
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

        # Recompute task_summary from disk for ALL 57 tasks (not just the ones
        # just run) so run_meta always reflects the true completed state of
        # the whole cell, including tasks that were already complete before
        # this invocation.
        task_summary = {}
        for task in TASKS:
            out_path = out_dir / f"{task}_{args.size}_answers.json"
            with open(out_path, "r", encoding="utf-8") as f:
                task_summary[task] = json.load(f)["accuracy"]

        meta = {
            **expected_meta,
            "provenance_note_RR": (
                "An RR-like condition was reported in an older results table under this "
                "project, but no protocol-complete, sample-level-reproducible source for "
                "it could be located (the historical logits_mdf_4/*_20_11_20.json files "
                "carry no E option in the rendered prompt, a null template field, and "
                "implausibly high invalid rates) -- so RR is RE-RUN from scratch here, "
                "not because it 'was never tested' but because its prior result could not "
                "be verified under this protocol."
            ),
            "steering_vector_path": str(vec_path) if vec_path else None,
            "steering_vector_layer_l2norms_unscaled": layer_norms_unscaled,
            "steering_vector_layer_l2norms_at_this_alpha": layer_norms_scaled,
            "injection_band_decoder_layers": "11-19",
            "model_and_tokenizer": {
                "model_dir": args.model_dir,
                **tokenizer_info,
            },
            "prompt_template": {
                "suite": "default",
                "use_E": True,
                "use_chat": args.use_chat,
                "note": "templates['neg'] via utils.construct_prompt for confident/unconfident "
                        "roles (any role containing 'confident' routes to the neg template); "
                        "bare-string, no chat template, no CoT.",
            },
            "task_list_n_tasks": len(TASKS),
            "task_list_source": "detection.task_list.TASKS",
            "decoding_config": {
                "method": "argmax over A-E option-token logits (regenerate_logits), "
                          "no sampling, temperature not applicable",
                "hook_mechanism": "VicundaModel.regenerate_logits -> _apply_diff_hooks, "
                                  "prefill-only, last tail_len prompt token(s)",
            },
            "task_summary": task_summary,
            "tasks_completed_this_invocation": tasks_to_run,
            "run_status": "complete" if len(tasks_to_run) == len(TASKS) or out_dir.exists() else "unknown",
        }
        # Overwrite run_meta with the fully up-to-date state (this file is
        # cheap metadata, not sample data -- rewriting it on every invocation,
        # including resume invocations, is intentional and safe; per-task
        # sample JSON files are never touched once complete).
        with open(out_dir / f"run_meta_{args.size}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"\nSaved run meta -> {out_dir / f'run_meta_{args.size}.json'}")

        # Final verification that the cell is now actually complete.
        complete, reason, incomplete_tasks = is_cell_complete(out_dir, mmlu_dir, expected_meta, args.size)
        if not complete:
            print(f"[WARNING] {cond_for_dir}/alpha_{alpha:g}: still incomplete after this run "
                  f"({reason}). Re-run the same command to complete remaining tasks: {incomplete_tasks}")
        else:
            print(f"[OK] {cond_for_dir}/alpha_{alpha:g}: verified complete (57/57 tasks).")

    print("\nAll requested alphas finished for condition:", args.condition)


if __name__ == "__main__":
    main()
