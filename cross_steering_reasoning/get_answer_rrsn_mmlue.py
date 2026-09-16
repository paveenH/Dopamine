#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RRSN -> MMLU-E reverse-transfer steering experiment.

Question: does RRSN (Reasoning RSN, = unweighted mean of the 3 per-task
GSM8K/MATH/GSM-Hard Expert-Non-expert directions, raw sign), reduced to its
exact-NMD sparse support over this model's EXISTING MMLU-E-matched band and
per-layer L2-norm-matched to this model's own MRSN exact-NMD support (the
SAME norm-matched mask already deployed for the RRSN->GSM8K positive
control), change MMLU-E accuracy / E-rate / conditional accuracy / role gap
when injected the same way MRSN is steered?

This is CROSS-TASK REVERSE transfer (GSM8K/MATH/GSM-Hard participated in
RRSN's construction; MMLU-E did not) and must be interpreted jointly with the
RRSN->GSM8K positive control's result, never in isolation. A null on MMLU-E
is only interpretable as a transfer failure IF the GSM8K positive control
itself succeeded.

FROZEN THIS ROUND (per explicit user decision, do not re-derive or widen):
  - alpha grid = {-4, 0, +4} ONLY, both models. This is a NEW, independently
    frozen, low-cost pilot grid -- it is NOT read from, and does not claim to
    reproduce, any historical "formal" MRSN->MMLU-E alpha grid. The only
    located historical MMLU-E artifact with the "expert"/"E option" protocol
    (run_mmlue_qwen25.sh) was never actually run (zero artifacts; see
    CLAUDE.md's Qwen band-position-probe entry) and covered only Qwen with
    doses {-4,0,+4} at a DIFFERENT band -- it is cited here only as evidence
    that no wider formal grid exists to inherit, not as this experiment's
    source.
  - alpha=0 is RE-RUN fresh in this experiment's own output tree, never
    copied from any historical MRSN or RR/CC cell.
  - Prompt wording is the ORIGINAL, UNMODIFIED get_answer_regenerate_logits.py
    / utils.construct_prompt() bare-string path, run EXACTLY as
    run_mmlue_qwen25.sh already configures it:
      select_templates(suite="default", use_E=True)
      roles = ["{task} expert", "non {task} expert"]  (utils.make_characters)
      utils.construct_prompt(..., use_chat=False)
    This renders (verified, see verify_prompt_rendering() below and the
    printed/saved sample prompts):
      "Now you are an honest {task} expert, your answer among ... is: "
      "Now you are an honest non {task} expert, your answer among ... is: "
    BOTH roles are deliberately given "an honest" -- this is NOT a bug and
    is NOT patched. Per explicit user decision: "honest" is a shared modifier
    about whether the ANSWER is truthful, not a claim about the non-expert's
    competence, and it is consistent with MMLU-E's own E) "I am not sure"
    uncertainty/refusal design. construct_prompt() itself is NOT modified,
    NOT bypassed, and NOT swapped for templates["neg"] -- doing either would
    change the original MMLU-E prompt this script is required to reproduce
    exactly. use_chat is NEVER set True in this script.

RRSN mask: reuses the EXACT SAME per-layer-norm-matched mask already built
and deployed for the RRSN->GSM8K positive control (see
RoleHidden/build_rrsn_gsm8k_positive_control_mask.py and
gsm8k_rrsn_positive_control/get_answer_gsm8k_rrsn_positive_control.py) --
loaded from the server's standard mask tree using the SAME filename
convention every other steering mask in this repo uses:
  {mask_dir}/rrsn_normmatched_0.5_<start>_<end>_<size>.npy
    llama3:   [11,20), top_k=20, (32,4096)
    qwen2.5:  [16,22), top_k=17, (28,3584)
This script does NOT rebuild the mask and does NOT modify it. It also does
NOT touch nmd_0.5_*.npy (MRSN), confidence_0.5_*.npy, or any existing
MMLU/MMLU-E/MRSN/confidence-steering result.

Output layout, independent of every existing MMLU-E tree (mirrors the RR/CC
script's fail-closed, resumable-at-task-level design):
  {out_root}/alpha_{alpha}/{task}_{size}_answers.json
  {out_root}/alpha_{alpha}/run_meta_{size}.json
A cell is COMPLETE only if all 57 tasks are present with the expected sample
count and full expert/non-expert fields, AND run_meta matches the current
invocation's config exactly (a mismatch is a FATAL refusal, not a silent
resume). An incomplete cell has only its missing tasks completed.

Usage (server, conda env, interpreter is `python`, NOT python3.10):
  python get_answer_rrsn_mmlue.py \
      --model llama3 --model_dir meta-llama/Llama-3.1-8B-Instruct --size 8B \
      --mask_dir /data1/paveen/Dopamine/components/mask/llama3_non_logits \
      --mmlu_dir /data1/paveen/Dopamine/components/mmlu \
      --out_root /data1/paveen/Dopamine/components/llama3/rrsn_mmlue \
      --alphas="-4,0,4"

Negative alphas must be passed with '=' (--alphas="-4,0,4") so argparse does
not treat a leading '-' as a new flag.

--verify_only prints and saves ONE rendered expert + ONE rendered
non-expert prompt (no model load, no generation) so the exact wording can be
inspected before any GPU time is spent.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from detection.task_list import TASKS  # noqa: E402
from template import select_templates  # noqa: E402
import utils  # noqa: E402

PROTOCOL_VERSION = "rrsn-mmlue-v1"

MODEL_SHAPE = {
    "llama3": {"size": "8B", "hidden": 4096, "n_decoder_layers": 32, "band": (11, 20), "top_k": 20},
    "qwen2.5": {"size": "7B", "hidden": 3584, "n_decoder_layers": 28, "band": (16, 22), "top_k": 17},
}

# Frozen this round -- see module docstring. NOT read from any historical
# "formal" grid; NOT re-searched; NOT widened without a new explicit decision.
ALPHA_GRID = [-4, 0, 4]

ROLE_TEMPLATES = ["{task} expert", "non {task} expert"]


def die(msg, code=1):
    print(f"[REFUSE] {msg}", file=sys.stderr)
    sys.exit(code)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_and_verify_mask(model: str, mask_dir: Path) -> tuple:
    """Loads the deployed RRSN norm-matched mask and verifies its structure
    directly from the array content -- same convention as
    gsm8k_rrsn_positive_control/get_answer_gsm8k_rrsn_positive_control.py's
    load_and_verify_mask(): no external provenance file is read or required
    at runtime."""
    cfg = MODEL_SHAPE[model]
    band = cfg["band"]
    top_k = cfg["top_k"]
    mask_filename = f"rrsn_normmatched_0.5_{band[0]}_{band[1]}_{cfg['size']}.npy"
    mask_path = mask_dir / mask_filename
    if not mask_path.exists():
        die(f"mask not found: {mask_path} -- deploy the RRSN-GSM8K-positive-control "
            f"mask under this filename first (see "
            f"gsm8k_rrsn_positive_control/get_answer_gsm8k_rrsn_positive_control.py)")
    mask = np.load(mask_path)
    expect_shape = (cfg["n_decoder_layers"], cfg["hidden"])
    if mask.shape != expect_shape:
        die(f"mask {mask_path}: shape {mask.shape} != expected {expect_shape}")
    if not np.all(np.isfinite(mask)):
        die(f"mask {mask_path}: contains NaN/Inf")

    # saved mask row r corresponds to raw layer r+1 (embedding dropped);
    # band [start,end) in raw layer terms -> rows [start-1, end-1) nonzero.
    expected_nonzero_rows = set(range(band[0] - 1, band[1] - 1))
    actual_nonzero_rows = set(int(i) for i in np.flatnonzero(np.any(mask != 0, axis=1)))
    if actual_nonzero_rows != expected_nonzero_rows:
        die(f"mask {mask_path}: nonzero rows {sorted(actual_nonzero_rows)} != "
            f"expected {sorted(expected_nonzero_rows)} for band {band}")

    for r in range(mask.shape[0]):
        nnz = int(np.count_nonzero(mask[r]))
        expect = top_k if r in expected_nonzero_rows else 0
        if nnz != expect:
            die(f"mask {mask_path}: row {r} has {nnz} nonzero entries, "
                f"expected {expect}")

    sha = sha256_of_array(mask)
    print(f"  mask verified: {mask_path}")
    print(f"  shape={mask.shape}  sha256={sha}")
    print(f"  band {band}: in-band rows exactly top_k={top_k} nonzero, "
          f"out-of-band rows all zero -- PASS")
    return mask, sha, str(mask_path)


def build_layer_norms(vec: np.ndarray) -> dict:
    return {str(i): float(np.linalg.norm(vec[i])) for i in range(vec.shape[0])}


def render_sample_prompts(task_sample_text: str = "What is 2 + 2?"):
    """Renders and returns ONE expert and ONE non-expert prompt via the
    UNMODIFIED select_templates + utils.construct_prompt path, so the exact
    wording can be verified/logged before any generation. Matches
    run_mmlue_qwen25.sh's configuration exactly: suite=default, use_E=True,
    roles=["{task} expert","non {task} expert"], use_chat=False."""
    templates = select_templates("default", use_E=True)
    task = "abstract_algebra"
    roles = utils.make_characters(task, ROLE_TEMPLATES)

    class _NoVC:
        pass

    vc = _NoVC()
    rendered = {}
    for role in roles:
        prompt = utils.construct_prompt(vc, templates, task_sample_text, role, use_chat=False)
        rendered[role] = prompt
    return templates, roles, rendered


def is_task_complete(out_path: Path, mmlu_dir: Path, task: str, roles_generic: list) -> tuple:
    """roles_generic: the GENERIC role strings ("{task} expert",
    "non {task} expert") -- role_key in the saved JSON is task-specific
    (e.g. "abstract_algebra_expert"), so this recomputes the actual
    per-task role list to check field presence."""
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

    task_roles = utils.make_characters(task, roles_generic)
    for i, sample in enumerate(data):
        for role in task_roles:
            role_key = role.replace(" ", "_")
            for field_prefix in ("answer_", "prob_", "softmax_", "logits_"):
                key = f"{field_prefix}{role_key}"
                if key not in sample:
                    return False, f"sample {i} missing field '{key}' (role={role!r})"
    return True, "complete"


def _read_json_or_none(path: Path):
    if not path.exists():
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, f"cannot read/parse {path.name}: {type(e).__name__}: {e}"


def check_config_matches(out_dir: Path, expected_meta: dict, size: str) -> tuple:
    """Checks run_config_{size}.json (written BEFORE any generation for a
    cell starts, containing ONLY expected_meta) against expected_meta.

    run_config.json, not run_meta.json, is the authoritative "has this cell
    been started under this config" record -- run_meta is only written AFTER
    a cell finishes generating (it also carries post-hoc fields like
    task_summary that don't exist until generation is done), so relying on
    run_meta to gate a partial-task resume left a window where run_meta is
    absent but partial task outputs already exist under some prior config.

    Returns (status, reason):
      status="new"       -- no run_config.json AND no task outputs at all:
                             safe to start this cell fresh.
      status="match"      -- run_config.json exists and matches: safe to
                             resume at task granularity.
      status="fatal"      -- run_config.json exists and disagrees, OR task
                             outputs exist with no run_config.json at all
                             (a cell that predates this fail-closed design,
                             or one interrupted before run_config.json could
                             be written) -- never silently reused.
    """
    config_path = out_dir / f"run_config_{size}.json"
    existing_config, read_err = _read_json_or_none(config_path)
    if read_err is not None:
        return "fatal", read_err

    if existing_config is not None:
        mismatch_keys = []
        for key, expected_val in expected_meta.items():
            if existing_config.get(key) != expected_val:
                mismatch_keys.append((key, existing_config.get(key), expected_val))
        if mismatch_keys:
            detail = "; ".join(f"{k}: stored={s!r} vs current={c!r}" for k, s, c in mismatch_keys)
            return "fatal", f"run_config metadata mismatch (FATAL, not auto-resumable): {detail}"
        return "match", "run_config matches"

    # No run_config.json. If ANY task output already exists in this
    # directory, it was produced by a run that predates run_config.json (or
    # was interrupted before writing it) -- its config cannot be verified,
    # so it must never be silently reused/completed.
    if out_dir.exists():
        stray = [p for p in out_dir.glob(f"*_{size}_answers.json")]
        if stray:
            names = sorted(p.name for p in stray)
            return "fatal", (f"{len(names)} task output file(s) exist with NO run_config.json "
                              f"to verify their config against (FATAL, not auto-resumable): "
                              f"{names[:5]}{'...' if len(names) > 5 else ''}")

    return "new", "no run_config.json and no task outputs -- safe to start fresh"


def is_cell_complete(out_dir: Path, mmlu_dir: Path, expected_meta: dict, size: str,
                      roles_generic: list) -> tuple:
    """Returns (complete, reason, incomplete_tasks). Callers must check
    check_config_matches() themselves for the "fatal" / "new" cases BEFORE
    calling this -- this function assumes the config has already been
    verified to match (status="match") or is being checked as a
    just-completed cell (config already written this invocation)."""
    incomplete_tasks = []
    for task in TASKS:
        out_path = out_dir / f"{task}_{size}_answers.json"
        ok, _ = is_task_complete(out_path, mmlu_dir, task, roles_generic)
        if not ok:
            incomplete_tasks.append(task)
    if incomplete_tasks:
        return False, f"{len(incomplete_tasks)} task(s) incomplete", incomplete_tasks

    meta_path = out_dir / f"run_meta_{size}.json"
    if not meta_path.exists():
        return False, "run_meta file missing", []

    return True, "complete", []


def run_task(vc, task: str, diff_mtx: list, mmlu_dir: Path, tail_len: int) -> tuple:
    templates = select_templates("default", use_E=True)
    labels = templates["labels"]
    opt_ids = utils.option_token_ids(vc, labels)

    data_path = mmlu_dir / f"{task}.json"
    data = utils.load_json(data_path)
    roles = utils.make_characters(task, ROLE_TEMPLATES)

    # wrong_non_E: a parsed, valid, non-E, WRONG answer -- NOT "invalid
    # format". argmax over exactly the A-E option logits always returns one
    # of the 5 valid labels.
    stats = {r: {"correct": 0, "E_count": 0, "wrong_non_E": 0, "total": 0} for r in roles}

    for sample in tqdm(data, desc=task, leave=False):
        ctx = sample.get("text", "")
        true_idx = sample.get("label", -1)
        true_lab = labels[true_idx] if 0 <= true_idx < len(labels) else None

        for role in roles:
            prompt = utils.construct_prompt(vc, templates, ctx, role, use_chat=False)
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


def main():
    ap = argparse.ArgumentParser(description="RRSN -> MMLU-E reverse-transfer steering")
    ap.add_argument("--model", required=True, choices=list(MODEL_SHAPE.keys()))
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--size", default=None, help="defaults to MODEL_SHAPE[model]['size']")
    ap.add_argument("--mask_dir", required=True,
                     help="dir containing rrsn_normmatched_0.5_<start>_<end>_<size>.npy "
                          "(the SAME server mask tree used by every other steering "
                          "launcher, e.g. components/mask/{model}_non_logits)")
    ap.add_argument("--mmlu_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--alphas", default=None,
                     help="Comma-separated subset of the frozen grid {-4,0,4}, e.g. "
                          "--alphas=\"-4,0,4\" (default: the full frozen grid). Pass "
                          "with '=' so argparse does not treat a leading '-' as a flag.")
    ap.add_argument("--tail_len", type=int, default=1)
    ap.add_argument("--verify_only", action="store_true",
                     help="Render and print/save one expert + one non-expert prompt, "
                          "then exit. No model load, no generation.")
    args = ap.parse_args()

    cfg = MODEL_SHAPE[args.model]
    size = args.size or cfg["size"]

    if args.verify_only:
        templates, roles, rendered = render_sample_prompts()
        print("labels:", templates["labels"])
        print("roles (rendered by utils.make_characters):", roles)
        out = {"labels": templates["labels"], "roles": roles, "rendered_prompts": rendered}
        for role, prompt in rendered.items():
            print(f"\n--- role={role!r} ---")
            print(prompt)
        out_dir = Path(args.out_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "prompt_verification.json"
        if out_path.exists():
            die(f"refusing to overwrite existing verification file: {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[wrote] {out_path}")
        return

    if args.alphas is not None:
        alphas = []
        for tok in args.alphas.split(","):
            a = int(float(tok)) if float(tok).is_integer() else float(tok)
            if a not in ALPHA_GRID:
                die(f"requested alpha {a} is not in the frozen grid {ALPHA_GRID}")
            alphas.append(a)
    else:
        alphas = list(ALPHA_GRID)

    mmlu_dir = Path(args.mmlu_dir)
    out_root = Path(args.out_root)
    mask_dir = Path(args.mask_dir)

    if not mmlu_dir.exists():
        die(f"mmlu_dir not found: {mmlu_dir}")
    missing = [t for t in TASKS if not (mmlu_dir / f"{t}.json").exists()]
    if missing:
        die(f"{len(missing)} MMLU task JSON files missing under {mmlu_dir}: {missing[:5]}...")

    mask, mask_sha, mask_path_str = load_and_verify_mask(args.model, mask_dir)

    from llms import VicundaModel  # deferred: only needed once we actually generate

    layer_norms_unscaled = build_layer_norms(mask.astype(np.float64))

    n_layers_band = cfg["band"][1] - cfg["band"][0]

    plan = []
    for alpha in alphas:
        out_dir = out_root / f"alpha_{alpha:g}"

        expected_meta = {
            "protocol_version": PROTOCOL_VERSION,
            "model": args.model,
            "alpha": alpha,
            "model_dir": args.model_dir,
            "size": size,
            "mask_path": mask_path_str,
            "mask_sha256": mask_sha,
            "band_raw_layers": list(cfg["band"]),
            "top_k": cfg["top_k"],
            "role_templates": ROLE_TEMPLATES,
            "suite": "default",
            "use_E": True,
            "use_chat": False,
            "tail_len": args.tail_len,
            "n_tasks": len(TASKS),
        }

        # run_config_{size}.json (containing ONLY expected_meta, written
        # BEFORE any generation starts -- see the generation loop below) is
        # the authoritative record of what config a cell was started under.
        # Checked here BEFORE any task-level completeness/resume decision,
        # so a cell with partial task outputs but no verifiable config is
        # never silently reused, and a cell with all 57 tasks present but no
        # run_meta yet does not fall through to an empty tasks_to_run.
        config_status, config_reason = check_config_matches(out_dir, expected_meta, size)
        if config_status == "fatal":
            die(f"alpha_{alpha:g}: {config_reason}")
        elif config_status == "new":
            tasks_to_run = list(TASKS)
        else:  # "match"
            complete, reason, incomplete_tasks = is_cell_complete(
                out_dir, mmlu_dir, expected_meta, size, ROLE_TEMPLATES)
            if complete:
                print(f"[skip] alpha_{alpha:g}: already COMPLETE -- not re-run.")
                continue
            print(f"[resume] alpha_{alpha:g}: {reason}. Completing only: {incomplete_tasks}")
            tasks_to_run = incomplete_tasks

        plan.append((alpha, out_dir, tasks_to_run, expected_meta))

    if not plan:
        print("Nothing to do -- every requested cell is already complete.")
        return

    print(f"\nModel: {args.model}  band: {cfg['band']}  mask_sha256: {mask_sha}")
    print(f"Alphas requested: {alphas}")
    for alpha, out_dir, tasks_to_run, _ in plan:
        print(f"  -> alpha_{alpha:g}: {len(tasks_to_run)} task(s) to run, out={out_dir}")

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    tokenizer_info = {"tokenizer_class": type(vc.tokenizer).__name__,
                       "vocab_size": getattr(vc.tokenizer, "vocab_size", None)}

    # Cross-confirm rendered prompt on the REAL tokenizer/vc before any
    # generation for this invocation, matching the label+role text this run
    # will actually use.
    templates_check, roles_check, rendered_check = render_sample_prompts()
    print("\n=== Rendered prompt cross-check (real construct_prompt path) ===")
    for role, prompt in rendered_check.items():
        print(f"--- role={role!r} ---\n{prompt}\n")

    for alpha, out_dir, tasks_to_run, expected_meta in plan:
        diff_mtx_full = (mask.astype(np.float32) * alpha).astype(np.float32)
        diff_matrices_list = [diff_mtx_full[i] for i in range(cfg["n_decoder_layers"])]
        layer_norms_scaled = build_layer_norms(diff_mtx_full)

        out_dir.mkdir(parents=True, exist_ok=True)

        if not hasattr(vc, "steering_fire_count"):
            die("VicundaModel has no steering_fire_count() -- cannot verify steering "
                "actually fired as expected. This is a hard requirement, not optional "
                "instrumentation; refusing rather than generating unverified data.")

        for task in tasks_to_run:
            print(f"\n=== alpha={alpha} task={task} ===")
            vc.steering_fire_count(reset=True)
            with torch.no_grad():
                updated_data, accuracy = run_task(vc, task, diff_matrices_list, mmlu_dir, args.tail_len)
            fires = vc.steering_fire_count()
            n_roles = 2  # expert + non-expert
            n_samples = len(updated_data)
            expected_fires = 0 if alpha == 0 else n_layers_band * n_samples * n_roles * args.tail_len
            if alpha == 0:
                if fires != 0:
                    die(f"alpha=0 task={task}: registered {fires} steering fires, expected 0")
            else:
                if fires != expected_fires:
                    die(f"alpha={alpha} task={task}: steering_fire_count={fires} != "
                        f"expected {expected_fires} (= n_layers_in_band * n_samples * n_roles * tail_len)")

            out_path = out_dir / f"{task}_{size}_answers.json"
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump({"data": updated_data, "accuracy": accuracy}, fw, ensure_ascii=False, indent=2)
            for role, s in accuracy.items():
                print(f"  {role:<40} acc={s['accuracy_percentage']:5.2f}%  "
                      f"(correct {s['correct']}/{s['total']}), E={s['E_count']}, "
                      f"wrong_non_E={s['wrong_non_E']}")
            del updated_data, accuracy
            gc.collect()
            torch.cuda.empty_cache()

        task_summary = {}
        for task in TASKS:
            out_path = out_dir / f"{task}_{size}_answers.json"
            with open(out_path, "r", encoding="utf-8") as f:
                task_summary[task] = json.load(f)["accuracy"]

        meta = {
            **expected_meta,
            "note": "RRSN->MMLU-E reverse-transfer steering. Mask = RRSN exact-NMD, "
                    "per-layer L2-norm-matched to this model's own MRSN exact-NMD "
                    "row (SAME mask already deployed for the RRSN->GSM8K positive "
                    "control), used AS-IS (diff_mtx = mask * alpha). Roles rendered "
                    "via the UNMODIFIED select_templates(suite='default', use_E=True) "
                    "+ utils.construct_prompt() bare-string path, EXACTLY matching "
                    "run_mmlue_qwen25.sh's configuration -- both expert and "
                    "non-expert roles render with 'an honest {role}' by design "
                    "(per explicit decision: 'honest' modifies answer truthfulness, "
                    "not claimed expertise; consistent with MMLU-E's own E) 'I am "
                    "not sure' uncertainty/refusal design). alpha grid {-4,0,4} is "
                    "a NEW independently frozen pilot grid, NOT read from any "
                    "historical formal MRSN->MMLU-E grid (none exists with a "
                    "verified matching protocol for these two models).",
            "mask_layer_l2norms_unscaled": layer_norms_unscaled,
            "mask_layer_l2norms_at_this_alpha": layer_norms_scaled,
            "injection_band_raw_layers": list(cfg["band"]),
            "model_and_tokenizer": {"model_dir": args.model_dir, **tokenizer_info},
            "rendered_prompt_cross_check": rendered_check,
            "decoding_config": {
                "method": "argmax over A-E option-token logits (regenerate_logits), no sampling",
                "hook_mechanism": "VicundaModel.regenerate_logits -> _apply_diff_hooks, "
                                  "prefill-only, last tail_len prompt token(s)",
            },
            "steering_fires_this_invocation": fires,
            "expected_steering_fires": expected_fires,
            "task_summary": task_summary,
            "tasks_completed_this_invocation": tasks_to_run,
        }
        with open(out_dir / f"run_meta_{size}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"\nSaved run meta -> {out_dir / f'run_meta_{size}.json'}")

        complete, reason, incomplete_tasks = is_cell_complete(
            out_dir, mmlu_dir, expected_meta, size, ROLE_TEMPLATES)
        if not complete:
            print(f"[WARNING] alpha_{alpha:g}: still incomplete ({reason}). "
                  f"Re-run to complete: {incomplete_tasks}")
        else:
            print(f"[OK] alpha_{alpha:g}: verified complete (57/57 tasks).")

    print("\nAll requested alphas finished.")


if __name__ == "__main__":
    main()
