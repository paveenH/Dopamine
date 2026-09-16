#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RRSN -> GSM8K own-direction positive-control generation.

Standalone, independent script. Does NOT modify get_answer_regenerate_gsm8k.py,
run_gsm8k.sh, run_gsm8k_qwen25.sh, or any existing GSM8K output tree. A new
script is needed because get_answer_regenerate_gsm8k.py can only load a mask
by the fixed filename convention {mask_type}_{percentage}_{start}_{end}_{size}.npy
under mask/{hs}_{type}_logits/ and multiply it by alpha -- it cannot load an
arbitrary pre-built, per-layer-norm-matched custom matrix. This script loads
the norm-matched RRSN mask built by
RoleHidden/build_rrsn_gsm8k_positive_control_mask.py directly and multiplies
it by alpha, otherwise reproducing get_answer_regenerate_gsm8k.py's frozen
GSM8K generation pipeline EXACTLY:

  - same benchmark file / question order (benchmark/gsm8k_test_sample.json)
  - same neutral, Bare (no chat template), No-CoT prompt
    (template.select_templates_gsm8k(suite="default", cot=False, wording="plain"))
  - same post-<|eot_id|> generation pipeline (VicundaModel._build_terminators())
  - same prefill-only, last-prompt-token injection (vc.regenerate(prefill_only=True,
    prefill_tail_len=1), the default)
  - same decoding (greedy, temperature=0.0), batch_size, and max_new_tokens
    as each model's existing frozen GSM8K No-CoT run (Llama: 768/bs=24,
    Qwen: 768/bs=24 -- both already 768/24 in run_gsm8k.sh / run_gsm8k_qwen25.sh)
  - the SAME per-model alpha grid as that model's existing frozen No-CoT sweep
    (Llama: -8,-6,-4,-2,0,2,4,6,8; Qwen: -8,-6,-4,-2,0,2,4,6,8,10,12)

The ONLY thing that differs from get_answer_regenerate_gsm8k.py's mechanics is
WHICH mask is loaded: this script loads the norm-matched RRSN mask from the
SERVER's standard mask tree,
  {base_dir}/mask/{model}_non_logits/rrsn_normmatched_0.5_<start>_<end>_{size}.npy
(a pre-built, per-layer-L2-norm-matched-to-this-model's-own-MRSN, signed RRSN
exact-NMD mask), deployed there as a plain copy of the array built and
verified locally by
RoleHidden/build_rrsn_gsm8k_positive_control_mask.py (output file
rrsn_scaled_{model}_{size}.npy in that script's own archive path -- see
RoleHidden/AdaResult/8.rrsn_gsm8k_positive_control/RUNBOOK.md for the
"local archive path" vs "server injection path" distinction). This mirrors
the on-server layout every other steering launcher in this repo uses
(mask/{model}_non_logits/nmd_0.5_<start>_<end>_<size>.npy for MRSN) instead
of a bespoke RoleHidden-relative path. The array CONTENT is unchanged by the
move -- only its filename and directory differ from the local archive copy
(never overwrites the existing nmd_0.5_*.npy MRSN mask files, which this
script never touches).

Output tree: components/{model}/gsm8k_rrsn_positive_control/mdf_<alpha>/
  gsm8k_rrsn_pc_{size}_answers_<top_k>_<start>_<end>.json
  run_config.json          (per-cell provenance: mask sha256, alpha, band,
                             steering_fires, prompt template, generation args)

Fail-closed checks performed by THIS script before/around generation:
  - mask file exists at {base_dir}/mask/{model}_non_logits/rrsn_normmatched_...npy,
    shape == (n_decoder_layers, hidden), matches this model's expected shape
  - mask sha256 (of the deployed server file's array content) matches the
    value recorded in rrsn_gsm8k_positive_control_mask_provenance.json (an
    artifact from the LOCAL build, read from --rolehidden_dir) -- refuses to
    proceed on mismatch. This verifies the deployed file is byte-identical
    (as an array) to the locally-built, locally-verified mask -- a filename/
    location change does not require rebuilding the mask or touching the
    provenance JSON itself.
  - alpha=0 must NOT register any steering fire (checked via
    vc.steering_fire_count() after the alpha=0 cell; VicundaModel builds a
    REAL all-zero diff_matrices at alpha=0, so hooks DO register -- fires
    read 0 only because every row is exactly zero)
  - alpha!=0 must register EXACTLY n_layers_in_band * n_samples * tail_len(=1)
    steering fires (prefill-only, last-token-only injection)
  - benchmark file question count/order recorded in provenance for the
    offline sample-alignment check across alpha (unified_behavior_* reuses
    this: same tree, same driver, alignment is verified there)
  - whole-cell overwrite guard: refuses to overwrite an existing output dir
    for this alpha unless --allow_overwrite is passed
"""
import argparse
import copy
import gc
import hashlib
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from llms import VicundaModel  # noqa: E402
from template import select_templates_gsm8k  # noqa: E402
import utils  # noqa: E402
from utils import extract_gsm8k_answer, is_correct_gsm8k  # noqa: E402

SCRIPT_VERSION = "get_answer_gsm8k_rrsn_positive_control-v1"

MODEL_SHAPE = {
    "llama3": {"size": "8B", "hidden": 4096, "n_decoder_layers": 32, "band": (11, 20)},
    "qwen2.5": {"size": "7B", "hidden": 3584, "n_decoder_layers": 28, "band": (16, 22)},
}

# Frozen per-model alpha grid -- SAME as each model's existing frozen No-CoT
# GSM8K sweep (run_gsm8k.sh / run_gsm8k_qwen25.sh). Not re-searched here.
ALPHA_GRID = {
    "llama3": [-8, -6, -4, -2, 0, 2, 4, 6, 8],
    "qwen2.5": [-8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12],
}

MAX_NEW_TOKENS = 768
TEMPERATURE = 0.0
TOP_P = 0.9
BATCH_SIZE = 24
SUITE = "default"
WORDING = "plain"
ANS_FILE = "gsm8k_rrsn_positive_control"


def die(msg, code=1):
    print(f"[REFUSE] {msg}", file=sys.stderr)
    sys.exit(code)


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_and_verify_mask(model, mask_dir, provenance_path):
    cfg = MODEL_SHAPE[model]
    band = cfg["band"]
    mask_filename = f"rrsn_normmatched_0.5_{band[0]}_{band[1]}_{cfg['size']}.npy"
    mask_path = os.path.join(mask_dir, mask_filename)
    if not os.path.exists(mask_path):
        die(f"mask not found: {mask_path} -- deploy it from the locally-built "
            f"rrsn_scaled_{model}_{cfg['size']}.npy (see "
            f"RoleHidden/build_rrsn_gsm8k_positive_control_mask.py and "
            f"RoleHidden/AdaResult/8.rrsn_gsm8k_positive_control/RUNBOOK.md)")
    mask = np.load(mask_path)
    expect_shape = (cfg["n_decoder_layers"], cfg["hidden"])
    if mask.shape != expect_shape:
        die(f"mask {mask_path}: shape {mask.shape} != expected {expect_shape}")
    if not np.all(np.isfinite(mask)):
        die(f"mask {mask_path}: contains NaN/Inf")

    # saved mask row r corresponds to raw layer r+1 (embedding dropped),
    # i.e. decoder_layer index r == raw layer (r+1); band [start,end) in raw
    # layer terms -> rows [start-1, end-1) nonzero.
    expected_nonzero_rows = set(range(band[0] - 1, band[1] - 1))
    actual_nonzero_rows = set(int(i) for i in np.flatnonzero(np.any(mask != 0, axis=1)))
    if actual_nonzero_rows != expected_nonzero_rows:
        die(f"mask {mask_path}: nonzero rows {sorted(actual_nonzero_rows)} != "
            f"expected {sorted(expected_nonzero_rows)} for band {band}")

    if not os.path.exists(provenance_path):
        die(f"mask-build provenance not found: {provenance_path} -- run "
            f"build_rrsn_gsm8k_positive_control_mask.py first")
    with open(provenance_path) as f:
        mask_prov = json.load(f)
    recorded_sha = mask_prov.get(model, {}).get("outputs", {}).get("rrsn_scaled_mask", {}).get("sha256")
    actual_sha = sha256_of_array(mask)
    if recorded_sha != actual_sha:
        die(f"mask {mask_path}: sha256 {actual_sha} does not match provenance-recorded "
            f"{recorded_sha} -- mask has changed since it was built/verified")
    print(f"  mask verified: {mask_path}")
    print(f"  mask shape={mask.shape}, sha256={actual_sha}, matches provenance -- PASS")
    return mask, actual_sha, mask_path


def run_one_alpha(vc, samples, diff_mtx, templates, batch_size):
    prompts = [templates["neutral"].format(context=s["question"]) for s in samples]
    generated_texts = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="GSM8K-RRSN-PC"):
        batch_prompts = prompts[i:i + batch_size]
        batch_out = vc.regenerate(
            batch_prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            diff_matrices=diff_mtx,
            batch_size=batch_size,
        )
        generated_texts.extend(batch_out)

    correct = 0
    for sample, generated in zip(samples, generated_texts):
        pred_answer = extract_gsm8k_answer(generated)
        is_ok = is_correct_gsm8k(pred_answer, sample["answer"])
        sample["generated_neutral"] = generated
        sample["pred_answer_neutral"] = pred_answer
        sample["correct_neutral"] = is_ok
        if is_ok:
            correct += 1
    inline_acc = correct / len(samples) * 100 if samples else 0.0
    return samples, inline_acc


def main():
    ap = argparse.ArgumentParser(description="RRSN -> GSM8K own-direction positive control (independent script)")
    ap.add_argument("--model", required=True, choices=list(MODEL_SHAPE.keys()))
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--base_dir", required=True,
                     help="server components dir, e.g. /data1/paveen/Dopamine/components")
    ap.add_argument("--rolehidden_dir", required=True,
                     help="path to the RoleHidden workspace holding "
                          "AdaResult/8.rrsn_gsm8k_positive_control/"
                          "rrsn_gsm8k_positive_control_mask_provenance.json "
                          "(the LOCAL build's provenance record, used only to "
                          "verify the deployed server mask's sha256 -- the mask "
                          "array itself is loaded from --base_dir/mask/, not "
                          "from this directory)")
    ap.add_argument("--gsm8k_file", default="benchmark/gsm8k_test_sample.json")
    ap.add_argument("--alphas", nargs="*", type=float, default=None,
                     help="subset of this model's frozen alpha grid to run "
                          "(default: the full frozen grid)")
    ap.add_argument("--allow_overwrite", action="store_true")
    args = ap.parse_args()

    cfg = MODEL_SHAPE[args.model]
    grid = ALPHA_GRID[args.model]
    if args.alphas is not None:
        alphas = []
        for a in args.alphas:
            a_int = int(a) if float(a).is_integer() else a
            if a_int not in grid:
                die(f"requested alpha {a_int} is not in the frozen grid for "
                    f"{args.model}: {grid}")
            alphas.append(a_int)
    else:
        alphas = grid

    mask_dir = os.path.join(args.base_dir, "mask", f"{args.model}_non_logits")
    provenance_path = os.path.join(args.rolehidden_dir, "AdaResult",
                                    "8.rrsn_gsm8k_positive_control",
                                    "rrsn_gsm8k_positive_control_mask_provenance.json")
    mask, mask_sha256, mask_path = load_and_verify_mask(args.model, mask_dir, provenance_path)

    gsm8k_path = os.path.join(args.base_dir, args.gsm8k_file)
    if not os.path.exists(gsm8k_path):
        die(f"benchmark file not found: {gsm8k_path}")
    all_samples = utils.load_json(gsm8k_path)
    print(f"Loaded {len(all_samples)} GSM8K samples from {gsm8k_path}")
    questions_sha256 = hashlib.sha256(
        json.dumps([s["question"] for s in all_samples], ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    templates = select_templates_gsm8k(suite=SUITE, cot=False, wording=WORDING)
    prompt_template_neutral = templates["neutral"]

    save_root = os.path.join(args.base_dir, args.model, ANS_FILE)
    os.makedirs(save_root, exist_ok=True)

    out_dirs = [os.path.join(save_root, f"mdf_{a}") for a in alphas]
    if not args.allow_overwrite:
        existing = [d for d in out_dirs if os.path.exists(d)]
        if existing:
            die("refusing to overwrite existing output dir(s) (pass --allow_overwrite "
                "to deliberately overwrite):\n  " + "\n  ".join(existing))

    print(f"\nModel: {args.model}  band: {cfg['band']}  mask_sha256: {mask_sha256}")
    print(f"Alpha grid this run: {alphas}")

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    n_layers_band = cfg["band"][1] - cfg["band"][0]

    for alpha in alphas:
        out_dir = os.path.join(save_root, f"mdf_{alpha}")
        os.makedirs(out_dir, exist_ok=True)

        diff_mtx = mask * alpha  # (n_decoder_layers, hidden); zeros outside band regardless of alpha
        print(f"\n=== alpha={alpha} | band={cfg['band']} ===")

        config_samples = copy.deepcopy(all_samples)

        vc.steering_fire_count(reset=True) if hasattr(vc, "steering_fire_count") else None
        with torch.no_grad():
            updated_data, inline_acc = run_one_alpha(vc, config_samples, diff_mtx, templates, BATCH_SIZE)

        fires = vc.steering_fire_count() if hasattr(vc, "steering_fire_count") else None
        expected_fires = 0 if alpha == 0 else n_layers_band * len(all_samples) * 1  # tail_len=1
        if fires is not None:
            if alpha == 0:
                if fires != 0:
                    die(f"alpha=0 registered {fires} steering fires, expected 0 "
                        f"(a real all-zero diff_matrices should not fire)")
            else:
                if fires != expected_fires:
                    die(f"alpha={alpha}: steering_fire_count={fires} != expected "
                        f"{expected_fires} (= n_layers_in_band * n_samples * tail_len)")
            print(f"  steering_fires={fires} (expected {expected_fires}) -- PASS")

        top_k = 20 if args.model == "llama3" else 17
        out_path = os.path.join(
            out_dir, f"gsm8k_rrsn_pc_{cfg['size']}_answers_{top_k}_{cfg['band'][0]}_{cfg['band'][1]}.json")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump({
                "data": updated_data,
                "inline_accuracy_percentage": round(inline_acc, 2),
                "template": prompt_template_neutral,
            }, fw, ensure_ascii=False, indent=2)
        print(f"  Saved -> {out_path}  (inline_acc={inline_acc:.2f}%, NOT the formal metric)")

        run_config = {
            "script": "get_answer_gsm8k_rrsn_positive_control.py",
            "script_version": SCRIPT_VERSION,
            "model": args.model, "model_dir": args.model_dir, "size": cfg["size"],
            "alpha": alpha, "band_raw_layers": list(cfg["band"]),
            "n_layers_in_band": n_layers_band,
            "mask_source": "RRSN exact-NMD, per-layer L2-norm-matched to this model's own MRSN "
                            "exact-NMD row (built by build_rrsn_gsm8k_positive_control_mask.py, "
                            "deployed to mask/{model}_non_logits/rrsn_normmatched_...npy)",
            "mask_path": mask_path,
            "mask_sha256": mask_sha256,
            "steering_fires": fires, "expected_steering_fires": expected_fires,
            "prompt_template_neutral": prompt_template_neutral,
            "suite": SUITE, "cot": False, "wording": WORDING, "role": "neutral",
            "prefill_only": True, "prefill_tail_len": 1,
            "max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE, "top_p": TOP_P,
            "batch_size": BATCH_SIZE,
            "n_samples": len(all_samples), "questions_sha256": questions_sha256,
            "gsm8k_file": args.gsm8k_file,
        }
        with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as fw:
            json.dump(run_config, fw, ensure_ascii=False, indent=2)

        del config_samples
        gc.collect()
        torch.cuda.empty_cache()

    print("\n[DONE] RRSN GSM8K positive-control generation finished for this invocation. "
          "Accuracy above is INLINE / process-state only -- formal accuracy must be "
          "computed offline by eval_gsm8k_rrsn_positive_control.py using the frozen "
          "GSM8K extractor.")


if __name__ == "__main__":
    main()
