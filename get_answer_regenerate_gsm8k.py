#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSM8K open-ended generation evaluation with neuron editing (regenerate).

Key difference from get_answer_gsm8k.py:
- Uses model.regenerate() instead of model.generate()
- Loads NMD diff masks and applies them during generation
- The diff hooks modify hidden states at each decoding step

Key difference from multiple-choice regenerate scripts:
- Uses regenerate() (full text generation) instead of regenerate_logits() (last token only)
- Extracts numeric answer from generated text via regex

@author: paveenhuang
"""

import os
import re
import gc
import csv
import json
import argparse
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from llms import VicundaModel
from template import select_templates_gsm8k
import utils
import gsm8k_1shot


# Answer extraction / correctness imported from utils (canonical, shared across scripts).
from utils import extract_gsm8k_answer, is_correct_gsm8k


# ───────────────────── Per-config runner ─────────────────────

FEWSHOT = {}  # filled in main() only when --fewshot_file is set
GEN_RETURNED = [0]  # replies actually returned by vc.regenerate (summed over batches)
PROMPT_AUDIT = {}  # filled only with --save_prompt_audit: role -> rendered-prompt record


def audit_prompts(vc, role, prompts):
    """--save_prompt_audit: hash the RENDERED prompts and read the real tokenisation.
    Fail-closed: one BOS (when the tokenizer has one) and ONE uniform final token that
    decodes to the ASCII space of the `Answer: ` anchor. Does not touch generation."""
    import hashlib
    tk = vc.tokenizer
    bos, tails, bos_counts = tk.bos_token_id, {}, set()
    for p_ in prompts:
        ids = tk(p_)["input_ids"]
        bos_counts.add(ids.count(bos) if bos is not None else 0)
        tails[ids[-1]] = tails.get(ids[-1], 0) + 1
    if bos is not None and bos_counts != {1}:
        raise SystemExit(f"[FATAL] role={role}: BOS counts per prompt {sorted(bos_counts)}, expected exactly 1")
    if len(tails) != 1 or tk.decode([next(iter(tails))]) != " ":
        raise SystemExit(f"[FATAL] role={role}: final prompt token(s) {sorted(tails)} are not one id decoding to ' '")
    if not all(p_.endswith("Answer: ") and not p_.endswith("Answer:  ") for p_ in prompts):
        raise SystemExit(f"[FATAL] role={role}: prompts must end in exactly one ASCII space after 'Answer:'")
    PROMPT_AUDIT[role] = {
        "n_prompts": len(prompts),
        "prompts_sha256": hashlib.sha256("\x1f".join(prompts).encode("utf-8")).hexdigest(),
        "first_prompt": prompts[0],
        "bos_token_id": bos,
        "final_token": {f"{k}:{tk.decode([k])!r}": v for k, v in tails.items()},
        "prompts": prompts,
    }
    print(f"[prompt-audit] {role!r}: n={len(prompts)} final={PROMPT_AUDIT[role]['final_token']} "
          f"sha={PROMPT_AUDIT[role]['prompts_sha256'][:16]}")


def run_gsm8k_regenerate(
    vc: VicundaModel,
    samples: List[dict],
    diff_mtx: np.ndarray,
    templates: dict,
    roles: List[str],
    batch_size: int = 1,
):
    """
    Run GSM8K with neuron editing applied during generation.
    Returns updated samples and accuracy stats.
    batch_size > 1 only takes effect when prefill_only=False (hooks active during full generation).
    For prefill_only=True (default), generation is sequential per sample.
    """
    stats = {r: {"correct": 0, "total": 0} for r in roles}

    for role in roles:
        # GSM8K has no "I am not sure" (E) option, so roles must NOT carry the
        # "honest" framing (that belongs only to E-option MMLU suites). Bypass
        # construct_prompt's default(+honest) branch: neutral → neutral template,
        # any other role → neg template ("Now you are {role}."). This matches
        # track_dopamine_signal.py so the two scripts produce identical prompts.
        if args.fewshot_file:
            if role not in ("neutral", "norole"):
                raise SystemExit("[FATAL] --fewshot_file supports the neutral role only")
            _prefix, prompts = gsm8k_1shot.build_prompts(
                templates["neutral"], FEWSHOT["demo"], [s["question"] for s in samples])
            FEWSHOT["prompts_sha256"] = gsm8k_1shot.prompts_sha256(prompts)
            FEWSHOT["n_prompts"] = len(prompts)
            if FEWSHOT.get("tok_checked") is None:
                tk = vc.tokenizer
                bos, tails = tk.bos_token_id, {}
                for p_ in prompts:
                    ids = tk(p_)["input_ids"]
                    if bos is not None and ids.count(bos) != 1:
                        raise SystemExit(f"[FATAL] prompt has {ids.count(bos)} BOS tokens, expected 1")
                    tails[ids[-1]] = tails.get(ids[-1], 0) + 1
                if len(tails) != 1 or tk.decode([next(iter(tails))]) != " ":
                    raise SystemExit(f"[FATAL] prompt final token(s) {sorted(tails)} are not one id decoding to ' '")
                FEWSHOT["tok_checked"] = True
                FEWSHOT["final_token_counts"] = {f"{k}:{tk.decode([k])!r}": v for k, v in tails.items()}
                print("[1shot] final prefill token ids:", FEWSHOT["final_token_counts"])
        elif role in ("neutral", "norole"):
            prompts = [templates["neutral"].format(context=s["question"]) for s in samples]
        else:
            prompts = [templates["neg"].format(character=role, context=s["question"]) for s in samples]
        if args.save_prompt_audit:
            audit_prompts(vc, role, prompts)
        generated_texts = []
        gen_meta = []
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"GSM8K-regen [{role}]"):
            batch_prompts = prompts[i : i + batch_size]
            batch_out = vc.regenerate(
                batch_prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                diff_matrices=diff_mtx,
                batch_size=args.batch_size,
                return_metadata=args.save_gen_meta,
            )
            if len(batch_out) != len(batch_prompts):
                raise SystemExit(f"[FATAL] regenerate returned {len(batch_out)} replies for "
                                 f"{len(batch_prompts)} prompts (role={role}, batch@{i})")
            GEN_RETURNED[0] += len(batch_out)
            if args.save_gen_meta:
                gen_meta.extend(batch_out)
                batch_out = [m["text"] for m in batch_out]
            generated_texts.extend(batch_out)

        rk = role.replace(" ", "_")
        for j, (sample, generated) in enumerate(zip(samples, generated_texts)):
            if args.save_gen_meta:
                sample[f"gen_tokens_{rk}"] = gen_meta[j]["generated_token_count"]
                sample[f"stop_reason_{rk}"] = gen_meta[j]["stop_reason"]
            pred_answer = extract_gsm8k_answer(generated)
            correct = is_correct_gsm8k(pred_answer, sample["answer"])
            sample[f"generated_{rk}"] = generated
            sample[f"pred_answer_{rk}"] = pred_answer
            sample[f"correct_{rk}"] = correct
            st = stats[role]
            st["total"] += 1
            if correct:
                st["correct"] += 1

    accuracy = {}
    for role, s in stats.items():
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        accuracy[role] = {**s, "accuracy_percentage": round(pct, 2)}
        print(f"{role:<25} acc={pct:5.2f}%  ({s['correct']}/{s['total']})")

    return samples, accuracy


# ───────────────────── Main ─────────────────────

def main():
    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("ALPHAS_START_END_PAIRS:", ALPHAS_START_END_PAIRS)

    # Load model
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Load GSM8K data
    all_samples = utils.load_json(DATA_DIR)
    print(f"Loaded {len(all_samples)} GSM8K samples.")

    # Templates
    templates = select_templates_gsm8k(suite=args.suite, cot=args.cot, wording=args.fmt_wording)

    if args.fewshot_file:
        FEWSHOT["demo"] = gsm8k_1shot.load_demo(os.path.join(BASE, args.fewshot_file), DATA_DIR)
        print("[1shot] demo train row", FEWSHOT["demo"]["train_row_index"],
              "sha", FEWSHOT["demo"]["content_sha256"][:16])
        if args.cot or args.fmt_wording != "plain" or args.suite != "default":
            raise SystemExit("[FATAL] 1-shot protocol is No-CoT / plain / default suite only")

    # Roles
    custom_roles = None
    if args.roles:
        custom_roles = [r.strip() for r in args.roles.split(",")]
    roles = utils.make_characters("math", custom_roles)

    # Outer loop: each alpha / layer range
    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, mask_name)
        diff_mtx = np.load(mask_path) * alpha
        TOP = max(1, int(args.percentage / 100 * diff_mtx.shape[1]))
        print(f"\n=== alpha={alpha} | layers={st}-{en} | TOP={TOP} ===")

        # Deep copy samples for this config
        import copy
        config_samples = copy.deepcopy(all_samples)

        if args.save_gen_meta:
            vc.steering_fire_count(reset=True)
            GEN_RETURNED[0] = 0
        with torch.no_grad():
            updated_data, accuracy = run_gsm8k_regenerate(
                vc=vc,
                samples=config_samples,
                diff_mtx=diff_mtx,
                templates=templates,
                roles=roles,
                batch_size=args.batch_size,
            )

        # Save JSON
        out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"gsm8k_{args.size}_answers_{TOP}_{st}_{en}.json")
        # Record the template ACTUALLY used per role (mirror the routing above):
        # neutral→neutral, any other role→neg (no honest). Do NOT use
        # utils.record_template here — it logs the default(+honest) template for
        # non-neg roles, which would mislabel GSM8K results as "an honest expert"
        # when the real prompt is "an expert".
        tmp_record = [
            templates["neutral"] if role in ("neutral", "norole") else templates["neg"]
            for role in roles
        ]
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump(
                {"data": updated_data, "accuracy": accuracy, "template": tmp_record},
                fw, ensure_ascii=False, indent=2,
            )
        print("Saved ->", out_path)

        # Save CSV
        csv_rows = []
        for role, s in accuracy.items():
            csv_rows.append({
                "model": args.model,
                "size": args.size,
                "alpha": alpha,
                "start": st,
                "end": en,
                "TOP": TOP,
                "task": "gsm8k",
                "role": role,
                "correct": s["correct"],
                "total": s["total"],
                "accuracy_percentage": s["accuracy_percentage"],
                "suite": args.suite,
                "cot": int(bool(args.cot)),
            })

        csv_path = os.path.join(out_dir, f"summary_gsm8k_{args.model}_{args.size}_{TOP}_{st}_{en}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model", "size", "alpha", "start", "end", "TOP",
                    "task", "role", "correct", "total", "accuracy_percentage",
                    "suite", "cot",
                ],
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[Saved CSV] {csv_path}")

        if args.save_gen_meta:
            # Run record + fail-closed injection check (opt-in; default schema unchanged).
            import hashlib
            fires = vc.steering_fire_count(reset=True)
            # Independent of the mask file: expected comes from the CONFIGURED band.
            want_rows = list(utils.decoder_layer_range(st, en))
            raw_mask = np.load(mask_path)
            mask_rows = sorted(int(r) for r in np.nonzero(np.any(raw_mask != 0, axis=1))[0])
            n_pool = len(all_samples)
            n_gen = GEN_RETURNED[0]
            n_steered = len(want_rows) if alpha != 0 else 0
            expected = n_steered * n_pool * len(roles) * 1  # layers x generations x tail_len=1
            hdm = getattr(vc.model, "hf_device_map", None)
            meta = {
                "alpha": alpha, "layers": [st, en], "n_generated": n_gen,
                "n_pool": n_pool, "expect_n": args.expect_n,
                "mask_nonzero_rows": mask_rows, "configured_rows": want_rows,
                "n_steered_layers": n_steered, "steering_fires": fires,
                "steering_fires_expected": expected,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "sharded": bool(hdm) and len(set(map(str, hdm.values()))) > 1,
                "devices_used": sorted(set(map(str, hdm.values()))) if hdm else None,
                "mask_sha256": hashlib.sha256(open(mask_path, "rb").read()).hexdigest(),
            }
            if args.fewshot_file:
                demo = FEWSHOT["demo"]
                gens = [x.get("generated_neutral", "") for x in config_samples]
                meta.update({
                    "protocol": gsm8k_1shot.PROTOCOL,
                    "demo_source": {k: demo[k] for k in (
                        "dataset", "config", "split", "dataset_revision", "train_row_index")},
                    "demo_content_sha256": demo["content_sha256"],
                    "demo_question": demo["question"], "demo_answer": demo["answer"],
                    "prompt_template": templates["neutral"],
                    "prompt_template_sha256": gsm8k_1shot.template_sha256(templates["neutral"], demo),
                    "prompts_sha256": FEWSHOT["prompts_sha256"],
                    "final_token_counts": FEWSHOT["final_token_counts"],
                    "test_file_sha256": hashlib.sha256(open(DATA_DIR, "rb").read()).hexdigest(),
                    "model_dir": args.model_dir,
                    "generation": {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                                   "top_p": args.top_p, "batch_size": args.batch_size},
                    "n_generations_containing_demo_answer": sum(demo["answer"] in g for g in gens),
                })
            if args.save_prompt_audit:
                import transformers
                meta["host"] = os.uname().nodename
                meta["library_versions"] = {"torch": torch.__version__, "transformers": transformers.__version__}
                meta["model_dir"] = args.model_dir
                meta["generation"] = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                                      "top_p": args.top_p, "batch_size": args.batch_size}
                meta["suite"], meta["fmt_wording"], meta["cot"] = args.suite, args.fmt_wording, bool(args.cot)
                meta["roles"] = list(roles)
                meta["templates_used"] = {
                    r: (templates["neutral"] if r in ("neutral", "norole") else templates["neg"]) for r in roles}
                meta["test_file_sha256"] = hashlib.sha256(open(DATA_DIR, "rb").read()).hexdigest()
                meta["prompt_audit"] = {r: {k: v for k, v in a.items() if k != "prompts"}
                                        for r, a in PROMPT_AUDIT.items()}
                with open(os.path.join(out_dir, "rendered_prompts_gsm8k.json"), "w", encoding="utf-8") as fp_:
                    json.dump({r: a["prompts"] for r, a in PROMPT_AUDIT.items()}, fp_, ensure_ascii=False)
            meta_path = os.path.join(out_dir, f"run_meta_gsm8k_{args.size}_{TOP}_{st}_{en}.json")
            with open(meta_path, "w", encoding="utf-8") as fm:
                json.dump(meta, fm, indent=2)
            print("[run_meta]", meta_path, "fires", fires, "expected", expected, "n_gen", n_gen)
            problems = []
            if args.expect_n is None or n_pool != args.expect_n:
                problems.append(f"pool {n_pool} != expect_n {args.expect_n}")
            if mask_rows != want_rows:
                problems.append(f"mask nonzero rows {mask_rows} != configured band rows {want_rows}")
            if n_gen != n_pool * len(roles):
                problems.append(f"replies returned {n_gen} != {n_pool * len(roles)}")
            if fires != expected:
                problems.append(f"steering fires {fires} != expected {expected}")
            meta["checks_passed"] = not problems
            meta["problems"] = problems
            with open(meta_path, "w", encoding="utf-8") as fm:
                json.dump(meta, fm, indent=2)
            if problems:
                raise SystemExit(f"[FATAL] alpha={alpha}: " + "; ".join(problems))

        del csv_rows, config_samples
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll configs finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSM8K generation with neuron editing (regenerate)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--hs", type=str, required=True)
    parser.add_argument("--size", type=str, required=True)
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["4-16-22"],
                        help="List of alpha-start-end triplets")
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--ans_file", type=str, default="answer_mdf_gsm8k")
    parser.add_argument("--suite", type=str, default="default", choices=["default", "vanilla"])
    parser.add_argument("--fmt_wording", type=str, default="plain", choices=["plain", "pushy"],
                        help="#### directive phrasing (default suite only). 'plain' = main line; "
                             "'pushy' = early-#### positive-control ablation. ('plain' not 'neutral' "
                             "to avoid colliding with the neutral role.)")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--use_chat", action="store_true")
    parser.add_argument("--data", type=str, default="default", choices=["data1", "data2"])
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--roles", type=str, default=None)
    # Generation parameters
    parser.add_argument("--expect_n", type=int, default=None,
                        help="with --save_gen_meta: required pool size (run aborts if different)")
    parser.add_argument("--save_gen_meta", action="store_true",
                        help="store generation-time token count + stop_reason per sample (default off: output schema unchanged)")
    parser.add_argument("--fewshot_file", type=str, default=None,
                        help="gsm8k-base-1shot-v1: path (under base_dir) of the frozen demo JSON from "
                             "prepare_gsm8k_1shot_demo.py. Default off = original prompts, unchanged. "
                             "Requires --save_gen_meta.")
    parser.add_argument("--save_prompt_audit", action="store_true",
                        help="role-correspondence-v1: hash + tokenise the rendered prompts per role (fail-closed on "
                             "BOS / final-token / trailing-space) and record host, library versions, templates in "
                             "run_meta; writes rendered_prompts_gsm8k.json. Default off: output unchanged. "
                             "Requires --save_gen_meta.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of prompts per forward pass (set >1 for speedup)")

    args = parser.parse_args()
    if args.fewshot_file and not args.save_gen_meta:
        parser.error("--fewshot_file requires --save_gen_meta (fail-closed run_meta)")
    if args.save_prompt_audit and not args.save_gen_meta:
        parser.error("--save_prompt_audit requires --save_gen_meta (fail-closed run_meta)")

    print("Model:", args.model)
    print("Import model from:", args.model_dir)
    print("HS:", args.hs)
    print("Mask Type:", args.mask_type)

    # Path setup
    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/RolePlaying/components"

    DATA_DIR = os.path.join(BASE, args.test_file)
    MASK_DIR = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()
