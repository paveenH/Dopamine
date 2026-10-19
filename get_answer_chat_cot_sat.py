#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_answer_chat_cot_sat.py -- Chat-CoT Saturation Check (protocol chat-cot-sat-v1), GSM8K / GSM-Hard x Llama / Qwen.

QUESTION: under the existing Native Chat interface, does adding explicit CoT leave room for RSN to change
output organisation and accuracy?  Added AFTER seeing the Native Chat No-CoT results; "saturation" is a
hypothesis here, never a fact.  Protocol: docs/PREREG_CHAT_COT_SAT_V1.md.

ONE script for the 4 (model x task) combinations (the ProofWriter-chat precedent): model facts live in the
MODELS table of chat_cot_sat_lib.py, not in knobs.  The existing Native Chat runners are NOT modified; their
verified helpers are IMPORTED (strip_leading_bos, assert_no_double_bos, die, GSM-Hard label-free loader).

Prompt = the existing Native Chat construction (neutral GSM8K body -> sole user turn ->
tokenizer.apply_chat_template(add_generation_prompt=True) -> duplicated BOS stripped) with the CoT template
body, which differs from the No-CoT body by exactly ONE line, "Let's think step by step.".  This is verified
three ways at run time: template level, per-prompt level, and WRAPPED-chat-string level (removing the line
from every wrapped CoT prompt must reproduce the wrapped No-CoT prompt byte for byte).  No system message
is added (the model's own template default is kept), no `Answer: ` anchor, no role, no few-shot.
Injection = prefill-only, tail_len=1, i.e. the LAST token of the assistant header; its id is READ from the real
tokenizer for ALL prompts (never the Bare token 220).

Modes:  --mode check  (everything except generation; writes nothing)
        --mode smoke  (first --smoke_n items, both alphas, to a SEPARATE _smoke dir)
        --mode run    (full 300 x 2 alphas, one model load)
Existing cells are skipped ONLY if verify_existing_cell() passes (same protocol/alpha/mask/prompts/fires/rows);
otherwise the run dies.  Hardware is recorded, never enforced.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys

import chat_cot_sat_lib as L
import utils
from get_answer_gsm8k_chat_sweep import assert_no_double_bos, die, strip_leading_bos
from template import select_templates_gsm8k
from utils import extract_gsm8k_answer, is_correct_gsm8k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(L.MODELS))
    ap.add_argument("--task", required=True, choices=sorted(L.TASKS))
    ap.add_argument("--mode", required=True, choices=["check", "smoke", "run"])
    ap.add_argument("--base_dir", required=True, help="components/ directory (benchmark/, mask/, <model>/ live here)")
    ap.add_argument("--smoke_n", type=int, default=8)
    ap.add_argument("--model_dir", default=None, help="override HF repo id (default from MODELS)")
    args = ap.parse_args()

    import numpy as np
    from tqdm import tqdm
    from llms import VicundaModel

    cfg, tcfg = L.MODELS[args.model], L.TASKS[args.task]
    model_dir = args.model_dir or cfg["model_dir"]
    smoke = args.mode == "smoke"
    q_path = os.path.join(args.base_dir, tcfg["questions_rel"])
    mask_path = os.path.join(args.base_dir, cfg["mask_rel"])
    out_dir = os.path.join(args.base_dir, cfg["out_name"], tcfg["out_subdir"] + ("_smoke" if smoke else ""))
    ls, le = cfg["band"]

    # ---- questions (GSM-Hard: label-free loader reused verbatim)
    if tcfg["label_free"]:
        from get_answer_gsm_hard_chat_sweep import EXPECTED_QUESTIONS_SHA256, load_questions
        qmeta, samples = load_questions(q_path, EXPECTED_QUESTIONS_SHA256)
        order_sha = hashlib.sha256("\n".join(str(s["sample_id"]) for s in samples).encode()).hexdigest()
    else:
        samples = utils.load_json(q_path)
        qmeta = {}
        order_sha = hashlib.sha256("\n".join(s["question"] for s in samples).encode()).hexdigest()
    if len(samples) != L.N_ITEMS:
        die(f"{len(samples)} questions in {q_path}, expected {L.N_ITEMS}")
    if smoke:
        samples = samples[:args.smoke_n]
    n = len(samples)
    q_file_sha = hashlib.sha256(open(q_path, "rb").read()).hexdigest()

    # ---- prompts: template pair, per-prompt pair, wrapped-string pair
    try:
        no_tmpl, co_tmpl = L.template_pair(select_templates_gsm8k)
        pairs = [L.render_pair(no_tmpl, co_tmpl, s["question"]) for s in samples]
    except ValueError as e:
        die(str(e))
    vc = VicundaModel(model_path=model_dir)
    vc.model.eval()
    tk = vc.tokenizer
    if tk.padding_side != "left":
        die(f"padding_side {tk.padding_side!r}, expected 'left'")
    if not getattr(tk, "chat_template", None):
        die("tokenizer has no chat_template")
    chat_hash = L.sha(tk.chat_template)

    def wrap(p):
        w = strip_leading_bos(vc, tk.apply_chat_template([{"role": "user", "content": p}],
                                                          tokenize=False, add_generation_prompt=True))
        return w
    wrapped_no = [wrap(p_no) for p_no, _ in pairs]
    wrapped_co = [wrap(p_co) for _, p_co in pairs]
    if wrapped_co[0] == pairs[0][1] or cfg["header"] not in wrapped_co[0]:
        die(f"chat template did not fire / header {cfg['header']!r} missing in the first wrapped prompt")
    for i, (wn, wc) in enumerate(zip(wrapped_no, wrapped_co)):
        assert_no_double_bos(vc, wc, f"cot sample {i}")
        if L.remove_cot_line(wc) != wn:
            die(f"sample {i}: removing the CoT line from the wrapped CoT prompt does NOT reproduce the wrapped "
                "No-CoT prompt (some other wording or system content differs)")
    try:
        tok_cot = L.check_prompt_tokens(tk, wrapped_co)
        tok_no = L.check_prompt_tokens(tk, wrapped_no)
    except ValueError as e:
        die(str(e))
    if tok_cot["final_token_id"] != tok_no["final_token_id"]:
        die("CoT and No-CoT prompts end on different tokens")
    prompt_sha = L.sha("\n".join(wrapped_co))
    nocot_chat_prompt_sha = L.sha("\n".join(wrapped_no))
    print(f"[tokens] BOS counts {tok_cot['bos_counts']}  final token {tok_cot['final_token_id']} "
          f"{tok_cot['final_token_text']!r}  (all {n} prompts)")
    print(f"[hash] chat_template={chat_hash[:16]} cot_prompts={prompt_sha[:16]} nocot_chat_prompts={nocot_chat_prompt_sha[:16]}")
    print("[prompt] first CoT chat prompt:", repr(wrapped_co[0][-260:]))

    # ---- mask
    raw_mask = np.load(mask_path)
    mask_sha = hashlib.sha256(open(mask_path, "rb").read()).hexdigest()
    n_dec = len(vc._find_decoder_layers())
    if n_dec != cfg["n_decoder"] or raw_mask.shape[0] != n_dec:
        die(f"decoder layers {n_dec} / mask rows {raw_mask.shape[0]} / expected {cfg['n_decoder']}")
    nz = sorted(int(i) for i in np.nonzero(np.any(raw_mask != 0, axis=1))[0])
    want = sorted(utils.decoder_layer_range(ls, le))
    if nz != want:
        die(f"mask non-zero rows {nz} != configured band rows {want}")

    identity = L.item_identity(args.task, samples)
    alphas = L.alphas_for(args.model)
    fname = L.cell_filename(args.model, args.task, cfg["band"])
    paths = {a: os.path.join(out_dir, L.alpha_dir(a), fname) for a in alphas}
    todo = []
    for a in alphas:
        if os.path.exists(paths[a]):
            probs = L.verify_existing_cell(paths[a], args.model, args.task, a, n, mask_sha, prompt_sha,
                                          model_dir, q_file_sha, order_sha, identity)
            if probs:
                die(f"existing cell {paths[a]} FAILED verification ({'; '.join(probs)}); refusing to skip or overwrite")
            print(f"[skip] alpha={a}: existing cell verified")
        else:
            todo.append(a)
    if args.mode == "check":
        print(f"[check] OK. cells to generate: {todo}; nothing written.")
        return

    try:
        devs = sorted({str(p.device) for p in vc.model.parameters()})
    except Exception:
        devs = []
    import torch, transformers
    prov = {"host": platform.node(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "param_devices": devs, "sharded": len(devs) > 1,
            "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "library_versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                                 "numpy": np.__version__},
            "code_version": L.code_version(os.path.dirname(os.path.abspath(__file__)))}
    os.makedirs(out_dir, exist_ok=True)
    diff_file = os.path.join(out_dir, "prompt_diff.json")
    if not os.path.exists(diff_file):
        L.atomic_write_json({
            "protocol": L.PROTOCOL, "cot_line": L.COT_LINE,
            "every_item_cot_line_removal_reproduces_nocot_user_message": True,
            "every_item_cot_line_removal_reproduces_nocot_wrapped_chat_prompt": True,
            "nocot_template": no_tmpl, "cot_template": co_tmpl,
            "examples": [{"nocot_user": pairs[i][0], "cot_user": pairs[i][1], "cot_wrapped": wrapped_co[i]}
                         for i in range(min(3, n))]}, diff_file)

    for a in todo:
        diff = raw_mask * a
        vc.steering_fire_count(reset=True)
        gen = []
        for i in tqdm(range(0, n, L.BATCH_SIZE), desc=f"chat-cot-sat {args.model}/{args.task} a={a}"):
            batch = wrapped_co[i:i + L.BATCH_SIZE]
            out = vc.regenerate(batch, max_new_tokens=L.MAX_NEW_TOKENS, temperature=L.TEMPERATURE, top_p=L.TOP_P,
                                diff_matrices=list(diff), batch_size=L.BATCH_SIZE, return_metadata=True)
            if len(out) != len(batch):
                die(f"regenerate returned {len(out)} replies for {len(batch)} prompts (alpha={a}, batch@{i})")
            gen.extend(out)
        if len(gen) != n:
            die(f"{len(gen)} replies for {n} prompts at alpha={a}")
        fires, expect = vc.steering_fire_count(), L.expected_fires(args.model, a, n)
        if fires != expect:
            die(f"steering_fires {fires} != {expect} at alpha={a}; cell unusable")
        rows = []
        for s, g in zip(samples, gen):
            if g["stop_reason"] not in L.STOP_REASONS:
                die(f"unknown stop_reason {g['stop_reason']!r}")
            if tcfg["label_free"]:
                rows.append({"sample_id": s["sample_id"], "question": s["question"], "generated": g["text"],
                             "generated_token_count": g["generated_token_count"], "stop_reason": g["stop_reason"]})
            else:
                pred = extract_gsm8k_answer(g["text"])
                rows.append({"question": s["question"], "answer": s["answer"], "generated_neutral": g["text"],
                             "pred_answer_neutral": pred, "correct_neutral": is_correct_gsm8k(pred, s["answer"]),
                             "generated_token_count": g["generated_token_count"], "stop_reason": g["stop_reason"]})
        meta = {
            "protocol": L.PROTOCOL, "model": args.model, "size": cfg["size"], "task": args.task, "alpha": a,
            "layers": list(cfg["band"]), "L": cfg["L"], "model_dir": model_dir, "smoke": smoke, "n": n,
            "mask_path": mask_path, "mask_sha256": mask_sha, "mask_nonzero_rows": nz,
            "max_new_tokens": L.MAX_NEW_TOKENS, "temperature": L.TEMPERATURE, "top_p": L.TOP_P,
            "batch_size": L.BATCH_SIZE, "suite": "default", "cot": True, "fmt_wording": "plain", "role": "neutral",
            "prefill_only": True, "prefill_tail_len": 1, "padding_side": tk.padding_side,
            "prompt_wrapper_id": cfg["wrapper_id"], "chat_template_applied": True, "chat_template_hash": chat_hash,
            "system_message": "none added (model template default kept)", "assistant_anchor": "none",
            "prompt_body_sha256": L.sha(co_tmpl), "nocot_prompt_body_sha256": L.sha(no_tmpl),
            "prompt_sha256": prompt_sha, "nocot_chat_prompt_sha256": nocot_chat_prompt_sha,
            "cot_line_removal_reproduces_nocot": True,
            "injection_token": tok_cot, "questions_file": q_path, "questions_file_sha256": q_file_sha,
            "sample_order_sha256": order_sha, "questions_sha256": qmeta.get("questions_sha256"),
            "contains_labels": not tcfg["label_free"],
            "steering_fires": fires, "steering_fires_expected": expect, "provenance": prov,
        }
        if not tcfg["label_free"]:
            meta["accuracy_inline_pct"] = round(sum(r["correct_neutral"] for r in rows) / n * 100, 2)
        os.makedirs(os.path.dirname(paths[a]), exist_ok=True)
        L.atomic_write_json({"meta": meta, "data": rows}, paths[a])
        print(f"  wrote {paths[a]}  fires={fires}/{expect}")
    print("\nFinished. Score OFFLINE with RoleAnswer/analyze_chat_cot_sat_v1.py (inline fields are process state).")


if __name__ == "__main__":
    main()
