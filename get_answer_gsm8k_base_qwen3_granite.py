#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_answer_gsm8k_base_qwen3_granite.py -- GSM8K Bare No-CoT on the BASE checkpoints of Qwen3-8B and Granite-3.3-8B (gsm8k-base-qwen3-granite-v1).

The instruct-version NMD mask is reused unchanged (read-only: it must exist and equal the mask rebuilt from the instruct means via
detection.nmd.get_nmd_mask; nothing is written or re-extracted). Two cells per model, alpha 0 and the instruct workpoint (Qwen3 +10,
Granite -6), in ONE process / one model load. BARE string, neutral/plain/zero-shot/No-CoT, 768 tokens, greedy, bs=24, prefill-only, tail 1,
last prompt token. An existing target cell is refused (no resume, never overwritten). BOS count is recorded, not asserted; the final prompt
token must be uniform. Score OFFLINE; inline fields are process state.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform

import chat_cot_sat_lib as L
import utils
from get_answer_gsm8k_chat_sweep import die
from template import select_templates_gsm8k
from utils import extract_gsm8k_answer, is_correct_gsm8k

PROTOCOL = "gsm8k-base-qwen3-granite-v1"
QUESTIONS_REL = "benchmark/gsm8k_test_sample.json"
CFG = {
    "qwen3": dict(model_dir="Qwen/Qwen3-8B-Base", band=(17, 26), n_decoder=36, alphas=[0, 10],
                  mask_rel="mask/qwen3_non_logits/nmd_0.5_17_26_8B.npy", mean_rel="hidden_states_mean/qwen3",
                  out_subdir="qwen3_base/gsm8k_bare_base_v1", fname="gsm8k_qwen3_base_8B_answers_17_26.json"),
    "granite33": dict(model_dir="ibm-granite/granite-3.3-8b-base", band=(11, 20), n_decoder=40, alphas=[0, -6],
                      mask_rel="mask/granite33_non_logits/nmd_0.5_11_20_8B.npy", mean_rel="hidden_states_mean/granite33/mmlue",
                      out_subdir="granite33_base/gsm8k_bare_base_v1", fname="gsm8k_granite33_base_8B_answers_11_20.json"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="components/ directory")
    ap.add_argument("--model", required=True, choices=sorted(CFG))
    args = ap.parse_args()
    C = CFG[args.model]
    ls, le = C["band"]
    n_layers = le - ls
    alphas = C["alphas"]

    import numpy as np
    import torch
    import transformers
    from tqdm import tqdm
    from llms import VicundaModel

    q_path = os.path.join(args.base_dir, QUESTIONS_REL)
    mask_path = os.path.join(args.base_dir, C["mask_rel"])
    out_dir = os.path.join(args.base_dir, C["out_subdir"])
    paths = {a: os.path.join(out_dir, L.alpha_dir(a), C["fname"]) for a in alphas}
    for a in alphas:
        if os.path.exists(paths[a]):
            die(f"target cell {paths[a]} already exists; refusing to overwrite or resume")

    # ---- instruct mask: read-only; must exist and equal the one rebuilt from the instruct means
    from detection.nmd import get_nmd_mask
    src = os.path.join(args.base_dir, C["mean_rel"])
    f_char, f_none = os.path.join(src, "diff_mean_8B.npy"), os.path.join(src, "none_diff_mean_8B.npy")
    d_char, d_none = np.load(f_char), np.load(f_none)
    if d_char.shape != d_none.shape or d_char.ndim != 4 or d_char.shape[:2] != (1, 1):
        die(f"means shapes {d_char.shape} / {d_none.shape} must be identical (1,1,L,H)")
    top_k = max(1, int(d_char.shape[-1] * 0.5 / 100.0))
    built = get_nmd_mask(d_char, d_none, top_k=top_k, start=ls, end=le)
    if not os.path.exists(mask_path):
        die(f"instruct mask {mask_path} not found; this runner never creates masks")
    if not np.array_equal(np.load(mask_path), built):
        die(f"mask {mask_path} differs from the one rebuilt from the instruct means")
    file_sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    mask_inputs = {"diff_mean_8B.npy": file_sha(f_char), "none_diff_mean_8B.npy": file_sha(f_none), "top_k": top_k,
                   "direction": "diff_char - diff_none", "mask_path": mask_path}

    samples = utils.load_json(q_path)
    if len(samples) != L.N_ITEMS:
        die(f"{len(samples)} questions in {q_path}, expected {L.N_ITEMS}")
    n = len(samples)
    order_sha = hashlib.sha256("\n".join(s["question"] for s in samples).encode()).hexdigest()
    q_sha = file_sha(q_path)
    tmpl = select_templates_gsm8k(suite="default", cot=False, wording="plain")["neutral"]

    vc = VicundaModel(model_path=C["model_dir"])
    vc.model.eval()
    tk = vc.tokenizer
    wrapped = [tmpl.format(context=s["question"]) for s in samples]      # BARE: no chat template
    bos, bos_counts, tails = tk.bos_token_id, set(), {}
    for t in wrapped:
        ids = tk(t, add_special_tokens=True)["input_ids"]
        bos_counts.add(ids.count(bos) if bos is not None else 0)
        tails[ids[-1]] = tails.get(ids[-1], 0) + 1
    if len(tails) != 1:
        die(f"final prompt token is not uniform: {sorted(tails.items())}")
    k = next(iter(tails))
    tok = {"bos_token_id": bos, "bos_counts": sorted(bos_counts), "n_prompts": len(wrapped), "final_token_id": int(k),
           "final_token_text": tk.decode([k]), "final_token_counts": {f"{kk}:{tk.decode([kk])!r}": v for kk, v in tails.items()}}
    prompt_sha = L.sha("\n".join(wrapped))
    print(f"[tokens] BOS {tok['bos_counts']} final token {tok['final_token_id']} {tok['final_token_text']!r}")
    print("[prompt] first bare prompt:", repr(wrapped[0][-260:]))

    raw_mask = np.load(mask_path)
    mask_sha = file_sha(mask_path)
    n_dec = len(vc._find_decoder_layers())
    if raw_mask.shape != (n_dec, vc.model.config.hidden_size):
        die(f"mask shape {raw_mask.shape} != (decoder layers {n_dec}, hidden_size {vc.model.config.hidden_size})")
    if n_dec != C["n_decoder"]:
        die(f"decoder layers {n_dec} != expected {C['n_decoder']}")
    nz = sorted(int(i) for i in np.nonzero(np.any(raw_mask != 0, axis=1))[0])
    if nz != sorted(utils.decoder_layer_range(ls, le)):
        die(f"mask non-zero rows {nz} != configured band")

    try:
        devs = sorted({str(p.device) for p in vc.model.parameters()})
    except Exception:
        devs = []
    prov = {"host": platform.node(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "param_devices": devs, "sharded": len(devs) > 1,
            "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "library_versions": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
            "code_version": L.code_version(os.path.dirname(os.path.abspath(__file__)))}
    os.makedirs(out_dir, exist_ok=True)

    for a in alphas:
        diff = raw_mask * a
        vc.steering_fire_count(reset=True)
        gen = []
        for i in tqdm(range(0, n, L.BATCH_SIZE), desc=f"{args.model}-base a={a}"):
            batch = wrapped[i:i + L.BATCH_SIZE]
            out = vc.regenerate(batch, max_new_tokens=L.MAX_NEW_TOKENS, temperature=L.TEMPERATURE, top_p=L.TOP_P,
                                diff_matrices=list(diff), batch_size=L.BATCH_SIZE, return_metadata=True)
            if len(out) != len(batch):
                die(f"regenerate returned {len(out)} replies for {len(batch)} prompts (alpha={a})")
            gen.extend(out)
        fires, expect = vc.steering_fire_count(), (0 if a == 0 else n_layers * n)
        if len(gen) != n or fires != expect:
            die(f"alpha={a}: {len(gen)} replies / steering_fires {fires} (expected {n} / {expect}); cell unusable")
        rows = []
        for idx, (s, g) in enumerate(zip(samples, gen)):
            if g["stop_reason"] not in L.STOP_REASONS:
                die(f"unknown stop_reason {g['stop_reason']!r}")
            pred = extract_gsm8k_answer(g["text"])
            rows.append({"sample_idx": idx, "question": s["question"], "answer": s["answer"], "generated_neutral": g["text"],
                         "pred_answer_neutral": pred, "correct_neutral": is_correct_gsm8k(pred, s["answer"]),
                         "generated_token_count": g["generated_token_count"], "stop_reason": g["stop_reason"]})
        meta = {
            "protocol": PROTOCOL, "checkpoint": "base", "it_mask_reused": True, "model": args.model, "size": "8B", "task": "gsm8k",
            "alpha": a, "layers": [ls, le], "L": n_layers, "model_dir": C["model_dir"], "n": n,
            "mask_path": mask_path, "mask_sha256": mask_sha, "mask_nonzero_rows": nz,
            "max_new_tokens": L.MAX_NEW_TOKENS, "temperature": L.TEMPERATURE, "top_p": L.TOP_P,
            "batch_size": L.BATCH_SIZE, "suite": "default", "cot": False, "fmt_wording": "plain", "role": "neutral",
            "prefill_only": True, "prefill_tail_len": 1, "padding_side": tk.padding_side,
            "chat_template_applied": False, "mask_inputs_sha256": mask_inputs, "system_message": "none", "assistant_anchor": "none",
            "prompt_body_sha256": L.sha(tmpl), "prompt_sha256": prompt_sha, "injection_token": tok,
            "questions_file": q_path, "questions_file_sha256": q_sha, "sample_order_sha256": order_sha,
            "steering_fires": fires, "steering_fires_expected": expect, "provenance": prov,
            "accuracy_inline_pct": round(sum(r["correct_neutral"] for r in rows) / n * 100, 2),
        }
        os.makedirs(os.path.dirname(paths[a]), exist_ok=True)
        L.atomic_write_json({"meta": meta, "data": rows}, paths[a])
        print(f"  wrote {paths[a]}  fires={fires}/{expect}")
    print("\nFinished. Score OFFLINE (inline fields are process state).")


if __name__ == "__main__":
    main()
