#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_answer_granite33_bare_transfer.py -- Granite33-8B Bare No-CoT fixed-workpoint transfer (granite33-bare-transfer-v1).

--task gsm_hard : alpha 0,-6,-4,+6  768 tok  bs 24  top_p 1.0  GSM-Hard neutral template  (####)
--task math     : alpha 0,-6,-4,+6  2048 tok bs 8   top_p 0.9  build_math_suite(cot=False)    (\\boxed{})
ibm-granite/granite-3.3-8b-instruct, BARE string (no chat template), prefill-only tail 1, greedy. Mask = the existing Granite33 NMD mask
(hidden [11,20) -> decoder rows 10..18, L=9), rebuilt from Granite33's own means and required identical to the stored file.
Fixed candidate dose -6 is carried over from the GSM8K sweep; -4 and +6 are additional diagnostic doses and do not redefine it. Primary transfer family: -6 vs 0 on GSM-Hard and MATH (Holm m=2). An expanded family includes -4 and +6 on both tasks (Holm m=6). No dose is re-searched. One model load per task, all alphas of the task in one process.
Generation reads questions only (no gold scoring here); score OFFLINE. Existing cells are reused only when settings/data/mask/prompt match
(environment differences, incl. CUDA_VISIBLE_DEVICES, only warn), otherwise abort (never overwritten).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform

import chat_cot_sat_lib as L
import utils
from get_answer_gsm8k_chat_sweep import die
from template import build_math_suite, select_templates_gsm8k

PROTOCOL = "granite33-bare-transfer-v1"
MODEL_DIR = "ibm-granite/granite-3.3-8b-instruct"
BAND, N_LAYERS, N_DECODER = (11, 20), 9, 40
MASK_REL = "mask/granite33_non_logits/nmd_0.5_11_20_8B.npy"
MEAN_REL = "hidden_states_mean/granite33/mmlue"
TASKS = {
    "gsm_hard": dict(alphas=[0, -6, -4, 6], max_new=768, bs=24, top_p=1.0, q_rel="benchmark/gsm_hard_p3_questions.json",
                     out="granite33/gsm_hard_bare_transfer_v1", fname="gsm_hard_granite33_bare_8B_11_20.json"),
    "math": dict(alphas=[0, -6, -4, 6], max_new=2048, bs=8, top_p=0.9, q_rel="benchmark/math_test_sample.json",
                 out="granite33/math_bare_transfer_v1", fname="math_granite33_bare_8B_11_20.json"),
}


def expected_fires(alpha, n):
    return 0 if alpha == 0 else N_LAYERS * n


def load_questions(task, path):
    d = json.load(open(path, encoding="utf-8"))
    if task == "gsm_hard":
        if d["meta"].get("contains_labels") is not False:
            die("GSM-Hard questions file must declare contains_labels=false")
        rows = [{"sample_id": s["sample_id"], "question": s["question"]} for s in d["data"]]
    else:
        rows = [{"sample_idx": i, "question": s["question"], "gold_solution": s["answer"],
                 "level": s.get("level", ""), "type": s.get("type", "")} for i, s in enumerate(d)]
    return rows


def ident(task, r):
    return [r["sample_id"], r["question"]] if task == "gsm_hard" else [r["question"], r["gold_solution"], r["level"], r["type"]]


def verify_existing(task, C, path, alpha, n, mask_sha, prompt_sha, q_sha, order_sha, identity, prov, mask_inputs):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        return [f"unreadable: {e}"]
    m, rows = d.get("meta", {}), d.get("data", [])
    P = []
    def need(c, msg):
        if not c:
            P.append(msg)
    need(m.get("protocol") == PROTOCOL and m.get("task") == task and m.get("alpha") == alpha, "protocol/task/alpha")
    need(m.get("model_dir") == MODEL_DIR and tuple(m.get("layers", ())) == BAND, "model/layers")
    need(m.get("mask_sha256") == mask_sha and m.get("prompt_sha256") == prompt_sha, "mask/prompt hash")
    need((m.get("max_new_tokens"), m.get("temperature"), m.get("top_p"), m.get("batch_size"),
          m.get("prefill_tail_len"), m.get("prefill_only"), m.get("cot")) ==
         (C["max_new"], L.TEMPERATURE, C["top_p"], C["bs"], 1, True, False), "generation settings")
    need(m.get("chat_template_applied") is False and m.get("mask_inputs_sha256") == mask_inputs, "bare flag / mask inputs")
    need(m.get("questions_file_sha256") == q_sha and m.get("sample_order_sha256") == order_sha, "question data")
    need(m.get("steering_fires") == expected_fires(alpha, n) == m.get("steering_fires_expected"), "steering_fires")
    need(len(rows) == n == m.get("n"), "row count")
    need([ident(task, r) for r in rows] == identity, "item identity/order")
    old = m.get("provenance", {})
    diff = [k for k in ("host", "cuda_visible_devices", "cuda_device_names", "param_devices", "sharded", "library_versions") if old.get(k) != prov[k]]
    if diff:
        print(f"[warn] alpha={alpha}: existing cell environment differs in {diff} (cross-run; not a reason to refuse)")
    for r in rows:
        if not isinstance(r.get("generated"), str) or r.get("stop_reason") not in L.STOP_REASONS \
                or not isinstance(r.get("generated_token_count"), int):
            P.append("incomplete row")
            break
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="components/ directory")
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--alphas", type=int, nargs="+", default=None, help="subset of the task alphas for this process (split across GPUs)")
    args = ap.parse_args()
    task, C = args.task, TASKS[args.task]
    ALPHAS = list(C["alphas"])
    if args.alphas is not None:
        if any(a not in ALPHAS for a in args.alphas) or len(set(args.alphas)) != len(args.alphas):
            die(f"--alphas must be distinct members of {ALPHAS}")
        ALPHAS = [a for a in ALPHAS if a in args.alphas]

    import numpy as np
    import torch
    import transformers
    from tqdm import tqdm
    from llms import VicundaModel

    q_path = os.path.join(args.base_dir, C["q_rel"])
    mask_path = os.path.join(args.base_dir, MASK_REL)
    out_dir = os.path.join(args.base_dir, C["out"])
    ls, le = BAND

    # ---- mask from Granite33's own means; must equal the stored GSM8K-line mask (never re-selected)
    from detection.nmd import get_nmd_mask
    src = os.path.join(args.base_dir, MEAN_REL)
    f_char, f_none = os.path.join(src, "diff_mean_8B.npy"), os.path.join(src, "none_diff_mean_8B.npy")
    d_char, d_none = np.load(f_char), np.load(f_none)
    if d_char.shape != d_none.shape or d_char.ndim != 4 or d_char.shape[:2] != (1, 1):
        die(f"means shapes {d_char.shape} / {d_none.shape} must be identical (1,1,L,H)")
    top_k = max(1, int(d_char.shape[-1] * 0.5 / 100.0))
    built = get_nmd_mask(d_char, d_none, top_k=top_k, start=ls, end=le)
    if not os.path.exists(mask_path):
        die(f"stored mask {mask_path} missing; this transfer test reuses the GSM8K-line mask")
    if not np.array_equal(np.load(mask_path), built):
        die(f"stored mask {mask_path} differs from the one rebuilt from the means; refusing")
    file_sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    mask_inputs = {"diff_mean_8B.npy": file_sha(f_char), "none_diff_mean_8B.npy": file_sha(f_none), "top_k": top_k,
                   "direction": "diff_char - diff_none", "mask_path": mask_path}

    samples = load_questions(task, q_path)
    if len(samples) != L.N_ITEMS:
        die(f"{len(samples)} questions in {q_path}, expected {L.N_ITEMS}")
    n = len(samples)
    order_sha = hashlib.sha256("\n".join(s["question"] for s in samples).encode()).hexdigest()
    q_sha = file_sha(q_path)
    if task == "gsm_hard":
        tmpl = select_templates_gsm8k(suite="default", cot=False, wording="plain")["neutral"]
    else:
        tmpl = build_math_suite(cot=False)["neutral"]

    vc = VicundaModel(model_path=MODEL_DIR)
    vc.model.eval()
    tk = vc.tokenizer
    wrapped = [tmpl.format(context=s["question"]) for s in samples]      # BARE: no chat template
    # native Bare encoding as vc.regenerate does it (add_special_tokens=True); BOS count is RECORDED, never forced or patched
    bos, bos_counts, tails = tk.bos_token_id, set(), {}
    for t in wrapped:
        ids = tk(t, add_special_tokens=True)["input_ids"]
        bos_counts.add(ids.count(bos) if bos is not None else 0)
        tails[ids[-1]] = tails.get(ids[-1], 0) + 1
    if len(tails) != 1:
        die(f"final prompt token is not uniform: {sorted(tails.items())}")
    if not all(w.endswith("Answer: ") for w in wrapped):
        die("rendered Bare prompt does not end with 'Answer: '")
    k = next(iter(tails))
    tok = {"bos_token_id": bos, "bos_counts": sorted(bos_counts), "n_prompts": len(wrapped), "final_token_id": int(k),
           "final_token_text": tk.decode([k]), "final_token_counts": {f"{kk}:{tk.decode([kk])!r}": v for kk, v in tails.items()}}
    prompt_sha = L.sha("\n".join(wrapped))
    print(f"[task] {task} alphas={ALPHAS} max_new={C['max_new']} bs={C['bs']} top_p={C['top_p']}")
    print(f"[tokens] BOS {tok['bos_counts']} final token {tok['final_token_id']} {tok['final_token_text']!r}")
    print("[prompt] first bare prompt:", repr(wrapped[0][-260:]))

    raw_mask = np.load(mask_path)
    mask_sha = file_sha(mask_path)
    n_dec = len(vc._find_decoder_layers())
    if raw_mask.shape != (n_dec, vc.model.config.hidden_size):
        die(f"mask shape {raw_mask.shape} != (decoder layers {n_dec}, hidden_size {vc.model.config.hidden_size})")
    if n_dec != N_DECODER:
        die(f"decoder layers {n_dec} expected {N_DECODER}")
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

    identity = [ident(task, s) for s in samples]
    paths = {a: os.path.join(out_dir, L.alpha_dir(a), C["fname"]) for a in ALPHAS}
    todo = []
    for a in ALPHAS:
        if os.path.exists(paths[a]):
            probs = verify_existing(task, C, paths[a], a, n, mask_sha, prompt_sha, q_sha, order_sha, identity, prov, mask_inputs)
            if probs:
                die(f"existing cell {paths[a]} failed verification ({'; '.join(probs)}); refusing to skip or overwrite")
            print(f"[skip] alpha={a}: existing cell verified")
        else:
            todo.append(a)
    os.makedirs(out_dir, exist_ok=True)

    for a in todo:
        diff = raw_mask * a
        vc.steering_fire_count(reset=True)
        gen = []
        for i in tqdm(range(0, n, C["bs"]), desc=f"granite33-{task} a={a}"):
            batch = wrapped[i:i + C["bs"]]
            out = vc.regenerate(batch, max_new_tokens=C["max_new"], temperature=L.TEMPERATURE, top_p=C["top_p"],
                                diff_matrices=list(diff), batch_size=C["bs"], return_metadata=True)
            if len(out) != len(batch):
                die(f"regenerate returned {len(out)} replies for {len(batch)} prompts (alpha={a})")
            gen.extend(out)
        fires, expect = vc.steering_fire_count(), expected_fires(a, n)
        if len(gen) != n or fires != expect:
            die(f"alpha={a}: {len(gen)} replies / steering_fires {fires} (expected {n} / {expect}); cell unusable")
        rows = []
        for s, g in zip(samples, gen):
            if g["stop_reason"] not in L.STOP_REASONS:
                die(f"unknown stop_reason {g['stop_reason']!r}")
            rows.append({**s, "generated": g["text"], "generated_token_count": g["generated_token_count"],
                         "stop_reason": g["stop_reason"]})
        meta = {
            "protocol": PROTOCOL, "model": "granite33", "size": "8B", "task": task, "alpha": a,
            "layers": list(BAND), "L": N_LAYERS, "model_dir": MODEL_DIR, "n": n,
            "mask_path": mask_path, "mask_sha256": mask_sha, "mask_nonzero_rows": nz,
            "max_new_tokens": C["max_new"], "temperature": L.TEMPERATURE, "top_p": C["top_p"],
            "batch_size": C["bs"], "suite": "default", "cot": False, "role": "neutral",
            "prefill_only": True, "prefill_tail_len": 1, "padding_side": tk.padding_side,
            "chat_template_applied": False, "mask_inputs_sha256": mask_inputs, "system_message": "none",
            "prompt_body_sha256": L.sha(tmpl), "prompt_sha256": prompt_sha, "injection_token": tok,
            "questions_file": q_path, "questions_file_sha256": q_sha, "sample_order_sha256": order_sha,
            "steering_fires": fires, "steering_fires_expected": expect, "provenance": prov,
        }
        os.makedirs(os.path.dirname(paths[a]), exist_ok=True)
        L.atomic_write_json({"meta": meta, "data": rows}, paths[a])
        print(f"  wrote {paths[a]}  fires={fires}/{expect}")
    print("\nFinished. Score OFFLINE (no inline correctness is stored).")


if __name__ == "__main__":
    main()
