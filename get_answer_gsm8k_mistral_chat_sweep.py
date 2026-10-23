#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_answer_gsm8k_mistral_chat_sweep.py -- Mistral GSM8K Native Chat No-CoT nine-point sweep (gsm8k-mistral-chat-sweep-v1).

Mistral-7B-Instruct-v0.3, mask nmd_0.5_14_22_7B (band [14,22), L=8), neutral/plain/zero-shot/No-CoT body as the single
user message through the model's own chat template (no system message, no assistant prefix). alpha order
0,-8,-6,-4,-2,2,4,6,8 in ONE process / one model load; no gating between doses. Injection = prefill-only, tail 1, i.e.
the last token of the wrapped prompt (id/text recorded). Score OFFLINE; inline fields are process state only.
Existing complete + identity/environment-matching cells are skipped; a mismatching one aborts (never overwritten).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform

import chat_cot_sat_lib as L
import utils
from get_answer_gsm8k_chat_sweep import assert_no_double_bos, die, strip_leading_bos
from template import select_templates_gsm8k
from utils import extract_gsm8k_answer, is_correct_gsm8k

PROTOCOL = "gsm8k-mistral-chat-sweep-v1"
MODEL_DIR = "mistralai/Mistral-7B-Instruct-v0.3"
BAND, N_LAYERS, N_DECODER = (14, 22), 8, 32
MASK_REL = "mask/mistral_non_logits/nmd_0.5_14_22_7B.npy"
QUESTIONS_REL = "benchmark/gsm8k_test_sample.json"
OUT_SUBDIR = "mistral/gsm8k_chat_sweep_v1"
ALPHAS = [0, -8, -6, -4, -2, 2, 4, 6, 8]
FNAME = "gsm8k_mistral_chat_7B_answers_14_22.json"


def expected_fires(alpha, n):
    return 0 if alpha == 0 else N_LAYERS * n


def verify_existing(path, alpha, n, mask_sha, prompt_sha, q_sha, order_sha, identity, prov):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        return [f"unreadable: {e}"]
    m, rows = d.get("meta", {}), d.get("data", [])
    P = []
    def need(c, msg):
        if not c:
            P.append(msg)
    need(m.get("protocol") == PROTOCOL and m.get("alpha") == alpha, "protocol/alpha")
    need(m.get("model_dir") == MODEL_DIR and tuple(m.get("layers", ())) == BAND, "model/layers")
    need(m.get("mask_sha256") == mask_sha and m.get("prompt_sha256") == prompt_sha, "mask/prompt hash")
    need((m.get("max_new_tokens"), m.get("temperature"), m.get("top_p"), m.get("batch_size"),
          m.get("prefill_tail_len"), m.get("prefill_only"), m.get("cot"), m.get("fmt_wording")) ==
         (L.MAX_NEW_TOKENS, L.TEMPERATURE, L.TOP_P, L.BATCH_SIZE, 1, True, False, "plain"), "generation settings")
    need(m.get("questions_file_sha256") == q_sha and m.get("sample_order_sha256") == order_sha, "question data")
    need(m.get("steering_fires") == expected_fires(alpha, n) == m.get("steering_fires_expected"), "steering_fires")
    need(len(rows) == n == m.get("n"), "row count")
    need([[r.get("question"), str(r.get("answer"))] for r in rows] == identity, "item order/gold")
    old = m.get("provenance", {})
    need(all(old.get(k) == prov[k] for k in ("host", "cuda_visible_devices", "cuda_device_names", "param_devices",
                                              "sharded", "library_versions")), "run environment differs")
    for r in rows:
        if not isinstance(r.get("generated_neutral"), str) or r.get("stop_reason") not in L.STOP_REASONS \
                or not isinstance(r.get("generated_token_count"), int):
            P.append("incomplete row")
            break
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="components/ directory")
    args = ap.parse_args()

    import numpy as np
    import torch
    import transformers
    from tqdm import tqdm
    from llms import VicundaModel

    q_path = os.path.join(args.base_dir, QUESTIONS_REL)
    mask_path = os.path.join(args.base_dir, MASK_REL)
    out_dir = os.path.join(args.base_dir, OUT_SUBDIR)
    ls, le = BAND

    samples = utils.load_json(q_path)
    if len(samples) != L.N_ITEMS:
        die(f"{len(samples)} questions in {q_path}, expected {L.N_ITEMS}")
    n = len(samples)
    order_sha = hashlib.sha256("\n".join(s["question"] for s in samples).encode()).hexdigest()
    q_sha = hashlib.sha256(open(q_path, "rb").read()).hexdigest()
    tmpl = select_templates_gsm8k(suite="default", cot=False, wording="plain")["neutral"]

    vc = VicundaModel(model_path=MODEL_DIR)
    vc.model.eval()
    tk = vc.tokenizer
    if not getattr(tk, "chat_template", None):
        die("tokenizer has no chat_template")
    chat_hash = L.sha(tk.chat_template)
    wrapped = [strip_leading_bos(vc, tk.apply_chat_template([{"role": "user", "content": tmpl.format(context=s["question"])}],
                                                             tokenize=False, add_generation_prompt=True)) for s in samples]
    for i, w in enumerate(wrapped):
        assert_no_double_bos(vc, w, f"sample {i}")
    try:
        tok = L.check_prompt_tokens(tk, wrapped)
    except ValueError as e:
        die(str(e))
    prompt_sha = L.sha("\n".join(wrapped))
    print(f"[tokens] BOS {tok['bos_counts']} final token {tok['final_token_id']} {tok['final_token_text']!r}")
    print("[prompt] first chat prompt:", repr(wrapped[0][-260:]))

    raw_mask = np.load(mask_path)
    mask_sha = hashlib.sha256(open(mask_path, "rb").read()).hexdigest()
    n_dec = len(vc._find_decoder_layers())
    if n_dec != N_DECODER or raw_mask.shape[0] != n_dec:
        die(f"decoder layers {n_dec} / mask rows {raw_mask.shape[0]} / expected {N_DECODER}")
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

    identity = [[s["question"], str(s["answer"])] for s in samples]
    paths = {a: os.path.join(out_dir, L.alpha_dir(a), FNAME) for a in ALPHAS}
    todo = []
    for a in ALPHAS:
        if os.path.exists(paths[a]):
            probs = verify_existing(paths[a], a, n, mask_sha, prompt_sha, q_sha, order_sha, identity, prov)
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
        for i in tqdm(range(0, n, L.BATCH_SIZE), desc=f"mistral-chat a={a}"):
            batch = wrapped[i:i + L.BATCH_SIZE]
            out = vc.regenerate(batch, max_new_tokens=L.MAX_NEW_TOKENS, temperature=L.TEMPERATURE, top_p=L.TOP_P,
                                diff_matrices=list(diff), batch_size=L.BATCH_SIZE, return_metadata=True)
            if len(out) != len(batch):
                die(f"regenerate returned {len(out)} replies for {len(batch)} prompts (alpha={a})")
            gen.extend(out)
        fires, expect = vc.steering_fire_count(), expected_fires(a, n)
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
            "protocol": PROTOCOL, "model": "mistral", "size": "7B", "task": "gsm8k", "alpha": a,
            "layers": list(BAND), "L": N_LAYERS, "model_dir": MODEL_DIR, "n": n,
            "mask_path": mask_path, "mask_sha256": mask_sha, "mask_nonzero_rows": nz,
            "max_new_tokens": L.MAX_NEW_TOKENS, "temperature": L.TEMPERATURE, "top_p": L.TOP_P,
            "batch_size": L.BATCH_SIZE, "suite": "default", "cot": False, "fmt_wording": "plain", "role": "neutral",
            "prefill_only": True, "prefill_tail_len": 1, "padding_side": tk.padding_side,
            "chat_template_applied": True, "chat_template_hash": chat_hash,
            "system_message": "none", "assistant_anchor": "none",
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
