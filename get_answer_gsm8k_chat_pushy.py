#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_answer_gsm8k_chat_pushy.py -- GSM8K Native Chat Pushy (protocol gsm8k-chat-pushy-v1).

Native Chat No-CoT with ONE line changed: the answer-format instruction
  plain: "Provide your final numeric answer after '####'."
  pushy: "Give your final answer as a single number after '####'."
Four fixed cells: llama3 alpha {0,-6}, qwen2.5 alpha {0,+8}; one model load per process, alphas run in order.
Protocol: docs/PREREG_GSM8K_CHAT_PUSHY_V1.md. Existing Chat No-CoT / Chat-CoT runners and trees are not touched;
generic helpers are imported from chat_cot_sat_lib / get_answer_gsm8k_chat_sweep.
Score OFFLINE (RoleAnswer/reasoningTaskAnalysis/chat_pushy_v1/); inline fields are process state only.
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

PROTOCOL = "gsm8k-chat-pushy-v1"
PLAIN_FMT = "Provide your final numeric answer after '####'."
PUSHY_FMT = "Give your final answer as a single number after '####'."
OUT_SUBDIR = "gsm8k_chat_pushy_v1"
QUESTIONS_REL = "benchmark/gsm8k_test_sample.json"


def verify_existing(path, model, alpha, n, mask_sha, prompt_sha, model_dir, q_sha, order_sha, identity, prov):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        return [f"unreadable: {e}"]
    m, rows, cfg = d.get("meta", {}), d.get("data", []), L.MODELS[model]
    P = []
    def need(c, msg):
        if not c:
            P.append(msg)
    need(m.get("protocol") == PROTOCOL, f"protocol {m.get('protocol')}")
    need(m.get("model") == model and m.get("alpha") == alpha, "model/alpha")
    need(m.get("cot") is False and m.get("fmt_wording") == "pushy" and m.get("chat_template_applied") is True,
         "cot/wording/chat flags")
    need(tuple(m.get("layers", ())) == tuple(cfg["band"]), "layers")
    need(m.get("mask_sha256") == mask_sha, "mask_sha256")
    need(m.get("prompt_sha256") == prompt_sha, "prompt_sha256")
    need((m.get("max_new_tokens"), m.get("temperature"), m.get("top_p"), m.get("batch_size"),
          m.get("prefill_tail_len"), m.get("prefill_only")) ==
         (L.MAX_NEW_TOKENS, L.TEMPERATURE, L.TOP_P, L.BATCH_SIZE, 1, True), "generation settings")
    need(m.get("model_dir") == model_dir, "model_dir")
    need(m.get("questions_file_sha256") == q_sha and m.get("sample_order_sha256") == order_sha, "question data")
    need(m.get("steering_fires") == L.expected_fires(model, alpha, n) == m.get("steering_fires_expected"), "steering_fires")
    need(len(rows) == n == m.get("n"), "row count")
    old = m.get("provenance", {})
    need(all(old.get(k) == prov[k] for k in ("host", "cuda_device_names", "param_devices", "sharded", "library_versions")),
         "run environment (host/GPU/devices/libraries) differs from this run; reuse would mix environments")
    need([[r.get("question"), str(r.get("answer"))] for r in rows] == identity, "item order/gold")
    for r in rows:
        if not isinstance(r.get("generated_neutral"), str) or r.get("stop_reason") not in L.STOP_REASONS \
                or not isinstance(r.get("generated_token_count"), int):
            P.append("incomplete row")
            break
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(L.MODELS))
    ap.add_argument("--base_dir", required=True, help="components/ directory")
    args = ap.parse_args()

    import numpy as np
    import torch
    import transformers
    from tqdm import tqdm
    from llms import VicundaModel

    cfg = L.MODELS[args.model]
    model_dir = cfg["model_dir"]
    q_path = os.path.join(args.base_dir, QUESTIONS_REL)
    mask_path = os.path.join(args.base_dir, cfg["mask_rel"])
    out_dir = os.path.join(args.base_dir, cfg["out_name"], OUT_SUBDIR)
    ls, le = cfg["band"]

    samples = utils.load_json(q_path)
    if len(samples) != L.N_ITEMS:
        die(f"{len(samples)} questions in {q_path}, expected {L.N_ITEMS}")
    n = len(samples)
    order_sha = hashlib.sha256("\n".join(s["question"] for s in samples).encode()).hexdigest()
    q_sha = hashlib.sha256(open(q_path, "rb").read()).hexdigest()

    # prompt: pushy template differs from the plain Chat No-CoT template by exactly the format line
    plain = select_templates_gsm8k(suite="default", cot=False, wording="plain")["neutral"]
    pushy = select_templates_gsm8k(suite="default", cot=False, wording="pushy")["neutral"]
    if plain.count(PLAIN_FMT) != 1 or pushy.count(PUSHY_FMT) != 1 or plain.replace(PLAIN_FMT, PUSHY_FMT) != pushy:
        die("pushy template != plain Chat No-CoT template with only the format line replaced")
    if "step by step" in pushy or not pushy.endswith("Answer: "):
        die("pushy template has a CoT line or lost the trailing 'Answer: '")

    vc = VicundaModel(model_path=model_dir)
    vc.model.eval()
    tk = vc.tokenizer
    if tk.padding_side != "left" or not getattr(tk, "chat_template", None):
        die("tokenizer must be left-padded and have a chat_template")
    chat_hash = L.sha(tk.chat_template)

    def wrap(p):
        return strip_leading_bos(vc, tk.apply_chat_template([{"role": "user", "content": p}],
                                                             tokenize=False, add_generation_prompt=True))
    users_plain = [plain.format(context=s["question"]) for s in samples]
    users_pushy = [pushy.format(context=s["question"]) for s in samples]
    wrapped_plain = [wrap(p) for p in users_plain]
    wrapped_pushy = [wrap(p) for p in users_pushy]
    if cfg["header"] not in wrapped_pushy[0]:
        die(f"chat template header {cfg['header']!r} missing")
    for i, (wp, wq) in enumerate(zip(wrapped_plain, wrapped_pushy)):
        assert_no_double_bos(vc, wq, f"pushy sample {i}")
        if wq.replace(PUSHY_FMT, PLAIN_FMT) != wp:      # question text may not contain either line (checked via equality)
            die(f"sample {i}: wrapped pushy prompt differs from wrapped plain prompt by more than the format line")
    try:
        tok = L.check_prompt_tokens(tk, wrapped_pushy)
    except ValueError as e:
        die(str(e))
    prompt_sha = L.sha("\n".join(wrapped_pushy))
    plain_prompt_sha = L.sha("\n".join(wrapped_plain))
    print(f"[tokens] BOS {tok['bos_counts']} final token {tok['final_token_id']} {tok['final_token_text']!r}")
    print("[prompt] first pushy chat prompt:", repr(wrapped_pushy[0][-260:]))

    raw_mask = np.load(mask_path)
    mask_sha = hashlib.sha256(open(mask_path, "rb").read()).hexdigest()
    n_dec = len(vc._find_decoder_layers())
    if n_dec != cfg["n_decoder"] or raw_mask.shape[0] != n_dec:
        die(f"decoder layers {n_dec} / mask rows {raw_mask.shape[0]} / expected {cfg['n_decoder']}")
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
    fname = f"gsm8k_chat_pushy_{cfg['size']}_answers_{ls}_{le}.json"
    alphas = L.alphas_for(args.model)
    paths = {a: os.path.join(out_dir, L.alpha_dir(a), fname) for a in alphas}
    todo = []
    for a in alphas:
        if os.path.exists(paths[a]):
            probs = verify_existing(paths[a], args.model, a, n, mask_sha, prompt_sha, model_dir, q_sha, order_sha, identity, prov)
            if probs:
                die(f"existing cell {paths[a]} failed verification ({'; '.join(probs)}); refusing to skip or overwrite")
            print(f"[skip] alpha={a}: existing cell verified")
        else:
            todo.append(a)

    os.makedirs(out_dir, exist_ok=True)
    diff_file = os.path.join(out_dir, "prompt_diff.json")
    if not os.path.exists(diff_file):
        L.atomic_write_json({"protocol": PROTOCOL, "plain_format_line": PLAIN_FMT, "pushy_format_line": PUSHY_FMT,
                             "every_item_differs_only_by_format_line": True,
                             "plain_template": plain, "pushy_template": pushy,
                             "examples": [{"plain_user": users_plain[i], "pushy_user": users_pushy[i],
                                           "pushy_wrapped": wrapped_pushy[i]} for i in range(3)]}, diff_file)

    for a in todo:
        diff = raw_mask * a
        vc.steering_fire_count(reset=True)
        gen = []
        for i in tqdm(range(0, n, L.BATCH_SIZE), desc=f"chat-pushy {args.model} a={a}"):
            batch = wrapped_pushy[i:i + L.BATCH_SIZE]
            out = vc.regenerate(batch, max_new_tokens=L.MAX_NEW_TOKENS, temperature=L.TEMPERATURE, top_p=L.TOP_P,
                                diff_matrices=list(diff), batch_size=L.BATCH_SIZE, return_metadata=True)
            if len(out) != len(batch):
                die(f"regenerate returned {len(out)} replies for {len(batch)} prompts (alpha={a})")
            gen.extend(out)
        fires, expect = vc.steering_fire_count(), L.expected_fires(args.model, a, n)
        if len(gen) != n or fires != expect:
            die(f"alpha={a}: {len(gen)} replies / steering_fires {fires} (expected {n} / {expect}); cell unusable")
        rows = []
        for s, g in zip(samples, gen):
            if g["stop_reason"] not in L.STOP_REASONS:
                die(f"unknown stop_reason {g['stop_reason']!r}")
            pred = extract_gsm8k_answer(g["text"])
            rows.append({"question": s["question"], "answer": s["answer"], "generated_neutral": g["text"],
                         "pred_answer_neutral": pred, "correct_neutral": is_correct_gsm8k(pred, s["answer"]),
                         "generated_token_count": g["generated_token_count"], "stop_reason": g["stop_reason"]})
        meta = {
            "protocol": PROTOCOL, "model": args.model, "size": cfg["size"], "task": "gsm8k", "alpha": a,
            "layers": list(cfg["band"]), "L": cfg["L"], "model_dir": model_dir, "n": n,
            "mask_path": mask_path, "mask_sha256": mask_sha, "mask_nonzero_rows": nz,
            "max_new_tokens": L.MAX_NEW_TOKENS, "temperature": L.TEMPERATURE, "top_p": L.TOP_P,
            "batch_size": L.BATCH_SIZE, "suite": "default", "cot": False, "fmt_wording": "pushy", "role": "neutral",
            "prefill_only": True, "prefill_tail_len": 1, "padding_side": tk.padding_side,
            "prompt_wrapper_id": cfg["wrapper_id"], "chat_template_applied": True, "chat_template_hash": chat_hash,
            "system_message": "none added (model template default kept)", "assistant_anchor": "none",
            "prompt_body_sha256": L.sha(pushy), "plain_prompt_body_sha256": L.sha(plain),
            "prompt_sha256": prompt_sha, "plain_chat_prompt_sha256": plain_prompt_sha,
            "injection_token": tok, "questions_file": q_path, "questions_file_sha256": q_sha,
            "sample_order_sha256": order_sha, "steering_fires": fires, "steering_fires_expected": expect,
            "provenance": prov,
            "accuracy_inline_pct": round(sum(r["correct_neutral"] for r in rows) / n * 100, 2),
        }
        os.makedirs(os.path.dirname(paths[a]), exist_ok=True)
        L.atomic_write_json({"meta": meta, "data": rows}, paths[a])
        print(f"  wrote {paths[a]}  fires={fires}/{expect}")
    print("\nFinished. Score OFFLINE (inline fields are process state).")


if __name__ == "__main__":
    main()
