#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinQA Chat 0-shot CoT generation, protocol `finqa-chat-0shot-cot-v1`.

A supplementary Native Chat experiment layered on the CLOSED `finqa-v0` Bare
line (which stays untouched). Designed AFTER seeing the Bare result, so it is
not a blind test. Fixed workpoints, no dose search:
    llama3   band [11,20)  alpha {0, -6}   expected fires {0, 2700}
    qwen2.5  band [16,22)  alpha {0, +8}   expected fires {0, 1800}

PROMPT: the body of finqa/prompt_finqa.py is reused verbatim (CoT line, answer
format, trailing `Answer: `) as the ONLY user message; wrapped with
tokenizer.apply_chat_template(add_generation_prompt=True), duplicated BOS
stripped. No system message is written here, no assistant-side anchor is
appended. Injection = prefill-only, tail=1, i.e. the last token of the model's
own assistant header; its id is READ from the real tokenizer for all prompts
(never the Bare token). NOTE: some chat templates (Llama-3.1) trim the user
content, so the trailing space of `Answer: ` may not survive wrapping; the
wrapped-prompt hash recorded in meta is what was actually fed.

Runs the full 300 items, both alphas, in one model load -> <root>/formal.
This script never reads gold and never computes accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import chat_cot_sat_lib as L  # noqa: E402  (stdlib-only helpers shared with the other Native Chat line)
from prompt_finqa import FINQA_TEMPLATE, build_finqa_prompt  # noqa: E402
from finqa_scoring import diagnostics_for_text  # noqa: E402

PROTOCOL = "finqa-chat-0shot-cot-v1"
N_ITEMS = 300
MAX_NEW_TOKENS, TEMPERATURE, TOP_P, BATCH_SIZE = 512, 0.0, 1.0, 8
STOP_REASONS = ("natural_eos", "budget_exhausted")
OUT_SUBDIR = "finqa/chat_0shot_cot_v1"

MODELS = {
    "llama3": dict(
        model_dir="meta-llama/Llama-3.1-8B-Instruct", size="8B",
        mask_rel="mask/llama3_non_logits/nmd_0.5_11_20_8B.npy", band=(11, 20), L=9, n_decoder=32,
        work_alpha=-6, header="<|start_header_id|>assistant<|end_header_id|>",
        wrapper_id="llama3-chat-template-v1"),
    "qwen2.5": dict(
        model_dir="Qwen/Qwen2.5-7B-Instruct", size="7B",
        mask_rel="mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy", band=(16, 22), L=6, n_decoder=28,
        work_alpha=8, header="<|im_start|>assistant", wrapper_id="qwen2.5-chat-template-v1"),
}


class FQError(Exception):
    pass


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def alphas_for(model):
    return [0, MODELS[model]["work_alpha"]]


def alpha_dir(alpha):
    return f"mdf_{int(alpha)}"          # signed: mdf_0, mdf_-6, mdf_8


def cell_filename(model):
    ls, le = MODELS[model]["band"]
    return f"finqa_chat_{MODELS[model]['size']}_{ls}_{le}.json"


def expected_fires(model, alpha, n):
    return 0 if alpha == 0 else MODELS[model]["L"] * n * 1


def file_sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def prompt_digest(samples):
    """Same formula as data_finqa.build_dataset's prompt_content_sha256_16."""
    return hashlib.sha256(json.dumps(
        [[d["question"], d["pre_text"], d["post_text"], d["table_linear"]] for d in samples],
        ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def load_samples(path):
    """Load finqa_formal.json, validate it against its own manifest, return (meta, all 300 samples)."""
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d["meta"], d["data"]
    if len(data) != N_ITEMS:
        raise FQError(f"{path}: {len(data)} rows, expected {N_ITEMS}")
    if [str(s["sample_id"]) for s in data] != [str(i) for i in range(N_ITEMS)]:
        raise FQError(f"{path}: sample_ids are not 0..{N_ITEMS - 1} in order")
    if len({s["id"] for s in data}) != N_ITEMS:
        raise FQError(f"{path}: duplicate official ids")
    for s in data:
        for k in ("id", "question", "pre_text", "post_text", "table_linear"):
            if not isinstance(s.get(k), str):
                raise FQError(f"{path}: sample {s.get('sample_id')} field {k!r} missing / not a string")
    if prompt_digest(data) != meta.get("prompt_content_sha256_16"):
        raise FQError(f"{path}: recomputed prompt_content_sha256_16 {prompt_digest(data)} != "
                      f"manifest {meta.get('prompt_content_sha256_16')} (file edited or built by other code)")
    return meta, data


def order_sha(samples):
    return hashlib.sha256("\n".join(f"{s['sample_id']}|{s['id']}" for s in samples).encode("utf-8")).hexdigest()


def identity_of(samples):
    return [[str(s["sample_id"]), s["id"], s["question"]] for s in samples]


# ----------------------------------------------------------------- chat wrapping (same logic as the other Native Chat runners)
def strip_leading_bos(tokenizer, text):
    bos = getattr(tokenizer, "bos_token", None)
    return text[len(bos):] if bos and text.startswith(bos) else text


def wrap_prompts(tokenizer, user_messages):
    """Sole user turn -> apply_chat_template(add_generation_prompt=True) -> duplicated BOS stripped."""
    return [strip_leading_bos(tokenizer, tokenizer.apply_chat_template(
        [{"role": "user", "content": m}], tokenize=False, add_generation_prompt=True)) for m in user_messages]


def assert_no_double_bos(tokenizer, texts):
    bos_id = getattr(tokenizer, "bos_token_id", None)
    for i, t in enumerate(texts):
        ids = tokenizer(t, add_special_tokens=True)["input_ids"]
        if bos_id is not None and len(ids) >= 2 and ids[:2] == [bos_id, bos_id]:
            raise FQError(f"double BOS in chat prompt {i} (head={ids[:4]})")


# ----------------------------------------------------------------- existing-cell verification (fail-closed)
def verify_existing_cell(path, model, alpha, samples, expect):
    """Problems (list of str) for an existing cell. Empty => complete AND same provenance as `expect`
    (a dict of the values this run would write). Never trusts a file merely because it parses."""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        return [f"unreadable: {e}"]
    m, rows = d.get("meta", {}), d.get("data", [])
    cfg, n = MODELS[model], len(samples)
    P = []

    def need(c, msg):
        if not c:
            P.append(msg)
    need(m.get("protocol") == PROTOCOL, f"protocol {m.get('protocol')}")
    need(m.get("model") == model and m.get("alpha") == alpha, f"model/alpha {m.get('model')}/{m.get('alpha')}")
    need(m.get("cot") is True and m.get("n_shot") == 0 and m.get("chat_template_applied") is True, "cot/n_shot/chat flags")
    need(m.get("accuracy_computed") is False, "accuracy_computed")
    need(tuple(m.get("layers", ())) == tuple(cfg["band"]), f"layers {m.get('layers')}")
    need((m.get("max_new_tokens"), m.get("temperature"), m.get("top_p"), m.get("batch_size"),
          m.get("prefill_only"), m.get("prefill_tail_len")) ==
         (MAX_NEW_TOKENS, TEMPERATURE, TOP_P, BATCH_SIZE, True, 1), "generation settings")
    for k in ("mask_sha256", "user_prompt_sha256", "wrapped_prompt_sha256", "chat_template_hash", "model_dir",
              "questions_file_sha256", "sample_order_sha256", "prompt_content_sha256_16", "mode"):
        need(m.get(k) == expect.get(k), f"{k} differs from this run")
    need(m.get("steering_fires") == expected_fires(model, alpha, n) == m.get("steering_fires_expected"),
         f"steering_fires {m.get('steering_fires')} vs expected {expected_fires(model, alpha, n)}")
    need(m.get("n_samples") == n == len(rows), f"rows {len(rows)} meta.n {m.get('n_samples')} expected {n}")
    inj = m.get("injection_token") or {}
    need(inj.get("final_token_id") == expect.get("final_token_id"), "injection_token final_token_id")
    got = [[str(r.get("sample_id")), r.get("id"), r.get("question")] for r in rows]
    need(got == identity_of(samples), "per-item sample_id/id/question order differs from the data file")
    for i, r in enumerate(rows):
        t = r.get("generated")
        if (not isinstance(t, str) or r.get("stop_reason") not in STOP_REASONS
                or not isinstance(r.get("generated_token_count"), int) or r["generated_token_count"] < 0):
            P.append(f"row {i} incomplete")
            break
        diag = diagnostics_for_text(t, MAX_NEW_TOKENS, r["generated_token_count"])
        if any(r.get(k) != v for k, v in diag.items()):
            P.append(f"row {i} stored diagnostics do not match its text")
            break
    return P


def make_rows(samples, gen):
    rows = []
    for s, g in zip(samples, gen):
        if g["stop_reason"] not in STOP_REASONS:
            raise FQError(f"unknown stop_reason {g['stop_reason']!r}")
        rows.append({"sample_id": s["sample_id"], "id": s["id"], "question": s["question"],
                     "generated": g["text"], "generated_token_count": g["generated_token_count"],
                     "stop_reason": g["stop_reason"],
                     **diagnostics_for_text(g["text"], MAX_NEW_TOKENS, g["generated_token_count"])})
    return rows


def build_meta(model, alpha, n, qmeta, q_path, model_dir, mask_path, nz, tok, chat_hash, expect, fires, exp,
               prov, padding_side):
    cfg = MODELS[model]
    ls, le = cfg["band"]
    return {
        "protocol": PROTOCOL, "model": model, "size": cfg["size"], "alpha": alpha,
        "layers": [ls, le], "L": cfg["L"], "n_samples": n, "mode": expect["mode"],
        "restricted_indices": None,
        "model_dir": model_dir, "mask_path": mask_path, "mask_sha256": expect["mask_sha256"],
        "mask_nonzero_rows": nz, "max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE,
        "top_p": TOP_P, "batch_size": BATCH_SIZE, "prefill_only": True, "prefill_tail_len": 1,
        "role": "neutral", "cot": True, "n_shot": 0, "padding_side": padding_side,
        "prompt_wrapper_id": cfg["wrapper_id"], "chat_template_applied": True, "chat_template_hash": chat_hash,
        "system_message": "none written (model template default kept)", "assistant_anchor": "none",
        "prompt_template_module": "finqa/prompt_finqa.FINQA_TEMPLATE",
        "prompt_template_sha256": L.sha(FINQA_TEMPLATE),
        "user_prompt_sha256": expect["user_prompt_sha256"], "wrapped_prompt_sha256": expect["wrapped_prompt_sha256"],
        "injection_token": tok, "questions_file": q_path, "questions_file_sha256": expect["questions_file_sha256"],
        "sample_order_sha256": expect["sample_order_sha256"],
        "prompt_content_sha256_16": expect["prompt_content_sha256_16"],
        "steering_fires": fires, "steering_fires_expected": exp, "accuracy_computed": False,
        "provenance": prov,
    }


def provenance(np, torch, transformers, vc):
    try:
        devs = sorted({str(p.device) for p in vc.model.parameters()})
    except Exception:
        devs = []
    return {"host": platform.node(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "param_devices": devs, "sharded": len(devs) > 1,
            "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "library_versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                                 "numpy": np.__version__},
            "code_version": L.code_version(ROOT)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--base_dir", required=True, help="components/ (benchmark/, mask/, <model>/ live here)")
    ap.add_argument("--questions", default=None, help="default <base_dir>/benchmark/finqa_formal.json")
    ap.add_argument("--model_dir", default=None)
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    import transformers
    import utils
    from llms import VicundaModel
    from tqdm import tqdm

    cfg = MODELS[args.model]
    model_dir = args.model_dir or cfg["model_dir"]
    q_path = args.questions or os.path.join(args.base_dir, "benchmark", "finqa_formal.json")
    mask_path = os.path.join(args.base_dir, cfg["mask_rel"])
    out_dir = os.path.join(args.base_dir, args.model, OUT_SUBDIR, "formal")
    ls, le = cfg["band"]
    try:
        qmeta, all_samples = load_samples(q_path)
    except FQError as e:
        die(str(e))
    samples = all_samples
    n = len(samples)

    vc = VicundaModel(model_path=model_dir)
    vc.model.eval()
    tk = vc.tokenizer
    if tk.padding_side != "left":
        die(f"padding_side {tk.padding_side!r}, expected 'left'")
    if not getattr(tk, "chat_template", None):
        die("tokenizer has no chat_template")
    users = [build_finqa_prompt(s) for s in samples]
    wrapped = wrap_prompts(tk, users)
    if wrapped[0] == users[0] or cfg["header"] not in wrapped[0]:
        die(f"chat template did not fire / header {cfg['header']!r} missing in the first wrapped prompt")
    try:
        assert_no_double_bos(tk, wrapped)
        tok = L.check_prompt_tokens(tk, wrapped)
    except (FQError, ValueError) as e:
        die(str(e))
    chat_hash = L.sha(tk.chat_template)
    print(f"[tokens] BOS counts {tok['bos_counts']}  final token {tok['final_token_id']} "
          f"{tok['final_token_text']!r}  (all {n} prompts)")
    print("[prompt] first wrapped prompt tail:", repr(wrapped[0][-220:]))

    raw_mask = np.load(mask_path)
    n_dec = len(vc._find_decoder_layers())
    if n_dec != cfg["n_decoder"] or raw_mask.shape[0] != n_dec:
        die(f"decoder layers {n_dec} / mask rows {raw_mask.shape[0]} / expected {cfg['n_decoder']}")
    nz = sorted(int(i) for i in np.nonzero(np.any(raw_mask != 0, axis=1))[0])
    want = sorted(utils.decoder_layer_range(ls, le))
    if nz != want:
        die(f"mask non-zero rows {nz} != configured band rows {want}")

    expect = {"mask_sha256": file_sha256(mask_path), "user_prompt_sha256": L.sha("\n".join(users)),
              "wrapped_prompt_sha256": L.sha("\n".join(wrapped)), "chat_template_hash": chat_hash,
              "model_dir": model_dir, "questions_file_sha256": file_sha256(q_path),
              "sample_order_sha256": order_sha(samples),
              "prompt_content_sha256_16": qmeta["prompt_content_sha256_16"],
              "mode": "formal", "final_token_id": tok["final_token_id"]}
    fname = cell_filename(args.model)
    paths = {a: os.path.join(out_dir, alpha_dir(a), fname) for a in alphas_for(args.model)}
    todo = []
    for a in alphas_for(args.model):
        if os.path.exists(paths[a]):
            probs = verify_existing_cell(paths[a], args.model, a, samples, expect)
            if probs:
                die(f"existing cell {paths[a]} FAILED verification ({'; '.join(probs)}); refusing to skip or overwrite")
            print(f"[skip] alpha={a}: existing cell verified")
        else:
            todo.append(a)
    prov = provenance(np, torch, transformers, vc)
    os.makedirs(out_dir, exist_ok=True)
    for a in todo:
        diff = raw_mask * a                       # alpha=0 -> a REAL all-zero matrix, same code path
        vc.steering_fire_count(reset=True)
        gen = []
        for i in tqdm(range(0, n, BATCH_SIZE), desc=f"finqa-chat {args.model} a={a}"):
            batch = wrapped[i:i + BATCH_SIZE]
            out = vc.regenerate(batch, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P,
                                diff_matrices=list(diff), prefill_only=True, prefill_tail_len=1,
                                batch_size=BATCH_SIZE, return_metadata=True)
            if len(out) != len(batch):
                die(f"regenerate returned {len(out)} replies for {len(batch)} prompts (alpha={a}, batch@{i})")
            gen.extend(out)
        fires, exp = vc.steering_fire_count(), expected_fires(args.model, a, n)
        if len(gen) != n or fires != exp:
            die(f"alpha={a}: {len(gen)} replies / steering_fires {fires} != {exp}; cell unusable")
        try:
            rows = make_rows(samples, gen)
        except FQError as e:
            die(str(e))
        meta = build_meta(args.model, a, n, qmeta, q_path, model_dir, mask_path, nz, tok, chat_hash, expect,
                          fires, exp, prov, tk.padding_side)
        os.makedirs(os.path.dirname(paths[a]), exist_ok=True)
        L.atomic_write_json({"meta": meta, "data": rows}, paths[a])
        print(f"  wrote {paths[a]}  fires={fires}/{exp}")
    print("\nFinished. Score OFFLINE with finqa/eval_finqa_chat.py (this script never reads gold).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
