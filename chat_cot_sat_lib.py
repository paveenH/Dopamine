#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chat_cot_sat_lib.py -- pure-stdlib logic for the Chat-CoT Saturation Check (protocol chat-cot-sat-v1).

Kept free of torch/numpy so the load-bearing invariants are unit-testable on a box without a GPU stack
(test_chat_cot_sat.py). The runner get_answer_chat_cot_sat.py imports everything from here.

The ONLY change versus the Native Chat No-CoT user message is the single line COT_LINE, placed (by the
existing template) before the final-answer format instruction. remove_cot_line() + the per-prompt
checks make that a verified fact rather than an assumption.
"""
import hashlib
import json
import os

PROTOCOL = "chat-cot-sat-v1"
COT_LINE = "Let's think step by step."
MAX_NEW_TOKENS, TEMPERATURE, TOP_P, BATCH_SIZE, N_ITEMS = 768, 0.0, 1.0, 24, 300
STOP_REASONS = ("natural_eos", "budget_exhausted")

MODELS = {
    "llama3": dict(
        model_dir="meta-llama/Llama-3.1-8B-Instruct", size="8B", out_name="llama3",
        mask_rel="mask/llama3_non_logits/nmd_0.5_11_20_8B.npy", band=(11, 20), L=9, n_decoder=32,
        work_alpha=-6, header="<|start_header_id|>assistant<|end_header_id|>",
        wrapper_id="llama3-chat-template-v1"),
    "qwen2.5": dict(
        model_dir="Qwen/Qwen2.5-7B-Instruct", size="7B", out_name="qwen2.5",
        mask_rel="mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy", band=(16, 22), L=6, n_decoder=28,
        work_alpha=8, header="<|im_start|>assistant", wrapper_id="qwen2.5-chat-template-v1"),
}
TASKS = {
    "gsm8k": dict(questions_rel="benchmark/gsm8k_test_sample.json", label_free=False,
                  out_subdir="gsm8k_chat_cot_sat_v1", text_key="generated_neutral"),
    "gsm_hard": dict(questions_rel="benchmark/gsm_hard_p3_questions.json", label_free=True,
                     out_subdir="gsm_hard_chat_cot_sat_v1", text_key="generated"),
}


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def alphas_for(model):
    """Fixed two-cell family: baseline first, then the model's frozen workpoint. Nothing else is allowed."""
    return [0, MODELS[model]["work_alpha"]]


def alpha_dir(alpha):
    return f"mdf_{int(alpha)}"          # signed: mdf_0, mdf_-6, mdf_8


def cell_filename(model, task, band):
    ls, le = band
    sz = MODELS[model]["size"]
    return (f"gsm8k_chat_cot_{sz}_answers_{ls}_{le}.json" if task == "gsm8k"
            else f"gsm_hard_chat_cot_{sz}_{ls}_{le}.json")


def expected_fires(model, alpha, n):
    return 0 if alpha == 0 else MODELS[model]["L"] * n * 1   # layers x generations x tail_len=1


# ------------------------------------------------------------------ prompts
def remove_cot_line(text):
    """Remove the LAST occurrence of 'COT_LINE\\n' (the template line), never a copy inside the question."""
    needle = COT_LINE + "\n"
    i = text.rfind(needle)
    if i < 0:
        raise ValueError("CoT line not found")
    return text[:i] + text[i + len(needle):]


def template_pair(select_templates_gsm8k, wording="plain"):
    """(No-CoT neutral body, CoT neutral body). Asserts they differ by exactly the one CoT line."""
    no = select_templates_gsm8k(suite="default", cot=False, wording=wording)["neutral"]
    co = select_templates_gsm8k(suite="default", cot=True, wording=wording)["neutral"]
    if co.count(COT_LINE) != 1 or no.count(COT_LINE) != 0:
        raise ValueError("CoT line count wrong in templates")
    if remove_cot_line(co) != no:
        raise ValueError("CoT template != No-CoT template + exactly one CoT line; refusing to assume other "
                         "wording is unchanged")
    if not co.index(COT_LINE) < co.index("Provide your final numeric answer"):
        raise ValueError("CoT line must precede the final-answer format instruction")
    return no, co


def render_pair(no_tmpl, co_tmpl, question):
    p_no, p_co = no_tmpl.format(context=question), co_tmpl.format(context=question)
    if remove_cot_line(p_co) != p_no:
        raise ValueError("per-prompt CoT removal does not reproduce the No-CoT user message")
    return p_no, p_co


def check_prompt_tokens(tokenizer, texts):
    """Tokenise exactly as vc.regenerate does (add_special_tokens=True). Returns a summary or raises:
    exactly one BOS when the tokenizer has a BOS id; ONE uniform final token across ALL prompts."""
    bos, bos_counts, tails = tokenizer.bos_token_id, set(), {}
    for t in texts:
        ids = tokenizer(t, add_special_tokens=True)["input_ids"]
        if bos is not None:
            bos_counts.add(ids.count(bos))
        tails[ids[-1]] = tails.get(ids[-1], 0) + 1
    if bos is not None and bos_counts != {1}:
        raise ValueError(f"BOS counts per prompt {sorted(bos_counts)}, expected exactly {{1}}")
    if len(tails) != 1:
        raise ValueError(f"final prompt token is not uniform: {sorted(tails.items())}")
    k = next(iter(tails))
    return {"bos_token_id": bos, "bos_counts": sorted(bos_counts), "n_prompts": len(texts),
            "final_token_id": int(k), "final_token_text": tokenizer.decode([k]),
            "final_token_counts": {f"{kk}:{tokenizer.decode([kk])!r}": v for kk, v in tails.items()}}


# ------------------------------------------------------------------ existing-cell verification
def item_identity(task, samples):
    """Per-item identity the cell must carry, in order: GSM8K (question, gold); GSM-Hard (sample_id, question)."""
    if task == "gsm8k":
        return [[s["question"], str(s["answer"])] for s in samples]
    return [[str(s["sample_id"]), s["question"]] for s in samples]


def verify_existing_cell(path, model, task, alpha, n, mask_sha256, prompt_sha256, model_dir, q_file_sha256,
                         order_sha256, identity):
    """Problems (list of str) for an existing cell. Empty list => complete AND from the same provenance
    (model repo id, data file hash, per-item question/gold/ID order, prefill/top_p settings included)."""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception as e:                      # unreadable == failed
        return [f"unreadable: {e}"]
    m, rows, cfg = d.get("meta", {}), d.get("data", []), MODELS[model]
    P = []
    def need(cond, msg):
        if not cond:
            P.append(msg)
    need(m.get("protocol") == PROTOCOL, f"protocol {m.get('protocol')}")
    need(m.get("model") == model and m.get("task") == task, f"model/task {m.get('model')}/{m.get('task')}")
    need(m.get("alpha") == alpha, f"alpha {m.get('alpha')}")
    need(m.get("cot") is True and m.get("chat_template_applied") is True, "cot/chat flags")
    need(tuple(m.get("layers", ())) == tuple(cfg["band"]), f"layers {m.get('layers')}")
    need(m.get("mask_sha256") == mask_sha256, "mask_sha256")
    need(m.get("prompt_sha256") == prompt_sha256, "prompt_sha256 (different prompts)")
    need((m.get("max_new_tokens"), m.get("temperature"), m.get("top_p"), m.get("batch_size"),
          m.get("prefill_tail_len"), m.get("prefill_only")) ==
         (MAX_NEW_TOKENS, TEMPERATURE, TOP_P, BATCH_SIZE, 1, True), "generation settings")
    need(m.get("model_dir") == model_dir, f"model_dir {m.get('model_dir')} != {model_dir}")
    need(m.get("questions_file_sha256") == q_file_sha256, "questions_file_sha256")
    need(m.get("sample_order_sha256") == order_sha256, "sample_order_sha256")
    need((m.get("suite"), m.get("fmt_wording"), m.get("role")) == ("default", "plain", "neutral"), "suite/wording/role")
    need(m.get("steering_fires") == expected_fires(model, alpha, n) == m.get("steering_fires_expected"),
         f"steering_fires {m.get('steering_fires')} vs expected {expected_fires(model, alpha, n)}")
    need(len(rows) == n == m.get("n"), f"rows {len(rows)} meta.n {m.get('n')} expected {n}")
    tk = TASKS[task]["text_key"]
    got_id = [[r.get("question"), str(r.get("answer"))] if task == "gsm8k"
              else [str(r.get("sample_id")), r.get("question")] for r in rows]
    need(got_id == identity, "per-item question/gold/sample_id order differs from the current data file")
    for i, r in enumerate(rows):
        if not isinstance(r.get(tk), str) or r.get("stop_reason") not in STOP_REASONS \
                or not isinstance(r.get("generated_token_count"), int) or r["generated_token_count"] < 0:
            P.append(f"row {i} incomplete")
            break
    return P


def atomic_write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def code_version(repo_dir):
    import subprocess
    try:
        h = subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "HEAD"], text=True, timeout=10).strip()
        dirty = bool(subprocess.check_output(["git", "-C", repo_dir, "status", "--porcelain"], text=True, timeout=10).strip())
        return f"{h}{'+dirty' if dirty else ''}"
    except Exception:
        return "unknown"
