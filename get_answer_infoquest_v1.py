#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_answer_infoquest_v1.py -- InfoQuest information-gap selectivity (infoquest-v1).

Question: does RSN change WHICH information a model asks for on its FIRST reply,
depending on whether the key facts are already in the request?

Design (see CLAUDE.md "InfoQuest" entry; no real result exists yet):
  * Native Chat, neutral role, ONE user message = the condition text
    (Incomplete = original request; Complete = request + factual supplement).
  * No instruction to ask questions, no JSON, no scoring hints.
  * Greedy, max_new_tokens=512, batch decoding; reply = the first assistant turn.
  * Prefill-only, tail=1, output-side injection (vc.regenerate prefill_only path),
    each model's OWN Role-derived RSN mask and band:
        llama3   [11,20)  decoder 10-18, L=9, mask nmd_0.5_11_20_8B.npy
        qwen2.5  [16,22)  decoder 15-20, L=6, mask nmd_0.5_16_22_7B.npy
  * pilot: alpha=0 only, 30 requests.  formal: alpha in {-4,0,+4}, 100 requests.
  * The native assistant header is kept as the last prefill token; NO anchor is
    added to recover token 220.

Fail-closed: the scenario file must match scenarios_v1.freeze.json (which exists
only if every scenario was reviewed); an existing cell is resumed ONLY when its
provenance key is identical, otherwise the run stops WITHOUT overwriting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL = "infoquest-v1"
MAX_NEW_TOKENS = 512
BATCH_SIZE = 24
ALPHAS = {"pilot": [0], "formal": [-4, 0, 4]}
N_EXPECT = {"pilot": 30, "formal": 100}
CONDITIONS = ("incomplete", "complete")

MODELS = {
    "llama3": {"model_dir": "meta-llama/Llama-3.1-8B-Instruct", "size": "8B", "band": (11, 20),
               "mask_rel": "mask/llama3_non_logits/nmd_0.5_11_20_8B.npy",
               "wrapper": "llama3-chat-template-v1", "out": "llama3"},
    "qwen2.5": {"model_dir": "Qwen/Qwen2.5-7B-Instruct", "size": "7B", "band": (16, 22),
                "mask_rel": "mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy",
                "wrapper": "qwen2.5-chat-template-v1", "out": "qwen2.5"},
}


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    with open(p, "rb") as f:
        return sha256_bytes(f.read())


def sha256_text(s):
    return sha256_bytes(s.encode("utf-8"))


def alpha_tag(a):
    return f"neg{-a}" if a < 0 else str(a)


# --------------------------------------------------------------------------- data
def load_frozen_scenarios(scenarios_path, freeze_path, stage):
    """Return (scenarios of this stage in file order, scenarios_sha256).
    Refuses unless a freeze manifest exists and matches the file byte-for-byte."""
    if not os.path.isfile(freeze_path):
        die(f"{freeze_path} missing: scenarios have not been reviewed and frozen "
            "(prepare_infoquest_v1.py freeze). Real runs need frozen material.")
    man = json.load(open(freeze_path, encoding="utf-8"))
    got = sha256_file(scenarios_path)
    if man.get("scenarios_sha256") != got:
        die("scenarios file does not match its freeze manifest "
            f"({got[:12]} vs {str(man.get('scenarios_sha256'))[:12]})")
    d = json.load(open(scenarios_path, encoding="utf-8"))
    sc = [s for s in d["scenarios"] if s["split"] == stage]
    if len(sc) != N_EXPECT[stage]:
        die(f"{len(sc)} {stage} scenarios, expected {N_EXPECT[stage]}")
    if any(s["review_status"] != "reviewed" for s in sc):
        die("unreviewed scenario in a frozen file")
    return sc, got


def build_jobs(scenarios):
    """Deterministic job order: scenario order x (incomplete, complete)."""
    jobs = []
    for s in scenarios:
        jobs.append((s, "incomplete", s["incomplete_text"]))
        jobs.append((s, "complete", s["complete_text"]))
    return jobs


# ------------------------------------------------------------------------ prompts
def strip_leading_bos(tok, text):
    bos = getattr(tok, "bos_token", None)
    return text[len(bos):] if bos and text.startswith(bos) else text


def render_chat(tok, user_text):
    msgs = [{"role": "user", "content": user_text}]
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return msgs, strip_leading_bos(tok, rendered)


def check_tokenization(tok, rendered, label):
    """No duplicated special tokens; returns the exact id list vc.regenerate will see."""
    ids = tok(rendered, add_special_tokens=True)["input_ids"]
    bos = getattr(tok, "bos_token_id", None)
    if bos is not None:
        n_bos = sum(1 for i in ids if i == bos)
        if ids[0] != bos or n_bos != 1:
            die(f"{label}: expected exactly one leading BOS, got {n_bos} (head={ids[:3]})")
    return ids


# ------------------------------------------------------------------- provenance
def provenance_key(meta):
    keys = ("protocol", "model", "stage", "alpha", "layer_start", "layer_end", "mask_sha256",
            "scenarios_sha256", "model_dir", "model_config_sha256", "chat_template_hash", "prompt_ids_sha256", "max_new_tokens",
            "temperature", "batch_size", "prefill_only", "prefill_tail_len", "n_rows")
    return {k: meta.get(k) for k in keys}


def cell_path(out_dir, stage, size, ls, le, alpha):
    return os.path.join(out_dir, stage, f"infoquest_v1_{stage}_{size}_{ls}_{le}_a{alpha_tag(alpha)}.json")


def try_resume(path, new_meta):
    """True -> identical complete cell already there (skip). False -> absent.
    Anything else (conflicting provenance, incomplete) -> stop, never overwrite."""
    if not os.path.exists(path):
        return False
    try:
        old = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        die(f"{path} exists but is unreadable ({e}); not overwriting")
    if provenance_key(old.get("meta", {})) != provenance_key(new_meta):
        die(f"{path} exists with DIFFERENT provenance; refusing to overwrite or merge. "
            "Move it away deliberately if a re-run under the new source is intended.")
    if len(old.get("data", [])) != new_meta["n_rows"]:
        die(f"{path} exists with the same provenance but {len(old.get('data', []))} rows; not appending")
    return True


# ------------------------------------------------------------------------- run
def run_cells(vc, model_key, stage, scenarios, scenarios_sha, mask, out_dir, mask_sha, mask_path,
              max_new_tokens=MAX_NEW_TOKENS, batch_size=BATCH_SIZE, decoder_layer_range=None, model_dir=None):
    import numpy as np
    cfg = MODELS[model_key]
    ls, le = cfg["band"]
    tok = vc.tokenizer
    if decoder_layer_range is None:
        import utils
        decoder_layer_range = utils.decoder_layer_range
    n_dec = len(vc._find_decoder_layers())
    if mask.shape[0] != n_dec:
        die(f"mask has {mask.shape[0]} rows but model has {n_dec} decoder layers (do not slice the mask)")
    nz = sorted(int(i) for i in np.nonzero(np.any(mask != 0, axis=1))[0])
    want = sorted(decoder_layer_range(ls, le))
    if nz != want:
        die(f"mask non-zero rows {nz} != decoder_layer_range({ls},{le}) {want}")
    L = len(want)
    if getattr(tok, "padding_side", "left") != "left":
        die("tokenizer.padding_side must be 'left'")

    jobs = build_jobs(scenarios)
    rendered, ids_list, msgs_list = [], [], []
    for s, cond, text in jobs:
        msgs, r = render_chat(tok, text)
        rendered.append(r)
        msgs_list.append(msgs)
        ids_list.append(check_tokenization(tok, r, f"{s['scenario_id']}/{cond}"))
    prompt_ids_sha = sha256_text(json.dumps(ids_list))
    chat_template = getattr(tok, "chat_template", None) or ""
    if not chat_template:
        die("tokenizer has no chat_template")
    first = rendered[0]
    tails = [{"id": int(i), "text": tok.decode([i])} for i in ids_list[0][-4:]]
    tail_ids_all = {ids[-1] for ids in ids_list}
    print(f"[prompts] n={len(jobs)} prompt_ids_sha256={prompt_ids_sha[:16]} "
          f"last-token ids across all prompts={sorted(tail_ids_all)}")
    print(f"[prompts] first rendered (repr, 300 chars): {first[:300]!r}")
    print(f"[inject] tail of first prompt = {tails}")

    try:
        devs = sorted({str(p.device) for p in vc.model.parameters()})
    except Exception:
        devs = []
    prov = {"host": platform.node(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "param_devices": devs, "sharded": len(devs) > 1}

    model_dir = model_dir or cfg["model_dir"]
    mc = getattr(vc.model, "config", None)
    model_cfg_sha = sha256_text(json.dumps(mc.to_dict(), sort_keys=True, default=str)) if hasattr(mc, "to_dict") else None
    alphas = ALPHAS[stage]
    plan = []
    for a in alphas:
        meta = {
            "protocol": PROTOCOL, "model": model_key, "size": cfg["size"], "stage": stage, "alpha": a,
            "layer_start": ls, "layer_end": le, "L": L, "mask_path": mask_path, "mask_sha256": mask_sha,
            "model_dir": model_dir, "model_config_sha256": model_cfg_sha,
            "scenarios_sha256": scenarios_sha, "n_scenarios": len(scenarios), "n_rows": len(jobs),
            "max_new_tokens": max_new_tokens, "temperature": 0.0, "top_p": 1.0, "batch_size": batch_size,
            "prefill_only": True, "prefill_tail_len": 1, "role": "neutral",
            "prompt_wrapper_id": cfg["wrapper"], "chat_template_applied": True,
            "chat_template_hash": sha256_text(chat_template), "prompt_ids_sha256": prompt_ids_sha,
            "system_message_added_by_template": "system" in first.split(msgs_list[0][0]["content"])[0].lower(),
            "injection_tail_tokens_first_prompt": tails,
            "injection_last_token_ids_all_prompts": sorted(int(i) for i in tail_ids_all),
            "extra_anchor_added": False, "terminators": [int(t) for t in getattr(vc, "terminators", [])],
            "provenance": prov, "padding_side": tok.padding_side,
        }
        path = cell_path(out_dir, stage, cfg["size"], ls, le, a)
        plan.append((a, path, meta, try_resume(path, meta)))

    written = []
    for a, path, meta, done in plan:
        if done:
            print(f"[resume] alpha={a}: identical complete cell exists -> skip ({path})")
            continue
        vc.steering_fire_count(reset=True)
        diff = list(mask * a)
        gen = []
        for i in range(0, len(rendered), batch_size):
            gen.extend(vc.regenerate(rendered[i:i + batch_size], max_new_tokens=max_new_tokens,
                                     temperature=0.0, top_p=1.0, diff_matrices=diff,
                                     prefill_only=True, prefill_tail_len=1,
                                     batch_size=batch_size, return_metadata=True))
        if len(gen) != len(jobs):
            die(f"alpha={a}: {len(gen)} generations for {len(jobs)} prompts")
        fires = vc.steering_fire_count()
        expect = 0 if a == 0 else L * len(jobs)
        if fires != expect:
            die(f"alpha={a}: steering_fires {fires} != {expect} (L={L}, n={len(jobs)})")
        rows = []
        for (s, cond, text), msgs, ids, g in zip(jobs, msgs_list, ids_list, gen):
            rows.append({
                "scenario_id": s["scenario_id"], "request_id": s["request_id"], "condition": cond,
                "alpha": a, "messages": msgs, "prompt_sha256": sha256_text(json.dumps(msgs)),
                "prompt_token_count": len(ids), "generated": g["text"],
                "generated_token_count": g["generated_token_count"], "stop_reason": g["stop_reason"],
                "hit_max_new_tokens": g["stop_reason"] == "budget_exhausted",
                "empty": not g["text"].strip(),
                "injection_expected_fires": 0 if a == 0 else L,
            })
        meta = {**meta, "steering_fires": fires}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "data": rows}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        written.append(path)
        print(f"[wrote] {path} steering_fires={fires}")
    return written


def main():
    ap = argparse.ArgumentParser(description="InfoQuest information-gap selectivity generation")
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--stage", required=True, choices=sorted(ALPHAS))
    ap.add_argument("--scenarios", default=None,
                    help="default: <base_dir>/benchmark/infoquest_v1/scenarios_v1.json")
    ap.add_argument("--freeze", default=None,
                    help="default: <base_dir>/benchmark/infoquest_v1/scenarios_v1.freeze.json")
    ap.add_argument("--base_dir", default=os.environ.get("BASE_DIR", "components"))
    ap.add_argument("--mask_path", default=None)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()

    import numpy as np
    from llms import VicundaModel
    cfg = MODELS[a.model]
    mask_path = a.mask_path or os.path.join(a.base_dir, cfg["mask_rel"])
    if not os.path.isfile(mask_path):
        die(f"mask not found: {mask_path}")
    out_dir = a.out_dir or os.path.join(a.base_dir, cfg["out"], "infoquest_v1")
    mat_dir = os.path.join(a.base_dir, "benchmark", "infoquest_v1")
    sc_path = a.scenarios or os.path.join(mat_dir, "scenarios_v1.json")
    fz_path = a.freeze or os.path.join(mat_dir, "scenarios_v1.freeze.json")
    for pth in (sc_path, fz_path):
        if not os.path.isfile(pth):
            die(f"not found: {pth} (looked exactly here; no fallback to the repo copy)")
    scenarios, sc_sha = load_frozen_scenarios(sc_path, fz_path, a.stage)
    mask = np.load(mask_path)
    vc = VicundaModel(model_path=a.model_dir or cfg["model_dir"])
    vc.model.eval()
    run_cells(vc, a.model, a.stage, scenarios, sc_sha, mask, out_dir, sha256_file(mask_path), mask_path,
              model_dir=a.model_dir or cfg["model_dir"])
    print("done. Next: export a BLIND file with RoleAnswer/infoquest_blind.py (no scoring happens here).")


if __name__ == "__main__":
    main()
