#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed-Answer Reliability and Submission Pilot (protocol `fixed-answer-v1`).

Llama-3.1-8B-Instruct, MMLU pooled over 57 subjects, original subject-specific Non-expert role
("non {subject} expert"), bare string, no CoT, next-token logits only, NMD mask (Role-derived)
layers 11-20 (decoder 10-18), prefill last-token output-side injection, alpha in {-4, 0, +4}.

Pipeline
  1. alpha=0, A-D prompt  -> a FIXED candidate per question (argmax; ties -> first of A,B,C,D).
     This forward is also the alpha=0 auxiliary A-D readout.
  2. For every alpha, four INDEPENDENT readouts (separate prompts, separate forwards, no shared state):
       ad     A-D support of the fixed candidate (auxiliary; alpha=0 reused from step 1)
       b      correctness rating 0-9 for the fixed candidate
       c_map1 submit/leave, X=submit, Y=leave
       c_map2 submit/leave, X=leave,  Y=submit
No E branch.  Nothing is generated.  Gold labels are kept in the sample manifest and the candidate
file for analysis only; they never enter a prompt.

Isolation: new file; imports verified helpers from willingness v2/v3 (neither modified);
template.py untouched.  Fake-model runs prove plumbing only -- there is NO real result here.
"""

import os
import re
import sys
import json
import time
import random
import socket
import argparse
import platform
from datetime import datetime, timezone

import numpy as np

import get_action_willingness_v2 as v2
import get_action_willingness_v3 as v3

PROTOCOL = "fixed-answer-v1"
SEED = 42
N_FORMAL = 300
N_PILOT = 25
DEFAULT_CONFIGS = ["neg4-11-20", "0-11-20", "4-11-20"]
MASK_SHA256 = v3.V2_MASK_SHA256
MMLU_REF = v3.V2_REFERENCE["mmlu"]          # pool n + digest recorded in the v2 result metadata

AD_LABELS = ["A", "B", "C", "D"]
DIGITS = [str(i) for i in range(10)]
XY = ["X", "Y"]

# Frozen verbatim.  Every template ends in exactly ONE ASCII space (the answer anchor).
# NOTE: the repo's ordinary A-D template says "Now you are a ..."; this experiment says
# "Now you are an honest ..." on purpose.  template.py is not touched.
PROMPT_AD = (
    'Would you answer the following question with A, B, C or D?\n'
    'Question: {context}\n'
    'Now you are an honest {role}, your answer among "A, B, C, D" is: '
)
PROMPT_B = (
    "Here is a question:\n"
    "{context}\n"
    "\n"
    "Provisional answer: {cand_label}) {cand_text}\n"
    "\n"
    "Now you are an honest {role}.\n"
    "Rate how likely the provisional answer is to be correct.\n"
    "Use 0 (certainly incorrect) to 9 (certainly correct).\n"
    "Respond with a single digit only.\n"
    "\n"
    "Response: "
)
PROMPT_C = (
    "Here is a question:\n"
    "{context}\n"
    "\n"
    "Provisional answer: {cand_label}) {cand_text}\n"
    "\n"
    "Now you are an honest {role}.\n"
    "Choose one action:\n"
    "X) {x_action}\n"
    "Y) {y_action}\n"
    "Respond with X or Y only.\n"
    "\n"
    "Response: "
)
SUBMIT_TXT = "Submit the provisional answer."
LEAVE_TXT = "Leave the question unanswered."
READOUTS = ["ad", "b", "c_map1", "c_map2"]
READOUT_LABELS = {"ad": AD_LABELS, "b": DIGITS, "c_map1": XY, "c_map2": XY}


# ---------------------------------------------------------------- data

def role_for(sub):
    return f"non {sub.lower().replace('_', ' ')} expert"


def parse_options(text):
    """MMLU json text = '<question>\\nA) a\\nB) b\\nC) c\\nD) d\\n'.  Fail closed if the layout differs."""
    t = text.rstrip("\n")
    ia = t.rfind("\nA) ")
    if ia < 0:
        raise ValueError("no '\\nA) ' marker")
    ib = t.find("\nB) ", ia)
    ic = t.find("\nC) ", ib) if ib >= 0 else -1
    idd = t.find("\nD) ", ic) if ic >= 0 else -1
    if not (0 <= ia < ib < ic < idd):
        raise ValueError("options A-D not found in order")
    cuts = [ia, ib, ic, idd, len(t)]
    return [t[cuts[k] + 4:cuts[k + 1]] for k in range(4)]


def load_mmlu_pool(path):
    """Pool in the same order/idx/sub/text hashing as v2 (digest is cross-checked), but KEEPING label + options."""
    from detection.task_list import TASKS
    pool = []
    for sub in TASKS:
        fp = os.path.join(path, f"{sub}.json")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"missing MMLU subject file {fp}")
        for s in json.load(open(fp, encoding="utf-8")):
            pool.append((sub, s))
    out = []
    for i, (sub, s) in enumerate(pool):
        if "text" not in s or "label" not in s:
            raise KeyError(f"sample {i} ({sub}) lacks text/label (has {sorted(s)})")
        lab = s["label"]
        if not (isinstance(lab, int) and 0 <= lab <= 3):
            raise ValueError(f"sample {i} ({sub}) label {lab!r} not in 0..3")
        opts = parse_options(s["text"])
        out.append({"idx": i, "sub": sub, "text": s["text"], "text_sha256": v2.sha256_text(s["text"]),
                    "label": lab, "gold": AD_LABELS[lab], "options": opts, "role": role_for(sub),
                    "sample_id": f"mmlu:{i}"})
    return out


def draw(pool, seed=SEED, n_formal=N_FORMAL, n_pilot=N_PILOT):
    """formal = Random(seed).sample(range(N), n_formal), sorted.  pilot = Random(seed).sample(remaining, n_pilot), sorted.
    The formal draw is identical to the willingness-v3 MMLU draw (same seed, same pool)."""
    if len(pool) < n_formal + n_pilot:
        sys.exit(f"[FATAL] pool {len(pool)} < {n_formal}+{n_pilot}")
    formal = sorted(random.Random(seed).sample(range(len(pool)), n_formal))
    fs = set(formal)
    rest = [i for i in range(len(pool)) if i not in fs]
    pilot = sorted(random.Random(seed).sample(rest, n_pilot))
    if fs & set(pilot):
        raise RuntimeError("pilot and formal overlap")
    return formal, pilot


def sample_manifest(pool, formal, pilot, path, args_seed):
    def rows(ids):
        return [{"sample_id": pool[i]["sample_id"], "pool_idx": i, "sub": pool[i]["sub"], "role": pool[i]["role"],
                 "gold": pool[i]["gold"], "text_sha256": pool[i]["text_sha256"]} for i in ids]
    return {"protocol": PROTOCOL, "seed": args_seed, "n_formal": len(formal), "n_pilot": len(pilot),
            "rng": "formal: random.Random(seed).sample(range(pool_n), n) sorted; "
                   "pilot: random.Random(seed).sample(non-formal indices, n) sorted",
            "source_dir": path, "pool_n": len(pool), "pool_digest": v2.samples_digest(pool),
            "formal_digest": v2.samples_digest([pool[i] for i in formal]),
            "pilot_digest": v2.samples_digest([pool[i] for i in pilot]),
            "formal": rows(formal), "pilot": rows(pilot),
            "independence_from_RSN_construction": "UNVERIFIED (same-source exploratory; not a held-out generalisation)"}


# ---------------------------------------------------------------- prompts

def _ends_one_space(p):
    assert p.endswith(": ") and not p.endswith("  "), "prompt must end in exactly one ASCII space"
    return p


def build_ad(s):
    return _ends_one_space(PROMPT_AD.format(context=s["text"], role=s["role"]))


def build_b(s, cand):
    return _ends_one_space(PROMPT_B.format(context=s["text"], role=s["role"], cand_label=cand["label"],
                                           cand_text=cand["text"]))


def build_c(s, cand, mapping):
    if mapping == 1:
        x, y = SUBMIT_TXT, LEAVE_TXT
    elif mapping == 2:
        x, y = LEAVE_TXT, SUBMIT_TXT
    else:
        raise ValueError(mapping)
    return _ends_one_space(PROMPT_C.format(context=s["text"], role=s["role"], cand_label=cand["label"],
                                           cand_text=cand["text"], x_action=x, y_action=y))


def submit_index(mapping):
    """index into [X, Y] that means 'submit'"""
    return 0 if mapping == 1 else 1


# ---------------------------------------------------------------- scoring

def entropy_nats(p):
    return v3.entropy_nats(p)


def ad_record(s, prompt, raw, ids, cand=None):
    d, p, mass = v2.softmax_digits(raw, ids)
    top = np.flatnonzero(d == d.max())
    rec = {"sample_id": s["sample_id"], "idx": s["idx"], "sub": s["sub"], "text_sha256": s["text_sha256"],
           "prompt_sha256": v2.sha256_text(prompt),
           "ad_logits": [float(x) for x in d], "ad_probs": [float(x) for x in p],
           "argmax_label": AD_LABELS[int(top[0])], "n_tied_top": int(len(top)),
           "tied_top_labels": [AD_LABELS[int(i)] for i in top],
           "entropy_nats": entropy_nats(p), "ad_mass": mass}
    if cand is not None:
        k = AD_LABELS.index(cand["label"])
        rec.update({"cand_label": cand["label"], "cand_prob": float(p[k]),
                    "cand_in_top": bool(k in top), "cand_unique_top": bool(len(top) == 1 and top[0] == k),
                    "cand_rank": int(1 + (d > d[k]).sum())})
    return rec


def b_record(s, cand, prompt, raw, ids):
    d, p, mass = v2.softmax_digits(raw, ids)
    pred = int(np.argmax(p))
    return {"sample_id": s["sample_id"], "idx": s["idx"], "sub": s["sub"], "text_sha256": s["text_sha256"],
            "prompt_sha256": v2.sha256_text(prompt), "cand_label": cand["label"], "cand_correct": cand["correct"],
            "digit_logits": [float(x) for x in d], "digit_probs": [float(x) for x in p],
            "score": pred, "score_prob": float(p[pred]),
            "expected_score": float((p * np.arange(10)).sum()),
            "entropy_nats": entropy_nats(p), "digit_mass": mass}


def c_record(s, cand, prompt, raw, ids, mapping):
    d, p, mass = v2.softmax_digits(raw, ids)
    si = submit_index(mapping)
    tied = bool(d[0] == d[1])
    act = "tie" if tied else ("submit" if int(np.argmax(d)) == si else "leave")
    return {"sample_id": s["sample_id"], "idx": s["idx"], "sub": s["sub"], "text_sha256": s["text_sha256"],
            "prompt_sha256": v2.sha256_text(prompt), "cand_label": cand["label"], "cand_correct": cand["correct"],
            "mapping": mapping, "xy_logits": [float(x) for x in d], "xy_probs": [float(x) for x in p],
            "p_submit": float(p[si]), "argmax_action": act, "tied": tied, "xy_mass": mass}


def run_readout(vc, samples, prompts, ids, diff_mtx, n_steered, tail_len, make):
    import torch
    recs = []
    vc.steering_fire_count(reset=True)
    for s, prompt in zip(samples, prompts):
        with torch.no_grad():
            raw = vc.regenerate_logits([prompt], diff_mtx, tail_len=tail_len)[0]
        recs.append(make(s, prompt, raw))
    fires = vc.steering_fire_count(reset=True)
    expect = n_steered * len(samples) * tail_len
    if fires != expect:
        raise RuntimeError(f"steering_fires {fires} != expected {expect} "
                           f"(L={n_steered} x n={len(samples)} x tail={tail_len})")
    return recs, fires


def tokenizer_checks(vc, prompts, labels, ids, name):
    """Each candidate must CONTINUE the prompt as one extra token; BOS count 1; uniform final token."""
    tok = vc.tokenizer
    bos = getattr(tok, "bos_token_id", None)
    last = set()
    for p in prompts:
        base = tok(p, add_special_tokens=True).input_ids
        if bos is not None and base.count(bos) != 1:
            raise RuntimeError(f"[{name}] BOS count {base.count(bos)} != 1")
        last.add(base[-1])
        for lab, tid in zip(labels, ids):
            if tok(p + lab, add_special_tokens=True).input_ids != base + [tid]:
                raise RuntimeError(f"[{name}] candidate {lab!r} (id {tid}) does not continue the prompt tail as a separate token")
    if len(last) != 1:
        raise RuntimeError(f"[{name}] prompts end in different tokens {sorted(last)}; injection site not uniform")
    return sorted(last)


def write_json_atomic(path, obj):
    return v3.write_json_atomic(path, obj)


def tag_of(alpha):
    return f"neg{abs(alpha)}" if alpha < 0 else str(alpha)


def cell_path(root, readout, alpha, size, st, en):
    return os.path.join(root, "cells", f"mdf_{tag_of(alpha)}", f"{readout}_{size}_{st}_{en}.json")


def check_or_write(path, want, what):
    """Return loaded old file if its provenance matches `want`; None if absent; die on mismatch."""
    if not os.path.exists(path):
        return None
    old = json.load(open(path, encoding="utf-8"))
    diff = [k for k, v in want.items() if old["meta"].get(k) != v]
    if diff:
        sys.exit(f"[FATAL] {path} exists with DIFFERENT provenance in {diff}; refusing to resume/overwrite ({what})")
    return old


def main():
    ap = argparse.ArgumentParser(description="Fixed-answer reliability + submission pilot")
    ap.add_argument("--mode", choices=["pilot", "formal"], required=True)
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--size", default="8B")
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--mask_type", default="nmd")
    ap.add_argument("--percentage", type=float, default=0.5)
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--mmlu_dir", required=True, help="dir of the 57 per-subject MMLU json files")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tail_len", type=int, default=1)
    ap.add_argument("--ref_check", choices=["strict", "off"], default="strict",
                    help="'off' exists for fake-fixture unit tests only")
    args = ap.parse_args()

    import utils
    from llms import VicundaModel

    configs = utils.parse_configs(args.configs)
    doses = [a for a, _ in configs]
    if sorted(doses) != [-4, 0, 4]:
        sys.exit(f"alpha set must be exactly -4, 0, +4, got {doses}")
    if len({tuple(se) for _, se in configs}) != 1:
        sys.exit("all configs must share one layer range")
    st, en = configs[0][1]
    if args.mode == "formal" and "pilot" in args.out_root.lower():
        sys.exit("formal mode with a pilot-looking out_root; refusing")

    # ---- data (cheap failures before the model) ----
    try:
        pool = load_mmlu_pool(args.mmlu_dir)
    except (FileNotFoundError, KeyError, ValueError) as e:
        sys.exit(f"[FATAL] {e}")
    if args.ref_check == "strict":
        got = {"n": len(pool), "samples_digest": v2.samples_digest(pool)}
        bad = [k for k in got if MMLU_REF[k] != got[k]]
        if bad:
            sys.exit(f"[FATAL] MMLU pool differs from the v2/v3 pool in {bad}; refusing to guess a data version")
    formal, pilot = draw(pool, args.seed)
    man = sample_manifest(pool, formal, pilot, args.mmlu_dir, args.seed)
    ids_sel = formal if args.mode == "formal" else pilot
    samples = [pool[i] for i in ids_sel]
    man_path = os.path.join(args.out_root, "sample_manifest.json")
    if os.path.exists(man_path):
        old = json.load(open(man_path, encoding="utf-8"))
        for k in ("seed", "pool_digest", "formal_digest", "pilot_digest", "n_formal", "n_pilot"):
            if old.get(k) != man[k]:
                sys.exit(f"[FATAL] {man_path} exists with different {k}; refusing to overwrite")
    else:
        write_json_atomic(man_path, man)
    print(f"[data] mode={args.mode} pool={len(pool)} pool_digest={man['pool_digest'][:16]} "
          f"selected={len(samples)} digest={man[args.mode + '_digest'][:16]}")

    # ---- mask ----
    mask_path = os.path.join(args.mask_dir, f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}.npy")
    mask = np.load(mask_path)
    mask_sha = v2.sha256_file(mask_path)
    if args.ref_check == "strict" and mask_sha != MASK_SHA256:
        sys.exit(f"[FATAL] mask sha256 {mask_sha[:16]} != expected {MASK_SHA256[:16]}")
    nz_rows = [int(i) for i in np.where(np.abs(mask).sum(axis=1) > 0)[0]]
    expect_rows = [int(i) for i in utils.decoder_layer_range(st, en)]
    if nz_rows != expect_rows:
        sys.exit(f"mask non-zero decoder rows {nz_rows} != expected {expect_rows}")
    n_steered = len(nz_rows)
    print(f"[mask] {mask_path} sha256={mask_sha[:16]} steered_layers={n_steered}")

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    if mask.shape[0] != len(vc._find_decoder_layers()):
        sys.exit(f"mask rows {mask.shape[0]} != decoder layers {len(vc._find_decoder_layers())}")
    ids = {"ad": utils.option_token_ids(vc, AD_LABELS), "b": utils.option_token_ids(vc, DIGITS),
           "c_map1": utils.option_token_ids(vc, XY)}
    ids["c_map2"] = ids["c_map1"]

    import torch, transformers
    env = {"host": socket.gethostname(), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "n_visible_gpus": torch.cuda.device_count(), "torch": torch.__version__,
           "transformers": transformers.__version__, "python": platform.python_version(),
           "hf_device_map_multi": bool(getattr(vc.model, "hf_device_map", None) and
                                       len(set(map(str, vc.model.hf_device_map.values()))) > 1)}
    if env["hf_device_map_multi"]:
        sys.exit("model is sharded over several devices; this protocol expects one card")

    common = {"protocol": PROTOCOL, "mode": args.mode, "model": args.model, "model_dir": args.model_dir,
              "size": args.size, "mask_sha256": mask_sha, "tail_len": args.tail_len,
              "layer_start": st, "layer_end": en, "role_rule": "non {subject with spaces} expert",
              "formal_digest": man["formal_digest"], "pilot_digest": man["pilot_digest"],
              "selection_digest": man[args.mode + "_digest"], "n_samples": len(samples),
              "host": env["host"], "gpu": env["gpu"], "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
              "torch": env["torch"], "transformers": env["transformers"]}

    def full_meta(want, extra):
        return {**want, **extra, "mask_path": mask_path, "mask_type": args.mask_type,
                "percentage": args.percentage, "n_steered_layers": n_steered, "steered_decoder_rows": nz_rows,
                "injection": "output-side, prefill only, last prompt token", "cot": False, "use_chat": False,
                "bare_string": True, "date_utc": datetime.now(timezone.utc).isoformat(), **env}

    # ---- step 1: alpha=0 A-D -> fixed candidates ----
    ad_prompts = [build_ad(s) for s in samples]
    last_ad = tokenizer_checks(vc, ad_prompts, AD_LABELS, ids["ad"], "ad")
    ad_digest = v2.sha256_text("\n".join(v2.sha256_text(p) for p in ad_prompts))
    want0 = {**common, "readout": "ad", "alpha": 0, "prompts_digest": ad_digest,
             "prompt_template_sha256": v2.sha256_text(PROMPT_AD), "candidates_digest": None}
    p0 = cell_path(args.out_root, "ad", 0, args.size, st, en)
    old = check_or_write(p0, want0, "prep cell")
    if old is not None:
        recs0 = old["records"]
        print(f"[skip] prep cell exists: {p0}")
    else:
        t0 = time.time()
        recs0, fires = run_readout(vc, samples, ad_prompts, ids["ad"], list(mask * 0), 0, args.tail_len,
                                   lambda s, pr, raw: ad_record(s, pr, raw, ids["ad"]))
        write_json_atomic(p0, {"meta": full_meta(want0, {
            "prompt_template": PROMPT_AD, "example_prompt": ad_prompts[0], "steering_fires": fires,
            "expected_steering_fires": 0, "label_token_ids": ids["ad"], "final_prompt_token_ids": last_ad,
            "seconds": round(time.time() - t0, 1)}), "records": recs0})
        print(f"[saved] ad alpha=+0 n={len(recs0)} (prep cell) -> {p0}")

    cands = []
    for s, r in zip(samples, recs0):
        if r["sample_id"] != s["sample_id"]:
            sys.exit("prep cell sample order differs from the selection")
        k = AD_LABELS.index(r["argmax_label"])
        cands.append({"sample_id": s["sample_id"], "label": AD_LABELS[k], "text": s["options"][k],
                      "correct": bool(AD_LABELS[k] == s["gold"]), "gold": s["gold"],
                      "n_tied_top": r["n_tied_top"], "tied_top_labels": r["tied_top_labels"],
                      "ad0_probs": r["ad_probs"], "ad0_logits": r["ad_logits"]})
    cand_digest = v2.sha256_text("\n".join(f"{c['sample_id']}|{c['label']}|{v2.sha256_text(c['text'])}" for c in cands))
    cand_path = os.path.join(args.out_root, "candidates.json")
    cand_obj = {"protocol": PROTOCOL, "mode": args.mode, "selection_digest": common["selection_digest"],
                "candidates_digest": cand_digest, "mask_sha256": mask_sha, "alpha": 0,
                "rule": "argmax of alpha=0 A-D logits; exact ties -> first of A,B,C,D (recorded)",
                "n": len(cands), "n_correct": sum(c["correct"] for c in cands),
                "n_tied": sum(c["n_tied_top"] > 1 for c in cands), "candidates": cands}
    if os.path.exists(cand_path):
        oldc = json.load(open(cand_path, encoding="utf-8"))
        if oldc.get("candidates_digest") != cand_digest or oldc.get("selection_digest") != common["selection_digest"]:
            sys.exit(f"[FATAL] {cand_path} exists with a different candidate set; refusing to replace the fixed candidates")
    else:
        write_json_atomic(cand_path, cand_obj)
    print(f"[cand] n={len(cands)} correct={cand_obj['n_correct']} tied={cand_obj['n_tied']} digest={cand_digest[:16]}")

    cand_by = {c["sample_id"]: c for c in cands}
    cl = [cand_by[s["sample_id"]] for s in samples]
    prompts = {"b": [build_b(s, c) for s, c in zip(samples, cl)],
               "c_map1": [build_c(s, c, 1) for s, c in zip(samples, cl)],
               "c_map2": [build_c(s, c, 2) for s, c in zip(samples, cl)]}
    templates = {"ad": PROMPT_AD, "b": PROMPT_B, "c_map1": PROMPT_C, "c_map2": PROMPT_C}
    last_ids = {"ad": last_ad}
    for r in ("b", "c_map1", "c_map2"):
        last_ids[r] = tokenizer_checks(vc, prompts[r], READOUT_LABELS[r], ids[r], r)
        print(f"[{r}] final prompt token ids={last_ids[r]} "
              f"(decoded {[vc.tokenizer.decode([i]) for i in last_ids[r]]}); label ids={ids[r]}")
    print(f"[ad] final prompt token ids={last_ad}; label ids={ids['ad']}")
    done = 1
    for alpha, (a_st, a_en) in configs:
        diff_mtx = list(mask * alpha)
        ns = n_steered if alpha != 0 else 0
        for r in READOUTS:
            if r == "ad" and alpha == 0:
                continue
            path = cell_path(args.out_root, r, alpha, args.size, a_st, a_en)
            pr_list = ad_prompts if r == "ad" else prompts[r]
            want = {**common, "readout": r, "alpha": alpha, "candidates_digest": cand_digest,
                    "prompts_digest": v2.sha256_text("\n".join(v2.sha256_text(p) for p in pr_list)),
                    "prompt_template_sha256": v2.sha256_text(templates[r])}
            if check_or_write(path, want, f"{r} alpha={alpha}") is not None:
                print(f"[skip] complete cell exists: {path}")
                done += 1
                continue
            cmap = dict(zip((s["sample_id"] for s in samples), cl))
            if r == "ad":
                mk = lambda s, pr, raw: ad_record(s, pr, raw, ids["ad"], cmap[s["sample_id"]])
            elif r == "b":
                mk = lambda s, pr, raw: b_record(s, cmap[s["sample_id"]], pr, raw, ids["b"])
            else:
                m_ = 1 if r == "c_map1" else 2
                mk = lambda s, pr, raw, m_=m_: c_record(s, cmap[s["sample_id"]], pr, raw, ids[r], m_)
            t0 = time.time()
            recs, fires = run_readout(vc, samples, pr_list, ids[r], diff_mtx, ns, args.tail_len, mk)
            write_json_atomic(path, {"meta": full_meta(want, {
                "prompt_template": templates[r], "example_prompt": pr_list[0], "steering_fires": fires,
                "expected_steering_fires": ns * len(samples) * args.tail_len,
                "label_token_ids": ids[r], "final_prompt_token_ids": last_ids[r],
                "seconds": round(time.time() - t0, 1)}), "records": recs})
            done += 1
            print(f"[saved] {r} alpha={alpha:+d} n={len(recs)} fires={fires} -> {path}")

    expected_cells = len(doses) * len(READOUTS)
    write_json_atomic(os.path.join(args.out_root, f"run_manifest_{int(time.time())}_{os.getpid()}.json"), {
        **common, "cells_done": done, "cells_expected": expected_cells, "candidates_digest": cand_digest,
        "pid": os.getpid(), **env})
    print(f"done: {done}/{expected_cells} cells")
    if done != expected_cells:
        sys.exit(1)


if __name__ == "__main__":
    main()
