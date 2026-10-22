#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinQA Chat 0-shot CoT offline evaluation, protocol `finqa-chat-0shot-cot-v1`.
CPU only. The ONLY script of this line that reads gold.

Requires EXACTLY four formal cells: llama3 alpha {0,-6}, qwen2.5 alpha {0,+8}. Anything else (
missing/extra/duplicate cell, another protocol, config/provenance drift inside a model, wrong fire count,
identity/order mismatch against the gold file, stored diagnostics that disagree with the text) aborts with
no output. Output goes to its own file and an existing --out is never overwritten.

SCORING reuses finqa/finqa_scoring.py unchanged: MAIN `first_acc` = first legal '#### <value>' over all 300
items (no legal marker = incorrect); `last_acc` is the sensitivity readout. No fallback, no looser scoring.
This is the CUSTOM direct-numeric-answer evaluator, NOT the official FinQA program/DSL execution scorer.

STATISTICS. Two contrasts, each model's workpoint vs ITS OWN alpha=0, paired by sample: exact McNemar, paired
bootstrap CI (B=10000, seed 0, pp, question unit), Holm over m=2 (this protocol's own family, separate from the
m=3 families of `finqa-v0`). Bare-vs-Chat (--bare_eval) is DESCRIPTIVE ONLY: no interface x steering
interaction test is run and none is claimed.

EXPLORATORY early-candidate / first-line audit. The frozen `earlycand-v1` detector (optional --detector_dir)
was never extended for FinQA surface forms, so alongside it this script classifies each reply's first
non-empty line by FORMAT (bare number, currency, percent, unit, 'Answer:'-style, question restatement, ...)
and lists detector misses on number-like first lines. A format-robust "answer-shaped first line" proxy is
reported next to the detector rate so a change in EC can be checked for format-driven artefacts. All of it is
visible-output-order description: not internal commitment, outside Holm, unadjusted.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import chat_cot_sat_lib as L  # noqa: E402
import get_answer_finqa_chat as G  # noqa: E402
from finqa_scoring import first_legal_answer, last_legal_answer, is_correct, diagnostics_for_text  # noqa: E402


def _load_pw_scoring():
    """Import proofwriter_owa/scoring.py BY PATH (the generic name `scoring` would collide): the same exact
    McNemar / Holm / question-paired bootstrap the other chat lines use, not a second implementation."""
    spec = importlib.util.spec_from_file_location("_pw_scoring", os.path.join(ROOT, "proofwriter_owa", "scoring.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SC = _load_pw_scoring()
PROTOCOL = G.PROTOCOL
EXPECTED = {("llama3", 0), ("llama3", -6), ("qwen2.5", 0), ("qwen2.5", 8)}
CROSS_CELL_FIELDS = ("mask_sha256", "wrapped_prompt_sha256", "user_prompt_sha256", "chat_template_hash",
                     "model_dir", "questions_file_sha256", "sample_order_sha256", "prompt_content_sha256_16",
                     "layers", "L", "max_new_tokens", "temperature", "top_p", "batch_size", "prefill_only",
                     "prefill_tail_len", "padding_side", "prompt_wrapper_id", "prompt_template_sha256",
                     "injection_token", "mask_nonzero_rows", "size")
PROV_FIELDS = ("host", "cuda_visible_devices", "param_devices", "sharded", "cuda_device_names", "library_versions",
               "code_version")


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(2)


# ------------------------------------------------------------------ first-line format audit
_NUM = r"-?\$?\s?-?[\d,]+(?:\.\d+)?"
_RULES = [
    ("answer_marker", re.compile(r"^####")),
    ("labeled_answer", re.compile(r"^(?:the\s+)?(?:final\s+)?answer\b|^result\s*[:=]|^therefore\b.*\d", re.I)),
    ("bare_number", re.compile(rf"^{_NUM}$")),
    ("percent", re.compile(rf"^{_NUM}\s*%$")),
    ("currency", re.compile(r"^-?\$\s?[\d,]+(?:\.\d+)?(?:\s*(?:thousand|million|billion|trillion))?$", re.I)),
    ("unit_number", re.compile(rf"^{_NUM}\s*(?:%|thousand|million|billion|trillion|[a-z]+)(?:\s+[a-z]+){{0,2}}$", re.I)),
    ("colon_number", re.compile(rf"^[^:\d]{{1,30}}:\s*{_NUM}")),
    ("question_restatement", re.compile(r"^(?:question|q)\s*:|\?$", re.I)),
    ("reasoning_opener", re.compile(r"^(?:step\s*\d|\d+[.)]\s|to\s+(?:find|calculate|determine|answer|compute)|first\b|let'?s\b|"
                                   r"we\s+(?:need|are|can|have)|the\s+question|given\b|according\b|from\s+the)", re.I)),
]
ANSWER_SHAPED = {"answer_marker", "labeled_answer", "bare_number", "percent", "currency", "unit_number", "colon_number"}
NUMBER_LIKE = ANSWER_SHAPED


def first_line(text):
    for ln in text.splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def classify_first_line(text):
    ln = first_line(text)
    if not ln:
        return "empty"
    if len(ln) > 60:
        return "long_line"
    for name, rx in _RULES:
        if rx.search(ln):
            return name
    return "other_short"


def load_detector(detector_dir):
    if not detector_dir:
        return None
    sys.path.insert(0, detector_dir)
    from early_candidate_detector import has_early_candidate  # noqa: E402
    return has_early_candidate


# ------------------------------------------------------------------ loading / validation
def load_cell(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["meta"], d["data"]


def validate(paths, gold_path):
    gmeta, gsamples = G.load_samples(gold_path)
    gold_sha = G.file_sha256(gold_path)
    cells, metas, pk = {}, {}, {}
    for p in paths:
        meta, rows = load_cell(p)
        if meta.get("protocol") != PROTOCOL:
            die(f"{p}: protocol {meta.get('protocol')!r} != {PROTOCOL!r}")
        if meta.get("mode") != "formal" or meta.get("restricted_indices") is not None:
            die(f"{p}: not a formal full-300 cell (mode={meta.get('mode')!r})")
        key = (meta.get("model"), meta.get("alpha"))
        if key not in EXPECTED:
            die(f"{p}: cell {key} is not one of the experiment's four cells {sorted(EXPECTED, key=str)}")
        if key in cells:
            die(f"duplicate cell {key}")
        mdl, al, cfg = key[0], key[1], G.MODELS[key[0]]
        if meta.get("n_samples") != G.N_ITEMS or len(rows) != G.N_ITEMS:
            die(f"{p}: {len(rows)} rows / meta n {meta.get('n_samples')}, expected {G.N_ITEMS}")
        if meta.get("cot") is not True or meta.get("n_shot") != 0 or meta.get("chat_template_applied") is not True:
            die(f"{p}: cot/n_shot/chat flags wrong")
        if meta.get("accuracy_computed") is not False:
            die(f"{p}: claims accuracy already computed")
        if tuple(meta.get("layers", ())) != tuple(cfg["band"]):
            die(f"{p}: layers {meta.get('layers')} != configured band {cfg['band']}")
        if meta.get("steering_fires") != G.expected_fires(mdl, al, G.N_ITEMS) or \
                meta.get("steering_fires_expected") != meta.get("steering_fires"):
            die(f"{p}: steering_fires {meta.get('steering_fires')} != expected {G.expected_fires(mdl, al, G.N_ITEMS)}")
        if meta.get("prompt_content_sha256_16") != gmeta["prompt_content_sha256_16"]:
            die(f"{p}: prompt_content_sha256_16 differs from the gold/data file (different questions/tables)")
        if meta.get("questions_file_sha256") != gold_sha:
            die(f"{p}: generated from a different file than --gold_file (sha256 differs)")
        if meta.get("sample_order_sha256") != G.order_sha(gsamples):
            die(f"{p}: sample order hash differs from the gold file")
        got = [[str(r.get("sample_id")), r.get("id"), r.get("question")] for r in rows]
        if got != G.identity_of(gsamples):
            die(f"{p}: per-item sample_id/id/question does not match the gold file in order")
        for i, r in enumerate(rows):
            if not isinstance(r.get("generated"), str) or r.get("stop_reason") not in G.STOP_REASONS \
                    or not isinstance(r.get("generated_token_count"), int):
                die(f"{p}: row {i} incomplete")
            diag = diagnostics_for_text(r["generated"], meta["max_new_tokens"], r["generated_token_count"])
            if any(r.get(k) != v for k, v in diag.items()):
                die(f"{p}: row {i} stored diagnostics disagree with its text")
        inj = meta.get("injection_token") or {}
        if inj.get("n_prompts") != G.N_ITEMS or not isinstance(inj.get("final_token_id"), int):
            die(f"{p}: injection_token report missing / incomplete")
        cells[key], metas[key], pk[key] = rows, meta, p
    if set(cells) != EXPECTED:
        die(f"cell set {sorted(cells, key=str)} != required {sorted(EXPECTED, key=str)}")
    for mdl in G.MODELS:
        a, b = metas[(mdl, 0)], metas[(mdl, G.MODELS[mdl]["work_alpha"])]
        for f in CROSS_CELL_FIELDS:
            if f not in a or f not in b or a[f] != b[f]:
                die(f"{mdl}: field {f!r} missing or different between alpha cells "
                    f"({a.get(f)!r} vs {b.get(f)!r})")
        pa, pb = a.get("provenance") or {}, b.get("provenance") or {}
        for f in PROV_FIELDS:
            if f not in pa or f not in pb or pa[f] in (None, "", [], {}) or pa[f] != pb[f]:
                die(f"{mdl}: provenance {f!r} missing/empty or different between alpha cells "
                    f"({pa.get(f)!r} vs {pb.get(f)!r}); the pair was not produced in one environment")
    ua = {metas[(m, 0)]["user_prompt_sha256"] for m in G.MODELS}
    ta = {metas[(m, 0)]["prompt_template_sha256"] for m in G.MODELS}
    if len(ua) != 1 or len(ta) != 1:
        die("the two models were not given the same user prompt body")
    return gsamples, gmeta, cells, metas, pk


# ------------------------------------------------------------------ statistics
def correct_vec(rows, gold, first=True):
    pick = first_legal_answer if first else last_legal_answer
    return [1 if is_correct(pick(r["generated"])[1], gold[r["sample_id"]]) else 0 for r in rows]


def cell_stats(rows, gold):
    n = len(rows)
    fa, la = correct_vec(rows, gold), correct_vec(rows, gold, first=False)
    valid = [not r["no_answer"] for r in rows]
    nv = sum(valid)

    def rate(x):
        return sum(x) / n
    loop, trunc = [r["loop"] for r in rows], [r["truncated"] for r in rows]
    nat = [r["stop_reason"] == "natural_eos" for r in rows]
    chars = sorted(r["gen_chars"] for r in rows)
    toks = sorted(r["generated_token_count"] for r in rows)
    v = [r for r in rows if not r["no_answer"]]
    nov = [r for r in rows if r["no_answer"]]

    def sub(rs, k):
        return (sum(1 for r in rs if r[k]) / len(rs)) if rs else None
    return {
        "n": n, "first_acc": rate(fa), "last_acc": rate(la),
        "first_correct": sum(fa), "last_correct": sum(la),
        "valid_submission_rate": nv / n, "n_valid": nv,
        "conditional_accuracy": (sum(f for f, ok in zip(fa, valid) if ok) / nv) if nv else None,
        "repeat_submission_rate": rate([r["multi_answer"] for r in rows]),
        "repeat_submission_rate_among_valid": sub(v, "multi_answer"),
        "loop_rate": rate(loop), "truncation_rate": rate(trunc), "natural_eos_rate": rate(nat),
        "loop_among_valid": sub(v, "loop"), "loop_among_no_answer": sub(nov, "loop"),
        "truncated_among_valid": sub(v, "truncated"), "truncated_among_no_answer": sub(nov, "truncated"),
        # post-first-answer loop vs failure before any answer: both kept in the scored denominator
        "valid_then_loop": sum(1 for r in v if r["loop"]), "valid_no_loop": sum(1 for r in v if not r["loop"]),
        "no_answer_loop": sum(1 for r in nov if r["loop"]), "no_answer_truncated_no_loop":
            sum(1 for r in nov if r["truncated"] and not r["loop"]),
        "no_answer_natural_eos": sum(1 for r in nov if r["stop_reason"] == "natural_eos"),
        "gen_chars_median": chars[n // 2], "gen_tokens_median": toks[n // 2],
    }


def paired(base_rows, wp_rows, gold):
    a, b = correct_vec(base_rows, gold), correct_vec(wp_rows, gold)
    g01, l10, p = SC.mcnemar_exact(a, b)
    lo, hi = SC.question_paired_bootstrap_ci(a, b, B=10000, seed=0)
    return {"delta_first_acc_pp": (sum(b) - sum(a)) / len(a) * 100.0, "alpha0_correct": sum(a),
            "workpoint_correct": sum(b), "wrong_to_right": g01, "right_to_wrong": l10,
            "mcnemar_exact_p": p, "bootstrap_ci95_pp": [lo, hi],
            "bootstrap": {"B": 10000, "seed": 0, "unit": "question"}}


def ec_audit(rows, detector, audit_n):
    cats, miss, flags = {}, [], []
    shaped = 0
    for r in rows:
        c = classify_first_line(r["generated"])
        cats[c] = cats.get(c, 0) + 1
        shaped += c in ANSWER_SHAPED
        f = bool(detector(r["generated"])) if detector else None
        flags.append(1 if f else 0)
        if detector and (c in NUMBER_LIKE) and not f and len(miss) < audit_n:
            miss.append({"sample_id": r["sample_id"], "first_line": first_line(r["generated"])[:80], "category": c})
    n = len(rows)
    out = {"first_line_categories": dict(sorted(cats.items())),
           "answer_shaped_first_line_rate": shaped / n, "answer_shaped_first_line_n": shaped}
    if detector:
        nm = sum(1 for r, f in zip(rows, flags) if not f and classify_first_line(r["generated"]) in NUMBER_LIKE)
        out.update({"earlycand_v1_rate": sum(flags) / n, "earlycand_v1_n": sum(flags),
                    "detector_misses_on_number_like_first_line": nm, "miss_examples": miss})
    return out, flags


def run(args):
    gsamples, gmeta, cells, metas, pk = validate(args.cells, args.gold_file)
    gold = {s["sample_id"]: float(s["gold_value"]) for s in gsamples}
    detector = load_detector(args.detector_dir)
    results, raw = {}, []
    for mdl in G.MODELS:
        wa = G.MODELS[mdl]["work_alpha"]
        base, wp = cells[(mdl, 0)], cells[(mdl, wa)]
        pr = paired(base, wp, gold)
        raw.append((mdl, pr["mcnemar_exact_p"]))
        e0, f0 = ec_audit(base, detector, args.audit_n)
        e1, f1 = ec_audit(wp, detector, args.audit_n)
        ec_pair = None
        if detector:
            g01, l10, p = SC.mcnemar_exact(f0, f1)
            ec_pair = {"alpha0": sum(f0), "workpoint": sum(f1), "off_to_on": g01, "on_to_off": l10,
                       "mcnemar_exact_p_unadjusted": p}
        results[mdl] = {"alpha_workpoint": wa,
                        "cells": {"alpha0": cell_stats(base, gold), "workpoint": cell_stats(wp, gold)},
                        "workpoint_vs_alpha0": pr,
                        "EXPLORATORY_first_line_audit": {"alpha0": e0, "workpoint": e1,
                                                         "earlycand_paired_unadjusted": ec_pair}}
    if len(raw) != 2:
        die("Holm family must have exactly m=2 members")
    adj = SC.holm(raw)
    for mdl in G.MODELS:
        results[mdl]["workpoint_vs_alpha0"]["holm_p_adj"] = adj[mdl]
    bare = None
    if args.bare_eval:
        bd = json.load(open(args.bare_eval, encoding="utf-8"))
        bare = {}
        for mdl in G.MODELS:
            pa = bd["results"][mdl]["per_alpha"]
            wa = G.MODELS[mdl]["work_alpha"]
            bare[mdl] = {"alpha0_first_acc": pa["0"]["first_acc"], "workpoint_first_acc": pa[str(wa)]["first_acc"],
                         "note": "Bare finqa-v0 values copied from its frozen evaluation; descriptive only"}
    out = {
        "protocol": PROTOCOL, "stage": "eval", "n_shot": 0, "cot": True, "n_items": G.N_ITEMS,
        "scoring": ("CUSTOM direct-numeric-answer evaluator (first legal '#### <value>', finqa_scoring normalisation "
                    "and tolerance, no marker = incorrect, denominator 300); NOT the official FinQA program/DSL "
                    "execution scorer"),
        "holm_family": {"id": "finqa-chat-0shot-cot-v1-workpoint", "m": 2,
                        "members": ["llama3 -6 vs 0", "qwen2.5 +8 vs 0"],
                        "separate_from": ["finqa-v0 (m=3 per model)"]},
        "results": results, "bare_vs_chat_DESCRIPTIVE": bare,
        "inputs": {"cells": {f"{m}:{a}": {"path": os.path.abspath(p), "sha256": G.file_sha256(p)}
                             for (m, a), p in sorted(pk.items(), key=str)},
                   "gold_file_sha256": G.file_sha256(args.gold_file),
                   "detector": "earlycand-v1 (imported, unchanged)" if detector else "not supplied"},
        "interpretation_limits": [
            "Designed after the Bare finqa-v0 result: a supplementary experiment, not a blind test.",
            "Bare-vs-Chat is a descriptive juxtaposition; no interface x steering interaction test was run.",
            "Early-candidate / first-line readouts describe visible output order, not internal commitment; they are exploratory, unadjusted and outside Holm.",
            "A loop after the first legal answer does not invalidate that answer; failures before any answer are scored incorrect and kept in the denominator.",
            "alpha values are nominal and not equivalent across models.",
        ],
    }
    L.atomic_write_json(out, args.out)
    print(f"wrote {args.out}")
    for mdl in G.MODELS:
        r, w = results[mdl], results[mdl]["workpoint_vs_alpha0"]
        c0, c1 = r["cells"]["alpha0"], r["cells"]["workpoint"]
        print(f"  {mdl}: first_acc {c0['first_acc']:.4f} -> {c1['first_acc']:.4f}  d={w['delta_first_acc_pp']:+.2f}pp "
              f"W->C {w['wrong_to_right']} C->W {w['right_to_wrong']}  p={w['mcnemar_exact_p']:.4g}  "
              f"holm={w['holm_p_adj']:.4g}  valid {c0['valid_submission_rate']:.3f}->{c1['valid_submission_rate']:.3f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--gold_file", required=True, help="the SAME finqa_formal.json the cells were generated from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--detector_dir", default=None, help="dir holding early_candidate_detector.py (RoleAnswer/)")
    ap.add_argument("--bare_eval", default=None, help="docs/finqa_formal_evaluation.json, for the descriptive side-by-side")
    ap.add_argument("--audit_n", type=int, default=15)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        die(f"{args.out} exists; refusing to overwrite")
    try:
        return run(args)
    except G.FQError as e:
        die(str(e))


if __name__ == "__main__":
    sys.exit(main())
