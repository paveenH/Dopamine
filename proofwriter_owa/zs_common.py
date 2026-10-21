#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared, STDLIB-ONLY logic for the ProofWriter-OWA chat zero-shot experiment
(protocol `proofwriter-owa-zs-chat-v1`): frozen configuration, cell
validation, per-cell metrics and the format gate. Kept free of numpy / torch
so it runs (and is tested) on a box without them; the generation entry point
imports the heavy libraries itself.

SCOPE. Independent of the closed bare-string line (`proofwriter-owa-v0`) and
of the 1-shot chat line (`proofwriter-owa-chat-v1`); nothing here touches
their results or defaults. The two workpoints (llama3 -6, qwen2.5 +8) are
READ FROM THE GSM8K RECORD and are not searched on ProofWriter.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from answer_parser import (_MARKER_RE_V2, find_all_markers_v2, get_marker_family,
                           is_correct, normalize_label, parse_final_answer_v2)  # noqa: E402
from commitment import per_sample_commitment  # noqa: E402
from eval_proofwriter_owa import is_loop  # noqa: E402
import label_early_candidate as lec  # noqa: E402
import prompt_zeroshot as pz  # noqa: E402

PROTOCOL = "proofwriter-owa-zs-chat-v1"
N_ITEMS = 300
LABELS = ("True", "False", "Unknown")

GEN_FIXED = {
    "max_new_tokens": 1024, "temperature": 0.0, "top_p": 1.0, "batch_size": 8,
    "prefill_only": True, "prefill_tail_len": 1, "n_shot": 0,
    "chat_template": True, "chat_template_applied": True, "padding_side": "left",
}

MODEL_CONFIG = {
    "llama3": {
        "size": "8B", "model_dir": "meta-llama/Llama-3.1-8B-Instruct",
        "wrapper_id": "llama3-chat-template-v1",
        "layer_start": 11, "layer_end": 20, "L": 9, "workpoint": -6,
        "mask_rel": "llama3_non_logits/nmd_0.5_11_20_8B.npy",
    },
    "qwen2.5": {
        "size": "7B", "model_dir": "Qwen/Qwen2.5-7B-Instruct",
        "wrapper_id": "qwen2.5-chat-template-v1",
        "layer_start": 16, "layer_end": 22, "L": 6, "workpoint": 8,
        "mask_rel": "qwen2.5_non_logits/nmd_0.5_16_22_7B.npy",
    },
}
CELLS = ("a0", "wp")

# FORMAT-STABILITY GATE (frozen 2026-10-21, BEFORE any zero-shot cell exists).
# Applied to each model's No-CoT alpha=0 cell and decided on submission
# format only -- accuracy and early-candidate are never read by the gate.
# Only the valid-submission rate gates. Loop / truncation are health
# DIAGNOSTICS split by whether the sample submitted a valid answer: a cell that
# submits validly and loops/truncates afterwards must not be blocked.
GATE_THRESHOLDS = {
    "valid_submission_rate_min": 0.90,   # >=1 strict line-anchored "#### label"
}

ROW_KEYS = ("sample_id", "key", "dataset", "official_theory_id", "official_qid",
            "generated", "generated_token_count", "stop_reason", "truncated")


class ZSError(Exception):
    """Fail-closed validation error."""


def alpha_of(model: str, cell: str) -> int:
    return 0 if cell == "a0" else MODEL_CONFIG[model]["workpoint"]


def alpha_tag(alpha) -> str:
    return f"neg{-alpha}" if alpha < 0 else str(alpha)


def expected_fires(model: str, alpha, n: int = N_ITEMS) -> int:
    return 0 if alpha == 0 else MODEL_CONFIG[model]["L"] * n


def cell_relpath(model: str, condition: str, alpha) -> str:
    c = MODEL_CONFIG[model]
    return os.path.join(
        model, "proofwriter_owa_zs", f"{condition}_mdf_{alpha_tag(alpha)}",
        f"proofwriter_owa_zs_{c['size']}_{c['layer_start']}_{c['layer_end']}.json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def atomic_write_json(path: str, obj, overwrite: bool = False) -> None:
    if os.path.exists(path) and not overwrite:
        raise ZSError(f"{path} exists; refusing to overwrite")
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_blind_manifest(path: str):
    d = json.load(open(path, encoding="utf-8"))
    if d["meta"].get("contains_labels") is not False:
        raise ZSError("manifest does not declare contains_labels=false; "
                      "point at manifest_blind.json")
    data = d["data"]
    if len(data) != N_ITEMS:
        raise ZSError(f"manifest has {len(data)} items, expected {N_ITEMS}")
    return d["meta"], data


def bare_prompts_for(samples, condition: str):
    cot = pz.condition_to_cot(condition)
    return [pz.build_user_prompt(s["theory_text"], s["question_text"], cot)
            for s in samples]


def bare_prompt_sha256(samples, condition: str) -> str:
    return sha256_text("\n".join(bare_prompts_for(samples, condition)))


# ───────────────────────── cell validation ─────────────────────────

def expected_fixed_meta(model: str, condition: str, alpha) -> dict:
    c = MODEL_CONFIG[model]
    tid = pz.TEMPLATE_IDS[condition]
    m = {
        "protocol": PROTOCOL, "prompt_template_id": tid,
        "marker_family": get_marker_family(tid)["marker_family"],
        "cot": pz.condition_to_cot(condition), "condition": condition,
        "model": model, "size": c["size"], "model_dir": c["model_dir"],
        "alpha": alpha, "layer_start": c["layer_start"],
        "layer_end": c["layer_end"], "L": c["L"],
        "prompt_wrapper_id": c["wrapper_id"],
        "expected_steering_fires": expected_fires(model, alpha),
        "steering_fires": expected_fires(model, alpha),
        "n": N_ITEMS, "contains_labels": False, "accuracy_computed": False,
    }
    m.update(GEN_FIXED)
    return m


def validate_cell(path: str, model: str, condition: str, alpha,
                  samples=None, extra_expect: dict | None = None):
    """Return (meta, rows) or raise ZSError. `samples` = blind manifest rows;
    when given, sample identity AND order are checked and bare_prompt_sha256 is
    recomputed. `extra_expect` adds exact-match fields (generation passes the
    wrapped-prompt hash, mask hash, ... which only it can compute)."""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception as e:  # unreadable / truncated file
        raise ZSError(f"{path}: cannot read ({e})")
    meta, rows = d.get("meta"), d.get("data")
    if not isinstance(meta, dict) or not isinstance(rows, list):
        raise ZSError(f"{path}: missing meta/data")
    want = expected_fixed_meta(model, condition, alpha)
    want.update(extra_expect or {})
    for k, v in want.items():
        if meta.get(k) != v or type(meta.get(k)) is not type(v):
            raise ZSError(f"{path}: meta.{k}={meta.get(k)!r} != expected {v!r}")
    if samples is not None:
        bp = bare_prompt_sha256(samples, condition)
        if meta.get("bare_prompt_sha256") != bp:
            raise ZSError(f"{path}: bare_prompt_sha256 does not match the "
                          f"{condition} prompt rendered from this manifest")
    if len(rows) != N_ITEMS:
        raise ZSError(f"{path}: {len(rows)} rows, expected {N_ITEMS}")
    for r in rows:
        for k in ROW_KEYS:
            if k not in r:
                raise ZSError(f"{path}: row missing {k!r}")
        if not isinstance(r["generated"], str):
            raise ZSError(f"{path}: non-string generation")
        if r["truncated"] != (r["stop_reason"] == "budget_exhausted"):
            raise ZSError(f"{path}: truncated flag disagrees with stop_reason")
        if r["stop_reason"] not in ("budget_exhausted", "natural_eos"):
            raise ZSError(f"{path}: unknown stop_reason {r['stop_reason']!r}")
    if samples is not None:
        for s, r in zip(samples, rows):
            for k in ("sample_id", "key", "dataset", "official_theory_id",
                      "official_qid"):
                if s[k] != r[k]:
                    raise ZSError(f"{path}: {k} mismatch at sample "
                                  f"{s['sample_id']} (identity/order differs "
                                  "from the manifest)")
    ns = sum(1 for r in rows if r["truncated"])
    if meta.get("n_truncated") != ns:
        raise ZSError(f"{path}: meta.n_truncated disagrees with rows")
    return meta, rows


CROSS_CELL_FIELDS = ("mask_path", "mask_sha256", "prompt_sha256",
                     "bare_prompt_sha256", "manifest_sha256_16",
                     "chat_template_hash", "model_dir")
PROVENANCE_FIELDS = ("host", "gpu_names", "sharded", "python", "torch",
                     "transformers", "numpy", "platform")


def check_cross_cell(metas: list[dict], label: str) -> None:
    """The alpha=0 and workpoint cells of one model must come from the same
    host / GPU model / software stack and identical prompt / mask / template
    / injection site."""
    a = metas[0]
    for m in metas:
        for k in CROSS_CELL_FIELDS + ("injection_site", "provenance"):
            if m.get(k) in (None, "", {}, []):
                raise ZSError(f"{label}: required meta.{k} is missing or empty")
        for k in ("final_token_id", "tail_token_ids"):
            if m["injection_site"].get(k) in (None, [], ""):
                raise ZSError(f"{label}: injection_site.{k} is missing or empty")
        for k in PROVENANCE_FIELDS:
            if k not in m["provenance"] or m["provenance"][k] in (None, "", []):
                if k in ("sharded",) and m["provenance"].get(k) is False:
                    continue
                raise ZSError(f"{label}: provenance.{k} is missing or empty")
    for m in metas[1:]:
        for k in CROSS_CELL_FIELDS:
            if a.get(k) != m.get(k):
                raise ZSError(f"{label}: meta.{k} differs between cells "
                              f"({a.get(k)!r} vs {m.get(k)!r})")
        if a.get("injection_site") != m.get("injection_site"):
            raise ZSError(f"{label}: injection_site differs between cells")
        pa, pb = a.get("provenance") or {}, m.get("provenance") or {}
        for k in PROVENANCE_FIELDS:
            if k not in pa or k not in pb:
                raise ZSError(f"{label}: provenance.{k} missing")
            if pa[k] != pb[k]:
                raise ZSError(f"{label}: provenance.{k} differs between cells "
                              f"({pa[k]!r} vs {pb[k]!r}); alpha=0 and the "
                              "workpoint must share host/GPU/software")


# ───────────────────────── per-cell metrics ─────────────────────────

def post_marker_class(text: str):
    """Fate of the text after the FIRST strict marker line:
    clean_stop | trailing_text (non-empty, no further strict marker) |
    additional_marker (another strict marker follows). None = no marker."""
    m = _MARKER_RE_V2.search(text)
    if not m:
        return None
    rest = text[m.end():].strip()
    if not rest:
        return "clean_stop"
    return "additional_marker" if find_all_markers_v2(rest) else "trailing_text"


def _med(vals):
    s = sorted(vals)
    return s[len(s) // 2] if s else None


def _rate(k, n):
    return {"n": k, "denominator": n, "rate": (k / n if n else None)}


def _split_rate(flags, valid):
    v = [f for f, ok in zip(flags, valid) if ok]
    nv = [f for f, ok in zip(flags, valid) if not ok]
    return {"with_valid_submission": _rate(sum(v), len(v)),
            "without_valid_submission": _rate(sum(nv), len(nv))}


def cell_stats(rows: list[dict], gold: dict | None = None) -> dict:
    """gold: {sample_id: label}; None = gold-free (gate) mode."""
    n = len(rows)
    parsed = [parse_final_answer_v2(r["generated"]) for r in rows]
    valid = [p.label is not None for p in parsed]
    n_valid = sum(valid)
    commits = [per_sample_commitment(r["generated"], "v2") for r in rows]
    cand = [lec.candidate(r["generated"]) for r in rows]
    cand40 = [lec.candidate(r["generated"], 40) for r in rows]
    cand80 = [lec.candidate(r["generated"], 80) for r in rows]
    af = [c["answer_first"] for c in commits]
    pm = [post_marker_class(r["generated"]) for r in rows]
    pm_counts = {k: sum(1 for x in pm if x == k)
                 for k in ("clean_stop", "trailing_text", "additional_marker")}
    af_clean = sum(1 for a, x in zip(af, pm) if a and x == "clean_stop")
    toks = [r["generated_token_count"] for r in rows]
    pred_dist = {k: sum(1 for p in parsed if p.label == k) for k in LABELS}
    pred_dist["none"] = n - n_valid
    out = {
        "n": n,
        "valid_submission": _rate(n_valid, n),
        "truncation": _rate(sum(1 for r in rows if r["truncated"]), n),
        "truncation_by_validity": _split_rate(
            [r["truncated"] for r in rows], valid),
        "loop_by_validity": _split_rate(
            [is_loop(r["generated"]) for r in rows], valid),
        "loop_note": ("loop/truncation are whole-generation flags; post_marker "
                      "does NOT locate where a loop starts, so a loop among "
                      "valid submissions is not shown to occur only after the "
                      "answer."),
        "natural_eos": _rate(sum(1 for r in rows
                                 if r["stop_reason"] == "natural_eos"), n),
        "loop": _rate(sum(1 for r in rows if is_loop(r["generated"])), n),
        "multiple_strict_markers": _rate(
            sum(1 for p in parsed if p.n_strict_markers >= 2), n),
        "gen_token_len": {"median": _med(toks), "mean": sum(toks) / n},
        "pred_label_counts": pred_dist,
        "early_candidate_label": {
            **_rate(sum(1 for c in cand if c), n),
            "detector": lec.VERSION, "cap_chars": lec.MAX_LINE_CHARS,
            "label_counts": {k: sum(1 for c in cand if c == k) for k in LABELS},
            "sensitivity_cap40": _rate(sum(1 for c in cand40 if c), n),
            "sensitivity_cap80": _rate(sum(1 for c in cand80 if c), n),
            "without_valid_submission": _rate(
                sum(1 for c, v in zip(cand, valid) if c and not v), n),
        },
        "answer_first": {
            **_rate(sum(1 for a, v in zip(af, valid) if a and v), n_valid),
            "denominator_note": "valid-submission subset",
        },
        "post_marker": {
            "denominator": n_valid, "counts": pm_counts,
            "answer_first_clean_stop": _rate(af_clean, sum(1 for a in af if a)),
            "note": ("clean_stop = nothing after the first strict marker; "
                     "trailing_text = explanation/other text, no further "
                     "marker; additional_marker = a further strict marker "
                     "(re-submission). The prompt has no stop instruction, so "
                     "these are observed behaviours; they do not locate the "
                     "start of any loop."),
        },
    }
    if gold is not None:
        corr = [is_correct(p.label, gold[r["sample_id"]]) for p, r in zip(parsed, rows)]
        last = [is_correct(p.all_strict_labels[-1] if p.all_strict_labels else None,
                           gold[r["sample_id"]]) for p, r in zip(parsed, rows)]
        n_vc = sum(1 for c, v in zip(corr, valid) if c and v)
        out["first_acc"] = _rate(sum(corr), n)
        out["conditional_accuracy"] = _rate(n_vc, n_valid)
        out["sensitivity_last_marker_acc"] = _rate(sum(last), n)
        out["gold_label_counts"] = {k: sum(1 for r in rows
                                           if gold[r["sample_id"]] == k) for k in LABELS}
    return out


def gate_verdict(stats: dict) -> dict:
    t = GATE_THRESHOLDS
    obs = {"valid_submission_rate": stats["valid_submission"]["rate"]}
    checks = {"valid_submission_rate":
              obs["valid_submission_rate"] >= t["valid_submission_rate_min"]}
    diag = {"truncation_rate": stats["truncation"]["rate"],
            "loop_rate": stats["loop"]["rate"],
            "truncation_by_validity": stats["truncation_by_validity"],
            "loop_by_validity": stats["loop_by_validity"]}
    return {"passed": all(checks.values()), "checks": checks, "observed": obs,
            "diagnostics_not_gated": diag, "thresholds": dict(t)}
