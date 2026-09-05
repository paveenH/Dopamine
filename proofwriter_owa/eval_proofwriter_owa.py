#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProofWriter OWA scoring. The ONLY script that reads gold. Protocol
`proofwriter-owa-v0`.

PRIMARY METRIC: Exact Label Accuracy against official True/False/Unknown
gold (proofwriter_owa/answer_parser.py, last-strict-marker, fail-closed).

ALSO REPORTED: D3 accuracy, D5 accuracy, per-label accuracy, parse-failure
rate, no-answer rate, invalid/multiple-final-answer rate, loop rate,
truncation rate, generation token length.

STATISTICS: each non-zero alpha vs that SAME model's own alpha=0. Exact
two-sided McNemar; Holm correction with m=3 PER MODEL (three non-zero alpha).
D3/D5 and per-label breakdowns are EXPLORATORY SUBGROUPS by default, reported
in full but not pooled into the primary Holm family. Confidence intervals
(when reported) are question-paired bootstrap. No dropping of unfavorable
depths or labels anywhere in this script.

WORKPOINT REPORTING: discrete argmax over the sampled alpha (ties -> smaller
|alpha|); near-optimal region defined only over sampled points, kept separate
from Holm significance. If no non-zero alpha survives Holm against alpha=0,
the frozen sentence below is used verbatim and no workpoint is declared.

THIS IS PROOFWRITER'S OWN TASK-SPECIFIC DOSE EXPLORATION -- there is no
"frozen GSM8K workpoint" being read here; every alpha in the sampled set is a
candidate.

@author: proofwriter_owa task
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from answer_parser import (normalize_label, is_correct, get_marker_family,
                           MARKER_FAMILIES)  # noqa: E402
from commitment import per_sample_commitment, aggregate_commitment  # noqa: E402
from scoring import mcnemar_exact, holm, question_paired_bootstrap_ci, \
    discrete_argmax, near_optimal_region  # noqa: E402

PROTOCOL = "proofwriter-owa-v0"
LABELS = ("True", "False", "Unknown")
NO_WORKPOINT_SENTENCE = (
    "No effective ProofWriter OWA workpoint was detected in the sampled dose set."
)

# Mirrors get_answer_proofwriter_owa.py's EXPECTED_CELLS exactly (frozen per
# PREREG_PROOFWRITER_OWA.md S5). A formal-run scoring call must see EXACTLY
# these four alphas per model -- not "however many files happened to be
# passed on the command line". Without this, a run missing one dose (or
# carrying an extra, unauthorized one) would silently score whatever showed
# up, report holm_family_m=3 in the output regardless, and nobody downstream
# could tell the difference from a complete, correctly-configured run.
EXPECTED_ALPHAS = {
    "llama3": {-6, -4, 0, 4},
    "qwen2.5": {-6, 0, 6, 8},
}

# meta fields that MUST be identical across every alpha cell of one model:
# a config drift here (different mask, different prompt, different batch
# size/token budget, different manifest) would silently mix incomparable
# cells into one "dose curve" and no downstream statistic would catch it --
# McNemar/Holm/bootstrap all just consume 0/1 accuracy vectors and have no
# way to know the vectors came from mismatched runs.
CONSISTENCY_FIELDS = (
    "model", "size", "layer_start", "layer_end", "L",
    "mask_path", "mask_sha256", "prompt_sha256", "prompt_template_id",
    "marker_family",
    "manifest_sha256_16", "batch_size", "max_new_tokens", "temperature",
    "top_p", "n_shot", "padding_side", "chat_template", "prefill_only",
    "prefill_tail_len",
)
# marker_family is DERIVABLE from prompt_template_id (answer_parser.
# get_marker_family), so checking both looks redundant -- it is kept
# deliberately as a second, independent field (review decision, 2026-09-05):
# prompt_template_id catching a drift relies on the registry lookup being
# correct and up to date; marker_family is what get_answer_proofwriter_owa.py
# ACTUALLY resolved and stored at generation time, so a stale or hand-edited
# meta.json with a mismatched pair (e.g. prompt_template_id says v2 but
# marker_family was hand-set to "v1") is caught here rather than silently
# trusted. Both fields must agree across every alpha cell of one model.

# A strict, GSM8K-loop-detector-style rule: the final 40-char block of the
# continuation recurring >=4 times in the full text. Matches the convention
# `analyze_loop_anxiety.py` / P4b's behaviour-panel note: a permissive n-gram
# proxy gives false positives on ordinary restatement.
def is_loop(text: str, tail_len: int = 40, min_repeats: int = 4) -> bool:
    text = text.strip()
    if len(text) < tail_len:
        return False
    tail = text[-tail_len:]
    return text.count(tail) >= min_repeats


def die(m):
    print(f"[FATAL] {m}", file=sys.stderr)
    raise SystemExit(2)


def load_gold(path):
    d = json.load(open(path, encoding="utf-8"))
    meta = d["meta"]
    if meta.get("contains_labels") is not True:
        die("gold file does not declare contains_labels=true; point --gold "
            "at manifest_gold.json")
    gold = {}
    for r in d["data"]:
        gold[r["sample_id"]] = {
            "answer": normalize_label(r["answer"]),
            "dataset": r["dataset"],
        }
    return meta, gold


def load_cell(path, protocol_expected=PROTOCOL):
    d = json.load(open(path, encoding="utf-8"))
    m = d["meta"]
    if m.get("protocol") != protocol_expected:
        die(f"{path}: protocol {m.get('protocol')!r} != {protocol_expected!r}")
    if m.get("accuracy_computed") is not False:
        die(f"{path}: generation file claims accuracy was already computed")
    # Resolve marker_family the SAME way get_answer_proofwriter_owa.py did
    # (from prompt_template_id, via the shared registry) and require it to
    # match what the cell's own meta already recorded. A mismatch means
    # either the generator and this evaluator disagree about what
    # prompt_template_id maps to (a registry drift), or the meta.json was
    # hand-edited/corrupted -- either way, scoring must stop rather than
    # silently pick one of the two disagreeing values.
    tpl_id = m.get("prompt_template_id")
    derived_family = get_marker_family(tpl_id)["marker_family"]
    recorded_family = m.get("marker_family")
    if recorded_family != derived_family:
        die(f"{path}: meta.marker_family={recorded_family!r} does not match "
            f"the family derived from prompt_template_id={tpl_id!r} "
            f"({derived_family!r}). This cell's marker family is "
            "ambiguous -- refusing to guess which one to score against.")
    return m, d["data"]


def score_cell(rows: list[dict], gold: dict, marker_family: str):
    """Returns per-sample records with parsed label / correctness / commitment
    fields, plus dataset-level summaries. `marker_family` ("v1" or "v2",
    answer_parser.MARKER_FAMILY_V1/_V2) is resolved by the CALLER from this
    cell's own prompt_template_id (see load_cell's cross-check above) and
    applied uniformly to every row in `rows` -- there is no per-row family
    guessing, and no default."""
    parse_fn = get_marker_family_functions(marker_family)
    out = []
    for r in rows:
        sid = r["sample_id"]
        if sid not in gold:
            die(f"sample_id {sid} in generation file has no gold entry")
        g = gold[sid]
        text = r["generated"]
        parsed = parse_fn(text)
        correct = is_correct(parsed.label, g["answer"])
        # pre_answer_reasoning_tokens needs a real tokenizer; this evaluator
        # runs offline with none loaded. get_answer_proofwriter_owa.py
        # computes it at generation time (the tokenizer is already loaded
        # there) and stores it per row as "pre_answer_reasoning_tokens" --
        # pass that through so the field is not silently None in every cell.
        # marker_family is threaded through explicitly (2026-09-05 fix) so
        # commitment timing is computed against the SAME family the marker
        # was actually parsed against, never a hardcoded default.
        commit = per_sample_commitment(
            text, marker_family,
            precomputed_tokens=r.get("pre_answer_reasoning_tokens"))
        out.append({
            "sample_id": sid, "dataset": g["dataset"], "gold": g["answer"],
            "pred": parsed.label, "correct": correct,
            "parse_status": parsed.status,
            "n_strict_markers": parsed.n_strict_markers,
            "n_loose_markers": parsed.n_loose_markers,
            "is_true_last_line": parsed.is_true_last_line,
            "truncated": r.get("truncated"),
            "generated_token_count": r.get("generated_token_count"),
            "loop": is_loop(text),
            "commit": commit,
        })
    return out


def get_marker_family_functions(marker_family: str):
    """Small local wrapper: return this family's parse_final_answer function
    from the registry. Kept as a one-line named function (rather than an
    inline dict lookup at each call site) purely so a future third family can
    be added by editing answer_parser.MARKER_FAMILIES alone -- no call site
    in this file needs to change."""
    for fam in MARKER_FAMILIES.values():
        if fam["marker_family"] == marker_family:
            return fam["parse_final_answer"]
    raise ValueError(f"no registered family has marker_family={marker_family!r}")


def summarize(scored: list[dict]):
    n = len(scored)
    acc = sum(1 for r in scored if r["correct"]) / n if n else None
    by_ds = {}
    for ds in ("D3", "D5"):
        rs = [r for r in scored if r["dataset"] == ds]
        by_ds[ds] = {"n": len(rs),
                    "accuracy": (sum(1 for r in rs if r["correct"]) / len(rs)
                                 if rs else None)}
    by_label = {}
    for lab in LABELS:
        rs = [r for r in scored if r["gold"] == lab]
        by_label[lab] = {"n": len(rs),
                         "accuracy": (sum(1 for r in rs if r["correct"]) / len(rs)
                                      if rs else None)}
    n_parse_fail = sum(1 for r in scored if r["parse_status"] != "ok")
    n_no_answer = sum(1 for r in scored if r["n_strict_markers"] == 0)
    n_multi = sum(1 for r in scored if r["n_strict_markers"] > 1
                  or r["n_loose_markers"] > 1)
    n_loop = sum(1 for r in scored if r["loop"])
    n_trunc = sum(1 for r in scored if r["truncated"])
    lens = sorted(r["generated_token_count"] for r in scored
                  if r["generated_token_count"] is not None)
    # Diagnostic (review finding #5, 2026-09-04): among samples that DID
    # parse (n_strict_markers>=1), what fraction actually put the scored
    # marker on the true last line, per the frozen prompt's literal
    # instruction. Denominator is parse-successful samples only -- a
    # no-marker sample has no "last marker" to check in the first place.
    ok_rows = [r for r in scored if r["parse_status"] == "ok"]
    n_true_last_line = sum(1 for r in ok_rows if r["is_true_last_line"])

    return {
        "n": n, "accuracy": acc,
        "by_dataset": by_ds, "by_label": by_label,
        "parse_failure_rate": n_parse_fail / n if n else None,
        "no_answer_rate": n_no_answer / n if n else None,
        "invalid_or_multiple_marker_rate": n_multi / n if n else None,
        "loop_rate": n_loop / n if n else None,
        "truncation_rate": n_trunc / n if n else None,
        "true_last_line_rate": (n_true_last_line / len(ok_rows)
                                if ok_rows else None),
        "gen_token_len_median": lens[len(lens) // 2] if lens else None,
        "gen_token_len_mean": sum(lens) / len(lens) if lens else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="manifest_gold.json")
    ap.add_argument("--generations", nargs="+", required=True,
                    help="cell JSONs written by get_answer_proofwriter_owa.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holm_m", type=int, default=3,
                    help="Holm family size per model (default 3 = the three "
                         "non-zero alpha in this task's own dose set)")
    ap.add_argument("--allow_partial_alphas", action="store_true",
                    help="Skip the frozen-dose-set completeness check. Only "
                         "for scoring a preflight (alpha=0 only) or pilot "
                         "(alpha=0 only) subset, where Holm/mcnemar-vs-alpha0 "
                         "is not meaningful anyway. A FORMAL run (the "
                         "300-item manifest, all four alpha per model) must "
                         "NEVER pass this flag -- doing so would let an "
                         "incomplete sweep silently get scored and reported "
                         "under the same holm_family_m=3 label as a real one.")
    a = ap.parse_args()

    if os.path.exists(a.out):
        die(f"{a.out} exists; refusing to overwrite")

    gmeta, gold = load_gold(a.gold)
    N = len(gold)

    cells = {}   # model -> alpha -> (meta, rows)
    for p in a.generations:
        m, rows = load_cell(p)
        mdl, al = m["model"], m["alpha"]
        # A cell's row count must equal N == len(gold) EXACTLY. This is not a
        # formal-vs-preflight distinction: N is whatever --gold file was
        # actually passed (the 300-row manifest_gold.json for a formal run,
        # or a smaller preflight/pilot gold subset for those stages), so
        # requiring an exact match against N already tolerates a legitimately
        # smaller gold file without a special case. The previous check --
        # `len(rows) not in (N, len([1 for _ in rows]))` -- is a tautology
        # (the second tuple element is always len(rows) itself, computed a
        # different way), so it could never be false and the `pass` branch
        # ran unconditionally: no cell was ever rejected for a short row
        # count.
        if len(rows) != N:
            die(f"{p}: {len(rows)} rows, but --gold has {N} entries; "
                f"a cell must have exactly one row per gold item.")
        ids_here = [r["sample_id"] for r in rows]
        dupes = {i for i in ids_here if ids_here.count(i) > 1}
        if dupes:
            die(f"{p}: duplicate sample_id(s) within this cell: "
                f"{sorted(dupes)[:5]}; every gold item must appear exactly "
                "once, or downstream per-id lookups (scored_by_alpha[al][i]) "
                "would silently pick one arbitrary duplicate.")
        ids_here = set(ids_here)
        missing = ids_here - set(gold)
        if missing:
            die(f"{p}: {len(missing)} sample_id(s) not present in gold, "
                f"e.g. {sorted(missing)[:5]}")
        exp_fires = 0 if al == 0 else m["L"] * len(rows)
        if m.get("steering_fires") != exp_fires:
            die(f"{p}: steering_fires {m.get('steering_fires')} != {exp_fires}; "
                "intervention unverified")
        if al in cells.get(mdl, {}):
            die(f"{mdl} alpha={al} supplied twice")
        cells.setdefault(mdl, {})[al] = (m, rows)

    results = {}
    for mdl, byalpha in sorted(cells.items()):
        if 0 not in byalpha:
            die(f"{mdl}: no alpha=0 cell; it is the baseline for every "
                "comparison in this evaluator")
        # COMPLETENESS CHECK (review findings #6 and #8, 2026-09-04). Without
        # this, a run missing one of the model's four frozen alpha (or
        # carrying an extra, unauthorized one) would silently score whichever
        # subset was passed on the command line, still label the Holm family
        # holm_family_m=3 in the output, and nothing downstream could tell a
        # complete formal sweep from a partial one.
        #
        # Two cases are legitimate WITHOUT any flag: (a) the PILOT's
        # alpha=0-ONLY case -- run_proofwriter_owa.sh's pilot stage invokes
        # this evaluator with exactly one alpha=0 cell and explicitly
        # documents "no mcnemar_vs_alpha0 pairs -- that is expected", never
        # passing --allow_partial_alphas, so requiring the flag there would
        # silently break the launcher's own documented workflow; and (b) the
        # FORMAL sweep's exact frozen 4-point family. Anything else --
        # missing one dose, or an extra unauthorized one -- is a protocol
        # violation and requires the human to explicitly acknowledge it via
        # --allow_partial_alphas (intended for a deliberate partial score,
        # e.g. inspecting one finished cell mid-sweep; never part of the
        # pilot or formal launcher paths).
        want = EXPECTED_ALPHAS[mdl]
        have = set(byalpha)
        if have != {0} and have != want and not a.allow_partial_alphas:
            die(f"{mdl}: alpha set {sorted(have)} is neither the pilot's "
                f"alpha=0-only case nor this model's frozen formal dose set "
                f"{sorted(want)} (missing {sorted(want - have)}, extra "
                f"{sorted(have - want)}). A formal-sweep scoring call must "
                "see all four frozen alpha for this model, or Holm m=3 is "
                "not a well-defined family. Pass --allow_partial_alphas only "
                "for a deliberate, human-invoked partial score.")
        if have != {0} and have != want and a.allow_partial_alphas:
            print(f"[eval] NOTE: {mdl} scored with a partial alpha set "
                  f"{sorted(have)} under --allow_partial_alphas; this run's "
                  "mcnemar_vs_alpha0/Holm output must NOT be cited as the "
                  "formal 4-point family.")
        # common sample_id set for THIS model across its own cells (formal
        # sweep uses the full manifest; preflight/pilot use a fixed subset --
        # either way every cell of one model must share exactly the same ids)
        id_sets = [set(r["sample_id"] for r in rows) for _, rows in byalpha.values()]
        if any(s != id_sets[0] for s in id_sets):
            die(f"{mdl}: cells do not all share the same sample_id set; "
                "cannot pair them")
        ids_sorted = sorted(id_sets[0])

        # ---- cross-alpha configuration consistency (review finding #8).
        # Every cell of one model must be a genuinely comparable point on
        # ONE dose curve: same mask, same rendered prompts, same manifest,
        # same batch/token/sampling config, same model checkpoint. Nothing
        # downstream (McNemar / Holm / bootstrap) can detect a config drift
        # on its own -- they only ever see 0/1 accuracy vectors -- so a
        # mismatched mask_sha256 or prompt_sha256 between two alpha cells
        # would silently produce a "dose-response" that is actually an
        # artifact of comparing two different experiments.
        metas_by_alpha = {al: m for al, (m, _) in byalpha.items()}
        ref_al = sorted(metas_by_alpha)[0]
        ref_meta = metas_by_alpha[ref_al]
        for al, m in metas_by_alpha.items():
            for field in CONSISTENCY_FIELDS:
                if m.get(field) != ref_meta.get(field):
                    die(f"{mdl}: alpha={al} cell's {field!r}={m.get(field)!r} "
                        f"differs from alpha={ref_al}'s {ref_meta.get(field)!r}; "
                        "cells of one model's dose curve must share an "
                        "identical configuration (mask/prompt/manifest/"
                        "batch/token-budget/model), or they are not a "
                        "comparable curve.")

        # marker_family is already guaranteed identical across every alpha
        # cell of this model by the CONSISTENCY_FIELDS check just above, so
        # it is safe to read it once here (from ref_meta) and apply it
        # uniformly to every score_cell call below -- no per-alpha family
        # resolution or guessing.
        model_marker_family = ref_meta["marker_family"]
        scored_by_alpha = {al: {r["sample_id"]: r
                                for r in score_cell(rows, gold, model_marker_family)}
                            for al, (_, rows) in byalpha.items()}

        def acc_vec(al):
            s = scored_by_alpha[al]
            return [1 if s[i]["correct"] else 0 for i in ids_sorted]

        acc0 = acc_vec(0)
        model_res = {
            "alphas_present": sorted(byalpha),
            "n_items": len(ids_sorted),
            "cells": {},
        }
        for al in sorted(byalpha):
            rows_scored = [scored_by_alpha[al][i] for i in ids_sorted]
            model_res["cells"][str(al)] = summarize(rows_scored)
            model_res["cells"][str(al)]["commitment"] = aggregate_commitment(
                [r["commit"] for r in rows_scored])

        non_zero = [al for al in byalpha if al != 0]
        pair_results = {}
        for al in non_zero:
            accA = acc_vec(al)
            b01, b10, p = mcnemar_exact(acc0, accA)
            lo, hi = question_paired_bootstrap_ci(acc0, accA)
            pair_results[al] = {
                "alpha": al,
                "acc_base": sum(acc0) / len(acc0),
                "acc_alpha": sum(accA) / len(accA),
                "dAcc_pp": (sum(accA) - sum(acc0)) / len(acc0) * 100,
                "discordant_0to1": b01, "discordant_1to0": b10,
                "p_raw": p, "ci95_pp": [lo, hi],
            }
        # holm() (scoring.py) always corrects over exactly len(holm_pairs) --
        # it has no notion of a declared family size, it just uses however
        # many (key, p) pairs it is handed. The PREVIOUS code stored
        # `a.holm_m` (the CLI flag, default 3) as "holm_family_m" regardless
        # of len(holm_pairs), so a run scored with only 2 of 3 non-zero doses
        # would still claim "holm_family_m": 3 in the output even though the
        # correction actually applied used m=2 -- an unadjusted-vs-formal
        # family-size mismatch invisible to anyone reading the JSON. The
        # completeness check above already guarantees len(non_zero) == 3 for
        # any run that isn't the pilot's alpha=0-only case (which never
        # reaches this branch: holm_pairs is empty when non_zero is empty),
        # so this assert is a second, independent confirmation rather than a
        # silent trust of the CLI default.
        holm_pairs = [(al, r["p_raw"]) for al, r in pair_results.items()]
        adj = holm(holm_pairs) if holm_pairs else {}
        actual_m = len(holm_pairs)
        if actual_m and actual_m != a.holm_m and not a.allow_partial_alphas:
            die(f"{mdl}: Holm was computed over {actual_m} non-zero alpha "
                f"{sorted(pair_results)}, but --holm_m={a.holm_m} was "
                "declared; these must match or the reported holm_family_m "
                "label would misdescribe the correction actually applied. "
                "This should be unreachable given the alpha-completeness "
                "check above -- if it fires, that check has a gap.")
        for al in pair_results:
            pair_results[al]["p_holm_adj"] = adj.get(al)
            pair_results[al]["holm_family_m"] = actual_m

        model_res["mcnemar_vs_alpha0"] = {str(al): v for al, v in pair_results.items()}

        # ---- exploratory D3/D5 and per-label subgroup McNemar (NOT pooled
        # into the primary Holm family above; reported in full)
        subgroup = {}
        for al in non_zero:
            subgroup[str(al)] = {}
            for ds in ("D3", "D5"):
                ids_ds = [i for i in ids_sorted if scored_by_alpha[0][i]["dataset"] == ds]
                if not ids_ds:
                    continue
                a0 = [1 if scored_by_alpha[0][i]["correct"] else 0 for i in ids_ds]
                aA = [1 if scored_by_alpha[al][i]["correct"] else 0 for i in ids_ds]
                b01, b10, p = mcnemar_exact(a0, aA)
                subgroup[str(al)][ds] = {
                    "n": len(ids_ds), "acc_base": sum(a0) / len(a0),
                    "acc_alpha": sum(aA) / len(aA), "p_raw": p,
                    "discordant_0to1": b01, "discordant_1to0": b10,
                    "scope": "exploratory subgroup, not in the primary Holm family",
                }
            for lab in LABELS:
                ids_lab = [i for i in ids_sorted if scored_by_alpha[0][i]["gold"] == lab]
                if not ids_lab:
                    continue
                a0 = [1 if scored_by_alpha[0][i]["correct"] else 0 for i in ids_lab]
                aA = [1 if scored_by_alpha[al][i]["correct"] else 0 for i in ids_lab]
                b01, b10, p = mcnemar_exact(a0, aA)
                subgroup[str(al)][f"label_{lab}"] = {
                    "n": len(ids_lab), "acc_base": sum(a0) / len(a0),
                    "acc_alpha": sum(aA) / len(aA), "p_raw": p,
                    "discordant_0to1": b01, "discordant_1to0": b10,
                    "scope": "exploratory subgroup, not in the primary Holm family",
                }
        model_res["exploratory_subgroups"] = subgroup

        # ---- workpoint reporting
        alpha_to_acc = {al: pair_results[al]["acc_alpha"] if al in pair_results
                        else sum(acc0) / len(acc0) for al in byalpha}
        argmax_alpha = discrete_argmax(alpha_to_acc)
        region = near_optimal_region(alpha_to_acc, tolerance_pp=0.0)

        # A dose only qualifies as a workpoint candidate if it is BOTH
        # Holm-significant AND an IMPROVEMENT (dAcc_pp > 0) vs alpha=0. The
        # previous `any_holm_pass = any(p_holm_adj < 0.05 ...)` accepted a
        # Holm-significant DEGRADATION just as readily as an improvement, and
        # then reported argmax_alpha (computed independently over raw
        # accuracy) as "the workpoint candidate" without ever checking that
        # argmax_alpha was itself among the Holm-passing doses -- so a
        # significant degradation at one alpha could report an unrelated,
        # non-significant alpha as the verdict, or even report alpha=0 itself
        # if every non-zero dose happened to score lower.
        qualifying = [al for al, v in pair_results.items()
                     if v["p_holm_adj"] is not None and v["p_holm_adj"] < 0.05
                     and v["dAcc_pp"] > 0]
        any_holm_pass_either_direction = any(
            v["p_holm_adj"] is not None and v["p_holm_adj"] < 0.05
            for v in pair_results.values())
        if qualifying:
            # Highest accuracy among the qualifying set; ties -> smaller
            # |alpha| (matches zebralogic/eval_zebralogic.py's convention).
            workpoint_alpha = max(qualifying,
                                  key=lambda al: (alpha_to_acc[al], -abs(al)))
        else:
            workpoint_alpha = None
        model_res["workpoint"] = {
            "alpha_to_accuracy": alpha_to_acc,
            "discrete_argmax_alpha": argmax_alpha,
            "near_optimal_region_exact_ties": region,
            "any_dose_holm_significant_either_direction": any_holm_pass_either_direction,
            "holm_significant_improvement_alphas": qualifying,
            "workpoint_alpha": workpoint_alpha,
            "verdict": (NO_WORKPOINT_SENTENCE if workpoint_alpha is None else
                       f"workpoint candidate alpha={workpoint_alpha} "
                       "(Holm-significant IMPROVEMENT vs this model's "
                       "alpha=0; see mcnemar_vs_alpha0 for the full family)"),
        }

        results[mdl] = model_res

    out = {
        "protocol": PROTOCOL,
        "not_a_transfer_test": True,
        "gold_manifest_sha256_16": gmeta.get("manifest_sha256_16"),
        "n_gold_items": N,
        "labels": list(LABELS),
        "owa_semantics": gmeta.get("owa_semantics"),
        "holm_family_m_declared": a.holm_m,
        "note_holm_family_m": ("the authoritative per-model family size is "
                               "results[model].mcnemar_vs_alpha0[*].holm_family_m "
                               "(derived from the actual non-zero alpha count "
                               "scored for that model); this top-level field "
                               "is only the --holm_m CLI default and can "
                               "legitimately differ for a pilot (0 non-zero "
                               "alpha) or an --allow_partial_alphas run."),
        "results": results,
        "note": ("D3/D5 and per-label breakdowns are exploratory subgroups, "
                "not pooled into the primary per-model Holm(m=3) family. "
                "Commitment-extractor fields are descriptive co-occurrence "
                "statistics only, never causal mediation evidence."),
        # Renamed from "note_llama3_loop_rate" to a model-neutral name
        # (review finding, 2026-09-04): this note is written into EVERY
        # report regardless of which model(s) it covers, and the original
        # name baked in an assumption -- "this is a llama3 thing" -- into a
        # field that could equally appear next to a qwen2.5-only result.
        # The interpretive rule it states applies uniformly to whichever
        # model/alpha actually shows an elevated loop_rate; it does not
        # presuppose llama3 is the (or the only) affected model.
        "note_loop_interpretation": (
            "Decided 2026-09-04 (PREREG_PROOFWRITER_OWA.md S6): the llama3 "
            "alpha=0 preflight (30 items) showed 100% truncation at the "
            "original max_new_tokens=768, with manual inspection finding a "
            "high, stable degenerate-repetition-loop tendency alongside "
            "genuine long-form reasoning. max_new_tokens was raised ONE "
            "TIME to a frozen 1024 for both models; loop_rate/"
            "truncation_rate are recorded per cell (see "
            "results[model].cells[alpha].loop_rate) but were NOT used as a "
            "preflight gate for either model. Consequence for reading this "
            "report: on ANY model/alpha with an elevated loop_rate, "
            "generation-length and commitment-timing diagnostics for that "
            "cell carry weaker interpretive weight; parseable-answer "
            "accuracy is scored identically to every other cell via the "
            "frozen parser and is unaffected."),
    }
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # A loop-rate threshold that prints a caveat, not a gate (frozen
    # 2026-09-04, PREREG_PROOFWRITER_OWA.md S6). Added after the llama3
    # alpha=0 preflight showed a high, stable degenerate-repetition-loop
    # tendency on this task, but the check itself is MODEL-NEUTRAL -- it
    # runs for every model in `results`, not llama3 specifically, since
    # nothing rules out the same pattern showing up elsewhere. loop_rate is
    # never used to filter, re-weight, or block anything -- this is purely
    # so a reader of the terminal summary sees the caveat inline instead of
    # having to separately notice it in the JSON's per-cell loop_rate field.
    LOOP_RATE_CAVEAT_THRESHOLD = 0.10

    print(f"\n=== PROOFWRITER OWA — task-specific dose exploration ===")
    for mdl, r in sorted(results.items()):
        print(f"\n[{mdl}]  n_items={r['n_items']}  alphas={r['alphas_present']}")
        high_loop_alphas = []
        for al in r["alphas_present"]:
            c = r["cells"][str(al)]
            print(f"  alpha={al:>3}  acc={c['accuracy']:.4f}  "
                  f"D3={c['by_dataset']['D3']['accuracy']}  "
                  f"D5={c['by_dataset']['D5']['accuracy']}  "
                  f"parse_fail={c['parse_failure_rate']:.3f}  "
                  f"loop={c['loop_rate']}  "
                  f"trunc={c['truncation_rate']}")
            if (c["loop_rate"] or 0) >= LOOP_RATE_CAVEAT_THRESHOLD:
                high_loop_alphas.append(al)
        if high_loop_alphas:
            print(f"  CAVEAT: {mdl} shows loop_rate >= "
                  f"{LOOP_RATE_CAVEAT_THRESHOLD:.0%} at alpha="
                  f"{high_loop_alphas} -- generation-length and "
                  "commitment-timing diagnostics for this model/alpha carry "
                  "weaker interpretive weight (known, stable degenerate-"
                  "repetition tendency, not a scoring defect). NOTE: the "
                  "MAIN accuracy above IS affected -- its denominator is "
                  "all items, so every parse failure scores incorrect. "
                  "Read accuracy together with parse_fail, never alone.")
        print(f"  workpoint verdict: {r['workpoint']['verdict']}")

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
