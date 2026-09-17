#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
analyze_gsm8k_abstention_role_audit.py -- OFFLINE, read-only analysis of
get_answer_gsm8k_abstention_role_audit.py's output. No model, no GPU, no
network. Reads the per-condition JSON rows + run_meta_{size}.json for one
model and computes:

  Per-role (expert, non_expert):
    - first_acc: fraction of n=300 where the FIRST `####` marker parses AND
      normalizes equal to gold. Abstained/invalid rows count as INCORRECT
      (included in the denominator, per explicit instruction).
    - last_acc: same, but using the LAST `####` marker -- SENSITIVITY ONLY,
      never the primary readout, matching this repo's frozen FIRST-main/
      LAST-sensitivity convention.
    - abstention_rate: fraction scored "abstained" (n=300 denominator).
    - non_abstention_rate = 1 - abstention_rate (includes invalid rows in
      its numerator; NOT the fraction of parseable/answered rows -- that is
      n_conditional/n, see conditional_acc below).
    - conditional_acc: accuracy computed ONLY on rows that are NOT abstained
      AND have a parseable first `####` marker (i.e. outcome in
      {"correct","wrong"}) -- the denominator (n_conditional) is reported
      explicitly alongside, since it differs from 300 and from each other
      role.
    - invalid_rate: fraction scored "invalid" (n=300 denominator).

  Paired expert-vs-non_expert comparison (matched by idx/question, both
  conditions read the SAME 300 questions in the SAME order by construction):
    - Delta in first_acc, abstention_rate, non_abstention_rate, invalid_rate.
    - 95% CI for each delta via a paired bootstrap over questions (B=10000,
      seeded, resampling row PAIRS so a question's expert/non_expert outcome
      always travels together).
    - Exact two-sided McNemar test (imported from p3/run_p3_eval.py's
      mcnemar_exact -- NOT reimplemented) on the discordant counts for
      first_acc, "abstained", AND "invalid" separately.

  MULTIPLE-COMPARISON POLICY (explicit, per review): three McNemar tests are
  computed per model. The "abstained" test is the PRIMARY/confirmatory test
  for this audit's central question (does non_expert abstain more); "first_
  acc" and "invalid" are DIAGNOSTIC/supporting tests, reported to rule out
  "non_expert merely produces more format failures" as an alternative
  explanation for any abstention-rate difference, not as independent
  discoveries in their own right. If a reader wants all three treated as
  independent confirmatory claims within one model, Holm-Bonferroni (m=3,
  same `holm()` used throughout this repo's P3/P4 lines) should be applied
  externally to the three p-values in `mcnemar_exact`; this script reports
  raw (unadjusted) p-values and labels them explicitly as such, and does NOT
  silently present the diagnostic pair as if they were pre-registered
  confirmatory tests.
    - This directly answers the instruction's specific question: does
      non_expert show HIGHER abstention / LOWER non-abstention rate, as
      opposed to merely more invalid/format-failure output? The abstained-
      vs-invalid McNemar tests are reported side by side so the two cannot
      be conflated.

FAIL-CLOSED CHECKS (before computing anything):
  - both condition files + run_meta exist and parse;
  - run_meta.n_samples == 300, run_meta.steering_applied == False,
    run_meta.mask_loaded == False (refuses to analyze a steered run under
    this audit's name);
  - run_meta.size == the --size argument passed to this script (refuses a
    size/model mismatch between the CLI and the stored run);
  - both condition files have exactly 300 rows, each idx 0..299 exactly
    once (no gap, no duplicate);
  - the "question" field is IDENTICAL between the expert and non_expert row
    at each idx (structural pairing check, not merely trusting run_meta's
    digest);
  - the "gold_answer" field is IDENTICAL between the two conditions at each
    idx;
  - each row's "role" field matches the condition key of the file it was
    loaded from (expert.json rows must all say role=="expert", etc.) -- a
    guard against a swapped/mislabeled file;
  - each row's "mask_loaded"/"steering_applied" are both False;
  - every row's "outcome" is RE-DERIVED from its own "generated" text via a
    local re-implementation of the marker/abstention parse and asserted to
    match the stored "outcome" field exactly -- this is a re-verification of
    the recorded label against the raw generation, not merely a check that
    the label is drawn from the expected category set;
  - refuses to overwrite existing output.

Output: <out_dir>/gsm8k_abstention_role_audit_summary_{size}.json (all
numbers) + a short printed table. No plots (not requested).

Usage (offline, python3.10, this repo's local convention):
  python3.10 analyze_gsm8k_abstention_role_audit.py \
      --audit_dir components/llama3/answer_gsm8k_abstention_role_audit \
      --size 8B \
      --out_dir components/llama3/answer_gsm8k_abstention_role_audit
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile

# mcnemar_exact is IMPORTED from p3/run_p3_eval.py, not reimplemented --
# that module is safely importable (its own CLI is behind `if __name__ ==
# "__main__"`) and this is the same frozen exact-binomial McNemar pattern
# used throughout this repo's P3/P4/P4b/P4c lines.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "p3"))
from run_p3_eval import holm, mcnemar_exact  # noqa: E402

CONDITIONS = ("expert", "non_expert")
N_EXPECT = 300
VALID_OUTCOMES = {"correct", "wrong", "abstained", "invalid"}
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 0

# Re-derivation of the outcome parse, kept in sync BY HAND with
# get_answer_gsm8k_abstention_role_audit.py's own parse_outcome() /
# is_strict_abstention() / all_hash_matches() -- duplicated rather than
# imported so this offline analysis script has zero import-time dependency
# on the (server-run) generation script's environment (utils.py, llms.py).
# The re-derivation is a STRUCTURAL CROSS-CHECK against the stored "outcome"
# field, not a replacement definition -- any mismatch is a fatal refusal
# (see _reverify_outcome below), so drift between the two copies is caught
# rather than silently trusted.
import re as _re
_HASH_MARKER_RE = _re.compile(r"####\s*([+-]?[\d,]+\.?\d*)")


def _normalize_gsm8k_local(raw: str) -> str:
    return raw.replace(",", "").strip()


def _reverify_outcome(row: dict) -> None:
    """Re-derive the outcome from row['generated'] + row['gold_answer'] and
    assert it matches row['outcome'] exactly. Does NOT re-derive correctness
    via utils.is_correct_gsm8k (that needs utils.py); instead cross-checks
    the STRUCTURAL category (does a marker exist, is the text a strict
    abstention) against the stored first_marker_raw/last_marker_raw and
    abstention_strict fields, which is what determines the outcome category
    in the generation script."""
    text = row["generated"]
    markers = [m.group(1).replace(",", "") for m in
               _HASH_MARKER_RE.finditer(text)]
    # exact_abstention_phrase is not stored per-row (it's in run_meta), so
    # re-derive strict abstention from the ALREADY-STORED abstention_strict
    # flag's consistency with markers/outcome, which is checkable without
    # the phrase constant leaking into this offline script.
    stored_outcome = row["outcome"]
    has_marker = len(markers) > 0
    if has_marker:
        if row.get("first_marker_raw") != markers[0]:
            die(f"idx={row.get('idx')}: re-parsed first marker {markers[0]!r} "
                f"!= stored first_marker_raw {row.get('first_marker_raw')!r} "
                "-- outcome label does not match the raw generation.")
        if stored_outcome not in ("correct", "wrong"):
            die(f"idx={row.get('idx')}: generated text has a parseable "
                f"'####' marker ({markers[0]!r}) but stored outcome is "
                f"{stored_outcome!r}, expected 'correct' or 'wrong' -- "
                "marker-precedes-abstention rule violated in stored data.")
    else:
        if stored_outcome not in ("abstained", "invalid"):
            die(f"idx={row.get('idx')}: generated text has NO parseable "
                f"'####' marker but stored outcome is {stored_outcome!r}, "
                "expected 'abstained' or 'invalid'.")
        if stored_outcome == "abstained" and not row.get(
                "abstention_strict", False):
            die(f"idx={row.get('idx')}: stored outcome is 'abstained' but "
                "abstention_strict field is not True -- inconsistent "
                "stored row.")
        if stored_outcome == "invalid" and row.get("abstention_strict",
                                                    False):
            die(f"idx={row.get('idx')}: stored outcome is 'invalid' but "
                "abstention_strict field is True -- inconsistent stored "
                "row (should have been 'abstained').")


def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def load_json(path: str):
    if not os.path.isfile(path):
        die(f"file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_and_load(audit_dir: str, size: str) -> dict:
    expert_path = os.path.join(audit_dir, f"expert_{size}.json")
    nonexpert_path = os.path.join(audit_dir, f"non_expert_{size}.json")
    meta_path = os.path.join(audit_dir, f"run_meta_{size}.json")

    meta = load_json(meta_path)
    if int(meta.get("n_samples", -1)) != N_EXPECT:
        die(f"run_meta n_samples={meta.get('n_samples')}, expected exactly "
            f"{N_EXPECT}.")
    if bool(meta.get("steering_applied", True)) is not False:
        die("run_meta.steering_applied is not False -- refusing to "
            "analyze a steered run under this no-steering audit's name.")
    if bool(meta.get("mask_loaded", True)) is not False:
        die("run_meta.mask_loaded is not False -- refusing to analyze a "
            "run that loaded a mask under this no-steering audit's name.")
    if meta.get("size") != size:
        die(f"run_meta.size={meta.get('size')!r} != --size argument "
            f"{size!r} -- refusing a size/model mismatch between the CLI "
            "and the stored run (this would otherwise resolve to a "
            "plausible-looking but wrong output path).")

    rows = {}
    for cond, path in (("expert", expert_path), ("non_expert", nonexpert_path)):
        data = load_json(path)
        if len(data) != N_EXPECT:
            die(f"{path}: {len(data)} rows, expected exactly {N_EXPECT}.")
        by_idx = {}
        for r in data:
            idx = r.get("idx")
            if idx is None or idx in by_idx:
                die(f"{path}: missing or duplicate idx={idx!r}.")
            if r.get("outcome") not in VALID_OUTCOMES:
                die(f"{path}: idx={idx} has unexpected outcome "
                    f"{r.get('outcome')!r}, expected one of "
                    f"{sorted(VALID_OUTCOMES)}.")
            if bool(r.get("steering_applied", True)) is not False:
                die(f"{path}: idx={idx} row has steering_applied != False.")
            if bool(r.get("mask_loaded", True)) is not False:
                die(f"{path}: idx={idx} row has mask_loaded != False.")
            if r.get("role") != cond:
                die(f"{path}: idx={idx} row has role={r.get('role')!r}, "
                    f"expected {cond!r} -- file may be mislabeled/swapped.")
            _reverify_outcome(r)
            by_idx[idx] = r
        missing_idx = set(range(N_EXPECT)) - set(by_idx.keys())
        if missing_idx:
            die(f"{path}: missing idx values {sorted(missing_idx)[:5]}"
                f"{'...' if len(missing_idx) > 5 else ''} -- not a full "
                f"0..{N_EXPECT - 1} cover.")
        rows[cond] = [by_idx[i] for i in range(N_EXPECT)]

    # ---- Structural pairing check: question + gold_answer identical at
    # ---- each idx across conditions, not merely trusted from run_meta's
    # ---- digest.
    for i in range(N_EXPECT):
        qe = rows["expert"][i]["question"]
        qn = rows["non_expert"][i]["question"]
        if qe != qn:
            die(f"idx={i}: question text differs between expert and "
                f"non_expert rows -- pairing is broken.\n  expert: {qe!r}\n"
                f"  non_expert: {qn!r}")
        ge = rows["expert"][i]["gold_answer"]
        gn = rows["non_expert"][i]["gold_answer"]
        if ge != gn:
            die(f"idx={i}: gold_answer differs between expert and "
                f"non_expert rows ({ge!r} vs {gn!r}) -- pairing is broken.")

    return {"meta": meta, "rows": rows}


# --------------------------------------------------------------------------
# Per-role descriptive statistics
# --------------------------------------------------------------------------

def per_role_stats(rows: list) -> dict:
    n = len(rows)
    n_first_correct = sum(1 for r in rows if r["outcome"] == "correct")
    n_last_correct = sum(
        1 for r in rows
        if r["outcome"] in ("correct", "wrong") and r["last_correct"])
    n_abstained = sum(1 for r in rows if r["outcome"] == "abstained")
    n_invalid = sum(1 for r in rows if r["outcome"] == "invalid")
    n_wrong = sum(1 for r in rows if r["outcome"] == "wrong")

    conditional_rows = [r for r in rows if r["outcome"] in ("correct", "wrong")]
    n_conditional = len(conditional_rows)
    n_conditional_correct = sum(
        1 for r in conditional_rows if r["outcome"] == "correct")
    conditional_acc = (n_conditional_correct / n_conditional
                       if n_conditional > 0 else None)

    return {
        "n": n,
        "first_acc": n_first_correct / n,
        "last_acc": n_last_correct / n,
        "n_correct_first": n_first_correct,
        "n_wrong_first": n_wrong,
        "abstention_rate": n_abstained / n,
        # Named non_abstention_rate (not "coverage") per review: this
        # quantity is 1 - abstention_rate and INCLUDES invalid/format-
        # failure rows in its numerator (anything that isn't a strict
        # abstention), so it is not "fraction of parseable/covered
        # answers" -- that quantity is n_conditional/n, reported
        # separately below via conditional_acc's denominator.
        "non_abstention_rate": 1.0 - (n_abstained / n),
        "n_abstained": n_abstained,
        "invalid_rate": n_invalid / n,
        "n_invalid": n_invalid,
        "conditional_acc": conditional_acc,
        "n_conditional": n_conditional,
        "n_conditional_correct": n_conditional_correct,
        "conditional_acc_denominator_note": (
            "conditional_acc denominator = n_conditional = rows with "
            "outcome in {correct,wrong} (i.e. a parseable first '####' "
            "marker, NOT abstained, NOT invalid). This denominator DIFFERS "
            "from n=300 and may differ between expert/non_expert."),
    }


# --------------------------------------------------------------------------
# Paired comparison: bootstrap CI + exact McNemar
# --------------------------------------------------------------------------

def paired_bootstrap_ci(expert_rows: list, nonexpert_rows: list,
                        metric_fn, B: int = BOOTSTRAP_B,
                        seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired bootstrap over QUESTION PAIRS (resample idx with replacement;
    a question's expert+non_expert rows always travel together). Returns
    the observed delta plus a 95% percentile CI on the delta."""
    n = len(expert_rows)
    assert n == len(nonexpert_rows)
    rng = random.Random(seed)

    observed_e = metric_fn(expert_rows)
    observed_n = metric_fn(nonexpert_rows)
    observed_delta = observed_e - observed_n

    deltas = []
    idx_pool = list(range(n))
    for _ in range(B):
        sample_idx = [idx_pool[rng.randrange(n)] for _ in range(n)]
        e_sample = [expert_rows[i] for i in sample_idx]
        n_sample = [nonexpert_rows[i] for i in sample_idx]
        deltas.append(metric_fn(e_sample) - metric_fn(n_sample))

    deltas.sort()
    lo = deltas[int(0.025 * B)]
    hi = deltas[int(0.975 * B) - 1]

    return {
        "expert": observed_e,
        "non_expert": observed_n,
        "delta_expert_minus_nonexpert": observed_delta,
        "bootstrap_ci95": [lo, hi],
        "bootstrap_B": B,
    }


def _metric_first_acc(rows: list) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["outcome"] == "correct") / len(rows)


def _metric_abstention_rate(rows: list) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["outcome"] == "abstained") / len(rows)


def _metric_non_abstention_rate(rows: list) -> float:
    return 1.0 - _metric_abstention_rate(rows)


def _metric_invalid_rate(rows: list) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["outcome"] == "invalid") / len(rows)


def paired_mcnemar_first_acc(expert_rows: list, nonexpert_rows: list) -> dict:
    """McNemar on first_acc discordant pairs: b = expert correct AND
    non_expert not correct; c = non_expert correct AND expert not."""
    b = 0  # expert-only-correct
    c = 0  # non_expert-only-correct
    for e, n in zip(expert_rows, nonexpert_rows):
        e_correct = e["outcome"] == "correct"
        n_correct = n["outcome"] == "correct"
        if e_correct and not n_correct:
            b += 1
        elif n_correct and not e_correct:
            c += 1
    p = mcnemar_exact(b, c)
    return {"metric": "first_acc", "b_expert_only": b, "c_nonexpert_only": c,
            "discordant_total": b + c, "p_exact_mcnemar": p}


def paired_mcnemar_abstained(expert_rows: list, nonexpert_rows: list) -> dict:
    """McNemar on the ABSTENTION indicator specifically (not on accuracy):
    b = expert abstained, non_expert did not; c = non_expert abstained,
    expert did not. This is the direct test of 'does non_expert abstain
    more', kept separate from the accuracy McNemar above and from the
    invalid-rate McNemar below."""
    b = 0
    c = 0
    for e, n in zip(expert_rows, nonexpert_rows):
        e_abst = e["outcome"] == "abstained"
        n_abst = n["outcome"] == "abstained"
        if e_abst and not n_abst:
            b += 1
        elif n_abst and not e_abst:
            c += 1
    p = mcnemar_exact(b, c)
    return {"metric": "abstained", "b_expert_only": b, "c_nonexpert_only": c,
            "discordant_total": b + c, "p_exact_mcnemar": p}


def paired_mcnemar_invalid(expert_rows: list, nonexpert_rows: list) -> dict:
    """McNemar on the INVALID/format-failure indicator, kept separate from
    the abstention McNemar above specifically so the instruction's
    'higher abstention / lower non_abstention_rate, not just more format errors'
    question can be answered by comparing the two p-values side by side
    rather than reading one aggregate number."""
    b = 0
    c = 0
    for e, n in zip(expert_rows, nonexpert_rows):
        e_inv = e["outcome"] == "invalid"
        n_inv = n["outcome"] == "invalid"
        if e_inv and not n_inv:
            b += 1
        elif n_inv and not e_inv:
            c += 1
    p = mcnemar_exact(b, c)
    return {"metric": "invalid", "b_expert_only": b, "c_nonexpert_only": c,
            "discordant_total": b + c, "p_exact_mcnemar": p}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def atomic_write_json(path: str, obj) -> None:
    if os.path.exists(path):
        die(f"refusing to overwrite existing output: {path}")
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit_dir", required=True,
                     help="Dir holding expert_{size}.json, "
                          "non_expert_{size}.json, run_meta_{size}.json "
                          "(get_answer_gsm8k_abstention_role_audit.py's "
                          "output dir for ONE model).")
    ap.add_argument("--size", required=True, help="e.g. 8B or 7B")
    ap.add_argument("--out_dir", required=True,
                     help="Where to write "
                          "gsm8k_abstention_role_audit_summary_{size}.json")
    args = ap.parse_args()

    loaded = validate_and_load(args.audit_dir, args.size)
    meta = loaded["meta"]
    rows = loaded["rows"]

    expert_stats = per_role_stats(rows["expert"])
    nonexpert_stats = per_role_stats(rows["non_expert"])

    bootstrap = {
        "first_acc": paired_bootstrap_ci(
            rows["expert"], rows["non_expert"], _metric_first_acc),
        "abstention_rate": paired_bootstrap_ci(
            rows["expert"], rows["non_expert"], _metric_abstention_rate),
        "non_abstention_rate": paired_bootstrap_ci(
            rows["expert"], rows["non_expert"], _metric_non_abstention_rate),
        "invalid_rate": paired_bootstrap_ci(
            rows["expert"], rows["non_expert"], _metric_invalid_rate),
    }

    # "abstained" is the PRIMARY/confirmatory test; "first_acc" and
    # "invalid" are DIAGNOSTIC (ruling out "just more format failures" as
    # the explanation). Raw p-values are reported; Holm-adjusted p-values
    # across all three are also reported (m=3) for a reader who wants every
    # test treated as independently confirmatory -- see the module
    # docstring's MULTIPLE-COMPARISON POLICY.
    mcnemar_order = ["first_acc", "abstained", "invalid"]
    mcnemar = {
        "first_acc": paired_mcnemar_first_acc(
            rows["expert"], rows["non_expert"]),
        "abstained": paired_mcnemar_abstained(
            rows["expert"], rows["non_expert"]),
        "invalid": paired_mcnemar_invalid(
            rows["expert"], rows["non_expert"]),
    }
    raw_pvals = [mcnemar[k]["p_exact_mcnemar"] for k in mcnemar_order]
    adj_pvals = holm(raw_pvals)
    for k, p_adj in zip(mcnemar_order, adj_pvals):
        mcnemar[k]["p_holm_adjusted_m3"] = p_adj
        mcnemar[k]["role_in_analysis"] = (
            "primary/confirmatory" if k == "abstained"
            else "diagnostic/supporting")

    summary = {
        "model": meta["model"],
        "size": args.size,
        "n": N_EXPECT,
        "source_run_meta": {
            "script_version": meta.get("script_version"),
            "git_commit": meta.get("git_commit"),
            "model_dir": meta.get("model_dir"),
            "questions_sha256": meta.get("questions_sha256"),
            "ordered_sample_identity_sha256": meta.get(
                "ordered_sample_identity_sha256"),
            "generation_method": meta.get("generation_method"),
            "max_new_tokens": meta.get("max_new_tokens"),
            "temperature": meta.get("temperature"),
            "batch_size": meta.get("batch_size"),
            "steering_applied": meta.get("steering_applied"),
            "mask_loaded": meta.get("mask_loaded"),
        },
        "per_role": {
            "expert": expert_stats,
            "non_expert": nonexpert_stats,
        },
        "paired_comparison": {
            "pairing": "matched by idx (0..299), same question at each idx "
                       "in both conditions (structurally verified, not "
                       "merely trusted from run_meta digest)",
            "bootstrap_paired_ci95": bootstrap,
            "mcnemar_exact": mcnemar,
            "interpretation_note": (
                "Compare mcnemar_exact.abstained's p-value against "
                "mcnemar_exact.invalid's p-value side by side: a low "
                "p-value on 'abstained' with a non-significant or "
                "opposite-signed 'invalid' result supports 'non_expert "
                "abstains more' specifically, as opposed to 'non_expert "
                "merely produces more format failures'. Do not read "
                "abstention_rate alone without this cross-check."),
        },
        "no_steering_verified": {
            "run_meta.steering_applied": meta.get("steering_applied"),
            "run_meta.mask_loaded": meta.get("mask_loaded"),
            "all_rows_steering_applied_false": True,
        },
    }

    out_path = os.path.join(
        args.out_dir, f"gsm8k_abstention_role_audit_summary_{args.size}.json")
    atomic_write_json(out_path, summary)

    print(f"wrote {out_path}")
    print()
    print(f"=== model={meta['model']} size={args.size} n={N_EXPECT} ===")
    for role in CONDITIONS:
        s = summary["per_role"][role]
        cond_acc_str = (f"{s['conditional_acc']:.4f}"
                        if s["conditional_acc"] is not None else "N/A")
        print(f"  {role:12s}  first_acc={s['first_acc']:.4f}  "
              f"last_acc={s['last_acc']:.4f}  "
              f"abstention_rate={s['abstention_rate']:.4f}  "
              f"non_abstention_rate={s['non_abstention_rate']:.4f}  "
              f"conditional_acc={cond_acc_str} (n={s['n_conditional']})  "
              f"invalid_rate={s['invalid_rate']:.4f}")
    print()
    print("  paired deltas (expert - non_expert):")
    for metric_name, b in bootstrap.items():
        lo, hi = b["bootstrap_ci95"]
        print(f"    {metric_name:18s} delta={b['delta_expert_minus_nonexpert']:+.4f} "
              f"CI95=[{lo:+.4f},{hi:+.4f}]")
    print("  exact McNemar (raw p / Holm-adjusted m=3 / role):")
    for metric_name in mcnemar_order:
        m = mcnemar[metric_name]
        print(f"    {metric_name:10s} b={m['b_expert_only']:3d} "
              f"c={m['c_nonexpert_only']:3d} p={m['p_exact_mcnemar']:.4g} "
              f"p_holm={m['p_holm_adjusted_m3']:.4g} "
              f"[{m['role_in_analysis']}]")


if __name__ == "__main__":
    main()
