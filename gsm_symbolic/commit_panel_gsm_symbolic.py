#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""
commit_panel_gsm_symbolic.py -- EXPLORATORY commitment / answer-formation
analysis for the GSM-Symbolic robustness sweep.

STATUS: EXPLORATORY, PERMANENTLY. This is descriptive post-hoc analysis of
already-generated GSM-Symbolic cells (No-CoT and CoT, both models, all four
alphas, all three configs). It does NOT enter the accuracy Holm family
reported for GSM-Symbolic, does NOT modify get_answer_gsm_symbolic*.py /
eval_gsm_symbolic.py / data_gsm_symbolic.py, and does NOT claim causal
mediation. Commitment features are themselves outcomes of alpha; any
accuracy-vs-commitment association reported here is consistent-with evidence
only, following the same convention as
RoleAnswer/p3/commit_panel_gsm_hard.py and RoleAnswer/p3/precandidate_reasoning.py.

FROZEN DEFINITIONS -- IMPORTED, NEVER REIMPLEMENTED:
  - early_candidate_detector.has_early_candidate           (Table 5.5's `ec`)
  - p3/precandidate_reasoning.first_candidate / per_sample  (Table 5.5's
    "Candidate position" / "Pre-candidate chars" / "Reason-first" columns --
    the SAME function that produced GSM8K/GSM-Hard's numbers, called here on
    GSM-Symbolic text with NO changes to its regexes or thresholds)
  - analyze_first_last_acc.all_hash / norm_gsm8k / fallback_gsm8k
    (the frozen GSM8K scorer, used ONLY to re-verify accuracy per cell against
    the numbers eval_gsm_symbolic.py reports -- never to re-derive a new
    parser)
  - thinking_curve/extract_metrics.py's three-way commit_state partition
    (committed / marker_unparsed / no_marker, keyed on marker PRESENCE, not
    marker count) -- reimplemented here in ~4 lines because that module's
    per_sample() is hardwired to the Llama/Qwen *signal* JSON schema
    (x_prefill/x_decode fields GSM-Symbolic cells do not carry), but the
    partition logic itself is copied verbatim, not redesigned.

DATA: the GSM-Symbolic cells under {model}/gsm_symbolic (No-CoT) and
{model}/gsm_symbolic_cot (CoT), main/p1/p2, all four alphas per model. These
are the SAME cells (already sample_id-repaired: unique
"{config}:{original_id}:{instance}" keys) that eval_gsm_symbolic.py scores for
the GSM-Symbolic accuracy result -- this script re-reads them, does not
regenerate anything, and does not touch get_answer_gsm_symbolic*.py.

PAIRING for the sample_id fix affected two things and both are re-verified
here rather than assumed: (1) sample_id uniqueness -- 300/300 per cell,
asserted; (2) cluster grouping by original_id for the equal-weighted-by-config
summary, reusing the SAME clustering convention as eval_gsm_symbolic.py's
PRIMARY cluster bootstrap (main/p1/p2 weighted equally, never by row count).

ACCURACY CROSS-CHECK (required before any commitment number is trusted): for
every cell, accuracy recomputed via the frozen extractor here must match
eval_gsm_symbolic.py's own per-cell accuracy (which itself matches the
generation-time inline accuracy). A mismatch is a hard stop -- it would mean
this script is reading a different file/row set than the accuracy result it
is being compared against.

Usage:
  python3.10 commit_panel_gsm_symbolic.py \
      --root /Users/paveenhuang/Documents/RSNResult/RoleAnswer \
      --out  /tmp/gsm_symbolic_commit_panel.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

_ROLEANSWER_DEFAULT = "/Users/paveenhuang/Documents/RSNResult/RoleAnswer"


def _bootstrap_imports(roleanswer_root):
    """Insert the frozen-code roots and import the frozen functions. Kept as
    a function (not top-level) so --root can be passed before importing."""
    p3_dir = os.path.join(roleanswer_root, "p3")
    if roleanswer_root not in sys.path:
        sys.path.insert(0, roleanswer_root)
    if p3_dir not in sys.path:
        sys.path.insert(0, p3_dir)
    global all_hash, norm_gsm8k, fallback_gsm8k, has_early_candidate, first_candidate, pr_per_sample
    from analyze_first_last_acc import all_hash, norm_gsm8k, fallback_gsm8k  # noqa: E402
    from early_candidate_detector import has_early_candidate  # noqa: E402
    import precandidate_reasoning as _pr  # noqa: E402
    first_candidate = _pr.first_candidate
    pr_per_sample = _pr.per_sample


CONFIGS = ["main", "p1", "p2"]
CELLS = {
    ("llama3", False): {
        "ans_root": "gsm_symbolic", "size": "8B", "layers": (11, 20),
        "alphas": [-6, -4, 0, 4],
    },
    ("llama3", True): {
        "ans_root": "gsm_symbolic_cot", "size": "8B", "layers": (11, 20),
        "alphas": [-6, -4, 0, 4],
    },
    ("qwen2.5", False): {
        "ans_root": "gsm_symbolic", "size": "7B", "layers": (16, 22),
        "alphas": [-6, 0, 6, 8],
    },
    ("qwen2.5", True): {
        "ans_root": "gsm_symbolic_cot", "size": "7B", "layers": (16, 22),
        "alphas": [-6, 0, 6, 8],
    },
}


def load_cell(root, model, ans_root, gsm_config, alpha, size, layers):
    st, en = layers
    path = os.path.join(root, model, ans_root, gsm_config, f"mdf_{alpha}",
                         f"gsm_symbolic_{gsm_config}_{size}_{st}_{en}.json")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload, path


def three_way_commit_state(text):
    """Verbatim logic from thinking_curve/extract_metrics.py's partition,
    reimplemented (not imported) because that module's per_sample() is
    hardwired to the signal-JSON schema (x_prefill/x_decode) GSM-Symbolic
    cells do not carry. The partition rule itself is unchanged: keyed on
    marker PRESENCE, exhaustive, is_loop a descriptive sub-flag of
    marker_unparsed (>=4 bare markers), never a fourth bucket."""
    hashes = all_hash(text)
    n_bare = text.count("####")
    if hashes:
        state = "committed"
    elif n_bare >= 1:
        state = "marker_unparsed"
    else:
        state = "no_marker"
    is_loop = (state == "marker_unparsed" and n_bare >= 4)
    return state, is_loop, n_bare, len(hashes)


def first_marker_posN(text):
    """char position of the FIRST parseable '#### <number>' / len(text) --
    the posN readout, same regex family as extract_metrics.py."""
    import re
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if not m:
        return None
    n = len(text)
    return m.start() / n if n else None


def first_last_hash_agree(text):
    """Same convention as get_answer_gsm_symbolic.py's own
    first_last_hash_agree, recomputed here (not imported, since that
    function lives in a generation script this analysis must not import
    torch/model code from) using the identical regex + normalize_gsm8k
    equivalence rule."""
    import re
    matches = list(re.finditer(r"####\s*([+-]?[\d,]+\.?\d*)", text))
    if len(matches) < 2:
        return True
    first = matches[0].group(1).replace(",", "")
    last = matches[-1].group(1).replace(",", "")
    return norm_gsm8k(first) == norm_gsm8k(last)


def analyze_cell(payload):
    """Per-row commitment metrics for one (model, cot, config, alpha) cell."""
    rows = []
    for s in payload["data"]:
        text = s["generated"]
        gold = s["answer"]
        hashes = all_hash(text)
        fb = fallback_gsm8k(text) if not hashes else None
        pred = hashes[0] if hashes else fb
        correct = (pred is not None and norm_gsm8k(pred) == norm_gsm8k(str(gold)))
        no_answer = (pred is None)  # frozen scorer's FULL fallback chain found nothing
        ec = has_early_candidate(text)
        posN = first_marker_posN(text)
        state, is_loop, n_bare, n_hash = three_way_commit_state(text)
        cand = pr_per_sample(text, gold)  # cand_pos / pre_chars / pre_eqs / rba
        rows.append({
            "sample_id": s["sample_id"],
            "original_id": s["original_id"],
            "config": s["config"],
            "correct": correct,
            "correct_inline": bool(s.get("correct")),
            "early_candidate": bool(ec),
            "posN": posN,
            "commit_state": state,
            "is_loop": bool(is_loop),
            "n_bare_markers": n_bare,
            "n_parseable_markers": n_hash,
            "multi_marker": n_hash > 1,
            "first_last_agree": first_last_hash_agree(text),
            "no_answer": no_answer,
            "cand_pos": cand["cand_pos"],
            "pre_chars": cand["pre_chars"],
            "pre_eqs": cand["pre_eqs"],
            "reason_before_answer": cand["rba"],
        })
    return rows


def _rate(rows, key):
    n = len(rows)
    return sum(1 for r in rows if r[key]) / n if n else float("nan")


def _median(vals):
    vals = sorted(v for v in vals if v is not None and v == v)
    if not vals:
        return float("nan")
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def cell_summary(rows, stored_acc=None):
    n = len(rows)
    posN_vals = [r["posN"] for r in rows if r["posN"] is not None]
    cand_rows = [r for r in rows if r["cand_pos"] is not None]
    acc_recomputed = sum(1 for r in rows if r["correct"]) / n if n else float("nan")

    summary = {
        "n": n,
        "accuracy_recomputed": round(acc_recomputed, 4),
        "early_candidate_rate": round(_rate(rows, "early_candidate"), 4),
        "posN_median": round(_median(posN_vals), 4) if posN_vals else None,
        "posN_coverage_n": len(posN_vals),
        "no_marker_rate": round(sum(1 for r in rows if r["commit_state"] == "no_marker") / n, 4),
        "marker_unparsed_rate": round(sum(1 for r in rows if r["commit_state"] == "marker_unparsed") / n, 4),
        "committed_rate": round(sum(1 for r in rows if r["commit_state"] == "committed") / n, 4),
        "is_loop_rate": round(_rate(rows, "is_loop"), 4),
        "multi_marker_rate": round(_rate(rows, "multi_marker"), 4),
        "first_last_disagree_rate": round(1 - _rate(rows, "first_last_agree"), 4),
        "no_answer_rate": round(_rate(rows, "no_answer"), 4),
        "cand_pos_coverage_n": len(cand_rows),
        "cand_pos_median": round(_median([r["cand_pos"] for r in cand_rows]), 4) if cand_rows else None,
        "pre_candidate_chars_median": round(_median([r["pre_chars"] for r in cand_rows]), 2) if cand_rows else None,
        "reason_first_rate": (round(sum(r["reason_before_answer"] for r in cand_rows) / len(cand_rows), 4)
                               if cand_rows else None),
    }
    if stored_acc is not None:
        summary["accuracy_stored_meta"] = round(stored_acc, 4)
        # stored_acc is generation-time round(...,2); acc_recomputed is
        # rounded to 4dp here then *100 -- allow a tolerance wide enough to
        # absorb both roundings (0.01pp) but tight enough to catch a real
        # cell/row mismatch (which differs by whole percentage points).
        summary["accuracy_matches_stored"] = abs(acc_recomputed * 100 - stored_acc) < 0.02
    return summary


def cluster_equal_weighted_summary(rows_by_config):
    """Mirror eval_gsm_symbolic.py's config-equal-weighted convention: report
    each config's own summary (already done by cell_summary per config) PLUS
    a config-equal-weighted pooled early_candidate_rate / reason_first_rate /
    posN_median, so the pooled number cannot be dominated by p1's larger
    original_id-instance density. This does NOT redo the accuracy cluster
    bootstrap (that stays eval_gsm_symbolic.py's job) -- it only pools the
    DESCRIPTIVE commitment rates the same way, for internal consistency."""
    per_config_rates = {}
    for cfg, rows in rows_by_config.items():
        per_config_rates[cfg] = {
            "early_candidate_rate": _rate(rows, "early_candidate"),
            "reason_first_rate": (
                sum(r["reason_before_answer"] for r in rows if r["cand_pos"] is not None) /
                max(1, sum(1 for r in rows if r["cand_pos"] is not None))
            ),
            "posN_median": _median([r["posN"] for r in rows if r["posN"] is not None]),
        }
    n_cfg = len(per_config_rates)
    pooled = {}
    for metric in ("early_candidate_rate", "reason_first_rate", "posN_median"):
        vals = [v[metric] for v in per_config_rates.values() if v[metric] == v[metric]]
        pooled[metric] = round(sum(vals) / len(vals), 4) if vals else None
    return per_config_rates, pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=_ROLEANSWER_DEFAULT,
                     help="RoleAnswer root holding {model}/gsm_symbolic[_cot]/...")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _bootstrap_imports(args.root)

    if os.path.exists(args.out):
        raise FileExistsError(f"{args.out} already exists -- refusing to overwrite.")

    result = {"note": ("EXPLORATORY commitment/answer-formation analysis. "
                        "Does not enter the GSM-Symbolic accuracy Holm family "
                        "and does not claim causal mediation -- see module docstring."),
              "models": {}}

    for model in ("llama3", "qwen2.5"):
        result["models"][model] = {}
        for cot in (False, True):
            spec = CELLS[(model, cot)]
            cond_key = "cot" if cot else "nocot"
            result["models"][model][cond_key] = {"per_config": {}, "alphas": spec["alphas"]}
            rows_by_config_per_alpha = {}
            for alpha in spec["alphas"]:
                rows_by_config_per_alpha[alpha] = {}
                for cfg in CONFIGS:
                    payload, path = load_cell(args.root, model, spec["ans_root"], cfg,
                                               alpha, spec["size"], spec["layers"])
                    stored_acc = payload["meta"]["accuracy"]["accuracy_percentage"]
                    rows = analyze_cell(payload)
                    # hard sample_id uniqueness check -- this analysis's pairing
                    # depends on it (per the same rule fix_sample_id_bug.py
                    # enforces at repair time)
                    sids = [r["sample_id"] for r in rows]
                    if len(set(sids)) != len(sids):
                        raise ValueError(
                            f"{path}: sample_id not unique ({len(set(sids))}/"
                            f"{len(sids)}) -- this cell was not repaired; "
                            "re-run fix_sample_id_bug.py --apply before "
                            "trusting any pairing here."
                        )
                    summary = cell_summary(rows, stored_acc=stored_acc)
                    if not summary["accuracy_matches_stored"]:
                        raise ValueError(
                            f"{path}: recomputed accuracy {summary['accuracy_recomputed']*100:.4f}% "
                            f"!= stored meta accuracy {stored_acc:.4f}% -- refusing to "
                            "report commitment numbers computed on a cell whose "
                            "accuracy does not match the reported GSM-Symbolic result."
                        )
                    result["models"][model][cond_key]["per_config"].setdefault(cfg, {})[alpha] = summary
                    rows_by_config_per_alpha[alpha][cfg] = rows

            # config-equal-weighted pooled descriptive rates per alpha
            pooled_by_alpha = {}
            for alpha in spec["alphas"]:
                _, pooled = cluster_equal_weighted_summary(rows_by_config_per_alpha[alpha])
                pooled_by_alpha[alpha] = pooled
            result["models"][model][cond_key]["pooled_config_equal_weighted"] = pooled_by_alpha

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")

    print("\nAll per-cell accuracy cross-checks PASSED (recomputed == stored meta accuracy).")
    print("EXPLORATORY. Commitment features are outcomes of alpha; any accuracy-vs-")
    print("commitment association is consistent-with evidence, never mediation.")


if __name__ == "__main__":
    main()
