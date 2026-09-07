#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinQA scoring, protocol `finqa-v0`. The ONLY script that reads gold.

Two modes, chosen by which generation files are supplied:

  PREFLIGHT (--preflight): alpha=0 cells only, restricted to the frozen
      30-item subset. Prints a per-item table plus accuracy/no-answer/
      loop/truncation. No statistics, no Holm -- this is a manual-inspection
      step, not the primary test.

  FORMAL: one alpha=0 cell plus up to 3 steered cells per model, all 300
      items. Per model: paired exact McNemar of each non-zero alpha vs that
      model's own alpha=0, Holm m=3 across the model's three steered doses.
      A dose only counts as an "effective workpoint" if it is a SIGNIFICANT
      IMPROVEMENT (accuracy higher, p_adj < .05) over alpha=0 -- a
      significant decrease or a non-significant change is not a workpoint.

MAIN metric = first-legal-marker numeric accuracy (`first_acc`), using the
same `is_correct` tolerance as the loader's numeric normalization. LAST is a
tail-pollution sensitivity readout, never the headline.

SCORING IS A CUSTOM DIRECT-NUMERIC-ANSWER EVALUATOR, NOT the official FinQA
program/DSL execution-based scorer. The model is never asked to produce a
program (`program_re`), and correctness is decided by parsing a '#### <value>'
marker and comparing its normalized value to gold within a small relative
tolerance -- not by executing an operation sequence. Every result this script
writes carries this disclosure verbatim so numbers are not mistaken for
official FinQA leaderboard accuracy.

CROSS-ALPHA CONSISTENCY (formal mode). Every cell of one model must share:
sample_id coverage (0..299), prompt_content_sha256_16 (the loader's full
question+table+report-text digest -- catches drift the older question-only
digest would miss), mask_sha256_16, layer band, max_new_tokens, batch_size,
and steering_fires == L*300 (0 at alpha=0). The alpha set itself must match
the frozen per-model set EXACTLY (REQUIRED_ALPHAS) -- a missing or extra dose
is refused rather than silently computing Holm at a different m.

@author: paveenhuang
"""

import argparse
import json
import os
import sys
from math import comb

sys.path.insert(0, os.path.dirname(__file__))
from finqa_scoring import first_legal_answer, last_legal_answer, is_correct  # noqa: E402

PROTOCOL = "finqa-v0"
N_FORMAL = 300
HOLM_M = 3
# Frozen alpha sets per model (task-specific exploration on FinQA -- swept
# fresh here, not read from the GSM8K record). Exactly these four cells (one
# alpha=0 baseline + HOLM_M=3 steered doses) are required per model; a
# missing or extra alpha is refused rather than silently producing a smaller
# or larger Holm family under the m=3 label.
REQUIRED_ALPHAS = {
    "llama3": {-6, -4, 0, 4},
    "qwen2.5": {-6, 0, 6, 8},
}


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def mcnemar_exact(a, b):
    """a, b: 0/1 correctness lists, same order. Returns (n01, n10, p)."""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n01, n10, min(1.0, 2 * tail)


def holm(pvals_by_key, m=HOLM_M):
    """Holm-Bonferroni with a FIXED family size m (default HOLM_M=3), not
    len(pvals_by_key) -- callers must pass exactly m p-values; a smaller or
    larger dict would silently compute a different-m adjustment under the
    m=3 label."""
    items = sorted(pvals_by_key.items(), key=lambda kv: kv[1])
    if len(items) != m:
        die(f"holm(): got {len(items)} p-values, expected exactly m={m}")
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = adj
    return out


def load_cell(path):
    d = json.load(open(path, encoding="utf-8"))
    if d["meta"].get("protocol") != PROTOCOL:
        die(f"{path}: protocol {d['meta'].get('protocol')!r} != {PROTOCOL!r}")
    if d["meta"].get("accuracy_computed") is not False:
        die(f"{path}: claims accuracy already computed")
    return d["meta"], {r["sample_id"]: r for r in d["data"]}


def score_row(row, gold_val):
    _, pred_first = first_legal_answer(row["generated"])
    _, pred_last = last_legal_answer(row["generated"])
    return {
        "first_correct": int(is_correct(pred_first, gold_val)),
        "last_correct": int(is_correct(pred_last, gold_val)),
        "no_answer": row["no_answer"],
        "multi_answer": row["multi_answer"],
        "loop": row["loop"],
        "truncated": row["truncated"],
        "gen_chars": row["gen_chars"],
    }


def run_preflight(gen_paths, gold_by_id, preflight_ids):
    print(f"\n=== FINQA PREFLIGHT (n={len(preflight_ids)}) ===")
    for path in gen_paths:
        meta, rows = load_cell(path)
        if meta["alpha"] != 0:
            die(f"{path}: preflight only accepts alpha=0 cells, got {meta['alpha']}")
        missing = [i for i in preflight_ids if i not in rows]
        if missing:
            die(f"{path}: missing sample_ids {missing[:5]}... "
                f"(cell was not generated with --indices matching the preflight set)")
        print(f"\n--- model={meta['model']}  alpha=0  "
              f"n_samples_in_cell={meta['n_samples']}  "
              f"max_new_tokens={meta['max_new_tokens']}  "
              f"steering_fires={meta['steering_fires']} (expect 0) ---")
        print(f"{'sid':>4} {'ok':>3} {'no_ans':>7} {'multi':>6} {'loop':>5} "
              f"{'trunc':>6} {'chars':>6}  pred / gold")
        n_correct = n_noans = n_multi = n_loop = n_trunc = 0
        for i in preflight_ids:
            row = rows[i]
            gold_val = gold_by_id[i]
            sc = score_row(row, gold_val)
            _, pred = first_legal_answer(row["generated"])
            n_correct += sc["first_correct"]
            n_noans += sc["no_answer"]
            n_multi += sc["multi_answer"]
            n_loop += sc["loop"]
            n_trunc += sc["truncated"]
            print(f"{i:4d} {'Y' if sc['first_correct'] else '.':>3} "
                  f"{'Y' if sc['no_answer'] else '.':>7} "
                  f"{'Y' if sc['multi_answer'] else '.':>6} "
                  f"{'Y' if sc['loop'] else '.':>5} "
                  f"{'Y' if sc['truncated'] else '.':>6} "
                  f"{sc['gen_chars']:6d}  {pred!r} / {gold_val!r}")
        n = len(preflight_ids)
        print(f"\n  accuracy={n_correct/n:.3f}  no_answer_rate={n_noans/n:.3f}  "
              f"multi_answer_rate={n_multi/n:.3f}  loop_rate={n_loop/n:.3f}  "
              f"truncated_rate={n_trunc/n:.3f}")


def run_formal(gen_paths, gold_by_id, gold_meta):
    cells, cmeta = {}, {}
    for path in gen_paths:
        meta, rows = load_cell(path)
        if len(rows) != N_FORMAL or sorted(rows) != list(range(N_FORMAL)):
            die(f"{path}: expected exactly sample_ids 0..{N_FORMAL-1}")
        mdl, al = meta["model"], meta["alpha"]
        if mdl not in REQUIRED_ALPHAS:
            die(f"{path}: unknown model {mdl!r}")
        if al not in REQUIRED_ALPHAS[mdl]:
            die(f"{path}: alpha {al} is not in {mdl}'s frozen set "
                f"{sorted(REQUIRED_ALPHAS[mdl])}; this protocol does not "
                "search doses beyond the four frozen per model")
        if meta.get("prompt_content_sha256_16") != gold_meta.get("prompt_content_sha256_16"):
            die(f"{path}: prompt_content_sha256_16 {meta.get('prompt_content_sha256_16')!r} "
                f"!= gold file's {gold_meta.get('prompt_content_sha256_16')!r}; "
                "this cell was generated from a different sample/table/report text")
        n_layers = meta["L"]
        expect_fires = 0 if al == 0 else n_layers * N_FORMAL
        if meta["steering_fires"] != expect_fires:
            die(f"{path}: steering_fires {meta['steering_fires']} != {expect_fires}")
        if al in cells.get(mdl, {}):
            die(f"{mdl} alpha={al} supplied twice")
        cells.setdefault(mdl, {})[al] = rows
        cmeta.setdefault(mdl, {})[al] = meta

    # cross-alpha consistency WITHIN each model: band, mask, batch size and
    # token budget must all agree, or a dose difference could be confounded
    # with a configuration difference rather than alpha itself.
    for mdl, metas in cmeta.items():
        ref_al = sorted(metas)[0]
        ref = metas[ref_al]
        for al, m in metas.items():
            for field in ("layer_start", "layer_end", "L", "mask_sha256_16",
                          "max_new_tokens", "batch_size", "size"):
                if m.get(field) != ref.get(field):
                    die(f"{mdl} alpha={al}: {field}={m.get(field)!r} differs "
                        f"from alpha={ref_al}'s {ref.get(field)!r}; cells of "
                        "one model must share band/mask/budget/batch_size")
        missing = REQUIRED_ALPHAS[mdl] - set(metas)
        extra = set(metas) - REQUIRED_ALPHAS[mdl]
        if missing or extra:
            die(f"{mdl}: alpha set {sorted(metas)} does not match the "
                f"required {sorted(REQUIRED_ALPHAS[mdl])} "
                f"(missing={sorted(missing)}, extra={sorted(extra)}); "
                "Holm m=3 requires exactly the frozen four cells")

    results = {}
    for mdl, byalpha in sorted(cells.items()):
        scored = {al: [score_row(byalpha[al][i], gold_by_id[i]) for i in range(N_FORMAL)]
                   for al in byalpha}
        acc0 = [s["first_correct"] for s in scored[0]]
        r = {"alphas": sorted(byalpha), "per_alpha": {}}
        for al in sorted(byalpha):
            s = scored[al]
            n = N_FORMAL
            r["per_alpha"][al] = {
                "first_acc": sum(x["first_correct"] for x in s) / n,
                "last_acc": sum(x["last_correct"] for x in s) / n,
                "no_answer_rate": sum(x["no_answer"] for x in s) / n,
                "multi_answer_rate": sum(x["multi_answer"] for x in s) / n,
                "loop_rate": sum(x["loop"] for x in s) / n,
                "truncated_rate": sum(x["truncated"] for x in s) / n,
                "gen_chars_med": sorted(x["gen_chars"] for x in s)[n // 2],
            }
        steered = [al for al in byalpha if al != 0]
        if len(steered) != HOLM_M:
            die(f"{mdl}: {len(steered)} non-zero alphas supplied; Holm family is m=3")
        pvals, deltas = {}, {}
        for al in steered:
            accA = [x["first_correct"] for x in scored[al]]
            n01, n10, p = mcnemar_exact(acc0, accA)
            pvals[al] = p
            deltas[al] = {
                "dAcc_pp": (sum(accA) - sum(acc0)) / N_FORMAL * 100,
                "discordant_0to1": n01, "discordant_1to0": n10, "p_raw": p,
            }
        adj = holm(pvals)
        r["holm_family_m"] = HOLM_M
        r["transfer"] = {}
        for al in steered:
            improved = deltas[al]["dAcc_pp"] > 0
            sig = adj.get(al, 1.0) < 0.05
            r["transfer"][al] = {
                **deltas[al], "p_adj": adj.get(al, 1.0),
                "effective_workpoint": bool(improved and sig),
            }
        results[mdl] = r

    print(f"\n=== FINQA FORMAL SWEEP (n={N_FORMAL} per cell) ===")
    print("    [scoring: custom direct-numeric-answer evaluator, NOT the "
          "official FinQA program/DSL scorer]")
    for mdl, r in sorted(results.items()):
        print(f"\n--- {mdl} ---")
        print(f"{'alpha':>6} {'first_acc':>10} {'last_acc':>9} {'no_ans':>7} "
              f"{'multi':>6} {'loop':>6} {'trunc':>6} {'chars':>6}")
        for al in r["alphas"]:
            p = r["per_alpha"][al]
            print(f"{al:6d} {p['first_acc']:10.4f} {p['last_acc']:9.4f} "
                  f"{p['no_answer_rate']:7.3f} {p['multi_answer_rate']:6.3f} "
                  f"{p['loop_rate']:6.3f} {p['truncated_rate']:6.3f} "
                  f"{p['gen_chars_med']:6d}")
        if r["transfer"]:
            print(f"\n  Holm m={r['holm_family_m']} (per-model family)")
            print(f"  {'alpha':>6} {'dAcc_pp':>9} {'0to1':>5} {'1to0':>5} "
                  f"{'p_raw':>9} {'p_adj':>9}  workpoint?")
            for al, t in sorted(r["transfer"].items()):
                print(f"  {al:6d} {t['dAcc_pp']:+9.2f} {t['discordant_0to1']:5d} "
                      f"{t['discordant_1to0']:5d} {t['p_raw']:9.4f} "
                      f"{t['p_adj']:9.4f}  "
                      f"{'YES' if t['effective_workpoint'] else 'no'}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", nargs="+", required=True)
    ap.add_argument("--gold_file", required=True,
                     help="finqa_formal.json from data_finqa.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()

    if os.path.exists(a.out):
        die(f"{a.out} exists; refusing to overwrite")

    gblob = json.load(open(a.gold_file, encoding="utf-8"))
    gold_by_id = {d["sample_id"]: d["gold_value"] for d in gblob["data"]}

    if a.preflight:
        preflight_ids = gblob["meta"]["preflight_indices"]
        run_preflight(a.generations, gold_by_id, preflight_ids)
        json.dump({"protocol": PROTOCOL, "mode": "preflight",
                    "preflight_indices": preflight_ids,
                    "note": ("no statistics computed in preflight mode; "
                              "scoring is a CUSTOM direct-numeric-answer "
                              "evaluator, NOT the official FinQA program/DSL "
                              "execution-based scorer")},
                   open(a.out, "w", encoding="utf-8"), indent=2)
    else:
        results = run_formal(a.generations, gold_by_id, gblob["meta"])
        json.dump({"protocol": PROTOCOL, "mode": "formal",
                    "n_formal": N_FORMAL, "holm_family_m": HOLM_M,
                    "required_alphas": {k: sorted(v) for k, v in REQUIRED_ALPHAS.items()},
                    "results": results,
                    "note": ("task-specific exploration on FinQA, not a GSM8K "
                              "fixed-workpoint transfer test; alpha swept "
                              "fresh per model, Holm m=3 within model. Scoring "
                              "is a CUSTOM direct-numeric-answer evaluator "
                              "(marker '#### <value>' + numeric normalization "
                              "+ relative-tolerance match), NOT the official "
                              "FinQA program/DSL execution-based scorer -- "
                              "numbers here are not directly comparable to "
                              "official FinQA leaderboard accuracy.")},
                   open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
