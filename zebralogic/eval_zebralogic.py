#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZebraLogic-Easy scoring. The ONLY script that reads gold. Protocol
zebralogic-easy-v0 (docs/PREREG_ZEBRALOGIC_EASY.md).

THREE MODES:

  --canary_check      Compare Puzzle Accuracy / Cell Accuracy / parse status /
                       truncation status across two or more canary generation
                       files (different --device_tag) run on different GPUs.
                       Reads private gold for just the small canary subset.
                       Frozen agreement criterion (prereg section 4): text
                       need not match verbatim, but those four summaries must
                       show NO SYSTEMATIC divergence. This does not compute
                       McNemar/Holm; it is a go/no-go check that must PASS
                       before any cross-GPU cell is pooled into one paired
                       alpha-curve.

  --preflight_check    Score the 5-item alpha=0 preflight cell against private
                       gold and print format/parse/truncation diagnostics
                       plus generation-length percentiles. Read-only sanity;
                       computes no formal statistics (n=5 is far too few).

  --formal (default,   Score the full 280-item, four-alpha sweep for ONE
   via --generations)   model: Puzzle Accuracy, Cell Accuracy, per-item
                       commitment metrics (incl. the two gold-dependent
                       revision metrics), paired McNemar of each non-zero
                       alpha against that model's own alpha=0, Holm m=3
                       within that model, paired bootstrap CIs, per-size
                       breakdowns, and the discrete-argmax / observed
                       near-optimal-region analysis per the frozen workpoint
                       rule (prereg section 8).

alpha is NEVER selected or searched here -- this script only scores whatever
cells are handed to it against the model's frozen four-point set
(zebralogic/get_answer_zebralogic.py's FROZEN_ALPHAS), and refuses anything
else.

Gold is loaded ONLY through zebralogic.data_zebralogic.load_private_gold(),
which hard-stops with an actionable message (never falls back to a
substitute) if allenai/ZebraLogicBench-private is not accessible. This
script inherits that behavior unchanged -- it does not add its own fallback.

@author: paveenhuang
"""

import argparse
import json
import os
import random
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from official_zebra_grid_scorer import build_solution_table, score_one_item  # noqa: E402
from commitment_metrics import compute_commitment_metrics, first_final_grid_agreement  # noqa: E402
from data_zebralogic import (  # noqa: E402
    load_private_gold, solution_shape, sha16, EXPECTED_EASY_IDS_SHA256,
)


def _expected_shapes(rows):
    """Build {id: solution_shape} from a generation cell's own
    `solution_shape` field (written label-free by get_answer_zebralogic.py
    from the public split), for load_private_gold()'s public/private
    shape-agreement check."""
    return {r["id"]: r["solution_shape"] for r in rows if "solution_shape" in r}

PROTOCOL = "zebralogic-easy-v0"
N_EASY = 280
B_BOOT, SEED = 10000, 0
FROZEN_ALPHAS = {
    "llama3": (-6, -4, 0, 4),
    "qwen2.5": (-6, 0, 6, 8),
}

# meta fields that MUST be identical across every alpha cell of one model's
# formal sweep: a config drift here (different mask, different token budget
# from the S3 escalation rule applied inconsistently, different batch size)
# would silently mix incomparable cells into one "dose curve", and nothing
# downstream (McNemar/Holm/bootstrap consume only 0/1 solved vectors) could
# tell. prompt_sha256 alone (the pre-existing check) does not catch a mask
# swap or a token-budget mismatch -- those never enter the rendered prompt.
CONSISTENCY_FIELDS = (
    "model", "size", "layer_start", "layer_end", "L",
    "mask_path", "mask_sha256", "max_new_tokens", "temperature", "top_p",
    "batch_size", "prefill_tail_len", "chat_template",
)


def die(m):
    print(f"[FATAL] {m}", file=sys.stderr)
    raise SystemExit(2)


def mcnemar_exact(a, b):
    """Exact two-sided paired McNemar on two equal-length 0/1 sequences,
    matching this repo's eval_bbh_numeric.py / eval_p3.py convention exactly
    (exact binomial tail on discordant pairs, not the chi-square
    approximation)."""
    b01 = sum(1 for x, y in zip(a, b) if not x and y)
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b01, b10, min(1.0, 2 * tail)


def boot_ci_puzzle(a, b, B=B_BOOT, seed=SEED):
    """Item(puzzle)-level paired bootstrap on the per-item Puzzle Accuracy
    difference, in pp. Matches eval_bbh_numeric.py's boot_ci exactly."""
    rng = random.Random(seed)
    n = len(a)
    d = [(y - x) * 100.0 for x, y in zip(a, b)]
    out = []
    for _ in range(B):
        out.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    out.sort()
    return out[int(.025 * B)], out[int(.975 * B)]


def boot_ci_cell(cell_rates_a, cell_rates_b, B=B_BOOT, seed=SEED):
    """PUZZLE-level paired bootstrap on Cell Accuracy. Resamples PUZZLES (not
    individual cells): a puzzle's cells are not independent draws of the same
    phenomenon (they are logically entangled by the puzzle's clue set), so a
    naive cell-level bootstrap would understate variance. `cell_rates_a/b`
    are per-puzzle (correct_cells/total_cells) ratios, one pair per puzzle;
    the resampled statistic is the mean of per-puzzle rate differences, in
    pp."""
    rng = random.Random(seed)
    n = len(cell_rates_a)
    d = [(y - x) * 100.0 for x, y in zip(cell_rates_a, cell_rates_b)]
    out = []
    for _ in range(B):
        out.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    out.sort()
    return out[int(.025 * B)], out[int(.975 * B)]


def holm(pairs):
    """pairs: list of (key, p_raw). Returns {key: p_adj}."""
    s = sorted(pairs, key=lambda t: t[1])
    m = len(s)
    out = {}
    run = 0.0
    for i, (k, p) in enumerate(s):
        adj = min(1.0, max(run, (m - i) * p))
        run = adj
        out[k] = adj
    return out


def med(xs):
    xs = sorted(xs)
    return None if not xs else xs[len(xs) // 2]


def pctl(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


def load_cell(path):
    d = json.load(open(path, encoding="utf-8"))
    m = d["meta"]
    if m.get("protocol") != PROTOCOL:
        die(f"{path}: protocol {m.get('protocol')!r} != {PROTOCOL!r}")
    if m.get("accuracy_computed") is not False:
        die(f"{path}: generation file claims accuracy was already computed")
    return m, d["data"]


def score_rows(rows, gold_by_id):
    """Returns a list of per-row dicts: puzzle-level solved/correct_cells/
    total_cells plus the gold-free commitment metrics plus (since gold is
    available here) the two revision metrics."""
    out = []
    for r in rows:
        gold_solution = gold_by_id[r["id"]]
        table = build_solution_table(gold_solution)
        sc = score_one_item(r["generated"], table)
        cm = compute_commitment_metrics(r["generated"])

        last_full = sc["prediction_table"] if sc["parsed"] else None
        agreement = first_final_grid_agreement(cm["_first_solution_grid"], last_full)

        rev_wrong_to_right = rev_right_to_wrong = None
        first_grid = cm["_first_solution_grid"]
        if isinstance(first_grid, dict) and last_full is not None:
            rev_wrong_to_right = 0
            rev_right_to_wrong = 0
            for house in table:
                for col in table[house]:
                    truth = table[house][col].lower().strip()

                    def _val(grid):
                        cell = grid.get(house, {}).get(col) if isinstance(grid.get(house), dict) else None
                        if cell is None:
                            return None
                        if isinstance(cell, list):
                            return str(cell[0]).lower().strip() if cell else None
                        return str(cell).lower().strip()

                    fv, lv = _val(first_grid), _val(last_full)
                    if fv is None or lv is None or fv == lv:
                        continue
                    f_correct = (fv == truth)
                    l_correct = (lv == truth)
                    if not f_correct and l_correct:
                        rev_wrong_to_right += 1
                    elif f_correct and not l_correct:
                        rev_right_to_wrong += 1

        out.append({
            "id": r["id"], "sample_id": r["sample_id"], "size": r["size"],
            "parsed": sc["parsed"], "solved": sc["solved"],
            "correct_cells": sc["correct_cells"], "total_cells": sc["total_cells"],
            "cell_rate": (sc["correct_cells"] / sc["total_cells"]) if sc["total_cells"] else None,
            "no_answer": not sc["parsed"],
            "truncated": bool(r.get("truncated")),
            "generated_token_count": r.get("generated_token_count"),
            "is_strict_loop": cm["is_strict_loop"],
            "solution_first_rate": cm["solution_first_rate"],
            "first_solution_pos": cm["first_solution_pos"],
            "pre_solution_chars": cm["pre_solution_chars"],
            "pre_solution_tokens": cm["pre_solution_tokens"],
            "reason_before_solution": cm["reason_before_solution"],
            "first_final_grid_agreement": agreement,
            "revision_wrong_to_right": rev_wrong_to_right,
            "revision_right_to_wrong": rev_right_to_wrong,
        })
    return out


def summarize(scored, sizes_order):
    n = len(scored)
    solved = [1 if x["solved"] else 0 for x in scored]
    cell_rate = [x["cell_rate"] if x["cell_rate"] is not None else 0.0 for x in scored]
    total_correct_cells = sum(x["correct_cells"] for x in scored)
    total_cells = sum(x["total_cells"] for x in scored)

    by_size = {}
    for sz in sizes_order:
        rows = [x for x in scored if x["size"] == sz]
        if not rows:
            continue
        by_size[sz] = {
            "n": len(rows),
            "puzzle_acc": sum(1 for x in rows if x["solved"]) / len(rows),
            "cell_acc": (sum(x["correct_cells"] for x in rows) /
                        sum(x["total_cells"] for x in rows)),
        }

    commit_defined = [x for x in scored if x["first_solution_pos"] is not None]
    n_tok = [x["generated_token_count"] for x in scored if x["generated_token_count"] is not None]

    return {
        "n": n,
        "puzzle_acc": sum(solved) / n,
        "cell_acc": total_correct_cells / total_cells if total_cells else None,
        "no_answer_rate": sum(1 for x in scored if x["no_answer"]) / n,
        "truncated_rate": sum(1 for x in scored if x["truncated"]) / n,
        "strict_loop_rate": sum(1 for x in scored if x["is_strict_loop"]) / n,
        "solution_first_rate": sum(1 for x in scored if x["solution_first_rate"]) / n,
        "reason_before_solution_rate": sum(1 for x in scored if x["reason_before_solution"]) / n,
        "commitment_defined_coverage": len(commit_defined) / n,
        "first_solution_pos_med": med([x["first_solution_pos"] for x in commit_defined]),
        "pre_solution_chars_med": med([x["pre_solution_chars"] for x in commit_defined]),
        "first_final_grid_agreement_med": med(
            [x["first_final_grid_agreement"] for x in scored
             if x["first_final_grid_agreement"] is not None]),
        "revision_wrong_to_right_sum": sum(
            x["revision_wrong_to_right"] for x in scored
            if x["revision_wrong_to_right"] is not None),
        "revision_right_to_wrong_sum": sum(
            x["revision_right_to_wrong"] for x in scored
            if x["revision_right_to_wrong"] is not None),
        "gen_tokens_med": med(n_tok), "gen_tokens_p95": pctl(n_tok, 0.95),
        "gen_tokens_p99": pctl(n_tok, 0.99),
        "by_size": by_size,
        "cell_rate_per_puzzle": cell_rate,  # kept for the cell-accuracy bootstrap
        "solved_per_puzzle": solved,        # kept for McNemar
    }


# ---------------------------------------------------------------- canary ---

def cmd_canary_check(args):
    files = args.canary_files
    if len(files) < 2:
        die("--canary_check needs at least 2 --canary_files (different device_tag)")
    metas, rowsets = [], []
    for p in files:
        m, rows = load_cell(p)
        if m.get("mode") != "canary":
            die(f"{p}: mode {m.get('mode')!r} != 'canary'")
        if m.get("alpha") != 0:
            die(f"{p}: canary check is defined at alpha=0 only, got {m.get('alpha')}")
        metas.append(m)
        rowsets.append(rows)

    ids = [r["id"] for r in rowsets[0]]
    for p, rows in zip(files[1:], rowsets[1:]):
        if [r["id"] for r in rows] != ids:
            die(f"{p}: canary item set/order differs from {files[0]}")

    gold = load_private_gold(ids, expected_shapes=_expected_shapes(rowsets[0]))

    summaries = []
    for m, rows in zip(metas, rowsets):
        scored = score_rows(rows, gold)
        summaries.append({
            "device_tag": m.get("device_tag"), "gpu_name": m.get("gpu_name"),
            "cuda_visible_devices": m.get("cuda_visible_devices"),
            "n": len(scored),
            "puzzle_acc": sum(1 for x in scored if x["solved"]) / len(scored),
            "cell_acc": (sum(x["correct_cells"] for x in scored) /
                        sum(x["total_cells"] for x in scored)),
            "no_answer_rate": sum(1 for x in scored if x["no_answer"]) / len(scored),
            "truncated_rate": sum(1 for x in scored if x["truncated"]) / len(scored),
            "per_item": [{"id": x["id"], "solved": x["solved"],
                          "no_answer": x["no_answer"], "truncated": x["truncated"]}
                        for x in scored],
        })

    print(f"\n=== CANARY CROSS-GPU CHECK  n_items={len(ids)}  n_devices={len(summaries)} ===")
    print(f"{'device_tag':15s} {'gpu':25s} {'puzzle_acc':>10} {'cell_acc':>10} "
          f"{'no_answer':>10} {'truncated':>10}")
    for s in summaries:
        print(f"{str(s['device_tag']):15s} {str(s['gpu_name'])[:25]:25s} "
              f"{s['puzzle_acc']:10.4f} {s['cell_acc']:10.4f} "
              f"{s['no_answer_rate']:10.4f} {s['truncated_rate']:10.4f}")

    # Systematic-divergence check: any pair disagreeing on solved/no_answer/
    # truncated status for the SAME item id is a per-item divergence; report
    # the count and let the human judge "systematic" per the frozen wording
    # (this script does not silently threshold it into pass/fail, since the
    # canary is only 8 items and a formal test would be underpowered).
    by_id_per_device = [{x["id"]: x for x in s["per_item"]} for s in summaries]
    diverging = []
    for iid in ids:
        vals = [d[iid] for d in by_id_per_device]
        if len({(v["solved"], v["no_answer"], v["truncated"]) for v in vals}) > 1:
            diverging.append(iid)
    print(f"\nitems with per-device disagreement (solved/no_answer/truncated): "
          f"{len(diverging)}/{len(ids)}  {diverging}")
    print("\nFrozen criterion: text need not match verbatim, but Puzzle Acc, "
          "Cell Acc, parse status and truncation status must show NO "
          "SYSTEMATIC divergence across devices. Judge the summary table and "
          "the disagreement list above against that; this script does not "
          "auto-decide pass/fail for n=8.")

    if args.out:
        json.dump({"protocol": PROTOCOL, "check": "canary_cross_gpu",
                   "item_ids": ids, "summaries": summaries,
                   "diverging_item_ids": diverging},
                  open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")


# ------------------------------------------------------------- preflight ---

def cmd_preflight_check(args):
    m, rows = load_cell(args.preflight_file)
    if m.get("mode") != "preflight":
        die(f"{args.preflight_file}: mode {m.get('mode')!r} != 'preflight'")
    if m.get("alpha") != 0:
        die(f"{args.preflight_file}: preflight format check is alpha=0 only, "
            f"got {m.get('alpha')}")
    ids = [r["id"] for r in rows]
    gold = load_private_gold(ids, expected_shapes=_expected_shapes(rows))
    scored = score_rows(rows, gold)

    print(f"\n=== PREFLIGHT (n={len(scored)}) model={m.get('model')} "
          f"alpha=0  max_new_tokens={m.get('max_new_tokens')} ===")
    for x in scored:
        print(f"  {x['id']:20s} parsed={x['parsed']!s:5} solved={x['solved']!s:5} "
              f"cells={x['correct_cells']}/{x['total_cells']} "
              f"truncated={x['truncated']!s:5} tokens={x['generated_token_count']}")

    n = len(scored)
    print(f"\nno_answer_rate  : {sum(1 for x in scored if x['no_answer'])/n:.3f}")
    print(f"truncated_rate  : {sum(1 for x in scored if x['truncated'])/n:.3f}")
    print(f"puzzle_acc      : {sum(1 for x in scored if x['solved'])/n:.3f}")
    print(f"steering_fires  : {m.get('steering_fires')}  (expect 0 at alpha=0)")
    print(f"prompt_sha256   : {m.get('prompt_sha256')}")
    print("\nThis is a FORMAT/PLUMBING check only, n=5. It must not be used to "
          "change the prompt, item set, doses, or scoring method based on "
          "accuracy -- see docs/PREREG_ZEBRALOGIC_EASY.md section 5.")


# ------------------------------------------------------------------ main ---

def cmd_formal(args):
    files = args.generations
    metas, rowsets = [], []
    for p in files:
        m, rows = load_cell(p)
        if m.get("mode") != "formal":
            die(f"{p}: mode {m.get('mode')!r} != 'formal'")
        if len(rows) != N_EASY:
            die(f"{p}: {len(rows)} rows, expected {N_EASY}")
        metas.append(m)
        rowsets.append(rows)

    model = metas[0]["model"]
    if any(m["model"] != model for m in metas):
        die("all --generations files must be the SAME model for one --formal run")
    frozen = set(FROZEN_ALPHAS.get(model, ()))
    if not frozen:
        die(f"unknown model {model!r}; expected one of {list(FROZEN_ALPHAS)}")

    ids_ref = sorted(r["id"] for r in rowsets[0])
    for p, rows in zip(files[1:], rowsets[1:]):
        if sorted(r["id"] for r in rows) != ids_ref:
            die(f"{p}: item id set differs from {files[0]}")

    # sample_id must be exactly 0..279, unique, per cell -- the frozen item
    # order every downstream sample_id-based pairing (McNemar/bootstrap) and
    # cross-check (item_ids_sha256) assumes. A gap or duplicate here would
    # silently misalign the paired comparison against alpha=0.
    for p, rows in zip(files, rowsets):
        sids = sorted(r["sample_id"] for r in rows)
        if sids != list(range(N_EASY)):
            die(f"{p}: sample_id is not exactly 0..{N_EASY-1} with no gaps/"
                "duplicates; the frozen item order is not intact.")

    # Cross-check the item set against the loader's own frozen id digest
    # (data_zebralogic.EXPECTED_EASY_IDS_SHA256), not just internal
    # consistency across the supplied cells -- internal consistency alone
    # would not catch every cell being generated from a stale/wrong blind
    # file that happens to be self-consistent.
    ids_digest = sha16("\n".join(ids_ref))
    if ids_digest != EXPECTED_EASY_IDS_SHA256:
        die(f"item id set digest {ids_digest} != frozen "
            f"EXPECTED_EASY_IDS_SHA256 {EXPECTED_EASY_IDS_SHA256}; these "
            "generations were not produced from the frozen easy-tier item set.")

    prompt_hash_ref = metas[0].get("prompt_sha256")
    item_ids_sha_ref = metas[0].get("item_ids_sha256")
    source_revision_ref = metas[0].get("source_revision")
    by_alpha = {}
    for m, rows in zip(metas, rowsets):
        al = m["alpha"]
        if al not in frozen:
            die(f"{files[metas.index(m)]}: alpha {al} not in frozen set "
                f"{sorted(frozen)} for {model}; this script does not search doses")
        if al in by_alpha:
            die(f"alpha={al} supplied twice")
        if m.get("prompt_sha256") != prompt_hash_ref:
            die(f"alpha={al}: prompt_sha256 differs from alpha={metas[0]['alpha']}; "
                "the rendered prompts are not identical across doses")
        if m.get("item_ids_sha256") != item_ids_sha_ref:
            die(f"alpha={al}: item_ids_sha256 differs from alpha={metas[0]['alpha']}; "
                "the generated item ORDER differs across doses even if the id "
                "set is the same -- sample_id-based pairing would misalign.")
        if m.get("source_revision") != source_revision_ref:
            die(f"alpha={al}: source_revision {m.get('source_revision')!r} "
                f"differs from alpha={metas[0]['alpha']}'s "
                f"{source_revision_ref!r}; the cells were generated from "
                "different dataset revisions.")
        exp_fires = 0 if al == 0 else m["L"] * len(rows)
        if m.get("steering_fires") != exp_fires:
            die(f"alpha={al}: steering_fires {m.get('steering_fires')} != "
                f"{exp_fires}; intervention unverified")
        by_alpha[al] = {"meta": m, "rows": {r["id"]: r for r in rows}}

    if 0 not in by_alpha:
        die(f"{model}: no alpha=0 cell supplied; it is required as the baseline")
    # COMPLETENESS: a formal-scope score MUST see all four frozen alpha for
    # this model, or Holm m=3 / the argmax-and-near-optimal-region workpoint
    # rule (prereg S8) is not a well-defined family. The previous behavior --
    # print a warning and score whichever subset was supplied -- let an
    # incomplete sweep produce a plausible-looking but statistically
    # meaningless "Holm m=3, INCOMPLETE" result (the cmd_formal body below
    # already tests `len(stats) == len(frozen) - 1` to withhold Holm, but the
    # near-optimal-region / argmax reporting still ran over a partial family,
    # and nothing forced a human to notice the printed warning before citing
    # the output). Hard-stop instead: an incomplete formal sweep is scored
    # only via a deliberate, explicit override.
    missing = frozen - set(by_alpha)
    if missing and not args.allow_partial_alphas:
        die(f"{model}: missing alpha cell(s) {sorted(missing)} -- this "
            f"protocol's Holm m=3 family and workpoint rule (prereg S8) "
            f"require all four frozen alpha {sorted(frozen)}. Pass "
            "--allow_partial_alphas only for a deliberate, human-invoked "
            "partial score (e.g. inspecting one finished cell mid-sweep); "
            "such a run's Holm/argmax/near-optimal-region output must NOT "
            "be cited as the formal four-point result.")
    if missing and args.allow_partial_alphas:
        print(f"[eval] NOTE: {model} scored with alpha cell(s) {sorted(missing)} "
              f"MISSING, under --allow_partial_alphas -- scoring only the "
              f"{sorted(by_alpha)} cells present. This run's Holm/argmax/"
              "near-optimal-region output must NOT be cited as the formal "
              "four-point result.")

    # CONSISTENCY: every cell of one model's formal sweep must be a
    # genuinely comparable point on ONE dose curve -- same mask, same token
    # budget, same batch size, same model/band. Only prompt_sha256 was
    # checked above; a mask swap or an inconsistently-applied S3
    # 2048->3072 token-budget escalation would otherwise pass silently.
    metas_by_alpha = {al: v["meta"] for al, v in by_alpha.items()}
    ref_al = sorted(metas_by_alpha)[0]
    ref_meta = metas_by_alpha[ref_al]
    for al, m in metas_by_alpha.items():
        for field in CONSISTENCY_FIELDS:
            if m.get(field) != ref_meta.get(field):
                die(f"{model}: alpha={al} cell's {field!r}={m.get(field)!r} "
                    f"differs from alpha={ref_al}'s {ref_meta.get(field)!r}; "
                    "cells of one model's dose curve must share an identical "
                    "configuration (mask/token-budget/batch-size/model), or "
                    "they are not a comparable curve.")

    # SAME PHYSICAL GPU per model (prereg S4): bf16 greedy is not byte-
    # reproducible across GPUs, and every alpha of one model is a paired
    # per-item contrast against that model's own alpha=0, so a cell generated
    # on a different card mixes a device difference into the alpha effect.
    # `cuda_visible_devices` is the authoritative field (it is what was
    # actually pinned at launch, per run_zebralogic.sh's require_card); a
    # missing value on any cell cannot be treated as "same as the others" --
    # that would silently accept an unpinned run precisely because it forgot
    # to record what card it used.
    cvd_by_alpha = {al: m.get("cuda_visible_devices") for al, m in metas_by_alpha.items()}
    if any(not v for v in cvd_by_alpha.values()):
        die(f"{model}: cuda_visible_devices missing/empty on cell(s) "
            f"{sorted(al for al, v in cvd_by_alpha.items() if not v)} -- "
            "cannot confirm all alpha cells of this model ran on one "
            "physical card (prereg S4 requirement).")
    cvd_set = set(cvd_by_alpha.values())
    if len(cvd_set) > 1:
        die(f"{model}: alpha cells report different cuda_visible_devices "
            f"{cvd_by_alpha} -- prereg S4 requires all of one model's alpha "
            "cells to share ONE physical GPU (paired per-item contrast, and "
            "bf16 greedy is not byte-reproducible across GPUs).")

    ids = ids_ref
    gold = load_private_gold(ids, expected_shapes=_expected_shapes(rowsets[0]))

    sizes_order = ("2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3")
    scored_by_alpha = {}
    for al, cell in sorted(by_alpha.items()):
        rows = [cell["rows"][i] for i in ids]
        scored_by_alpha[al] = score_rows(rows, gold)

    summaries = {al: summarize(sc, sizes_order) for al, sc in scored_by_alpha.items()}

    print(f"\n=== FORMAL RESULTS  model={model}  n={len(ids)} ===")
    print(f"{'alpha':>6} {'puzzle_acc':>10} {'cell_acc':>10} {'no_answer':>10} "
          f"{'truncated':>10} {'sol_first':>10} {'loop':>7} {'tok_med':>8}")
    for al in sorted(summaries):
        s = summaries[al]
        print(f"{al:>6} {s['puzzle_acc']:10.4f} {s['cell_acc']:10.4f} "
              f"{s['no_answer_rate']:10.4f} {s['truncated_rate']:10.4f} "
              f"{s['solution_first_rate']:10.4f} {s['strict_loop_rate']:7.3f} "
              f"{str(s['gen_tokens_med']):>8}")

    # ---- per-size breakdown
    print(f"\n=== PER-SIZE Puzzle Accuracy (n=40 each; descriptive only, NOT "
          f"independently significant) ===")
    print(f"{'alpha':>6} " + " ".join(f"{sz:>7}" for sz in sizes_order))
    for al in sorted(summaries):
        row = summaries[al]["by_size"]
        print(f"{al:>6} " + " ".join(
            f"{row[sz]['puzzle_acc']:7.3f}" if sz in row else "     NA"
            for sz in sizes_order))

    # ---- paired stats: each non-zero alpha vs alpha=0
    stats = {}
    if 0 in scored_by_alpha:
        base = scored_by_alpha[0]
        base_solved = [x["solved"] for x in sorted(base, key=lambda r: r["sample_id"])]
        base_cellrate = [x["cell_rate"] for x in sorted(base, key=lambda r: r["sample_id"])]
        for al, sc in scored_by_alpha.items():
            if al == 0:
                continue
            sc_sorted = sorted(sc, key=lambda r: r["sample_id"])
            solved = [x["solved"] for x in sc_sorted]
            cellrate = [x["cell_rate"] for x in sc_sorted]
            a01 = [1 if x else 0 for x in base_solved]
            a11 = [1 if x else 0 for x in solved]
            b01, b10, p = mcnemar_exact(a01, a11)
            lo, hi = boot_ci_puzzle(a01, a11)
            clo, chi = boot_ci_cell(base_cellrate, cellrate)
            stats[al] = {
                "puzzle_acc_base": sum(a01) / len(a01),
                "puzzle_acc_steer": sum(a11) / len(a11),
                "dPuzzleAcc_pp": (sum(a11) - sum(a01)) / len(a01) * 100,
                "discordant_0to1": b01, "discordant_1to0": b10,
                "p_raw": p, "ci95_puzzle_pp": [lo, hi],
                "cell_acc_base": summaries[0]["cell_acc"],
                "cell_acc_steer": summaries[al]["cell_acc"],
                "dCellAcc_pp": (summaries[al]["cell_acc"] - summaries[0]["cell_acc"]) * 100,
                "ci95_cell_pp": [clo, chi],
            }

        adj = holm([(al, stats[al]["p_raw"]) for al in stats]) if len(stats) == len(frozen) - 1 else None
        holm_complete = adj is not None
        print(f"\n=== PAIRED vs alpha=0  ({'Holm m=3' if holm_complete else 'INCOMPLETE -- Holm withheld'}) ===")
        print(f"{'alpha':>6} {'acc0':>8} {'accA':>8} {'dPuzzle':>9} {'0>1':>4} "
              f"{'1>0':>4} {'p_raw':>9} {'p_adj':>9}  CI95(puzzle)  |  dCell  CI95(cell)")
        for al in sorted(stats):
            t = stats[al]
            pa = f"{adj[al]:9.4f}" if adj else "  WITHHELD"
            print(f"{al:>6} {t['puzzle_acc_base']:8.4f} {t['puzzle_acc_steer']:8.4f} "
                  f"{t['dPuzzleAcc_pp']:+9.2f} {t['discordant_0to1']:4d} "
                  f"{t['discordant_1to0']:4d} {t['p_raw']:9.4f} {pa}  "
                  f"[{t['ci95_puzzle_pp'][0]:+.2f}, {t['ci95_puzzle_pp'][1]:+.2f}]  |  "
                  f"{t['dCellAcc_pp']:+6.2f}  [{t['ci95_cell_pp'][0]:+.2f}, {t['ci95_cell_pp'][1]:+.2f}]")

        # ---- workpoint rule (frozen, prereg section 8)
        all_alphas = sorted(summaries)
        argmax_alpha = max(all_alphas, key=lambda a: (summaries[a]["puzzle_acc"], -abs(a)))
        # rule 3 is written generally ("any point whose paired difference
        # from the argmax is not detected") with no carve-out for alpha=0 --
        # excluding alpha=0 unconditionally (the previous `or al == 0:
        # continue`) meant baseline could never be reported as statistically
        # indistinguishable from the argmax, which is exactly the wrong bias
        # in the case rule 5 exists for: "no dose actually beats baseline".
        # argmax_alpha itself is trivially indistinguishable from itself and
        # is seeded into the set below without a redundant self-comparison.
        near_optimal = [argmax_alpha]
        for al in all_alphas:
            if al == argmax_alpha:
                continue
            # "not detected vs the argmax": exploratory, NOT the Holm family
            # above (that family is strictly alpha-vs-0).
            a_argmax = [1 if x["solved"] else 0 for x in
                        sorted(scored_by_alpha[argmax_alpha], key=lambda r: r["sample_id"])]
            a_other = [1 if x["solved"] else 0 for x in
                       sorted(scored_by_alpha[al], key=lambda r: r["sample_id"])]
            _, _, p_vs_argmax = mcnemar_exact(a_argmax, a_other)
            lo, hi = boot_ci_puzzle(a_argmax, a_other)
            if p_vs_argmax > 0.05 or (lo <= 0 <= hi):
                near_optimal.append(al)

        # A dose only "clears Holm" toward a workpoint claim if it is BOTH
        # Holm-significant AND an IMPROVEMENT over alpha=0 (dPuzzleAcc_pp >
        # 0). The previous `holm_pass = any(adj[al] < 0.05 ...)` accepted a
        # Holm-significant DEGRADATION just as readily as an improvement --
        # so a dose that significantly HURT accuracy could make the verdict
        # report the numerically-highest (but possibly non-significant, or
        # even alpha=0 itself) point as an established "workpoint", which
        # rule 5 exists specifically to forbid.
        qualifying = [al for al in stats
                     if adj is not None and adj[al] < 0.05
                     and stats[al]["dPuzzleAcc_pp"] > 0]
        if qualifying:
            verdict = (f"argmax alpha={argmax_alpha} (puzzle_acc="
                      f"{summaries[argmax_alpha]['puzzle_acc']:.4f}); "
                      f"Holm-significant improving alpha(s) = {sorted(qualifying)}; "
                      f"near-optimal region (exploratory) = {sorted(set(near_optimal))}")
        else:
            verdict = ("NO non-zero alpha cleared Holm as a SIGNIFICANT "
                      "IMPROVEMENT vs alpha=0 -- per the frozen workpoint "
                      "rule, the conclusion is 'no effective workpoint "
                      "detected among the four sampled points'. The "
                      f"numerically highest point (alpha={argmax_alpha}) may "
                      "NOT be reported as an established workpoint, even if "
                      "some other alpha is a significant DEGRADATION.")
        print(f"\n=== WORKPOINT VERDICT ===\n{verdict}")
    else:
        holm_complete = False
        adj = None
        argmax_alpha = None
        near_optimal = []
        qualifying = []
        verdict = "alpha=0 cell missing -- no paired analysis possible."
        print(f"\n{verdict}")

    out = {
        "protocol": PROTOCOL, "model": model, "n": len(ids),
        "alphas_present": sorted(summaries), "frozen_alphas": sorted(frozen),
        "summaries": summaries, "paired_vs_zero": stats,
        # holm_family_m reflects the ACTUAL number of non-zero-alpha pairs
        # scored (len(stats)), not a hardcoded 3 -- a hardcoded value would
        # misdescribe an --allow_partial_alphas run's correction (which
        # applies over fewer pairs) as if it were the full m=3 family.
        "holm_family_m": len(stats), "holm_complete": holm_complete, "p_adj": adj,
        "argmax_alpha": argmax_alpha,
        "holm_significant_improvement_alphas": sorted(qualifying),
        "near_optimal_region": sorted(set(near_optimal)),
        "verdict": verdict,
        "note": ("Four sampled points only; this is NOT a dose-response curve. "
                 "near_optimal_region is EXPLORATORY and excluded from the "
                 "alpha-vs-0 Holm family. Cell Accuracy improving without "
                 "Puzzle Accuracy improving supports only 'localized grid-"
                 "filling quality improved', never a solved-more-puzzles claim."),
    }
    if args.out:
        if os.path.exists(args.out):
            die(f"{args.out} exists; refusing to overwrite")
        json.dump(out, open(args.out, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    ap.add_argument("--generations", nargs="+",
                    help="formal mode: cell JSONs from get_answer_zebralogic.py --mode formal, one model")
    ap.add_argument("--out", help="output JSON path (formal/canary_check)")
    ap.add_argument("--canary_check", action="store_true")
    ap.add_argument("--canary_files", nargs="+", default=[],
                    help="canary_check: 2+ --mode canary cell JSONs, different --device_tag")
    ap.add_argument("--preflight_check", action="store_true")
    ap.add_argument("--preflight_file", help="preflight_check: one --mode preflight cell JSON")
    ap.add_argument("--allow_partial_alphas", action="store_true",
                    help="formal mode: score a model with fewer than the "
                         "frozen four alpha cells present. Only for a "
                         "deliberate, human-invoked partial score (e.g. "
                         "inspecting one finished cell mid-sweep) -- the "
                         "formal launcher path must NEVER pass this, and "
                         "such a run's Holm/argmax/near-optimal-region "
                         "output must not be cited as the formal result.")
    a = ap.parse_args()

    if a.canary_check:
        cmd_canary_check(a)
    elif a.preflight_check:
        if not a.preflight_file:
            die("--preflight_check requires --preflight_file")
        cmd_preflight_check(a)
    else:
        if not a.generations:
            die("--generations is required for the formal scoring mode "
                "(or pass --canary_check / --preflight_check)")
        cmd_formal(a)


if __name__ == "__main__":
    main()
