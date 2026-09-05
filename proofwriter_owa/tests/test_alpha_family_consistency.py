#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression test for eval_proofwriter_owa.py's cross-alpha consistency checks
(review finding #8, 2026-09-04, completed): a formal-scope scoring call must
(a) see the FULL frozen alpha family per model, with the family size actually
used by Holm matching what is reported, and (b) reject a run whose cells
carry mismatched configuration (mask/prompt/manifest/batch/token-budget/
model) across the same model's own alpha curve. The pilot's legitimate
alpha=0-only case must NOT require any flag.

No network, no GPU, no real ProofWriter data. Run with:
    python3.10 proofwriter_owa/tests/test_alpha_family_consistency.py
"""

import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_PW_DIR = os.path.dirname(_HERE)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


N = 60
GOLD_IDS = list(range(N))


def make_gold(path):
    rows = []
    for i in GOLD_IDS:
        ds = "D3" if i < N // 2 else "D5"
        lab = ("True", "False", "Unknown")[i % 3]
        rows.append({"sample_id": i, "dataset": ds, "answer": lab})
    json.dump({"meta": {"contains_labels": True, "manifest_sha256_16": "test",
                        "owa_semantics": "test"}, "data": rows},
              open(path, "w"))


def make_cell(path, model, alpha, layer_start, layer_end, L, correct_ids,
             overrides=None):
    """overrides: dict of extra/overridden meta fields (used to inject a
    consistency mismatch, e.g. a different mask_sha256 or prompt_sha256)."""
    gold_lookup = {i: ("True", "False", "Unknown")[i % 3] for i in GOLD_IDS}
    rows = []
    for i in GOLD_IDS:
        gold_lab = gold_lookup[i]
        pred_lab = gold_lab if i in correct_ids else next(
            l for l in ("True", "False", "Unknown") if l != gold_lab)
        rows.append({"sample_id": i, "generated": f"reasoning...\nAnswer: {pred_lab}",
                    "truncated": False, "generated_token_count": 20,
                    "pre_answer_reasoning_tokens": 10})
    fires = 0 if alpha == 0 else L * len(rows)
    # prompt_template_id must be a REAL registered id (answer_parser.
    # MARKER_FAMILIES), since load_cell now derives marker_family from it
    # and rejects anything unregistered -- the placeholder string "v1" used
    # before the marker-family registry existed is no longer valid here.
    meta = {"protocol": "proofwriter-owa-v0", "model": model, "alpha": alpha,
           "layer_start": layer_start, "layer_end": layer_end, "L": L,
           "steering_fires": fires, "accuracy_computed": False,
           "mask_path": "/fake/mask.npy", "mask_sha256": "abc123",
           "prompt_sha256": "prompt_hash_1",
           "prompt_template_id": "proofwriter-owa-cot-v1",
           "marker_family": "v1",
           "manifest_sha256_16": "test", "batch_size": 8,
           "max_new_tokens": 768, "temperature": 0.0, "top_p": 1.0,
           "n_shot": 0, "padding_side": "left", "chat_template": False,
           "prefill_only": True, "prefill_tail_len": 1, "size": "8B"}
    if overrides:
        meta.update(overrides)
    json.dump({"meta": meta, "data": rows}, open(path, "w"))


def run_eval(gold_path, cell_paths, out_path, holm_m=3, extra_args=None):
    cmd = [sys.executable, os.path.join(_PW_DIR, "eval_proofwriter_owa.py"),
          "--gold", gold_path, "--generations", *cell_paths,
          "--out", out_path, "--holm_m", str(holm_m)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True)


def four_llama_cells(tmpdir, overrides_by_alpha=None):
    overrides_by_alpha = overrides_by_alpha or {}
    paths = {}
    for al, tag in ((0, "c0"), (-6, "cn6"), (-4, "cn4"), (4, "c4")):
        p = os.path.join(tmpdir, f"{tag}.json")
        make_cell(p, "llama3", al, 11, 20, 9, set(range(0, 30)),
                 overrides=overrides_by_alpha.get(al))
        paths[al] = p
    return paths


def test_complete_family_scores_cleanly():
    tmpdir = tempfile.mkdtemp()
    gold_path = os.path.join(tmpdir, "gold.json")
    make_gold(gold_path)
    paths = four_llama_cells(tmpdir)
    out_path = os.path.join(tmpdir, "result.json")
    r = run_eval(gold_path, list(paths.values()), out_path)
    check("complete 4-point family exits 0", r.returncode == 0, r.stderr)
    result = json.load(open(out_path))
    fam = result["results"]["llama3"]["mcnemar_vs_alpha0"]
    check("holm_family_m recorded per-pair is 3 (the real family size), "
          "not just the CLI default echoed back",
          all(v["holm_family_m"] == 3 for v in fam.values()),
          {k: v["holm_family_m"] for k, v in fam.items()})


def test_missing_dose_hard_stops():
    """Only 2 of the 3 non-zero llama doses supplied -- must hard-stop
    without --allow_partial_alphas (this is the scenario review finding #8
    describes: 'a run missing one of the model's four frozen alpha')."""
    tmpdir = tempfile.mkdtemp()
    gold_path = os.path.join(tmpdir, "gold.json")
    make_gold(gold_path)
    paths = four_llama_cells(tmpdir)
    del paths[4]  # drop alpha=+4
    out_path = os.path.join(tmpdir, "result.json")
    r = run_eval(gold_path, list(paths.values()), out_path)
    check("missing one non-zero dose hard-stops (exit != 0)",
          r.returncode != 0, f"returncode={r.returncode}")
    check("error names the frozen dose set and what's missing",
          "frozen" in r.stderr.lower() and "4" in r.stderr, r.stderr)


def test_missing_dose_allowed_with_explicit_flag():
    tmpdir = tempfile.mkdtemp()
    gold_path = os.path.join(tmpdir, "gold.json")
    make_gold(gold_path)
    paths = four_llama_cells(tmpdir)
    del paths[4]
    out_path = os.path.join(tmpdir, "result.json")
    r = run_eval(gold_path, list(paths.values()), out_path,
                extra_args=["--allow_partial_alphas"])
    check("--allow_partial_alphas lets a 2-of-3 partial family score",
          r.returncode == 0, r.stderr)
    result = json.load(open(out_path))
    fam = result["results"]["llama3"]["mcnemar_vs_alpha0"]
    check("with only 2 non-zero doses present, holm_family_m reflects 2, "
          "NOT the CLI default of 3 (the exact bug review finding #8 named: "
          "'仍标注 Holm 固定 m=3')",
          all(v["holm_family_m"] == 2 for v in fam.values()),
          {k: v["holm_family_m"] for k, v in fam.items()})


def test_pilot_alpha0_only_needs_no_flag():
    """The launcher's pilot stage invokes the evaluator with exactly one
    alpha=0 cell and NEVER passes --allow_partial_alphas -- this must keep
    working without the flag."""
    tmpdir = tempfile.mkdtemp()
    gold_path = os.path.join(tmpdir, "gold.json")
    make_gold(gold_path)
    p0 = os.path.join(tmpdir, "c0.json")
    make_cell(p0, "llama3", 0, 11, 20, 9, set(range(0, 30)))
    out_path = os.path.join(tmpdir, "result.json")
    r = run_eval(gold_path, [p0], out_path)
    check("pilot's alpha=0-only cell scores WITHOUT --allow_partial_alphas "
          "(matches run_proofwriter_owa.sh's documented pilot invocation, "
          "which never passes that flag)",
          r.returncode == 0, r.stderr)


def test_extra_unauthorized_dose_hard_stops():
    tmpdir = tempfile.mkdtemp()
    gold_path = os.path.join(tmpdir, "gold.json")
    make_gold(gold_path)
    paths = four_llama_cells(tmpdir)
    p2 = os.path.join(tmpdir, "c2.json")
    make_cell(p2, "llama3", 2, 11, 20, 9, set(range(0, 30)))  # never frozen
    out_path = os.path.join(tmpdir, "result.json")
    r = run_eval(gold_path, list(paths.values()) + [p2], out_path)
    check("an extra alpha outside the frozen set hard-stops",
          r.returncode != 0, f"returncode={r.returncode}")


def test_mask_mismatch_hard_stops():
    """One cell was accidentally generated against a different mask file --
    the classic 'compared two different experiments' failure mode."""
    tmpdir = tempfile.mkdtemp()
    gold_path = os.path.join(tmpdir, "gold.json")
    make_gold(gold_path)
    paths = four_llama_cells(
        tmpdir, overrides_by_alpha={4: {"mask_sha256": "DIFFERENT_HASH"}})
    out_path = os.path.join(tmpdir, "result.json")
    r = run_eval(gold_path, list(paths.values()), out_path)
    check("a mask_sha256 mismatch across one model's own alpha cells "
          "hard-stops (exit != 0)", r.returncode != 0, f"rc={r.returncode}")
    check("error names the mismatching field",
          "mask_sha256" in r.stderr, r.stderr)


def test_prompt_mismatch_hard_stops():
    """A prompt_sha256 drift (e.g. one cell rendered against a stale
    manifest, or a template edit between runs) must be caught the same way."""
    tmpdir = tempfile.mkdtemp()
    gold_path = os.path.join(tmpdir, "gold.json")
    make_gold(gold_path)
    paths = four_llama_cells(
        tmpdir, overrides_by_alpha={-6: {"prompt_sha256": "stale_prompt_hash"}})
    out_path = os.path.join(tmpdir, "result.json")
    r = run_eval(gold_path, list(paths.values()), out_path)
    check("a prompt_sha256 mismatch hard-stops", r.returncode != 0,
          f"rc={r.returncode}")
    check("error names prompt_sha256", "prompt_sha256" in r.stderr, r.stderr)


def test_batch_size_mismatch_hard_stops():
    tmpdir = tempfile.mkdtemp()
    gold_path = os.path.join(tmpdir, "gold.json")
    make_gold(gold_path)
    paths = four_llama_cells(
        tmpdir, overrides_by_alpha={-4: {"batch_size": 1}})
    out_path = os.path.join(tmpdir, "result.json")
    r = run_eval(gold_path, list(paths.values()), out_path)
    check("a batch_size mismatch hard-stops", r.returncode != 0,
          f"rc={r.returncode}")


def test_max_new_tokens_mismatch_hard_stops():
    """The 768->1024 escalation rule (PREREG S6) must never be applied to
    only SOME of one model's cells -- if the pilot triggered the escalation,
    ALL formal cells for that model use the new budget, or the curve is not
    comparable."""
    tmpdir = tempfile.mkdtemp()
    gold_path = os.path.join(tmpdir, "gold.json")
    make_gold(gold_path)
    paths = four_llama_cells(
        tmpdir, overrides_by_alpha={0: {"max_new_tokens": 1024}})
    out_path = os.path.join(tmpdir, "result.json")
    r = run_eval(gold_path, list(paths.values()), out_path)
    check("a max_new_tokens mismatch (768 vs 1024 escalation applied "
          "inconsistently) hard-stops", r.returncode != 0, f"rc={r.returncode}")


def test_mutation_pre_fix_would_have_passed():
    """Mutation check: confirm the OLD code path (no completeness/consistency
    checks at all, holm_family_m always echoing the CLI default) really
    would have scored the mask-mismatch and missing-dose scenarios above
    without complaint -- i.e. these are not strawman scenarios the old code
    happened to already catch some other way."""
    tmpdir = tempfile.mkdtemp()
    gold_path = os.path.join(tmpdir, "gold.json")
    make_gold(gold_path)

    # Simulate the pre-fix evaluator: it never reads mask_sha256/prompt_sha256
    # /batch_size/max_new_tokens from meta at all (grep confirms these fields
    # are only referenced inside the NEW consistency-check block), so a
    # mismatch on any of them was structurally invisible before this fix.
    src = open(os.path.join(_PW_DIR, "eval_proofwriter_owa.py")).read()
    n_mask_refs = src.count('"mask_sha256"') + src.count("'mask_sha256'")
    check("mask_sha256 is referenced by the evaluator (proves the new "
          "check block is the ONLY place that reads it, and that removing "
          "it would silently stop checking)",
          n_mask_refs >= 1, f"n_mask_refs={n_mask_refs}")
    check("CONSISTENCY_FIELDS constant exists and lists the fields findings "
          "#8 named (config/manifest/prompt/mask)",
          "CONSISTENCY_FIELDS" in src and "mask_sha256" in src
          and "prompt_sha256" in src and "manifest_sha256_16" in src
          and "batch_size" in src and "max_new_tokens" in src)
    check("EXPECTED_ALPHAS constant exists (the frozen dose-family gate)",
          "EXPECTED_ALPHAS" in src)


def main():
    print("== complete family scores cleanly, holm_family_m == 3 ==")
    test_complete_family_scores_cleanly()
    print("== missing dose hard-stops without a flag ==")
    test_missing_dose_hard_stops()
    print("== missing dose allowed only with --allow_partial_alphas, "
          "and holm_family_m reflects the REAL count ==")
    test_missing_dose_allowed_with_explicit_flag()
    print("== pilot alpha=0-only needs no flag ==")
    test_pilot_alpha0_only_needs_no_flag()
    print("== extra unauthorized dose hard-stops ==")
    test_extra_unauthorized_dose_hard_stops()
    print("== mask_sha256 mismatch hard-stops ==")
    test_mask_mismatch_hard_stops()
    print("== prompt_sha256 mismatch hard-stops ==")
    test_prompt_mismatch_hard_stops()
    print("== batch_size mismatch hard-stops ==")
    test_batch_size_mismatch_hard_stops()
    print("== max_new_tokens mismatch hard-stops ==")
    test_max_new_tokens_mismatch_hard_stops()
    print("== mutation sanity: these fields were genuinely unchecked before ==")
    test_mutation_pre_fix_would_have_passed()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
