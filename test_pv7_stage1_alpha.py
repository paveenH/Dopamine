#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""Local checks for the pv7 Stage-1 alpha diagnostic. No GPU, no server.

Exits non-zero on the first failure, like the other test_*.py in this repo.

WHAT THIS PROTECTS
------------------
The evaluator's guarantees are fail-closed assertions that only fire on bad
input. Without a test they degrade into comments the moment someone edits the
run loop. Each check below corresponds to a guarantee that, if silently lost,
would produce a result file that looks fine and is uninterpretable:

  * Stage 2 is NEVER steered            -> would mix policy and execution
  * alpha=0 registers no hook at all    -> "unsteered" != "steered by zero"
  * steered Stage 1 fires n_layers      -> wrong site count = wrong injection
  * critical_arm comes from the bank    -> sort order mislabels seeds 3, 26
  * the Stage 1 prompt is alpha-shared  -> otherwise cells are not comparable
  * inference clusters by seed          -> 123 states are not 123 samples

THE FAKE MODEL IS DELIBERATELY STRICT
-------------------------------------
`generate` rejects `diff_matrices` and `regenerate` raises on None, mirroring
llms.py:821. A permissive fake would hide exactly the confusion these checks
exist to catch. It models INTENT, not arithmetic: it cannot validate the real
hook internals, only that the evaluator asks for the right thing.

Usage
-----
    python3.10 test_pv7_stage1_alpha.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import bandit_reference as br
import eval_pv7_stage1_alpha as E

BANK = Path(__file__).with_name("bandit_pv7_lockin_states.json")
N_LAYERS = 9

_RATIONALE = ("Evidence: Button A has 1 trial and is uncertain.\n"
              "Policy: EXPLORE Button A because it has only 1 trial.")


def make_vc(fire_rationale=N_LAYERS, fire_action=0, text=_RATIONALE):
    """A contract-faithful stand-in for VicundaModel."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    class FakeVC:
        tokenizer = tok

        def __init__(self):
            self.n = 0
            self.calls = []

        def steering_fire_count(self, reset=False):
            v = self.n
            if reset:
                self.n = 0
            return v

        def generate(self, inputs, **kw):
            assert "diff_matrices" not in kw, \
                "generate() takes no diff_matrices (llms.py contract)"
            self.calls.append(("stage1_unsteered", None))
            self.n += 0            # unsteered: no sites
            return [text]

        def regenerate(self, inputs, diff_matrices=None, **kw):
            if diff_matrices is None:
                raise ValueError("regenerate raises on None (llms.py:821)")
            assert kw.get("prefill_only") is True
            assert kw.get("prefill_tail_len") == 1
            self.calls.append(("stage1_steered", True))
            self.n += fire_rationale
            return [text]

        def regenerate_logits_teacher_forcing(self, prompts, answer_token_ids,
                                              diff_matrices=None):
            self.calls.append(("stage2_score", diff_matrices))
            self.n += fire_action
            return [[np.random.RandomState(i).randn(128256)]
                    for i in range(len(prompts))]

    return FakeVC()


def _bank():
    return json.loads(BANK.read_text())


def _state(pred=None):
    for s in _bank()["states"]:
        if pred is None or pred(s):
            return s
    raise AssertionError("no matching state")


def check_stage2_never_steered():
    st = _state()
    vc = make_vc()
    env = br.get_environment("easy")
    diff = {-4.0: [1] * 32, 0.0: None, 4.0: [1] * 32}
    E.run_state(vc, st, env, diff, N_LAYERS, dry_run=False)
    seen = {c[1] for c in vc.calls if c[0] == "stage2_score"}
    assert seen == {None}, f"Stage 2 received {seen}, must always be None"
    paths = [c[0] for c in vc.calls if c[0] != "stage2_score"]
    assert paths == ["stage1_steered", "stage1_unsteered", "stage1_steered"], \
        f"alpha=0 must use generate(), steered must use regenerate(): {paths}"
    print("  stage2 never steered; alpha=0 uses the no-hook path   OK")


def check_fires_acceptance():
    env = br.get_environment("easy")
    diff = {-4.0: [1] * 32, 0.0: None, 4.0: [1] * 32}
    st = _state()
    for label, vc in [
        ("steered but 0 sites", make_vc(fire_rationale=0)),
        ("stage 2 leaked sites", make_vc(fire_action=N_LAYERS * 4)),
        ("wrong site count", make_vc(fire_rationale=N_LAYERS + 1)),
    ]:
        try:
            E.run_state(vc, st, env, diff, N_LAYERS, dry_run=False)
        except SystemExit:
            continue
        raise AssertionError(f"accepted a bad cell: {label}")
    print("  rejects no-fire / stage2-leak / wrong-count           OK")


def check_critical_arm_from_bank():
    bank = _bank()
    crit = [s for s in bank["states"]
            if s["tags"]["is_critical_one_shot_zero"]]
    assert len(crit) == 5, f"expected 5 critical states, got {len(crit)}"
    mism = [(s["seed"], s["diagnostics"]["critical_arm"],
             s["tags"]["one_shot_zero_arms"][0])
            for s in crit
            if s["diagnostics"]["critical_arm"]
            != s["tags"]["one_shot_zero_arms"][0]]
    # This is the whole point: sort order is NOT the abandoned arm. If this
    # list is ever empty the bank changed, and the test must be re-derived
    # rather than deleted.
    assert mism, ("no critical state distinguishes critical_arm from "
                  "sorted-first; the regression this guards is untestable")
    for s in crit:
        ca = s["diagnostics"]["critical_arm"]
        assert ca in s["tags"]["one_shot_zero_arms"]
        assert ca == s["diagnostics"]["best_arm"]
    print(f"  critical_arm stored, differs from sorted-first on "
          f"{len(mism)}/5      OK")


def check_critical_metric_uses_stored_arm():
    env = br.get_environment("easy")
    diff = {0.0: None}
    st = _state(lambda s: s["tags"]["is_critical_one_shot_zero"]
                and s["diagnostics"]["critical_arm"]
                != s["tags"]["one_shot_zero_arms"][0])
    crit_arm = st["diagnostics"]["critical_arm"]
    wrong = st["tags"]["one_shot_zero_arms"][0]

    saved_alphas, saved_score = E.ALPHAS, E.ep.score_candidates_pv7
    E.ALPHAS = (0.0,)
    try:
        for forced, expect in [(crit_arm, True), (wrong, False)]:
            vc = make_vc(fire_rationale=0)
            # Return the full candidate set (K scores) so norm_entropy is
            # defined; only the argmax is forced.
            def _forced(v, p, e, d, _f=forced, _order=st["arm_order"]):
                return ({a: (1.0 if a == _f else 0.0) for a in _order}, _f)
            E.ep.score_candidates_pv7 = _forced
            rec = E.run_state(vc, st, env, diff, N_LAYERS, dry_run=False)
            got = rec["cells"]["0.0"]["critical_arm_revisit_choice"]
            assert got is expect, (
                f"chose {forced!r}: critical_arm_revisit_choice={got}, "
                f"expected {expect}")
    finally:
        E.ALPHAS = saved_alphas
        E.ep.score_candidates_pv7 = saved_score
    print("  critical_arm_revisit_choice keys off the stored arm   OK")


def check_shared_stage1_prompt():
    st = _state()
    env = br.get_environment("easy")
    import bandit_pv7 as p7
    hist = [(e["arm"], e["reward"]) for e in st["history"]]
    prompts = {p7.build_rationale_prompt(
        st["arm_order"], hist, st["round_idx"], env,
        prompt_variant=p7.PROMPT_P1B) for _ in range(3)}
    assert len(prompts) == 1, "Stage 1 prompt is not deterministic"
    only = prompts.pop()
    assert only.endswith(" ") and not only.endswith("  "), \
        "Stage 1 prompt must end in exactly one space (token 220)"
    print("  Stage 1 prompt shared across cells, single trailing space  OK")


def check_cluster_bootstrap_unit():
    """The resampling unit must be the seed, not the state."""
    # 20 seeds x 6 states; alpha moves the metric in HALF the seeds only.
    states = []
    for seed in range(20):
        for j in range(6):
            flip = seed < 10
            states.append({
                "seed": seed, "state_id": f"s{seed}-{j}",
                "cells": {"0.0": {"m": False},
                          "4.0": {"m": bool(flip)}},
            })
    r = E.cluster_bootstrap_delta(states, "m", "0.0", "4.0", n_boot=2000)
    assert r["n_clusters"] == 20, r
    assert r["n_states"] == 120, r
    assert abs(r["delta"] - 0.5) < 1e-9, r
    # With clustering, 20 units of 6 identical states must give a WIDER
    # interval than pretending there are 120 independent states.
    flat = []
    for i, s in enumerate(states):
        flat.append({**s, "seed": i})          # every state its own cluster
    r_flat = E.cluster_bootstrap_delta(flat, "m", "0.0", "4.0", n_boot=2000)
    w_clust = r["hi"] - r["lo"]
    w_flat = r_flat["hi"] - r_flat["lo"]
    assert w_clust > w_flat * 1.5, (
        f"clustered CI ({w_clust:.3f}) must be materially wider than the "
        f"state-level one ({w_flat:.3f}); the seed clustering is not applied")
    print(f"  cluster bootstrap: unit=seed, CI {w_clust:.3f} vs flat "
          f"{w_flat:.3f}  OK")


def check_low_sample_matches_bank():
    bank = _bank()
    for s in bank["states"]:
        assert (sorted(E.low_sample_arms(s, 1))
                == sorted(s["tags"]["one_shot_zero_arms"])), s["state_id"]
    print(f"  low_sample_arms agrees with the frozen tag on "
          f"{len(bank['states'])} states  OK")


def main() -> int:
    print("pv7 Stage-1 alpha diagnostic -- local checks")
    check_low_sample_matches_bank()
    check_critical_arm_from_bank()
    check_shared_stage1_prompt()
    check_cluster_bootstrap_unit()
    check_stage2_never_steered()
    check_fires_acceptance()
    check_critical_metric_uses_stored_arm()
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
