#!/usr/bin/env python3.10
"""PV10 episode-runner invariants via a contract-faithful fake model.

The fake mirrors the REAL VicundaModel contract on purpose:
  * generate() takes no diff_matrices and registers no hooks (0 fires)
  * regenerate() RAISES on diff_matrices=None, like llms.py:821
  * steering_fire_count() reports SITES and resets

A fake more permissive than the real API hides exactly the bugs these tests
exist to catch. If you extend it, keep it at least as strict.

CAVEAT, inherited from the pv6 lesson: a FakeVC models INTENT, not arithmetic.
It reimplements the site formula rather than executing llms.py, so it will
happily agree with a wrong counter. Anything depending on real hook internals
must be driven against the actual closures, not this fake.
"""

from __future__ import annotations

import sys

import bandit_pv10 as p10
import bandit_pv10_episode as ep
from bandit_reference import Environment

FAILURES: list[str] = []
N_LAYERS = 9


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILURES.append(msg)


ENV = Environment(name="pv10_bai_candidate", k=4,
                  probs=(0.60, 0.50, 0.40, 0.30), horizon=100,
                  is_reference=False, competence_eligible=False)


class FakeTokenizer:
    bos_token_id = 128000

    def encode(self, text, add_special_tokens=False):
        # Only the invariant matters here: a prompt ending in one space ends on
        # 220, and a stripped one does not.
        ids = [1] * max(1, len(text) // 12)
        return ids + ([220] if text.endswith(" ") else [25])

    def decode(self, ids):
        return " " if ids == [220] else "?"


class FakeVC:
    """Contract-faithful stand-in. `script` maps decision_index -> raw text."""

    def __init__(self, script, layers=N_LAYERS):
        self.tokenizer = FakeTokenizer()
        self.script = script
        self.layers = layers
        self._fires = 0
        self.calls = 0
        self.saw_generate = 0
        self.saw_regenerate = 0
        self.prompts = []
        self.tail_lens = []

    def _next(self, prompt):
        self.prompts.append(prompt)
        i = self.calls
        self.calls += 1
        out = self.script(i, prompt) if callable(self.script) else self.script[i]
        return [out]

    def generate(self, inputs, max_new_tokens=128, temperature=0.0, **kw):
        if "diff_matrices" in kw:
            raise TypeError("generate() takes no diff_matrices")
        self.saw_generate += 1
        return self._next(inputs[0])          # registers NO hooks: 0 fires

    def regenerate(self, inputs, diff_matrices=None, prefill_only=True,
                   prefill_tail_len=1, max_new_tokens=128, temperature=0.0,
                   stop_strings=None, **kw):
        if diff_matrices is None:             # llms.py:821 behaviour
            raise ValueError("regenerate requires diff_matrices")
        self.saw_regenerate += 1
        self.tail_lens.append(prefill_tail_len)
        self._fires += self.layers * len(inputs) * prefill_tail_len
        return self._next(inputs[0])

    def steering_fire_count(self, reset=False):
        v = self._fires
        if reset:
            self._fires = 0
        return v


class NoCounterVC(FakeVC):
    """A stale wrapper: steering_fire_count returns None."""
    def steering_fire_count(self, reset=False):
        return None


class BrokenVC(FakeVC):
    def generate(self, inputs, **kw):
        raise RuntimeError("CUDA out of memory")


def _orders(seed=0):
    return p10.assign_orders(list(range(20)), 4)[seed]


def _run(script, alpha=0.0, vc_cls=FakeVC, **kw):
    vc = vc_cls(script)
    diff = None if alpha == 0 else [object()] * 32
    res = ep.run_pv10_episode(vc, seed=0, env=ENV, orders=_orders(),
                              diff_mtx=diff, alpha=alpha,
                              n_steered_layers=N_LAYERS, **kw)
    return vc, res


# ───────────────────────────── termination ──────────────────────────────────

def test_autonomous_commit():
    print("\n[autonomous commit]")
    # Sample twice, then commit.
    script = ["r\nPolicy: SAMPLE Button A",
              "r\nPolicy: SAMPLE Button B",
              "r\nPolicy: COMMIT Button C"]
    vc, res = _run(script)
    check(res["termination_reason"] == "autonomous_commit", "reason is autonomous")
    check(res["autonomous_commit"] is True, "autonomous_commit flag set")
    check(res["committed_arm"] == "C", "committed arm recorded")
    check(res["tau"] == 6, "tau = 4 forced + 2 adaptive samples")
    check(res["n_at_invalid"] is None, "no invalid exit time")
    check(res["model_calls"] == 3, "3 generations for 2 samples + 1 commit")
    check(res["commit_correct"] in (0, 1), "commit correctness is scored")
    check(len(res["forced_init"]) == 4, "forced init pulled each arm once")
    check(res["history"][:4] == list(_orders().initial_pull_order),
          "history opens in initial_pull_order")


def test_forced_commit():
    print("\n[forced commit at T_max]")
    def script(i, prompt):
        # Sample forever; the terminal prompt withdraws SAMPLE.
        if "No samples remain." in prompt:
            return "r\nPolicy: COMMIT Button A"
        return "r\nPolicy: SAMPLE Button A"
    vc, res = _run(script)
    check(res["termination_reason"] == "forced_commit", "reason is forced")
    check(res["autonomous_commit"] is False, "forced commit is NOT autonomous")
    check(res["tau"] == 100, "tau = T_max")
    check(res["n_samples_used"] == 100, "budget fully consumed")
    check(res["model_calls"] == 97,
          f"97 model calls (tau-3), got {res['model_calls']}")


def test_invalid_terminates():
    print("\n[invalid policy]")
    script = ["r\nPolicy: SAMPLE Button A",
              "I think Button B is good.",          # no policy line
              "r\nPolicy: COMMIT Button A"]         # never reached
    vc, res = _run(script)
    check(res["termination_reason"] == "invalid_policy", "reason is invalid")
    check(res["tau"] is None, "tau is NULL -- no fabricated stopping time")
    check(res["n_at_invalid"] == 5, "n_at_invalid is the competing-exit time")
    check(res["invalid_kind"] == "no_policy", "invalid_kind recorded")
    check(res["committed_arm"] is None and res["commit_correct"] is None,
          "no committed arm, no correctness")
    check(res["autonomous_commit"] is False, "invalid is not a commit")
    check(vc.calls == 2, "episode stopped -- no retry of the same state")


def test_infra_failure_is_not_invalid():
    print("\n[infra vs model failure]")
    try:
        _run(["x"], vc_cls=BrokenVC)
        check(False, "infra failure must raise")
    except ep.EpisodeInfrastructureError as e:
        check("CUDA out of memory" in str(e),
              "infra failure raises EpisodeInfrastructureError, not invalid")

    try:
        _run(["r\nPolicy: COMMIT Button A"], vc_cls=NoCounterVC)
        check(False, "missing fire counter must raise")
    except ep.EpisodeInfrastructureError as e:
        check("stale" in str(e).lower(),
              "None fire count is a FAILURE, not a skip")


# ───────────────────────────── steering ─────────────────────────────────────

def test_alpha_zero_registers_no_hook():
    print("\n[alpha = 0]")
    vc, res = _run(["r\nPolicy: COMMIT Button A"], alpha=0.0)
    check(vc.saw_generate == 1 and vc.saw_regenerate == 0,
          "alpha=0 uses generate(), which registers no hooks")
    check(res["steering_fires"] == 0, "alpha=0 gives exactly 0 fires")
    check(res["steered"] is False, "steered flag false")

    try:
        ep.run_pv10_episode(FakeVC(["x"]), seed=0, env=ENV, orders=_orders(),
                            diff_mtx=[object()], alpha=0.0)
        check(False, "alpha=0 with a diff matrix must raise")
    except ValueError as e:
        check("NO hook" in str(e),
              "alpha=0 refuses a diff matrix (zero-matrix != unsteered)")

    try:
        ep.run_pv10_episode(FakeVC(["x"]), seed=0, env=ENV, orders=_orders(),
                            diff_mtx=None, alpha=4.0)
        check(False, "alpha!=0 without a matrix must raise")
    except ValueError:
        check(True, "alpha!=0 requires a diff matrix")


def test_steered_fires_and_attestation():
    print("\n[steering attestation]")
    script = ["r\nPolicy: SAMPLE Button A", "r\nPolicy: COMMIT Button B"]
    vc, res = _run(script, alpha=4.0)
    check(vc.saw_regenerate == 2 and vc.saw_generate == 0,
          "alpha!=0 uses regenerate()")
    check(set(vc.tail_lens) == {1}, "prefill_tail_len is 1 (frozen)")
    check(res["steering_fires"] == 2 * N_LAYERS,
          f"2 calls x 9 layers = 18 fires, got {res['steering_fires']}")
    check(res["steering_fires"] == res["steering_fires_expected"],
          "observed matches expected")

    # A miscounting wrapper must fail closed, not be believed.
    vc = FakeVC(script, layers=32)      # e.g. zero rows wrongly counted
    try:
        ep.run_pv10_episode(vc, seed=0, env=ENV, orders=_orders(),
                            diff_mtx=[object()] * 32, alpha=4.0,
                            n_steered_layers=N_LAYERS)
        check(False, "fire mismatch must raise")
    except ep.EpisodeInfrastructureError as e:
        check("attestation FAILED" in str(e),
              "a wrong fire count fails closed")


def test_full_budget_fires():
    print("\n[full-budget fires = 873]")
    def script(i, prompt):
        if "No samples remain." in prompt:
            return "r\nPolicy: COMMIT Button A"
        return "r\nPolicy: SAMPLE Button A"
    vc, res = _run(script, alpha=-4.0)
    check(res["model_calls"] == 97, "97 model calls")
    check(res["steering_fires"] == 873,
          f"873 fires, NOT PV9's 900 (got {res['steering_fires']})")


# ───────────────────────── records / evidence ───────────────────────────────

def test_round_records():
    print("\n[per-round records]")
    script = ["r1\nPolicy: SAMPLE Button A", "r2\nPolicy: COMMIT Button B"]
    vc, res = _run(script, alpha=4.0)
    r0, r1 = res["rounds"]

    for f in ("decision_index", "n_pre", "n_post", "pre_counts",
              "empirical_leader", "empirical_challenger", "prompt_sha256",
              "prompt_token_count", "raw", "valid", "action", "arm",
              "format_exact", "trailing_period_tolerated",
              "native_ends_after_policy", "steering_fires_before",
              "steering_fires_after", "steering_fires_delta"):
        check(f in r0, f"round record carries {f}")

    check(r0["n_pre"] == 4 and r0["n_post"] == 5, "SAMPLE advances n by 1")
    check(r0["sampled_arm"] == "A" and r0["reward"] in (0, 1),
          "sampled arm and reward stored")
    check(r0["arm_pull_index"] == 1,
          "arm_pull_index is 1 (forced init was pull 0)")
    check(r1["n_pre"] == r1["n_post"] == 5, "COMMIT does not consume a sample")
    check(r1["sampled_arm"] is None, "COMMIT records no sampled arm")
    check(r0["steering_fires_delta"] == N_LAYERS, "per-round fire delta is 9")
    check(len({r["prompt_sha256"] for r in res["rounds"]}) == 2,
          "prompt hashes differ across rounds (state advanced)")
    check(r0["pre_counts"]["A"] == [
        sum(1 for x in res["forced_init"] if x["arm"] == "A" and x["reward"]), 1],
        "pre-decision counts include the forced init")


def test_commit_evidence():
    print("\n[commit evidence]")
    # Commit immediately: the arm has only its forced-init pull.
    vc, res = _run(["r\nPolicy: COMMIT Button A"])
    ce = res["commit_evidence"]
    check(ce["committed_arm_trials"] == 1, "committed arm has 1 trial")
    check(ce["committed_arm_adaptive_trials"] == 0,
          "0 ADAPTIVE trials -- a valid premature commitment, not blocked")
    check(res["termination_reason"] == "autonomous_commit",
          "committing to a barely-sampled arm is still a valid commit")
    check("committed_is_empirical_leader" in ce, "leader/challenger flags stored")

    vc, res = _run(["nope"])
    check(res["commit_evidence"] is None, "invalid episode has no commit evidence")


def test_episode_provenance():
    print("\n[episode provenance]")
    vc, res = _run(["r\nPolicy: COMMIT Button A"], interface_tag="tag123")
    for f in ("protocol_version", "policy_parser_version", "order_version",
              "interface_tag", "seed", "alpha", "environment", "display_order",
              "initial_pull_order", "tape_id", "true_best", "final_counts",
              "history", "model_calls"):
        check(f in res, f"episode carries {f}")
    check(res["interface_tag"] == "tag123", "interface tag propagates")
    check(res["true_best"] in ("A", "B", "C", "D"), "true best is a bare letter")
    check(sum(t for _, t in res["final_counts"].values())
          == res["n_samples_used"], "final counts reconcile with samples used")


def test_terminal_prompt_used_at_budget():
    print("\n[terminal prompt]")
    def script(i, prompt):
        if "No samples remain." in prompt:
            return "r\nPolicy: COMMIT Button A"
        return "r\nPolicy: SAMPLE Button A"
    vc, res = _run(script)
    check(sum("No samples remain." in p for p in vc.prompts) == 1,
          "the terminal prompt is used exactly once")
    check(all(p.endswith(p10.REASON_ANCHOR) for p in vc.prompts),
          "EVERY prompt ends at the Reason anchor, terminal included")


def test_sample_at_terminal_is_invalid():
    print("\n[SAMPLE at terminal]")
    def script(i, prompt):
        return "r\nPolicy: SAMPLE Button A"      # never commits
    vc, res = _run(script)
    check(res["termination_reason"] == "invalid_policy",
          "SAMPLE at the terminal prompt is invalid, not silently forced")
    check(res["invalid_kind"] == "sample_at_terminal", "invalid_kind is specific")
    check(res["tau"] is None, "no fabricated tau")
    check(res["n_at_invalid"] == 100, "exits at the budget boundary")


def main():
    test_autonomous_commit()
    test_forced_commit()
    test_invalid_terminates()
    test_infra_failure_is_not_invalid()
    test_alpha_zero_registers_no_hook()
    test_steered_fires_and_attestation()
    test_full_budget_fires()
    test_round_records()
    test_commit_evidence()
    test_episode_provenance()
    test_terminal_prompt_used_at_budget()
    test_sample_at_terminal_is_invalid()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("all PV10 episode invariants hold")


if __name__ == "__main__":
    main()
