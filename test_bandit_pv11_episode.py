#!/usr/bin/env python3.10
"""PV11 episode-runner invariants via a contract-faithful FakeVC. No GPU.

The fake mirrors the REAL VicundaModel contract on purpose: `generate` takes no
`diff_matrices`, `regenerate` raises when `diff_matrices` is None (llms.py:821),
and `steering_fire_count` is unconditional. A fake more permissive than the
real API hides exactly the bugs these tests exist to catch.

A FakeVC models INTENT, not arithmetic: it reimplements the site formula rather
than executing llms.py, so it will happily agree with a wrong counter. That is
why the fire checks here assert the RELATION (calls x L, and 0 when unsteered)
rather than trusting a number the fake produced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/paveenhuang/Downloads/Dopamine")

import bandit_pv11 as p11
import bandit_pv11_episode as ep

HERE = Path("/Users/paveenhuang/Downloads/Dopamine")
FAILURES: list[str] = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


class FakeVC:
    """Scripted replies. Records every prompt for byte-comparison."""

    def __init__(self, replies, n_layers=9):
        self.replies = list(replies)
        self.n_layers = n_layers
        self.prompts = []
        self._pending = 0
        self.calls = []

    def _next(self, prompt):
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("FakeVC ran out of scripted replies")
        return [self.replies.pop(0)]

    def generate(self, inputs, max_new_tokens=None, temperature=None,
                 stop_strings=None, **kw):
        if "diff_matrices" in kw:
            raise AssertionError("generate() does not accept diff_matrices")
        self.calls.append(("generate", stop_strings))
        self._pending = 0                       # no hooks registered at all
        return self._next(inputs[0])

    def regenerate(self, inputs, diff_matrices=None, prefill_only=False,
                   prefill_tail_len=1, max_new_tokens=None, temperature=None,
                   stop_strings=None, **kw):
        if diff_matrices is None:
            raise ValueError("regenerate requires diff_matrices")  # llms.py
        self.calls.append(("regenerate", stop_strings))
        self._pending = self.n_layers * len(inputs) * prefill_tail_len
        return self._next(inputs[0])

    def steering_fire_count(self, reset=False):
        v = self._pending
        if reset:
            self._pending = 0
        return v


def _state(uid):
    bank = json.loads((HERE / "pv11_state_bank.json").read_text())
    return [s for s in bank["states"] if s["state_uid"] == uid][0]


def R(action, arm):
    return f"the evidence is mixed.\nPolicy: {action} Button {arm}"


def main() -> int:
    st = _state("A-low_n-low_rate-00")           # H=20, probe A
    comm = _state("C-strong-short-00")           # H=5

    # ── 1. immediate COMMIT: one call, no forced init ───────────────────────
    vc = FakeVC([R("COMMIT", "B")])
    r = ep.run_pv11_episode(vc, st)
    check(r["secondary_trajectory"]["model_calls"] == 1,
          "immediate commit made more than one model call")
    check(r["secondary_trajectory"]["n_samples"] == 0,
          "a forced initialization pull was taken")
    check(r["opening_counts"] == {a: list(st["displayed_counts"][a])
                                  for a in st["display_order"]},
          "opening counts are not the synthetic state")
    check(r["first_action"]["kind"] == "commit", "first_action kind wrong")
    check(r["secondary_trajectory"]["termination_reason"]
          == "autonomous_commit", "termination not autonomous_commit")

    # ── 2. first_action is captured BEFORE any state update ────────────────
    vc = FakeVC([R("SAMPLE", "A"), R("SAMPLE", "A"), R("COMMIT", "A")])
    r = ep.run_pv11_episode(vc, st)
    fa = r["first_action"]
    check(fa["kind"] == "sample" and fa["arm"] == "A",
          "first_action did not record the opening SAMPLE")
    check(fa["is_probe"] is True, "probe not flagged in first_action")
    check(fa["is_displayed_best"] is False,
          "probe wrongly flagged as the displayed best (B is, at .55)")
    check(fa["is_true_best"] is True,
          "probe is the true best in this state and was not flagged")
    # The opening prompt must be the state's, unaffected by later rounds.
    check(fa["prompt_sha256"] ==
          __import__("hashlib").sha256(vc.prompts[0].encode()).hexdigest(),
          "first_action prompt hash is not the opening prompt")

    # ── 3. rewards are REAL, tape-derived, and update the counts ───────────
    sec = r["secondary_trajectory"]
    check(sec["n_samples"] == 2, "sample count wrong")
    rewards = [x["reward"] for x in sec["rounds"] if x["action"] == "SAMPLE"]
    check(all(x in (0, 1) for x in rewards), "reward not Bernoulli")
    s0, t0 = st["displayed_counts"]["A"]
    check(sec["final_counts"]["A"] == [s0 + sum(rewards), t0 + 2],
          f"counts not updated by real rewards: {sec['final_counts']['A']}")
    # pull_index restarts at 0 after the synthetic state
    idx = [x["pull_index"] for x in sec["rounds"] if x["action"] == "SAMPLE"]
    check(idx == [0, 1], f"pull_index does not restart at 0: {idx}")

    # ── 4. tape pairs across the four cells and across alpha ───────────────
    want = ep.tape_reward("A-00", "A", 0, 0.6)
    for cell in ("low_n/high_rate", "matched_n/low_rate",
                 "matched_n/high_rate"):
        uid = "A-" + cell.replace("/", "-") + "-00"
        other = _state(uid)
        check(other["tape_key"] == st["tape_key"],
              f"{uid}: tape_key differs across cells")
        check(ep.tape_reward(other["tape_key"], "A", 0,
                             other["latent_probs"]["A"]) == want,
              f"{uid}: first pull of A differs across cells")
    # determinism across processes: recompute, must be identical
    check(ep.tape_reward("A-00", "A", 0, 0.6) == want, "tape not deterministic")

    # `prob` must NOT enter the key. The four cells of a state_id share one
    # latent, so comparing them cannot detect a prob-keyed tape (verified by
    # mutation: adding prob to the key left that check green). The property is
    # that the underlying UNIFORM is a function of the tape position alone, so
    # the draw must be monotone in prob at a FIXED position -- true for a
    # shared u, false if prob is hashed into the key.
    for key, arm in (("A-00", "A"), ("C-07", "B"), ("A-13", "D")):
        for idx in range(6):
            seq = [ep.tape_reward(key, arm, idx, p / 100.0)
                   for p in range(0, 101, 5)]
            check(seq == sorted(seq),
                  f"tape draw is not monotone in prob at {key}/{arm}/{idx} -- "
                  f"prob has leaked into the hash key")
            check(seq[0] == 0 and seq[-1] == 1,
                  f"tape draw at {key}/{arm}/{idx} is not p=0 -> 0, p=1 -> 1")

    # ── 5. invalid TERMINATES and stays in the denominator ─────────────────
    vc = FakeVC(["Reason: x\nPolicy: COMMIT Button A because it looks best"])
    r = ep.run_pv11_episode(vc, st)
    check(r["first_action"]["kind"] == "invalid",
          "prose-after-arm was not recorded as invalid")
    check(r["secondary_trajectory"]["termination_reason"] == "invalid_policy",
          "invalid did not terminate the episode")
    check(r["secondary_trajectory"]["model_calls"] == 1,
          "episode continued after an invalid policy")
    check(r["secondary_trajectory"]["committed_arm"] is None,
          "an invalid episode recorded a commit")

    # ── 6. H=0 forces a terminal prompt; SAMPLE there is INVALID ───────────
    vc = FakeVC([R("SAMPLE", "A")] * 5 + [R("SAMPLE", "A")])
    r = ep.run_pv11_episode(vc, comm)             # H=5
    sec = r["secondary_trajectory"]
    check(sec["n_samples"] == 5, f"took {sec['n_samples']} samples, not 5")
    check(sec["rounds"][-1]["terminal_prompt"] is True,
          "the last call did not use the terminal prompt")
    check(sec["termination_reason"] == "invalid_policy",
          "SAMPLE at a terminal decision was not judged invalid")
    check(sec["rounds"][-1]["valid"] is False, "terminal SAMPLE marked valid")

    # ── 7. H=0 COMMIT is a forced_commit ───────────────────────────────────
    vc = FakeVC([R("SAMPLE", "A")] * 5 + [R("COMMIT", "A")])
    r = ep.run_pv11_episode(vc, comm)
    check(r["secondary_trajectory"]["termination_reason"] == "forced_commit",
          "exhausting the horizon then committing is not forced_commit")

    # ── 8. alpha=0 registers NO hook; fires must be exactly 0 ──────────────
    vc = FakeVC([R("COMMIT", "B")])
    r = ep.run_pv11_episode(vc, st, alpha=0.0, diff_mtx=None)
    check(r["attestation"]["steering_fires"] == 0,
          "alpha=0 recorded non-zero fires")
    check(vc.calls[0][0] == "generate",
          "alpha=0 did not take the no-hook generate path")
    check(vc.calls[0][1] == ["#"], "alpha=0 lost the stop marker")

    # ── 9. steered fires == model_calls * L, not the ceiling ──────────────
    vc = FakeVC([R("SAMPLE", "A"), R("COMMIT", "A")])
    r = ep.run_pv11_episode(vc, st, alpha=4.0, diff_mtx=[[0.0]])
    att = r["attestation"]
    check(att["model_calls"] == 2, "wrong call count")
    check(att["steering_fires"] == 2 * 9,
          f"fires {att['steering_fires']} != calls*L")
    check(att["ceiling_fires"] == 21 * 9, "ceiling is not (H+1)*L")
    check(att["steering_fires"] < att["ceiling_fires"],
          "an early-committing episode was attested against the ceiling")
    check(all(c[0] == "regenerate" for c in vc.calls),
          "a steered episode used the generate path")
    check(all(c[1] == ["#"] for c in vc.calls),
          "steered path lost the stop marker (PV10-A v1 stop-parity bug)")

    # ── 9b. attestation must REJECT a wrong count ─────────────────────────
    # Mutation-checked: with the attestation disabled, groups 1-9 all still
    # pass, because FakeVC computes fires with the same formula the runner
    # verifies. A fake that models INTENT cannot test ARITHMETIC. These fakes
    # report a WRONG number, so only a live attestation catches them.
    class MiscountVC(FakeVC):
        """Counts hook CALLS, not sites -- the bug pv6 actually shipped."""

        def regenerate(self, inputs, **kw):
            out = FakeVC.regenerate(self, inputs, **kw)
            self._pending = 1                    # one call, regardless of L
            return out

    try:
        ep.run_pv11_episode(MiscountVC([R("COMMIT", "B")]), st,
                            alpha=4.0, diff_mtx=[[0.0]])
        check(False, "attestation accepted a call-counter (1 != L)")
    except ep.EpisodeInfrastructureError:
        pass

    class ZeroRowVC(FakeVC):
        """Counts zero-rows too: 32 layers instead of the steered 9."""

        def regenerate(self, inputs, **kw):
            out = FakeVC.regenerate(self, inputs, **kw)
            self._pending = 32
            return out

    try:
        ep.run_pv11_episode(ZeroRowVC([R("COMMIT", "B")]), st,
                            alpha=4.0, diff_mtx=[[0.0]])
        check(False, "attestation accepted 32 sites where 9 layers are steered")
    except ep.EpisodeInfrastructureError:
        pass

    class SilentVC(FakeVC):
        """Never fires: the hook was never registered."""

        def regenerate(self, inputs, **kw):
            out = FakeVC.regenerate(self, inputs, **kw)
            self._pending = 0
            return out

    try:
        ep.run_pv11_episode(SilentVC([R("COMMIT", "B")]), st,
                            alpha=4.0, diff_mtx=[[0.0]])
        check(False, "attestation accepted a steered episode with 0 fires")
    except ep.EpisodeInfrastructureError:
        pass

    # An UNSTEERED episode that somehow fires must also be rejected.
    class LeakVC(FakeVC):
        def generate(self, inputs, **kw):
            out = FakeVC.generate(self, inputs, **kw)
            self._pending = 9
            return out

    try:
        ep.run_pv11_episode(LeakVC([R("COMMIT", "B")]), st)
        check(False, "attestation accepted fires in an alpha=0 episode")
    except ep.EpisodeInfrastructureError:
        pass

    # ── 10. stop parity: both paths pass identical stop_strings ────────────
    v0 = FakeVC([R("COMMIT", "B")])
    ep.run_pv11_episode(v0, st)
    v4 = FakeVC([R("COMMIT", "B")])
    ep.run_pv11_episode(v4, st, alpha=4.0, diff_mtx=[[0.0]])
    check(v0.calls[0][1] == v4.calls[0][1],
          "alpha=0 and alpha=4 were given different stop_strings")

    # ── 11. alpha/diff_mtx pairing is enforced both ways ───────────────────
    for kw, label in (({"alpha": 4.0}, "alpha!=0 without diff_mtx"),
                      ({"alpha": 0.0, "diff_mtx": [[0.0]]},
                       "alpha=0 with a diff_mtx")):
        try:
            ep.run_pv11_episode(FakeVC([R("COMMIT", "B")]), st, **kw)
            check(False, f"{label} did not raise")
        except ValueError:
            pass

    # ── 12. infra failures are NOT model invalids ─────────────────────────
    class Dead(FakeVC):
        def generate(self, **kw):
            raise RuntimeError("CUDA is unhappy")

    try:
        ep.run_pv11_episode(Dead([]), st)
        check(False, "an infrastructure exception was swallowed")
    except ep.EpisodeInfrastructureError:
        pass

    class Stale(FakeVC):
        def steering_fire_count(self, reset=False):
            return None

    try:
        ep.run_pv11_episode(Stale([R("COMMIT", "B")]), st)
        check(False, "a missing fire count was treated as a skip")
    except ep.EpisodeInfrastructureError:
        pass

    # ── 13. the opening prompt is byte-identical across the four cells ─────
    # except for the probe's OPTION line -- the manipulation itself.
    opens = {}
    for cell in ("low_n-low_rate", "low_n-high_rate",
                 "matched_n-low_rate", "matched_n-high_rate"):
        s = _state(f"A-{cell}-00")
        vc = FakeVC([R("COMMIT", "B")])
        ep.run_pv11_episode(vc, s)
        opens[cell] = vc.prompts[0]
    lines = [set(p.splitlines()) for p in opens.values()]
    common = set.intersection(*lines)
    diff = [sorted(set(p.splitlines()) - common) for p in opens.values()]
    check(all(len(d) == 1 and d[0].startswith("- Button A:") for d in diff),
          f"cells differ outside the probe OPTION line: {diff}")

    # ── 14. the same state under different alpha opens identically ────────
    a = FakeVC([R("COMMIT", "B")])
    ep.run_pv11_episode(a, st)
    b = FakeVC([R("COMMIT", "B")])
    ep.run_pv11_episode(b, st, alpha=-4.0, diff_mtx=[[0.0]])
    check(a.prompts[0] == b.prompts[0],
          "the opening prompt differs across alpha cells")

    if FAILURES:
        print(f"FAIL ({len(FAILURES)})")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ok  test_bandit_pv11_episode.py  (14 groups)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
