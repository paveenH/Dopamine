#!/usr/bin/env python3
"""pv7 episode-runner checks. Standalone; exits non-zero on failure.

The FakeVC mirrors the REAL VicundaModel contract deliberately, following
`test_bandit_pv6_episode.py`: `generate()` accepts no diff_matrices, and
`regenerate()` raises on diff_matrices=None exactly like llms.py:821. A fake
more permissive than the real API hides the bugs these tests exist to catch.

It uses the REAL Llama-3.1 tokenizer (local HF cache), because every token
invariant under test -- tail 220, candidates 32-35, ID-level concatenation --
is a property of that tokenizer and cannot be faked.

CAVEAT INHERITED FROM pv6: a FakeVC models INTENT, not arithmetic. The site
counter here reimplements the formula rather than executing llms.py hooks, so
it will happily agree with a wrong counter. It verifies that the runner asks
for the right thing, never that llms.py computes it correctly.
"""

from __future__ import annotations

import sys

import bandit_reference as br
import bandit_pv7 as p7
import bandit_pv7_episode as ep

FAILS: list[str] = []


def check(cond, label):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILS.append(label)


class FakeVC:
    """Contract-faithful stand-in. Scores are deterministic and arbitrary."""

    def __init__(self, tokenizer, reply="Evidence: B looks weak.\n"
                                        "Policy: EXPLORE Button B because untried.",
                 n_layers=9):
        self.tokenizer = tokenizer
        self.reply = reply
        self.n_layers = n_layers
        self._fires = 0
        self.rationale_prompts: list[str] = []
        self.action_prompts: list[str] = []
        self.generate_calls = 0
        self.regenerate_calls = 0

    # -- real API shape: generate takes NO diff_matrices ---------------------
    def generate(self, inputs, max_new_tokens=64, temperature=0.0):
        self.generate_calls += 1
        self.rationale_prompts.append(inputs[0])
        return [self.reply]

    def regenerate(self, inputs, diff_matrices, prefill_only=False,
                   prefill_tail_len=1, max_new_tokens=64, temperature=0.0):
        if diff_matrices is None:                      # llms.py:821 raises
            raise ValueError("regenerate requires diff_matrices")
        self.regenerate_calls += 1
        self.rationale_prompts.append(inputs[0])
        self._fires += self.n_layers * len(inputs) * prefill_tail_len
        return [self.reply]

    def regenerate_logits_teacher_forcing(self, prompts, answer_token_ids,
                                          diff_matrices):
        import numpy as np
        self.action_prompts.append(prompts[0])
        if diff_matrices is not None:
            self._fires += self.n_layers * len(prompts)
        V = 128256
        out = []
        for i, ids in enumerate(answer_token_ids):
            lg = np.zeros((len(ids), V), dtype=np.float32)
            for k, tid in enumerate(ids):
                lg[k][tid] = 10.0 - i        # deterministic: first arm wins
            out.append(lg)
        return out

    def steering_fire_count(self, reset=False):
        n = self._fires
        if reset:
            self._fires = 0
        return n


def main() -> int:
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    except Exception as exc:                                    # noqa: BLE001
        print(f"cannot load the Llama-3.1 tokenizer ({exc}); these checks are "
              "about real token ids and cannot run without it")
        return 2

    env = br.get_environment("easy")
    T = 6                                    # short horizon; shape not scale
    env = br._replace_horizon(env, T) if hasattr(br, "_replace_horizon") else env
    horizon = env.horizon

    print("[1] pv6 is untouched and pv7 is independent")
    # The real property is that importing the runner does not DRAG IN pv6 --
    # grepping the source would pass on a module that imports it lazily and
    # fail on one that only names it in a comment.
    import subprocess
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys, bandit_pv7_episode; "
         "print('bandit_pv6_episode' in sys.modules)"],
        capture_output=True, text=True)
    check(probe.stdout.strip() == "False",
          "importing the runner does not import bandit_pv6_episode")
    check(ep.PROTOCOL_VERSION == "pv7", "protocol version is pv7")

    print("\n[2] both stages end at token 220 and use bare candidates")
    vc = FakeVC(tok)
    rec = ep.run_pv7_episode(vc, None, seed=0, env=env, attest=True)
    for stage, prompts in (("rationale", vc.rationale_prompts),
                           ("action", vc.action_prompts)):
        tails = {tok.encode(p, add_special_tokens=False)[-1] for p in prompts}
        check(tails == {220}, f"{stage} prompts all end at token 220")
    sfx = p7.candidate_suffixes(env)
    check([tok.encode(s, add_special_tokens=False) for s in sfx]
          == [[32], [33], [34], [35]], "candidates are bare tokens 32-35")

    print("\n[3] both stages see the SAME fixed OPTIONS order (no rotation)")
    def options_block(p):
        body = p.split("OPTIONS\n", 1)[1]
        return [l for l in body.splitlines() if l.startswith("- ")]
    same = all(options_block(r) == options_block(a) for r, a in
               zip(vc.rationale_prompts, vc.action_prompts))
    check(same, "Stage 1 and Stage 2 render identical OPTIONS rows")
    first = options_block(vc.action_prompts[0])
    order_stable = all([l.split(":")[0] for l in options_block(a)]
                       == [l.split(":")[0] for l in first]
                       for a in vc.action_prompts)
    check(order_stable, "the row order never changes within an episode")

    print("\n[4] Stage 2 carries the S1 instruction, Stage 1 does not")
    check(all(ep._ACTION_INSTRUCTION in a for a in vc.action_prompts),
          "every action prompt contains the S1 instruction")
    check(not any(ep._ACTION_INSTRUCTION in r for r in vc.rationale_prompts),
          "no rationale prompt contains it")

    print("\n[5] alpha=0 registers NO hook in either stage")
    check(vc.regenerate_calls == 0, "alpha=0 never calls regenerate for Stage 1")
    check(vc.generate_calls == horizon, "alpha=0 uses generate for every round")
    check(rec["steering_fires"] == {"rationale": 0, "action": 0},
          "alpha=0 fires no injection sites")
    check(rec["steered_rationale"] is False and rec["steered_action"] is False,
          "alpha=0 records both stages as unsteered")

    print("\n[6] site counts match the acceptance formula")
    K, L = env.k, 9
    diff = [None] * 32
    v2 = FakeVC(tok, n_layers=L)
    r2 = ep.run_pv7_episode(v2, diff, seed=0, env=env, rationale_alpha=4.0)
    check(r2["steering_fires"] == {"rationale": L * horizon, "action": 0},
          f"rationale-only -> {{{L * horizon}, 0}}")
    check(v2.regenerate_calls == horizon and v2.generate_calls == 0,
          "a steered Stage 1 goes through regenerate, not generate")
    v3 = FakeVC(tok, n_layers=L)
    r3 = ep.run_pv7_episode(v3, diff, seed=0, env=env, action_alpha=4.0)
    check(r3["steering_fires"] == {"rationale": 0, "action": L * K * horizon},
          f"action-only -> {{0, {L * K * horizon}}}")
    check(ep.expected_fires(4.0, 0.0, L, K, horizon)
          == {"rationale": L * horizon, "action": 0},
          "expected_fires agrees with the observed rationale-only count")

    print("\n[7] the gate's read-only contract is satisfied additively")
    for f in ("choices", "feedbacks", "arm_map", "best_arm", "seed",
              "environment"):
        check(f in rec, f"record carries `{f}` for evaluate_competence_gate")
    check(len(rec["choices"]) == horizon, "trajectory length == horizon")
    gate_src = open("evaluate_competence_gate.py").read()
    check("bandit_pv7" not in gate_src, "the frozen gate evaluator is unmodified")

    print("\n[8] decision quality and execution consistency are separable")
    r0 = rec["rounds"][0]
    for f in ("policy_stance", "policy_target", "policy_parsed",
              "action_follows_policy", "action", "reward", "candidate_scores",
              "margin", "norm_entropy", "rationale_raw", "rationale_clean"):
        check(f in r0, f"per-round record carries `{f}`")
    check("policy_parse_rate" in rec and "action_follows_policy_rate" in rec,
          "episode summarises Stage 1 quality and Stage 2 consistency apart")

    print("\n[9] the parser records intent but never picks the action")
    # The fake scores ALWAYS make the first-listed arm win, while every
    # rationale names Button B. If the parser were choosing, actions would be
    # B; they must instead follow the scores.
    first_arm = rec["arm_order"][0]
    check(all(r["action"] == first_arm for r in rec["rounds"]),
          "action comes from candidate scoring, not from policy_target")
    check(all(r["policy_target"] == "Button B" for r in rec["rounds"]),
          "policy_target is still recorded")
    check(any(r["action_follows_policy"] is False for r in rec["rounds"])
          or first_arm == "Button B",
          "action_follows_policy can be False -- it is not a tautology")

    print("\n[10] two different non-zero alphas are rejected, not averaged")
    try:
        ep.run_pv7_episode(FakeVC(tok), diff, seed=0, env=env,
                           rationale_alpha=4.0, action_alpha=-4.0)
        check(False, "mixed non-zero alphas must raise")
    except ValueError:
        check(True, "mixed non-zero alphas raise instead of silently reusing")

    print("\n[11] resume key separates every behaviour-affecting version")
    k0 = ep.resume_key("easy", 0, 0, 11, 20, 20)
    check(ep.STAGE1_INSTRUCTION_VERSION in k0
          and ep.STAGE2_INSTRUCTION_VERSION in k0,
          "both instruction versions appear in the resume key")
    check(k0 != ep.resume_key("easy", 4, 0, 11, 20, 20),
          "a different rationale_alpha yields a different key")
    check(ep.resume_key("easy", 4, 0, 11, 20, 20)
          != ep.resume_key("easy", 0, 4, 11, 20, 20),
          "rationale-only and action-only do not collide")

    print("\n[12] a mis-anchored prompt stops the run")
    class BadVC(FakeVC):
        pass
    bad = BadVC(tok)
    try:
        ep.audit_pv7_prompt(bad, "no trailing space here", "action")
        check(False, "a prompt not ending at token 220 must raise")
    except AssertionError:
        check(True, "a prompt not ending at token 220 raises")

    print("\n" + ("ALL PV7 EPISODE CHECKS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILED: " + "; ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
