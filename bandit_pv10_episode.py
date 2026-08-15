#!/usr/bin/env python3.10
"""PV10-B episode runner: self-paced BAI with subjective commitment.

One episode = one seed. The environment force-samples each arm once, then the
model repeatedly either SAMPLEs an arm or COMMITs to one, in a SINGLE generation
per decision. No Stage 2 (see bandit_pv10 for why), so the effect measured here
is a JOINT REASONING-DECISION EFFECT -- alpha is injected at `Reason: ` and the
same generation produces the action.

THREE TERMINATION REASONS, mutually exclusive:

    autonomous_commit  the model committed on its own. tau is a real stopping
                       time and enters the accuracy-sample tradeoff.
    forced_commit      it sampled to T_max and the terminal prompt required a
                       commit. Censored -- NOT an autonomous stopping time.
    invalid_policy     a generation could not be parsed. The episode ENDS:
                       tau is null, n_at_invalid records where it exited, and
                       survival analysis treats it as a COMPETING EXIT. It is
                       never budget censoring and never gets a fabricated tau.

TWO-TIER FAILURE SEPARATION, load-bearing:

    a MODEL failure   the generation succeeded but its content does not parse
                      -> invalid_policy, a real interface-validity outcome that
                      may itself be an alpha effect
    an INFRA failure  server exception, empty return, interrupted inference
                      -> EpisodeInfrastructureError, the runner fails closed and
                      resume re-runs the cell

Conflating them would let a flaky GPU masquerade as a model behaviour, and would
put an infrastructure artifact into the invalid-rate table that the capability
check reads.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

import bandit_pv10 as p10
from bandit_reference import Environment, RewardTape

try:                                  # torch is absent on the analysis box
    import torch
except Exception:                     # pragma: no cover
    torch = None


class EpisodeInfrastructureError(RuntimeError):
    """Generation itself failed. NOT a model invalid -- see the module docstring.

    Raised so the runner fails closed and the cell is re-run on resume, rather
    than recording an infrastructure artifact as model behaviour.
    """


# ─────────────────────────────── helpers ────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _letter(name: str) -> str:
    """bandit_reference keys arms as 'Button A'; PV10 uses the bare letter."""
    return name.rsplit(" ", 1)[-1]


def _empirical_leader(counts: dict[str, tuple[int, int]],
                      display_order) -> str:
    """Highest empirical rate, display-order tie-break.

    Tie-tolerant scan in display order, matching evaluate_competence_gate.py:115
    and the frozen prescreen convention. A bare argmax over a dict would return
    the first-inserted key instead, which is a different rule.
    """
    best, best_val = None, -1.0
    for a in display_order:
        s, t = counts[a]
        v = s / t if t else 0.0
        if v > best_val:
            best, best_val = a, v
    return best


def _empirical_challenger(counts, display_order, leader) -> str:
    rest = [a for a in display_order if a != leader]
    return _empirical_leader({a: counts[a] for a in rest}, rest)


def _audit(vc, prompt: str) -> dict:
    """Assert the prompt ends on the injection token. Fails closed."""
    tok = getattr(vc, "tokenizer", None)
    if tok is None:
        raise EpisodeInfrastructureError("model wrapper exposes no tokenizer")
    return p10.audit_prompt_tokens(tok, prompt)


# ─────────────────────────────── episode ────────────────────────────────────

def run_pv10_episode(
    vc,
    seed: int,
    env: Environment,
    orders: p10.EpisodeOrders,
    diff_mtx=None,
    alpha: float = 0.0,
    total_budget: int = p10.TOTAL_BUDGET,
    max_new_tokens: int = p10.RATIONALE_MAX_TOKENS,
    n_steered_layers: int = 9,
    interface_tag: str = "",
    prompt_module=p10,
) -> dict:
    """Run one PV10-B episode.

    `prompt_module` selects the protocol's prompt surface. It defaults to
    bandit_pv10 (PV10-B) so every existing caller is byte-identical; PV10-A
    passes bandit_pv10a, which withholds COMMIT until the budget is spent and
    then delegates to THIS module's frozen terminal prompt. Only the prompt
    varies -- parser, anchor, steering, tapes and records are shared, which is
    what makes an A-vs-B difference attributable to the stopping decision.

    `orders` must come from the frozen bank-level `assign_orders`, never from a
    per-seed `make_orders` -- see that function's binding-consequence note.

    alpha == 0 must leave `diff_mtx` None: with no matrix `vc.generate` is
    called, which registers no hooks at all. "Unsteered" is deliberately a
    different code path from "steered by zero", and that is what makes an
    alpha=0 cell reusable.
    """
    steered = alpha != 0
    if steered and diff_mtx is None:
        raise ValueError(f"alpha={alpha} requires diff_matrices")
    if not steered and diff_mtx is not None:
        raise ValueError(
            "alpha=0 must pass diff_mtx=None so NO hook is registered; "
            "a zero matrix would still register hooks and is not the same")

    k = env.k
    display_order = list(orders.display_order)
    if len(display_order) != k:
        raise ValueError(f"orders cover {len(display_order)} arms, env has {k}")

    # bandit_reference keys the tape by "Button X"; translate at this boundary.
    tape = RewardTape(seed, env, length=total_budget)
    tape_arm_map = tape.arm_map
    letter_to_full = {_letter(n): n for n in tape_arm_map}
    true_best = _letter(max(tape_arm_map, key=tape_arm_map.get))

    counts: dict[str, tuple[int, int]] = {a: (0, 0) for a in display_order}
    history: list[str] = []
    per_round: list[dict] = []

    def pull(arm: str) -> tuple[int, int]:
        """Consume the tape. Returns (reward, 0-based pull index for this arm)."""
        idx = counts[arm][1]
        r = tape.pull(letter_to_full[arm])
        s, t = counts[arm]
        counts[arm] = (s + r, t + 1)
        history.append(arm)
        return r, idx

    # ---- forced initialization: consumes samples, but NO generation ---------
    init_records = []
    for arm in orders.initial_pull_order:
        for _ in range(p10.FORCED_INIT_PER_ARM):
            r, idx = pull(arm)
            init_records.append({"arm": arm, "reward": r, "arm_pull_index": idx})
    n = len(history)

    fires_total = 0
    can_count = hasattr(vc, "steering_fire_count")
    if can_count:
        vc.steering_fire_count(reset=True)

    termination = None
    committed_arm = None
    tau = None
    n_at_invalid = None
    invalid_kind = None
    decision_index = 0

    while termination is None:
        terminal = (n >= total_budget)
        prompt = prompt_module.build_decision_prompt(
            display_order, counts, history, n=n, total_budget=total_budget)
        audit = _audit(vc, prompt)

        leader = _empirical_leader(counts, display_order)
        challenger = _empirical_challenger(counts, display_order, leader)
        pre_counts = {a: counts[a] for a in display_order}

        fires_before = fires_total
        if torch is not None:
            # Sampling luck is matched across alpha cells at the same decision
            # index, exactly as pv7/pv9 seed per (seed, round).
            torch.manual_seed(seed * 100_003 + decision_index)

        try:
            if steered:
                out = vc.regenerate(
                    inputs=[prompt], diff_matrices=diff_mtx,
                    prefill_only=True, prefill_tail_len=1,
                    max_new_tokens=max_new_tokens, temperature=0.0,
                    stop_strings=list(p10.STOP_STRINGS))
            else:
                out = vc.generate(inputs=[prompt],
                                  max_new_tokens=max_new_tokens,
                                  temperature=0.0)
        except Exception as e:      # INFRA, not model behaviour
            raise EpisodeInfrastructureError(
                f"generation failed at seed={seed} decision={decision_index} "
                f"n={n}: {type(e).__name__}: {e}") from e

        raw = out[0] if isinstance(out, list) else out
        if raw is None:
            raise EpisodeInfrastructureError(
                f"generation returned None at seed={seed} n={n}")

        fired = vc.steering_fire_count(reset=True) if can_count else None
        if fired is None:
            # steering_fire_count is unconditional in llms.py; a missing count
            # means a stale server llms.py. Sync it -- do not add a bypass.
            raise EpisodeInfrastructureError(
                "steering_fire_count unavailable: the model wrapper is stale, "
                "so injection cannot be attested")
        fires_total += fired

        pol = p10.parse_policy(raw, display_order, terminal=terminal)

        rec = {
            "decision_index": decision_index,
            "model_call_index": decision_index,
            "n_pre": n,
            "terminal_prompt": terminal,
            "pre_counts": {a: list(pre_counts[a]) for a in display_order},
            "history_len_pre": len(history),
            "empirical_leader": leader,
            "empirical_challenger": challenger,
            "prompt_sha256": _sha256(prompt),
            "prompt_token_count": audit["n_tokens"],
            "prompt_tail_token_id": audit["tail_token_id"],
            "interface_tag": interface_tag,
            "raw": raw,
            "generation_char_count": len(raw),
            "generation_hit_stop_marker": any(s in raw for s in p10.STOP_STRINGS),
            "valid": pol.valid,
            "action": pol.action,
            "arm": pol.arm,
            "reason": pol.reason,
            "invalid_kind": pol.invalid_kind,
            "n_policy_lines": pol.n_policy_lines,
            "format_exact": pol.format_exact,
            "trailing_period_tolerated": pol.trailing_period_tolerated,
            "native_ends_after_policy": pol.native_ends_after_policy,
            "steering_fires_before": fires_before,
            "steering_fires_after": fires_total,
            "steering_fires_delta": fired,
        }

        if not pol.valid:
            termination = "invalid_policy"
            n_at_invalid = n
            invalid_kind = pol.invalid_kind
            rec["n_post"] = n
            rec["sampled_arm"] = None
            rec["reward"] = None
            rec["arm_pull_index"] = None
            per_round.append(rec)
            decision_index += 1
            break

        if pol.action == "COMMIT":
            termination = "forced_commit" if terminal else "autonomous_commit"
            committed_arm = pol.arm
            tau = n
            rec["n_post"] = n
            rec["sampled_arm"] = None
            rec["reward"] = None
            rec["arm_pull_index"] = None
            per_round.append(rec)
            decision_index += 1
            break

        r, idx = pull(pol.arm)
        n = len(history)
        rec["n_post"] = n
        rec["sampled_arm"] = pol.arm
        rec["reward"] = r
        rec["arm_pull_index"] = idx
        per_round.append(rec)
        decision_index += 1

    model_calls = decision_index

    # ---- committed-arm evidence -------------------------------------------
    # Every arm has at least the forced-init pull, so "never sampled" is not
    # possible. The meaningful quantity is ADAPTIVE trials: a commit to an arm
    # the model never chose to re-sample is a valid, and interesting, premature
    # commitment -- the interface must not block it.
    commit_ev = None
    if committed_arm is not None:
        cs, ct = counts[committed_arm]
        leader_at_commit = _empirical_leader(counts, display_order)
        commit_ev = {
            "committed_arm_trials": ct,
            "committed_arm_adaptive_trials": ct - p10.FORCED_INIT_PER_ARM,
            "committed_arm_successes": cs,
            "committed_arm_rate": (cs / ct) if ct else None,
            "committed_is_empirical_leader": committed_arm == leader_at_commit,
            "committed_is_empirical_challenger": committed_arm == _empirical_challenger(
                counts, display_order, leader_at_commit),
        }

    expected = p10.expected_fires(alpha, model_calls,
                                  n_steered_layers=n_steered_layers,
                                  tail_tokens=1)
    if fires_total != expected:
        raise EpisodeInfrastructureError(
            f"steering attestation FAILED: observed {fires_total} sites, "
            f"expected {expected} (alpha={alpha}, model_calls={model_calls}, "
            f"layers={n_steered_layers}). Behaviour must not be read off a "
            f"cell whose intervention is unverified.")

    return {
        # -- identity / provenance ------------------------------------------
        # from prompt_module, NOT p10: a PV10-A episode must not label itself
        # "pv10" or its records would be pooled with PV10-B's.
        "protocol_version": getattr(
            prompt_module, "PROTOCOL_VERSION", p10.PROTOCOL_VERSION),
        "stage1_instruction_version": getattr(
            prompt_module, "STAGE1_INSTRUCTION_VERSION",
            p10.STAGE1_INSTRUCTION_VERSION),
        "policy_parser_version": p10.POLICY_PARSER_VERSION,
        "order_version": p10.ORDER_VERSION,
        "interface_tag": interface_tag,
        "seed": seed,
        "alpha": alpha,
        "environment": {"name": env.name, "k": env.k,
                        "probs": list(env.probs), "total_budget": total_budget},
        "display_order": display_order,
        "initial_pull_order": list(orders.initial_pull_order),
        "tape_id": tape.tape_id,
        "arm_true_probs": {_letter(a): p for a, p in tape_arm_map.items()},
        "true_best": true_best,

        # -- outcome ---------------------------------------------------------
        "termination_reason": termination,
        "autonomous_commit": termination == "autonomous_commit",
        "committed_arm": committed_arm,
        "commit_correct": (None if committed_arm is None
                           else int(committed_arm == true_best)),
        "simple_regret": (None if committed_arm is None else
                          tape_arm_map[letter_to_full[true_best]]
                          - tape_arm_map[letter_to_full[committed_arm]]),
        "tau": tau,                      # null unless an episode committed
        "n_at_invalid": n_at_invalid,    # competing-exit time for invalids
        "invalid_kind": invalid_kind,
        "n_samples_used": len(history),
        "model_calls": model_calls,

        # -- attestation -----------------------------------------------------
        "steering_fires": fires_total,
        "steering_fires_expected": expected,
        "steered": steered,
        "n_steered_layers": n_steered_layers,

        # -- state -----------------------------------------------------------
        "forced_init": init_records,
        "final_counts": {a: list(counts[a]) for a in display_order},
        "history": history,
        "commit_evidence": commit_ev,
        "rounds": per_round,
    }
