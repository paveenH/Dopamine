#!/usr/bin/env python3.10
"""PV11 micro-episode runner: controlled evidence states, real rewards.

Deliberately NOT a wrapper around `run_pv10_episode`. That function bakes in a
forced initialization phase, an online-accumulated choice history and the
`tau - k*FORCED_INIT + 1` fire formula, all three of which are wrong here:

    forced init : PV11 opens with the synthetic counts already in place. There
                  is nothing to initialize, and a forced pull would overwrite
                  the manipulated evidence state.
    history     : PV11's frozen prohibition -- a label history binds sample
                  size to token frequency, and the Acquisition block IS a
                  sample-size contrast.
    fires       : PV11 makes one call per decision with no forced pulls, so
                  the count is `model_calls * L`, not PV10's formula.

A short standalone runner makes those three differences visible instead of
inheriting them silently.

THE PRIMARY ESTIMAND IS THE FIRST ACTION
----------------------------------------
`first_action` is recorded as its own top-level block. It is the ONLY strictly
state-matched quantity in the episode: every cell of a state_id, and every
alpha, sees a byte-identical opening prompt, so a difference there is
attributable to the manipulation. From the moment the first action is taken,
action and sampled reward fork the trajectory -- two cells are no longer
answering the same question. Everything after it is stored under
`secondary_trajectory` and must never be reported as a state-matched effect.

REWARDS ARE REAL
----------------
A SAMPLE returns an actual Bernoulli draw from the arm's LATENT probability,
which the model never sees. This is not decoration: the pv7 frozen-state bank
gave one choice with no feedback, so sampling could not pay off, and its nulls
could only ever be read as "did not change the immediate choice in an already
locked state". Making rewards real is what turns a null here into evidence
about acquisition rather than about the model correctly noticing that sampling
was inert.

TAPES
-----
`(tape_key, arm, pull_index)` -> a fixed reward, derived by SHA256 of that
triple. Never the builtin `hash()`: PYTHONHASHSEED is randomized per process,
so the same cell would draw different rewards on every run and cross-alpha
pairing would silently dissolve. Because the key omits the cell and alpha, the
n-th pull of a given arm is the SAME latent outcome for all four cells of a
state_id and for every alpha -- so a cross-cell or cross-alpha difference can
never be reward luck.

`pull_index` restarts at 0 at the FIRST SAMPLE AFTER the synthetic state. The
displayed counts are a DISPLAY, not a sampled history; there is no earlier tape
to continue from, and pretending otherwise would imply an unobserved prefix.

INVALID
-------
An invalid policy TERMINATES the episode immediately, and the episode is
retained in the ITT denominator with `first_action.kind == "invalid"`. It is
model behaviour, distinct from `EpisodeInfrastructureError` (a stale wrapper,
an exception, an empty return), which fails closed so the cell is re-run.
"""

from __future__ import annotations

import hashlib
import json

import bandit_pv11 as p11

try:
    import torch
except Exception:                                  # noqa: BLE001
    torch = None

# The generation cap and the parser/stop contract are PV10-v2's, shared.
RATIONALE_MAX_TOKENS = 128


class EpisodeInfrastructureError(RuntimeError):
    """Generation itself failed. NOT a model invalid.

    Raised so the runner fails closed and the cell is re-run on resume, rather
    than recording an infrastructure artifact as model behaviour.
    """


# ─────────────────────────────── reward tape ────────────────────────────────

def tape_reward(tape_key: str, arm: str, pull_index: int, prob: float) -> int:
    """Deterministic Bernoulli draw for one (tape_key, arm, pull_index).

    SHA256 of the triple -> a uniform in [0, 1) from the leading 64 bits ->
    compare against `prob`. Stable across processes, machines and Python
    versions, which the builtin hash() is not.

    `prob` is NOT part of the key. That is deliberate: the uniform draw is a
    property of the tape position alone, so two cells sharing a state_id draw
    the same u and differ only through their latent probability. Keying on
    prob would decouple them and reintroduce reward luck.
    """
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob out of range: {prob}")
    if pull_index < 0:
        raise ValueError(f"pull_index must be >= 0, got {pull_index}")
    digest = hashlib.sha256(
        f"{tape_key}|{arm}|{pull_index}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return 1 if u < prob else 0


# ─────────────────────────────── the episode ────────────────────────────────

def run_pv11_episode(
    vc,
    state: dict,
    diff_mtx=None,
    alpha: float = 0.0,
    max_new_tokens: int = RATIONALE_MAX_TOKENS,
    n_steered_layers: int = 9,
    interface_tag: str = "",
    prompt_module=p11,
) -> dict:
    """Run one PV11 micro-episode from a synthetic evidence state.

    `state` is one entry of `pv11_state_bank.json`, used verbatim: its
    `displayed_counts` open the episode, its `latent_probs` generate rewards,
    its `tape_key` pairs those rewards across cells and alphas, and its
    `remaining_horizon` bounds the number of SAMPLEs.

    alpha == 0 must leave `diff_mtx` None: with no matrix `vc.generate` is
    called, which registers no hooks at all. "Unsteered" is deliberately a
    different code path from "steered by zero" -- that is what makes an alpha=0
    cell reusable and what lets attestation assert exactly 0 fires.
    """
    steered = alpha != 0
    if steered and diff_mtx is None:
        raise ValueError(f"alpha={alpha} requires diff_matrices")
    if not steered and diff_mtx is not None:
        raise ValueError(
            "alpha=0 must pass diff_mtx=None so NO hook is registered; "
            "a zero matrix would still register hooks and is not the same")

    display_order = list(state["display_order"])
    counts = {a: tuple(state["displayed_counts"][a]) for a in display_order}
    latent = dict(state["latent_probs"])
    horizon = int(state["remaining_horizon"])
    tape_key = state["tape_key"]
    if set(counts) != set(latent):
        raise ValueError("displayed_counts and latent_probs disagree on arms")

    # A stable per-state integer for torch seeding, so sampling luck is matched
    # across alpha cells at the same decision index. Derived from the state_uid
    # rather than passed in, so two cells of one state_id cannot drift apart.
    state_seed = int.from_bytes(
        hashlib.sha256(state["state_uid"].encode()).digest()[:4], "big")

    can_count = hasattr(vc, "steering_fire_count")
    if can_count:
        vc.steering_fire_count(reset=True)

    pulls_taken = 0                 # SAMPLEs so far == next pull_index
    per_arm_pulls = {a: 0 for a in display_order}
    fires_total = 0
    model_calls = 0
    rounds: list[dict] = []
    first_action: dict | None = None
    termination = None
    committed_arm = None

    while True:
        remaining = horizon - pulls_taken
        terminal = (remaining == 0)
        prompt = prompt_module.build_decision_prompt(
            display_order, counts, remaining)

        pre_counts = {a: counts[a] for a in display_order}
        if torch is not None:
            torch.manual_seed(state_seed * 100_003 + model_calls)

        try:
            if steered:
                out = vc.regenerate(
                    inputs=[prompt], diff_matrices=diff_mtx,
                    prefill_only=True, prefill_tail_len=1,
                    max_new_tokens=max_new_tokens, temperature=0.0,
                    stop_strings=list(p11.STOP_STRINGS))
            else:
                # stop_strings MUST match the steered branch. PV10-A v1 lost
                # 58/60 episodes to exactly this asymmetry: alpha=0 took the
                # `generate` path with no stop marker while its own +-4 cells
                # stopped on "#", so the two arms of one experiment had
                # different generation boundaries.
                out = vc.generate(
                    inputs=[prompt], max_new_tokens=max_new_tokens,
                    temperature=0.0, stop_strings=list(p11.STOP_STRINGS))
        except Exception as e:                     # INFRA, not model behaviour
            raise EpisodeInfrastructureError(
                f"generation failed at state={state['state_uid']} "
                f"call={model_calls}: {type(e).__name__}: {e}") from e

        raw = out[0] if isinstance(out, list) else out
        if raw is None:
            raise EpisodeInfrastructureError(
                f"generation returned None at state={state['state_uid']} "
                f"call={model_calls}")

        fired = vc.steering_fire_count(reset=True) if can_count else None
        if fired is None:
            # Unconditional in llms.py; a missing count means a stale server
            # llms.py. Sync it -- do not add a bypass flag.
            raise EpisodeInfrastructureError(
                "steering_fire_count unavailable: the model wrapper is stale, "
                "so injection cannot be attested")
        fires_total += fired
        model_calls += 1

        stopped = p11.apply_stop_boundary(raw)
        pol = p11.parse_policy(stopped, display_order, terminal=terminal)

        rec = {
            "call_index": model_calls - 1,
            "pulls_taken_pre": pulls_taken,
            "remaining_pre": remaining,
            "terminal_prompt": terminal,
            "pre_counts": {a: list(pre_counts[a]) for a in display_order},
            "valid": pol.valid,
            "action": pol.action,
            "arm": pol.arm,
            "invalid_kind": pol.invalid_kind,
            "reason": pol.reason,
            "raw_generation": raw,
            "generation_stopped": stopped,
            "stop_marker_truncated": stopped != raw,
            "fires": fired,
            "reward": None,
        }

        if first_action is None:
            # THE PRIMARY ESTIMAND. Captured before any state update, so it is
            # a pure function of the opening prompt.
            first_action = {
                "kind": ("invalid" if not pol.valid
                         else pol.action.lower()),   # "sample" | "commit"
                "valid": pol.valid,
                "action": pol.action,
                "arm": pol.arm,
                "invalid_kind": pol.invalid_kind,
                "is_probe": (state.get("probe_label") is not None
                             and pol.arm == state.get("probe_label")),
                "is_displayed_best": pol.arm == state.get("displayed_best"),
                "is_true_best": pol.arm == state.get("true_best"),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }

        if not pol.valid:
            # Model behaviour, not infrastructure. Terminates the episode; the
            # episode stays in the ITT denominator.
            rounds.append(rec)
            termination = "invalid_policy"
            break

        if pol.action == "COMMIT":
            rounds.append(rec)
            committed_arm = pol.arm
            termination = "forced_commit" if terminal else "autonomous_commit"
            break

        # SAMPLE: draw a REAL reward from the latent probability.
        arm = pol.arm
        reward = tape_reward(tape_key, arm, per_arm_pulls[arm], latent[arm])
        rec["reward"] = reward
        rec["pull_index"] = per_arm_pulls[arm]
        rounds.append(rec)

        s, t = counts[arm]
        counts[arm] = (s + reward, t + 1)
        per_arm_pulls[arm] += 1
        pulls_taken += 1

    # ── attestation against the REALIZED call count, not the ceiling ────────
    expected = (p11.expected_fires_for_calls(model_calls, n_steered_layers)
                if steered else 0)
    ceiling = p11.expected_fires_max(horizon, n_steered_layers)
    if fires_total != expected:
        raise EpisodeInfrastructureError(
            f"steering attestation failed at state={state['state_uid']} "
            f"alpha={alpha}: fires={fires_total} expected={expected} "
            f"(model_calls={model_calls}, L={n_steered_layers})")
    if steered and fires_total > ceiling:
        raise EpisodeInfrastructureError(
            f"fires {fires_total} exceed the per-episode ceiling {ceiling}")

    return {
        "protocol_version": p11.PROTOCOL_VERSION,
        "state_bank_version": p11.STATE_BANK_VERSION,
        "policy_parser_version": p11.POLICY_PARSER_VERSION,
        "interface_tag": interface_tag,
        "state_uid": state["state_uid"],
        "block": state["block"],
        "state_id": state["state_id"],
        "cell": state["cell"],
        "alpha": alpha,
        "steered": steered,
        "display_order": display_order,
        "opening_counts": {a: list(state["displayed_counts"][a])
                           for a in display_order},
        "latent_probs": latent,
        "tape_key": tape_key,
        "true_best": state.get("true_best"),
        "displayed_best": state.get("displayed_best"),
        "probe_label": state.get("probe_label"),
        "horizon": horizon,
        # ── PRIMARY ────────────────────────────────────────────────────────
        "first_action": first_action,
        # ── SECONDARY: forks after the first action; descriptive only ──────
        "secondary_trajectory": {
            "termination_reason": termination,
            "committed_arm": committed_arm,
            "commit_correct": (None if committed_arm is None
                               else committed_arm == state.get("true_best")),
            "n_samples": pulls_taken,
            "per_arm_pulls": per_arm_pulls,
            "final_counts": {a: list(counts[a]) for a in display_order},
            "model_calls": model_calls,
            "rounds": rounds,
        },
        "attestation": {
            "steering_fires": fires_total,
            "expected_fires": expected,
            "ceiling_fires": ceiling,
            "model_calls": model_calls,
            "n_steered_layers": n_steered_layers,
        },
    }
