#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""Freeze 20 seeds x 6 prompt-independent states from pv6 Easy-bare alpha=0.

Prompt selection must be based on validity, grounding, completion, and cost —
never on which wording selects the true best arm on these same trajectories.
The frozen file therefore stores true probabilities only under ``diagnostics``;
pv7 prompt renderers consume ``arm_order`` and ``history`` only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import bandit_reference as br


STATE_BANK_VERSION = "pv7-state-bank-v1"
DEFAULT_SOURCE = Path(
    "/Users/paveenhuang/Documents/RSNResult/RoleAnswer/llama3/bandit/pv6/"
    "pv6_easy_bare/mdf_0/bandit_pv6_reference_easy_8B_20_11_20.json")
DEFAULT_OUTPUT = Path(__file__).with_name("bandit_pv7_frozen_states.json")

STATE_TYPES = (
    "round_1",
    "first_partial_tried_untried",
    "after_first_positive_reward",
    "after_first_two_consecutive_zero_rewards",
    "round_50",
    "round_100",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_prefix_with_two_but_not_all(choices: list[str], k: int) -> int:
    for n in range(1, len(choices) + 1):
        n_seen = len(set(choices[:n]))
        if 2 <= n_seen < k:
            return n
    raise ValueError("trajectory never reaches a partial 2..K-1 arm state")


def _first_positive_prefix(feedbacks: list[int]) -> int:
    for i, reward in enumerate(feedbacks):
        if reward == 1:
            return i + 1
    raise ValueError("trajectory contains no positive reward")


def _first_two_zero_prefix(feedbacks: list[int]) -> int:
    for i in range(len(feedbacks) - 1):
        if feedbacks[i] == feedbacks[i + 1] == 0:
            return i + 2
    raise ValueError("trajectory contains no consecutive zero rewards")


def _prefixes(run: dict, horizon: int) -> dict[str, int]:
    choices, feedbacks = run["choices"], run["feedbacks"]
    if len(choices) != horizon or len(feedbacks) != horizon:
        raise ValueError(
            f"seed {run.get('seed')}: expected T={horizon}, got "
            f"{len(choices)} choices / {len(feedbacks)} feedbacks")
    return {
        "round_1": 0,
        "first_partial_tried_untried": _first_prefix_with_two_but_not_all(
            choices, len(run["arm_order"])),
        "after_first_positive_reward": _first_positive_prefix(feedbacks),
        "after_first_two_consecutive_zero_rewards": _first_two_zero_prefix(
            feedbacks),
        # State immediately BEFORE the named choice.
        "round_50": 49,
        "round_100": 99,
    }


def build_state_bank(source: Path) -> dict:
    doc = json.loads(source.read_text())
    env = br.get_environment("easy")
    expected_seeds = br.build_seed_bank(env)
    runs = doc.get("runs", [])
    seeds = [r.get("seed") for r in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("source has duplicate seeds")
    if sorted(seeds) != expected_seeds:
        raise ValueError(
            f"source seeds {sorted(seeds)} != frozen Easy bank {expected_seeds}")
    if any(r.get("protocol") != "pv6" for r in runs):
        raise ValueError("source must contain pv6 trajectories")

    states = []
    fingerprints = []
    for run in sorted(runs, key=lambda r: r["seed"]):
        choices = run["choices"]
        feedbacks = run["feedbacks"]
        for state_type, round_idx in _prefixes(run, env.horizon).items():
            history = [
                {"arm": arm, "reward": reward}
                for arm, reward in zip(
                    choices[:round_idx], feedbacks[:round_idx])
            ]
            fingerprint_payload = {
                "seed": run["seed"], "round_idx": round_idx,
                "history": history,
            }
            fingerprint = hashlib.sha256(
                json.dumps(fingerprint_payload, sort_keys=True,
                           separators=(",", ":")).encode()).hexdigest()[:16]
            fingerprints.append(fingerprint)
            states.append({
                "state_id": f"easy-s{run['seed']}-{state_type}",
                "state_type": state_type,
                "seed": run["seed"],
                "round_idx": round_idx,
                "round_number": round_idx + 1,
                "arm_order": list(run["arm_order"]),
                "history": history,
                "state_fingerprint": fingerprint,
                "tags": {
                    "has_untried": len(set(choices[:round_idx])) < env.k,
                    "after_positive": bool(round_idx and feedbacks[round_idx - 1] == 1),
                    "after_failures": bool(
                        round_idx >= 2 and feedbacks[round_idx - 2:round_idx] == [0, 0]),
                    "true_best_discovered": run["best_arm"] in choices[:round_idx],
                    "round_bin": (
                        "early" if round_idx < env.horizon // 3 else
                        "middle" if round_idx < 2 * env.horizon // 3 else "late"),
                },
                # Diagnostic-only: prompt construction must never read this.
                "diagnostics": {
                    "best_arm": run["best_arm"],
                    "best_position": run["best_position"],
                    "true_probs_by_arm": dict(run["arm_map"]),
                    "source_tape_id": run["tape_id"],
                },
            })

    type_counts = Counter(s["state_type"] for s in states)
    duplicate_fingerprints = {
        fp: n for fp, n in Counter(fingerprints).items() if n > 1}
    if len(states) != 120 or any(type_counts[t] != 20 for t in STATE_TYPES):
        raise AssertionError(
            f"expected 120 states and 20/type, got {len(states)} / {type_counts}")

    return {
        "state_bank_version": STATE_BANK_VERSION,
        "purpose": (
            "Prompt selection by validity, evidence grounding, rationale "
            "completion, and cost only; never select by true-best outcome."),
        "analysis_unit": (
            "Pair prompt/intervention conditions within the exact snapshot and "
            "report the six state types separately. Event definitions can land "
            "on the same history prefix, so do not pool 120 slots as 120 "
            "independent states. TWO DIFFERENT UNITS: (a) the six per-type "
            "tables each use their own 20 slots -- keep all 120 there, because "
            "one history answers a different diagnostic question under each "
            "type; (b) any POOLED/overall statistic must deduplicate on "
            "'state_fingerprint' (the dedup key: a bijection with "
            "(seed, history, arm_order)) and reports n=107, not 120."),
        "source": {
            "file": source.name,
            "sha256": _sha256(source),
            "protocol": "pv6",
            "environment": "reference_easy",
            "alpha": 0,
            "interface": "bare",
        },
        "environment": {
            "name": env.name,
            "k": env.k,
            "horizon": env.horizon,
            "probability_multiset": list(env.probs),
        },
        "seed_bank": expected_seeds,
        "state_type_definitions": {
            "round_1": "before choice 1; empty history",
            "first_partial_tried_untried": (
                "first state with 2..K-1 distinct tried arms"),
            "after_first_positive_reward": (
                "first state immediately after observing reward 1"),
            "after_first_two_consecutive_zero_rewards": (
                "first state immediately after observing two adjacent zeros"),
            "round_50": "before choice 50; 49 observations",
            "round_100": "before choice 100; 99 observations",
        },
        "n_states": len(states),
        "n_unique_state_fingerprints": len(set(fingerprints)),
        "duplicate_state_fingerprints": duplicate_fingerprints,
        "state_type_counts": dict(type_counts),
        "states": states,
    }


def _canonical(obj: dict) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    bank = build_state_bank(args.source)
    rendered = _canonical(bank)
    if args.check:
        if not args.output.exists() or args.output.read_text() != rendered:
            raise SystemExit(f"MISMATCH: {args.output}")
        print(f"OK: {args.output} ({bank['n_states']} states)")
        return
    args.output.write_text(rendered)
    print(
        f"wrote {args.output}: {bank['n_states']} states, "
        f"{bank['n_unique_state_fingerprints']} unique fingerprints")


if __name__ == "__main__":
    main()
