#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""Freeze pv7 Easy-bare alpha=0 states for the Stage-1 alpha diagnostic.

WHY A SECOND BANK
-----------------
`bandit_pv7_frozen_states.json` was sampled from pv6 trajectories to SELECT a
pv7 prompt. This bank is sampled from pv7's own alpha=0 Easy-bare cell and
exists to ask a different question: does Stage-1 alpha change the transition
from recognizing uncertainty to acting on it?

Both banks are kept. They are not interchangeable: a state here carries a pv7
history, so a pv7-vs-pv6 prompt comparison must use the older file.

WHAT IS FROZEN
--------------
  * 20 seeds x 6 fixed rounds = 120 states. Fixed rounds, NOT event-triggered:
    the six pv6 state types were event-defined (first positive reward, ...),
    which makes each seed's slot land at a different round and couples the
    sample to the trajectory's own dynamics. Here the question is about
    lock-in over TIME, so the grid is time-based and identical for every seed.
  * 5 `critical_one_shot_zero` states -- the state immediately after the true
    best arm returned 0 on its first and only pull, on the five seeds where
    that arm was then never pulled again (3, 26, 31, 46, 50).

The five critical states are mostly ADDITIONAL rather than a tag on the 120,
but not entirely: a critical state whose round happens to equal a grid round
IS the same history, hence the same prompt. Storing it twice would evaluate
one prompt twice under each alpha and double-count it in any pooled figure, so
those are recorded as a cross-reference (`critical_refs`) to the existing grid
slot instead of a new state. On the current source this happens for seeds 3
and 46 (best arm pulled at r3 -> critical round 4 = the r4 grid slot), giving
123 stored states: 120 grid + 3 additional critical, with 5 critical slots in
total once the two references are resolved.

SELECTION IS NOT OUTCOME-BASED
------------------------------
The 120 are every seed at every grid round -- no filtering at all, so no
selection is possible. The 5 critical states ARE selected by a trajectory
property, and that property is deliberately defined WITHOUT reward
information: "an arm at n=1 whose single observation was 0, never pulled
again". That this arm happens to be the true best is a diagnostic fact stored
under `diagnostics`, never a selection criterion and never visible to a prompt
renderer. The same definition applied to a non-best arm would qualify equally.

WHAT THE CRITICAL SUBSET CAN AND CANNOT SHOW
--------------------------------------------
At rounds 1-4 no lock has formed yet, so these states test the UPSTREAM
question -- does alpha make the model re-sample an arm that just returned 0 --
and CANNOT show a lock being broken. Breaking an established lock is what the
r30/r50/r75/r95 slots of those same five seeds test. Keep the two separate.

ANALYSIS UNITS
--------------
Same rule as the pv6 bank: per-type tables keep all their slots; any POOLED
statistic deduplicates on `state_fingerprint`. Fixed rounds make collisions
between grid slots impossible, but a critical state and a grid slot could in
principle share a prefix, so the dedup count is computed, not assumed.

Usage
-----
    python3.10 freeze_pv7_lockin_states.py            # write
    python3.10 freeze_pv7_lockin_states.py --check    # byte-compare
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import bandit_reference as br


STATE_BANK_VERSION = "pv7-lockin-state-bank-v1"
DEFAULT_SOURCE = Path(
    "/Users/paveenhuang/Documents/RSNResult/RoleAnswer/llama3/bandit/pv7/"
    "pv7_easy_bare/bandit_pv7_easy_8B_11_20.json")
DEFAULT_OUTPUT = Path(__file__).with_name("bandit_pv7_lockin_states.json")

# Rounds are 0-indexed; a state at round_idx r is the state BEFORE choice r+1.
# r99 is the last live decision (T=100 => rounds 0..99). There is deliberately
# no r100 slot: it would be post-terminal, with no decision to make.
GRID_ROUNDS = (4, 10, 30, 50, 75, 95)
GRID_TYPES = tuple(f"round_{r + 1}" for r in GRID_ROUNDS)
CRITICAL_TYPE = "critical_one_shot_zero"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint(seed: int, round_idx: int, history: list[dict]) -> str:
    payload = {"seed": seed, "round_idx": round_idx, "history": history}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()[:16]


def _one_shot_zero_arms(choices: list[str], feedbacks: list[int],
                        round_idx: int) -> list[str]:
    """Arms with exactly one observation, that observation 0, as of round_idx.

    Reward-blind with respect to WHICH arm: it reads the arm's own observed
    history only. Used both to define the critical subset and as a tag.
    """
    seen = Counter(choices[:round_idx])
    won = Counter()
    for arm, reward in zip(choices[:round_idx], feedbacks[:round_idx]):
        won[arm] += reward
    return sorted(a for a, n in seen.items() if n == 1 and won[a] == 0)


def _critical_round(run: dict) -> int | None:
    """Round index right after the true best arm's single zero, if abandoned.

    Returns None unless the best arm was pulled exactly once in the whole
    episode AND that pull returned 0. `diagnostics.best_arm` is read here --
    this is the one place the bank consults it, and the resulting slots are
    labelled as an oracle-selected subset in the output.
    """
    choices, feedbacks, best = run["choices"], run["feedbacks"], run["best_arm"]
    idx = [i for i, c in enumerate(choices) if c == best]
    if len(idx) != 1 or feedbacks[idx[0]] != 0:
        return None
    return idx[0] + 1


def _make_state(run: dict, env, state_type: str, round_idx: int) -> dict:
    choices, feedbacks = run["choices"], run["feedbacks"]
    history = [{"arm": a, "reward": r}
               for a, r in zip(choices[:round_idx], feedbacks[:round_idx])]
    tried = set(choices[:round_idx])
    low = _one_shot_zero_arms(choices, feedbacks, round_idx)
    counts = Counter(choices[:round_idx])
    dominant = max(counts.values()) / round_idx if round_idx else 0.0
    return {
        "state_id": f"easy-s{run['seed']}-{state_type}",
        "state_type": state_type,
        "seed": run["seed"],
        "round_idx": round_idx,
        "round_number": round_idx + 1,
        "arm_order": list(run["arm_order"]),
        "history": history,
        "state_fingerprint": _fingerprint(run["seed"], round_idx, history),
        "tags": {
            "has_untried": len(tried) < env.k,
            "n_one_shot_zero_arms": len(low),
            "one_shot_zero_arms": low,
            "dominant_arm_fraction": round(dominant, 4),
            "round_bin": ("early" if round_idx < env.horizon // 3 else
                          "middle" if round_idx < 2 * env.horizon // 3
                          else "late"),
        },
        # Diagnostic-only. A prompt renderer must never read this block; it
        # exists so an oracle-assisted SECONDARY metric can be computed after
        # the primary reward-blind one.
        "diagnostics": {
            "best_arm": run["best_arm"],
            "best_position": run["best_position"],
            "true_probs_by_arm": dict(run["arm_map"]),
            "source_tape_id": run["tape_id"],
            "best_arm_tried": run["best_arm"] in tried,
            "best_arm_pulls_so_far": counts.get(run["best_arm"], 0),
            "best_arm_is_one_shot_zero": run["best_arm"] in low,
            "episode_suffix_failure": run["suffix_failure"],
        },
    }


def build_state_bank(source: Path) -> dict:
    doc = json.loads(source.read_text())
    if doc.get("protocol") != "pv7":
        raise ValueError(f"source declares protocol {doc.get('protocol')!r}")
    if float(doc.get("rationale_alpha", 0)) or float(doc.get("action_alpha", 0)):
        raise ValueError("source must be an alpha=0 cell")

    env = br.get_environment("easy")
    expected_seeds = br.build_seed_bank(env)
    runs = doc.get("runs", [])
    seeds = [r.get("seed") for r in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("source has duplicate seeds")
    if sorted(seeds) != expected_seeds:
        raise ValueError(
            f"source seeds {sorted(seeds)} != frozen Easy bank {expected_seeds}")

    states: list[dict] = []
    critical_seeds: list[int] = []
    critical_refs: dict[str, str] = {}
    for run in sorted(runs, key=lambda r: r["seed"]):
        if len(run["choices"]) != env.horizon:
            raise ValueError(f"seed {run['seed']}: T != {env.horizon}")
        grid_at = {}
        for round_idx, state_type in zip(GRID_ROUNDS, GRID_TYPES):
            st = _make_state(run, env, state_type, round_idx)
            grid_at[round_idx] = st
            states.append(st)
        crit = _critical_round(run)
        if crit is None:
            continue
        critical_seeds.append(run["seed"])
        if crit in grid_at:
            # Same history => same prompt. Reference the grid slot instead of
            # storing a duplicate; evaluating one prompt twice would waste a
            # generation per alpha and double-count it in pooled figures.
            shared = grid_at[crit]
            shared["tags"]["is_critical_one_shot_zero"] = True
            critical_refs[str(run["seed"])] = shared["state_id"]
        else:
            st = _make_state(run, env, CRITICAL_TYPE, crit)
            st["tags"]["is_critical_one_shot_zero"] = True
            states.append(st)

    for s in states:
        s["tags"].setdefault("is_critical_one_shot_zero", False)

    counts = Counter(s["state_type"] for s in states)
    if any(counts[t] != len(expected_seeds) for t in GRID_TYPES):
        raise AssertionError(f"grid slots not 20/type: {counts}")
    n_stored_crit = counts.get(CRITICAL_TYPE, 0)
    if n_stored_crit + len(critical_refs) != len(critical_seeds):
        raise AssertionError("critical slots lost between stored and referenced")
    if len(states) != len(GRID_TYPES) * len(expected_seeds) + n_stored_crit:
        raise AssertionError(f"unexpected state count {len(states)}")
    flagged = [s for s in states if s["tags"]["is_critical_one_shot_zero"]]
    if len(flagged) != len(critical_seeds):
        raise AssertionError(
            f"{len(flagged)} states flagged critical, expected "
            f"{len(critical_seeds)}")
    for s in flagged:
        if not s["diagnostics"]["best_arm_is_one_shot_zero"]:
            raise AssertionError(
                f"{s['state_id']} flagged critical but its best arm is not a "
                "one-shot-zero arm in that state")

    fps = [s["state_fingerprint"] for s in states]
    dupes = {fp: n for fp, n in Counter(fps).items() if n > 1}

    return {
        "state_bank_version": STATE_BANK_VERSION,
        "purpose": (
            "Stage-1 alpha diagnostic on frozen pv7 states. Measures whether "
            "alpha changes the transition from recognizing uncertainty to "
            "acting on it. States are NOT selected by alpha effect or by "
            "true-best outcome."),
        "analysis_unit": (
            "TWO UNITS. (a) Per-type tables use their own slots: 20 per grid "
            "round, 5 for the critical subset -- select the latter by the tag "
            "'is_critical_one_shot_zero', NOT by state_type, because a "
            "critical state that coincides with a grid round is stored once "
            "under its grid type and cross-referenced in 'critical_refs'. "
            "Report the critical subset on its own denominator of 5, never "
            "folded into a whole-bank total. (b) Any POOLED statistic "
            "deduplicates on 'state_fingerprint'. Compare alpha conditions "
            "WITHIN a state: the same state under -4/0/+4 is one paired "
            "observation, not three independent ones."),
        "critical_refs": critical_refs,
        "critical_subset_note": (
            "Select the critical subset by the tag 'is_critical_one_shot_zero' "
            "(5 states); 'critical_refs' maps a seed to the grid slot that "
            "carries its critical state when the two rounds coincide. "
            "These states sit at rounds 1-4, BEFORE any lock "
            "has formed. They test whether alpha makes the model re-sample an "
            "arm that just returned 0 -- they CANNOT show an established lock "
            "being broken. Lock-breaking is tested by the r31/r51/r76/r96 "
            "slots of the same five seeds. Selection uses the reward-blind "
            "property 'best arm pulled exactly once, observed 0, never again'; "
            "that the arm is the true best is recorded, not used to select."),
        "source": {
            "file": source.name,
            "sha256": _sha256(source),
            "protocol": "pv7",
            "environment": "reference_easy",
            "rationale_alpha": 0,
            "action_alpha": 0,
            "interface": "bare",
            "stage1_instruction_version":
                doc["config"]["stage1_instruction_version"],
            "stage2_instruction_version":
                doc["config"]["stage2_instruction_version"],
        },
        "environment": {
            "name": env.name,
            "k": env.k,
            "horizon": env.horizon,
            "probability_multiset": list(env.probs),
        },
        "seed_bank": expected_seeds,
        "grid_rounds": list(GRID_ROUNDS),
        "critical_seeds": sorted(critical_seeds),
        "state_type_definitions": {
            **{t: f"before choice {r + 1}; {r} observations"
               for t, r in zip(GRID_TYPES, GRID_ROUNDS)},
            CRITICAL_TYPE: (
                "state immediately after the true best arm's single pull "
                "returned 0, on seeds where it was never pulled again"),
        },
        "n_states": len(states),
        "n_grid_states": len(GRID_TYPES) * len(expected_seeds),
        "n_critical_states": len(critical_seeds),
        "n_critical_states_stored_separately": n_stored_crit,
        "n_critical_states_shared_with_grid": len(critical_refs),
        "n_unique_state_fingerprints": len(set(fps)),
        "duplicate_state_fingerprints": dupes,
        "state_type_counts": dict(counts),
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
        if not args.output.exists():
            raise SystemExit(f"MISSING: {args.output}")
        if args.output.read_text() != rendered:
            raise SystemExit(f"MISMATCH: {args.output}")
        print(f"OK: {args.output} ({bank['n_states']} states, "
              f"{bank['n_unique_state_fingerprints']} unique)")
        return
    args.output.write_text(rendered)
    print(f"wrote {args.output}: {bank['n_states']} states stored "
          f"({bank['n_grid_states']} grid + "
          f"{bank['n_critical_states_stored_separately']} critical; "
          f"{bank['n_critical_states_shared_with_grid']} critical share a grid "
          f"slot), {bank['n_critical_states']} critical slots total, "
          f"{bank['n_unique_state_fingerprints']} unique fingerprints; "
          f"critical seeds {bank['critical_seeds']}")


if __name__ == "__main__":
    main()
