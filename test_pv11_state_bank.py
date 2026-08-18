#!/usr/bin/env python3.10
"""Structural invariants of the PV11 state bank. No model, no GPU, ~1s.

Scope is deliberately the DATA LAYER only: counts legality, the balance claims,
within-state_id sharing, and reproducibility of the hash. Prompt/anchor/parser
and runner invariants belong in `test_bandit_pv11.py`, which does not exist yet
-- keeping them in separate files means a state-bank edit and a prompt edit
cannot mask each other.

The balance checks are the point of this file. `build_pv11_state_bank.py`
CLAIMS exact 5/5/5/5 marginals and pairwise crossings within a max-min
difference of 1; a claim in a docstring that nothing verifies is how the pv6
launcher's seed list drifted from its manifest. Here it is executable.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import build_pv11_state_bank as B

HERE = Path(__file__).resolve().parent
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


def main() -> int:
    bank = json.loads((HERE / "pv11_state_bank.json").read_text())
    manifest = json.loads((HERE / "pv11_state_manifest.json").read_text())
    states = bank["states"]

    # ── reproducibility ─────────────────────────────────────────────────────
    fresh = B.build_bank()
    check(B.canonical(fresh) == B.canonical(bank),
          "stored bank does not reproduce from the builder")
    check(manifest["state_bank_sha256"] == B.sha256(bank),
          "manifest state_bank_sha256 does not match the stored bank")

    # ── sizes ───────────────────────────────────────────────────────────────
    comm = [s for s in states if s["block"] == "commitment"]
    acq = [s for s in states if s["block"] == "acquisition"]
    check(len(states) == 160, f"expected 160 states, got {len(states)}")
    check(len(comm) == 80, f"expected 80 commitment states, got {len(comm)}")
    check(len(acq) == 80, f"expected 80 acquisition states, got {len(acq)}")
    uids = [s["state_uid"] for s in states]
    check(len(set(uids)) == len(uids), "state_uid is not unique")

    # 4 cells x 20 matched state_ids in each block
    for block, group in (("commitment", comm), ("acquisition", acq)):
        cells = Counter(s["cell"] for s in group)
        check(len(cells) == 4, f"{block}: expected 4 cells, got {len(cells)}")
        check(set(cells.values()) == {20},
              f"{block}: cells are not all 20 states: {dict(cells)}")
        for cell in cells:
            sids = sorted(s["state_id"] for s in group if s["cell"] == cell)
            check(sids == list(range(20)),
                  f"{block}/{cell}: state_ids are not 0..19")

    # ── counts legality ─────────────────────────────────────────────────────
    for s in states:
        counts = s["displayed_counts"]
        check(set(counts) == set(B.ARM_LABELS),
              f"{s['state_uid']}: displayed_counts keys != arm labels")
        check(sorted(s["display_order"]) == sorted(B.ARM_LABELS),
              f"{s['state_uid']}: display_order is not a permutation")
        for lab, (succ, trials) in counts.items():
            check(isinstance(succ, int) and isinstance(trials, int),
                  f"{s['state_uid']}/{lab}: counts are not ints")
            check(trials > 0, f"{s['state_uid']}/{lab}: trials <= 0")
            check(0 <= succ <= trials,
                  f"{s['state_uid']}/{lab}: successes out of range")
        check(set(s["latent_probs"]) == set(B.ARM_LABELS),
              f"{s['state_uid']}: latent_probs keys != arm labels")
        check(all(0.0 < p < 1.0 for p in s["latent_probs"].values()),
              f"{s['state_uid']}: latent prob outside (0,1)")
        check(s["remaining_horizon"] > 0,
              f"{s['state_uid']}: non-positive remaining horizon")
        # true_best must actually be the latent argmax
        best = max(s["latent_probs"].values())
        check(s["latent_probs"][s["true_best"]] == best,
              f"{s['state_uid']}: true_best is not the latent argmax")

    # ── Commitment: sample size held constant, only the gap varies ──────────
    for s in comm:
        trials = {t for _, t in s["displayed_counts"].values()}
        check(trials == {20},
              f"{s['state_uid']}: commitment arms are not all at 20 trials")
        rates = sorted((c[0] / c[1] for c in s["displayed_counts"].values()),
                       reverse=True)
        gap = round(rates[0] - rates[1], 10)
        want = 0.05 if s["evidence"] == "weak" else 0.20
        check(abs(gap - want) < 1e-9,
              f"{s['state_uid']}: gap {gap} != {want}")
    # short/long see IDENTICAL counts -- horizon must be the only difference
    for sid in range(20):
        for ev in ("weak", "strong"):
            pair = [s for s in comm
                    if s["state_id"] == sid and s["evidence"] == ev]
            check(len(pair) == 2, f"C-{ev}-{sid}: expected 2 horizon cells")
            a, b = pair
            check(a["displayed_counts"] == b["displayed_counts"],
                  f"C-{ev}-{sid}: horizon cells differ in counts")
            check(a["display_order"] == b["display_order"],
                  f"C-{ev}-{sid}: horizon cells differ in display order")
            check(a["remaining_horizon"] != b["remaining_horizon"],
                  f"C-{ev}-{sid}: horizon cells share a horizon")

    # ── Acquisition: only the probe differs across the four cells ──────────
    for sid in range(20):
        group = [s for s in acq if s["state_id"] == sid]
        check(len(group) == 4, f"A-{sid}: expected 4 cells")
        probe = {s["probe_label"] for s in group}
        check(len(probe) == 1, f"A-{sid}: probe_label differs across cells")
        plabel = probe.pop()
        for other in B.ARM_LABELS:
            if other == plabel:
                continue
            vals = {tuple(s["displayed_counts"][other]) for s in group}
            check(len(vals) == 1,
                  f"A-{sid}/{other}: non-probe arm differs across cells")
        probe_counts = {tuple(s["displayed_counts"][plabel]) for s in group}
        check(len(probe_counts) == 4,
              f"A-{sid}: the four cells do not have 4 distinct probe counts")
        for s in group:
            succ, trials = s["displayed_counts"][plabel]
            rate = succ / trials
            want_rate = 0.60 if s["probe_rate_level"] == "high_rate" else 0.40
            check(abs(rate - want_rate) < 1e-9,
                  f"{s['state_uid']}: probe rate {rate} != {want_rate}")
            want_n = 5 if s["probe_sample_size"] == "low_n" else 20
            check(trials == want_n,
                  f"{s['state_uid']}: probe trials {trials} != {want_n}")
            check(s["display_order"][s["probe_display_row"] - 1] == plabel,
                  f"{s['state_uid']}: probe not at its declared display row")

    # ── within-state_id sharing (both blocks) ───────────────────────────────
    for block, group in (("commitment", comm), ("acquisition", acq)):
        for sid in range(20):
            cells = [s for s in group if s["state_id"] == sid]
            latents = {B.canonical(s["latent_probs"]) for s in cells}
            check(len(latents) == 1,
                  f"{block}/{sid}: cells do not share latent_probs")
            tapes = {s["tape_key"] for s in cells}
            check(len(tapes) == 1,
                  f"{block}/{sid}: cells do not share a tape_key")

    # ── balance: EXACT marginals ────────────────────────────────────────────
    # One row per state_id (the four cells share these attributes).
    by_sid = {}
    for s in acq:
        by_sid[s["state_id"]] = s
    ranks = Counter(by_sid[i]["probe_true_rank"] for i in range(20))
    labels = Counter(by_sid[i]["probe_label"] for i in range(20))
    rows = Counter(by_sid[i]["probe_display_row"] for i in range(20))
    check(set(ranks.values()) == {5}, f"probe_true_rank not 5/5/5/5: {ranks}")
    check(set(labels.values()) == {5}, f"probe_label not 5/5/5/5: {labels}")
    check(set(rows.values()) == {5}, f"probe_display_row not 5/5/5/5: {rows}")

    # ── balance: pairwise crossings within max-min <= 1 (NOT exact) ────────
    pairs = (("probe_true_rank", "probe_label"),
             ("probe_true_rank", "probe_display_row"),
             ("probe_label", "probe_display_row"))
    for a, b in pairs:
        cross = Counter((by_sid[i][a], by_sid[i][b]) for i in range(20))
        # 16 possible cells, 20 states -> counts must be 1 or 2
        lo = min(cross.get((x, y), 0)
                 for x in {by_sid[i][a] for i in range(20)}
                 for y in {by_sid[i][b] for i in range(20)})
        hi = max(cross.values())
        check(hi - lo <= 1,
              f"{a} x {b} crossing spread {hi - lo} > 1: {dict(cross)}")

    # Commitment's leading LABEL is balanced too (it rotates with state_id).
    lead = Counter(by_sid_c["true_best"] for by_sid_c in
                   {s["state_id"]: s for s in comm}.values())
    check(set(lead.values()) == {5},
          f"commitment true_best label not 5/5/5/5: {lead}")

    # ── manifest agrees with the bank ───────────────────────────────────────
    check(manifest["n_states"] == len(states),
          "manifest n_states disagrees with the bank")
    check(manifest["blocks"]["commitment"]["n_states"] == 80,
          "manifest commitment n_states != 80")
    check(manifest["blocks"]["acquisition"]["n_states"] == 80,
          "manifest acquisition n_states != 80")
    # The gate thresholds are frozen; a silent edit here would change PASS/FAIL.
    gate = manifest["manipulation_gate"]
    check(gate["M1_commitment_sensitivity"]["threshold"] == 0.15,
          "M1 threshold is no longer 0.15")
    check(gate["M2_acquisition_sensitivity"]["threshold"] == 0.15,
          "M2 threshold is no longer 0.15")
    check(gate["denominator_convention"] == "ITT",
          "gate denominator convention is no longer ITT")
    check("does not trigger prompt iteration" in gate["on_failure"],
          "the frozen gate-failure wording has been altered")

    if FAILURES:
        print(f"FAIL ({len(FAILURES)})")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ok  test_pv11_state_bank.py  (160 states, all invariants hold)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
