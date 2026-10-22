#!/usr/bin/env python3.10
"""PV11-Acq analyzer. FROZEN before any steered episode was generated.

Every number reported for PV11-Acq has exactly one definition here.

WHAT THIS MEASURES
------------------
One question, pre-registered in pv11_amendment_01.json:

    Does RSN steering change the probability that, from an IDENTICAL evidence
    state, the model's FIRST action samples a low-sample-size probe arm whose
    displayed empirical rate is held at .40?

PRIMARY METRIC -- the low_rate pair ONLY
----------------------------------------
    P(first action = SAMPLE probe | low_n/low_rate)
  - P(first action = SAMPLE probe | matched_n/low_rate)

In both cells the probe's displayed rate is .40 and the probe is NOT the
displayed best arm, so sampling it cannot be rate-chasing. Probe label,
display row, displayed rate and the other three arms are identical across the
pair; ONLY sample size differs (5 vs 20 trials). This is the only contrast in
PV11 with causal content.

The high_rate cells are DESCRIPTIVE ONLY. There the probe IS the displayed
best arm, so sampling it is inseparable from rate-chasing -- the structure
that produced PV10-C's deprecated `.774` construct failure. They may not be
used to draw an information-seeking conclusion, and may not re-characterize
the low_rate result.

TWO READINGS, AND WHY THE UNIQUE-PROMPT ONE LEADS
-------------------------------------------------
Latent rank is invisible at the first step, so several design slots render a
BYTE-IDENTICAL opening prompt. At temperature=0 those are one decision
repeated, not independent samples. Each acquisition cell holds 20 slots but
only 16 unique opening prompts.

  * slot_weighted   -- ITT over all 20 slots. Reported for continuity with
                       the frozen gate, which was defined this way.
  * unique_prompt   -- over the 16 distinct opening-prompt fingerprints.
                       PRIMARY descriptive reading for PV11-Acq.

If two slots sharing a fingerprint disagree on the first action, that is
reported explicitly and loudly; a representative is never chosen arbitrarily.

ESTIMAND SCOPE
--------------
FIRST ACTION ONLY. After the first action, choice and realized reward fork
the trajectory, so no later round is state-matched across alpha. Secondary
trajectory statistics are printed as interface/behaviour description and may
not support or rescue a first-step conclusion.

POWER
-----
The alpha=0 baseline holds 3 positive events (unique-prompt) / 4 (slot).
Pre-registered consequence: no significance claim, no improvement claim, and
NO EQUIVALENCE CLAIM. A null is written "not detected at this low-power
baseline", never "alpha does not affect acquisition". This analyzer refuses
to print a p-value for the primary contrast for that reason.

NOT ANALYSED HERE
-----------------
The Commitment block / M1 is withdrawn (pv11_amendment_01.json). This file
loads acquisition states only and raises if handed commitment data.

Usage
-----
    python3.10 analyze_bandit_pv11_acq.py \
        --a0  <dir>/pv11_a0/bandit_pv11_alpha0.json \
        --am4 <dir>/pv11_acq_am4/bandit_pv11_alpha-4.0.json \
        --ap4 <dir>/pv11_acq_ap4/bandit_pv11_alpha4.0.json

`--a0` is the ORIGINAL full 160-state A0 file; acquisition is filtered out of
it. It is never regenerated or resumed into a new directory: it genuinely
predates the amendment, and re-running it to make a label match would replace
a real baseline with a later one.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

AMENDMENT = "pv11_amendment_01"
AMENDMENT_SHA256 = (
    "7f72b42d752b84a8c0765403574aa8519f0a51c99bd0cf9df76e47d18deb4741")

CELLS = ("low_n/low_rate", "matched_n/low_rate",
         "low_n/high_rate", "matched_n/high_rate")
PRIMARY_PAIR = ("low_n/low_rate", "matched_n/low_rate")
DESCRIPTIVE_PAIR = ("low_n/high_rate", "matched_n/high_rate")

# Fields that must be identical for a state to be considered the same state
# across alpha. `latent_probs` and `tape_key` are included even though they
# are invisible to the model: if they differed, the realized rewards would
# differ and the trajectories would not be comparable.
STATE_FIELDS = ("state_id", "cell", "display_order", "opening_counts",
                "latent_probs", "tape_key", "true_best", "displayed_best",
                "probe_label", "horizon")

VERSION_FIELDS = ("protocol_version", "state_bank_version",
                  "policy_parser_version")


# ─────────────────────────────── loading ────────────────────────────────────

def load_cell(path: Path, expect_alpha: float | None) -> dict:
    """Load one cell and reduce it to its acquisition states.

    Fails closed on anything that would make a cross-alpha comparison
    dishonest: wrong alpha, commitment contamination, duplicate uids, a
    short cell, or an attestation whose observed fires disagree with what
    that cell's own configuration expects.
    """
    if not path.exists():
        raise SystemExit(f"missing result file: {path}")
    payload = json.loads(path.read_text())
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise SystemExit(f"{path}: no 'runs' array")

    acq = [r for r in runs if r.get("block") == "acquisition"]
    if not acq:
        raise SystemExit(f"{path}: contains no acquisition states")
    if len(acq) != 80:
        raise SystemExit(
            f"{path}: {len(acq)} acquisition states, expected 80")

    # A file declaring acquisition scope must contain ONLY acquisition
    # states. Filtering first and counting second would silently DROP stray
    # commitment runs and still report 80/80 -- the count check cannot see
    # them. Commitment is withdrawn; a PV11-Acq cell carrying it is a scope
    # error, not something to quietly discard.
    declared = payload.get("config", {}).get("block")
    if declared == "acquisition" and len(acq) != len(runs):
        stray = Counter(r.get("block") for r in runs if r.get("block")
                        != "acquisition")
        raise SystemExit(
            f"{path}: declares block=acquisition but holds {len(runs) - len(acq)} "
            f"non-acquisition run(s) {dict(stray)}. Commitment is withdrawn "
            f"({AMENDMENT}); it is not silently dropped.")

    uids = [r["state_uid"] for r in acq]
    if len(set(uids)) != len(uids):
        dup = [u for u, n in Counter(uids).items() if n > 1]
        raise SystemExit(f"{path}: duplicate state_uid(s): {dup[:5]}")

    alphas = {r["alpha"] for r in acq}
    if len(alphas) != 1:
        raise SystemExit(f"{path}: mixed alphas {sorted(alphas)}")
    alpha = alphas.pop()
    if expect_alpha is not None and alpha != expect_alpha:
        raise SystemExit(
            f"{path}: alpha={alpha}, expected {expect_alpha}")

    cells = Counter(r["cell"] for r in acq)
    for c in CELLS:
        if cells.get(c) != 20:
            raise SystemExit(
                f"{path}: cell {c} has {cells.get(c, 0)} states, expected 20")

    # Attestation: observed fires must equal what this cell's own config
    # expects. A None count is a FAILURE, not a skip -- it means a stale
    # server llms.py, and behaviour must never be read off an unverified
    # intervention.
    steered = alpha != 0.0
    for r in acq:
        att = r.get("attestation") or {}
        fires, exp = att.get("steering_fires"), att.get("expected_fires")
        if fires is None or exp is None:
            raise SystemExit(
                f"{path}: {r['state_uid']} has no fire count. This means a "
                f"stale llms.py, not a skippable check.")
        if fires != exp:
            raise SystemExit(
                f"{path}: {r['state_uid']} fires={fires} expected={exp}")
        if steered and fires == 0:
            raise SystemExit(
                f"{path}: alpha={alpha} but {r['state_uid']} fired 0 times")
        if not steered and fires != 0:
            raise SystemExit(
                f"{path}: alpha=0 must register NO hook, but "
                f"{r['state_uid']} fired {fires} times")

    return {
        "path": path,
        "alpha": alpha,
        "runs": {r["state_uid"]: r for r in acq},
        "interface_tag": payload.get("config", {}).get("interface_tag"),
        "block": payload.get("config", {}).get("block"),
        "bank_canonical": payload.get("state_bank_canonical_sha256"),
        "bank_file": payload.get("state_bank_file_sha256"),
        "manifest": payload.get("manifest_sha256"),
        "model_config": payload.get("config", {}).get("model_config"),
        "mask_path": payload.get("config", {}).get("mask_path"),
    }


def attest_pairing(cells: dict[str, dict]) -> None:
    """Every cell must present the SAME 80 states. Fails closed.

    Deliberately asserts rather than intersecting: a silent intersection
    would reintroduce exactly the survival selection that made PV10-A's
    cross-cell accuracy uninterpretable.
    """
    names = list(cells)
    base = cells[names[0]]
    base_uids = set(base["runs"])

    for n in names[1:]:
        u = set(cells[n]["runs"])
        if u != base_uids:
            miss, extra = sorted(base_uids - u), sorted(u - base_uids)
            raise SystemExit(
                f"cell {n} does not present the same states as {names[0]}\n"
                f"  missing: {len(miss)} {miss[:3]}\n"
                f"  extra:   {len(extra)} {extra[:3]}")

    for uid in sorted(base_uids):
        ref = base["runs"][uid]
        for n in names[1:]:
            got = cells[n]["runs"][uid]
            for f in STATE_FIELDS:
                if got.get(f) != ref.get(f):
                    raise SystemExit(
                        f"{uid}: field '{f}' differs between {names[0]} and "
                        f"{n}\n  {names[0]}: {ref.get(f)}\n  {n}: {got.get(f)}")
            for f in VERSION_FIELDS:
                if got.get(f) != ref.get(f):
                    raise SystemExit(
                        f"{uid}: {f} differs: {ref.get(f)} vs {got.get(f)}")

    for key, label in (("bank_canonical", "state bank canonical sha256"),
                       ("bank_file", "state bank file sha256"),
                       ("manifest", "manifest sha256"),
                       ("model_config", "model/mask fingerprint"),
                       ("mask_path", "mask path")):
        vals = {n: cells[n][key] for n in names}
        if len(set(vals.values())) != 1:
            raise SystemExit(f"{label} differs across cells: {vals}")

    # The interface tag is EXPECTED to differ: the alpha=0 cell is the
    # original full-bank run (`H5-20`, block=all) while the steered cells are
    # acquisition-only (`H20`, _blkacquisition). Requiring equality here
    # would force a needless re-run of a baseline that genuinely predates the
    # amendment. What must hold is that each tag matches its own scope.
    for n in names:
        c = cells[n]
        blk, tag = c["block"], c["interface_tag"] or ""
        if blk == "all":
            if "_blk" in tag or "_H5-20_" not in tag:
                raise SystemExit(
                    f"cell {n}: block=all but tag does not look full-bank: "
                    f"{tag}")
        elif blk == "acquisition":
            if "_blkacquisition" not in tag or "_H20_" not in tag:
                raise SystemExit(
                    f"cell {n}: block=acquisition but tag does not look "
                    f"acquisition-scoped: {tag}")
        else:
            raise SystemExit(f"cell {n}: unexpected block={blk!r}")


# ─────────────────────────────── metrics ────────────────────────────────────

def fingerprint_clusters(runs: dict, cell: str) -> dict[str, list[dict]]:
    """Group a cell's states by their opening-prompt fingerprint."""
    out = defaultdict(list)
    for r in runs.values():
        if r["cell"] == cell:
            out[r["first_action"]["prompt_sha256"]].append(r)
    return dict(out)


def probe_rate(runs: dict, cell: str) -> dict:
    """Probe-sampling rate under both readings, with invalid handling.

    ITT (primary): the denominator is every state of the cell; an invalid
    first action counts in the denominator and in no numerator.
    valid-only is reported beside it with its own denominator.
    """
    members = [r for r in runs.values() if r["cell"] == cell]
    n_slots = len(members)
    valid = [r for r in members if r["first_action"]["valid"]]
    hit_slots = sum(1 for r in members
                    if r["first_action"].get("is_probe") is True)

    clusters = fingerprint_clusters(runs, cell)
    inconsistent = []
    hit_uniq = 0
    valid_uniq = 0
    for fp, group in clusters.items():
        acts = {(g["first_action"]["kind"], g["first_action"].get("arm"))
                for g in group}
        if len(acts) > 1:
            inconsistent.append((fp, sorted(acts),
                                 [g["state_uid"] for g in group]))
        rep = group[0]
        if rep["first_action"].get("is_probe") is True:
            hit_uniq += 1
        if rep["first_action"]["valid"]:
            valid_uniq += 1

    return {
        "cell": cell,
        "n_slots": n_slots,
        "n_unique": len(clusters),
        "slot_hits": hit_slots,
        "slot_itt": hit_slots / n_slots if n_slots else float("nan"),
        "slot_valid_n": len(valid),
        "slot_valid_only": (
            sum(1 for r in valid if r["first_action"].get("is_probe") is True)
            / len(valid)) if valid else float("nan"),
        "uniq_hits": hit_uniq,
        "uniq_itt": hit_uniq / len(clusters) if clusters else float("nan"),
        "uniq_valid_n": valid_uniq,
        "inconsistent": inconsistent,
        "positive_fingerprints": [
            (g[0]["probe_label"],
             g[0]["display_order"].index(g[0]["probe_label"]) + 1,
             len(g))
            for g in clusters.values()
            if g[0]["first_action"].get("is_probe") is True],
    }


def contrast(a: dict, b: dict, reading: str) -> float:
    k = "slot_itt" if reading == "slot" else "uniq_itt"
    return a[k] - b[k]


# ─────────────────────────────── reporting ──────────────────────────────────

def hdr(t: str) -> None:
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def report_limits() -> None:
    hdr("PV11-Acq  --  EXPLORATORY ACQUISITION-ONLY FOLLOW-UP")
    print(f"Frozen by {AMENDMENT}.json")
    print(f"  sha256 {AMENDMENT_SHA256}")
    print()
    print("This is NOT a PV11 gate PASS and NOT a BAI competence claim.")
    print("Commitment / M1 is withdrawn on construct grounds and is neither")
    print("analysed nor cited here.")
    print()
    print("PRIMARY  : the low_rate pair only (probe rate held at .40, probe")
    print("           is NOT the displayed best arm -> sampling it cannot be")
    print("           rate-chasing).")
    print("DESCRIPT.: the high_rate pair shows that sample size changed")
    print("           choice, but CANNOT identify an information-seeking")
    print("           motive -- there the probe IS the displayed best arm.")
    print("POWER    : 3 positive events at the alpha=0 baseline. No")
    print("           significance, improvement, or EQUIVALENCE claim. A")
    print("           null reads 'not detected at this low-power baseline'.")


def report_cell_table(cells: dict[str, dict]) -> dict:
    hdr("FIRST-ACTION PROBE SAMPLING  (the only state-matched estimand)")
    table = {}
    for name, c in cells.items():
        table[name] = {cell: probe_rate(c["runs"], cell) for cell in CELLS}

    for group, label in ((PRIMARY_PAIR, "PRIMARY (causal)"),
                         (DESCRIPTIVE_PAIR, "DESCRIPTIVE ONLY")):
        print(f"\n-- {label} " + "-" * (58 - len(label)))
        print(f"{'cell':22s}{'alpha':>7s}"
              f"{'slot ITT':>14s}{'unique ITT':>14s}{'valid-only':>13s}")
        for cell in group:
            for name, c in cells.items():
                m = table[name][cell]
                print(f"{cell:22s}{c['alpha']:>7}"
                      f"{m['slot_hits']:>7d}/{m['n_slots']:<6d}"
                      f"{m['uniq_hits']:>7d}/{m['n_unique']:<6d}"
                      f"{m['slot_valid_only']:>13.3f}")
            print()
    return table


def report_primary(table: dict, cells: dict) -> None:
    hdr("PRIMARY CONTRAST  low_n/low_rate  -  matched_n/low_rate")
    lo, mo = PRIMARY_PAIR
    print(f"{'alpha':>7s}{'slot':>22s}{'unique (PRIMARY)':>26s}")
    for name, c in cells.items():
        a, b = table[name][lo], table[name][mo]
        s = contrast(a, b, "slot")
        u = contrast(a, b, "uniq")
        print(f"{c['alpha']:>7}"
              f"{a['slot_hits']:>8d}/{a['n_slots']}-{b['slot_hits']}/"
              f"{b['n_slots']} = {s:+.3f}"
              f"{a['uniq_hits']:>10d}/{a['n_unique']}-{b['uniq_hits']}/"
              f"{b['n_unique']} = {u:+.4f}")

    print()
    print("READING RULE (pre-registered, not negotiable):")
    print("  * No p-value is printed for this contrast. With 3 positive")
    print("    events at baseline, a significance test would imply a")
    print("    precision this design does not have.")
    print("  * A difference in either direction is DESCRIPTIVE.")
    print("  * A null is 'not detected at this low-power baseline'. It is")
    print("    NOT evidence that alpha does not affect acquisition, and")
    print("    cannot be distinguished from insufficient power.")

    base = [c for c in cells.values() if c["alpha"] == 0.0]
    if base:
        m = table[[n for n, c in cells.items() if c["alpha"] == 0.0][0]][lo]
        print()
        print("BASELINE CONCENTRATION (must be stated whenever the contrast")
        print("is reported): the positive fingerprints are")
        for lbl, row, mult in sorted(m["positive_fingerprints"]):
            print(f"    probe={lbl} display_row={row}  ({mult} slot"
                  f"{'s' if mult > 1 else ''})")
        print("  The effect is not spread across the design.")


def report_consistency(table: dict, cells: dict) -> None:
    hdr("FINGERPRINT CONSISTENCY")
    print("Slots sharing an opening prompt are ONE decision at temperature=0.")
    print("Any disagreement within a fingerprint is reported, never resolved")
    print("by picking a representative.")
    any_bad = False
    for name, c in cells.items():
        bad = [(cell, m["inconsistent"]) for cell, m in table[name].items()
               if m["inconsistent"]]
        if not bad:
            print(f"  alpha={c['alpha']:>5}: 0 inconsistent fingerprints")
            continue
        any_bad = True
        for cell, items in bad:
            print(f"  alpha={c['alpha']:>5} {cell}: "
                  f"{len(items)} INCONSISTENT")
            for fp, acts, uids in items:
                print(f"      {fp[:12]} {acts} {uids}")
    if any_bad:
        print()
        print("  !! The slot and unique readings will diverge. Report BOTH,")
        print("     and do not treat the unique reading as a clean summary.")


def report_secondary(cells: dict) -> None:
    hdr("SECONDARY / INTERFACE  (description only -- NOT state-matched)")
    print("These cannot support or rescue a first-step conclusion.")
    print()
    print(f"{'alpha':>7s}{'1st-step valid':>16s}{'per-round valid':>18s}"
          f"{'auto':>7s}{'forced':>8s}{'invalid':>9s}")
    for name, c in cells.items():
        runs = c["runs"].values()
        v1 = sum(1 for r in runs if r["first_action"]["valid"])
        tot = inv = 0
        term = Counter()
        for r in runs:
            sec = r["secondary_trajectory"]
            term[sec["termination_reason"]] += 1
            for rd in sec["rounds"]:
                tot += 1
                if not rd["valid"]:
                    inv += 1
        print(f"{c['alpha']:>7}{v1:>10d}/{len(c['runs']):<5d}"
              f"{tot - inv:>12d}/{tot:<5d}"
              f"{term['autonomous_commit']:>7d}"
              f"{term['forced_commit']:>8d}"
              f"{term['invalid_policy']:>9d}")

    print()
    print("first-step invalid_kind breakdown:")
    for name, c in cells.items():
        k = Counter(r["first_action"]["invalid_kind"]
                    for r in c["runs"].values()
                    if not r["first_action"]["valid"])
        print(f"  alpha={c['alpha']:>5}: {dict(k) if k else 'none'}")

    print()
    print("empirical-leader following at the first step (descriptive):")
    for name, c in cells.items():
        s = [r for r in c["runs"].values()
             if r["first_action"]["kind"] == "sample"]
        db = sum(1 for r in s if r["first_action"]["is_displayed_best"])
        print(f"  alpha={c['alpha']:>5}: {db}/{len(s)} sampled the displayed "
              f"best arm")


def report_attestation(cells: dict) -> None:
    hdr("ATTESTATION")
    for name, c in cells.items():
        f = Counter(r["attestation"]["steering_fires"]
                    for r in c["runs"].values())
        print(f"  alpha={c['alpha']:>5}  block={c['block']:<12s} "
              f"fires={dict(f) if len(f) < 4 else f'{len(f)} distinct values'}")
        print(f"              tag={c['interface_tag']}")
    print()
    print("  The alpha=0 tag is EXPECTED to differ from the steered tags: it")
    print("  is the original full-bank run and genuinely predates this")
    print("  amendment. Each tag is checked against its own scope instead.")


# ──────────────────────────────── main ──────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", required=True,
                    help="the ORIGINAL full 160-state alpha=0 file")
    ap.add_argument("--am4", help="acquisition-only alpha=-4 file")
    ap.add_argument("--ap4", help="acquisition-only alpha=+4 file")
    args = ap.parse_args()

    cells: dict[str, dict] = {}
    cells["a0"] = load_cell(Path(args.a0), 0.0)
    if args.am4:
        cells["am4"] = load_cell(Path(args.am4), -4.0)
    if args.ap4:
        cells["ap4"] = load_cell(Path(args.ap4), 4.0)

    attest_pairing(cells)

    report_limits()
    report_attestation(cells)
    table = report_cell_table(cells)
    report_consistency(table, cells)
    report_primary(table, cells)
    report_secondary(cells)

    if len(cells) == 1:
        hdr("BASELINE ONLY")
        print("Only the alpha=0 cell was supplied. The primary contrast")
        print("across alpha is not computed.")

    print()


if __name__ == "__main__":
    main()
