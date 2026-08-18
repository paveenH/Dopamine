#!/usr/bin/env python3.10
"""PV11 Controlled Evidence-State Micro-Episodes: state bank builder. PURE.

PV11 is a NEW PROTOCOL, not PV10-D. The online PV10-A/B/C line is closed (see
AdaBandit.md 4.4): four representation-layer interventions -- Stage-1 alpha,
choice history, the Beta calculator and the PV10-C competitor cue -- each
raised recognition without moving acquisition. PV11 therefore stops asking the
model to generate its own evidence and instead HANDS it a controlled evidence
state, so that "willing to continue" and "which arm to sample" can be
separated at a state-matched first action.

WHY THE STATES ARE SYNTHETIC, NOT LIFTED FROM PV10
--------------------------------------------------
The pv7 state bank was sampled from stored alpha=0 trajectories and that is
exactly what made it weak. Two failures, both structural:

  * SELECTION BIAS. A state only appears in an alpha=0 trajectory if the
    model's own policy walked there. PV10's policy almost never samples a
    low-count arm, so "challenger at 2 trials with a low empirical rate" --
    the single cell PV11 most needs -- is the rarest state in its own history.
  * NON-ORTHOGONALITY. In a real trajectory, evidence strength and sample
    balance are correlated by construction (an arm pulled more has tighter
    evidence). Lifting states makes the design observational and the two
    factors inseparable.

So every count in this bank is WRITTEN, not observed. PV10 data enters this
file in exactly one place: `calibration_basis` in the manifest, which records
the DISTRIBUTIONS (quantiles) that bound the synthetic values to magnitudes
the model could actually meet in a T=100 environment. Distributions, never
states.

THE `.774` LESSON IS THE REASON FOR THE 2x2
-------------------------------------------
PV10-C's `next_sample_targets_competitor` = .774 is deprecated as
CONSTRUCT-INVALID: a one-shot `1/1` arm becomes the empirical-rate leader, so
the confirmed incumbent got classified as "competitor" and continuing to
sample it scored as alignment. 291 of its 308 "aligned" rounds sampled the
MOST-sampled arm and only 4 the least.

The lesson is that "did it sample the challenger" is unreadable unless
sampling the challenger is decoupled from chasing its rate. Hence the
Acquisition block crosses `probe sample size` with `probe empirical rate`:

    low-n  / high-rate  -> ambiguous (information OR rate-chasing)
    low-n  / low-rate   -> CLEAN: sampling it can only be information-seeking
    matched-n / high-rate -> rate-chasing control
    matched-n / low-rate  -> sample-size control

The name `probe_arm` (never `challenger`) is deliberate: under high-rate the
probe IS the empirical leader, and a metric named "challenger" would then be
self-contradicting -- which is how .774 happened.

TWO LAYERS PER STATE
--------------------
    displayed : the counts/rates the model is shown. Synthetic.
    latent    : each arm's true probability, used to emit REAL rewards after a
                SAMPLE and to score the final commit.

They are deliberately allowed to disagree; that disagreement is what builds
the low-n/low-rate cell. Rewards after SAMPLE are real, because a model that
knows sampling is inert would rationally refuse to sample and the resulting
null would be uninterpretable (the pv7 frozen-state scope limit).

BALANCE
-------
`probe true rank`, `probe label` and `probe display row` are each EXACTLY
5/5/5/5 marginally over the 20 Acquisition state_ids. Their pairwise crossings
are balanced only to within a max-min count difference of 1 -- with 20 states
and three 4-level factors, exact joint orthogonality is not attainable, and
this file does not claim it. `test_pv11_state_bank.py` asserts the marginals
exactly and the crossings at the <=1 tolerance.

Run:  python3.10 build_pv11_state_bank.py          # writes bank + manifest
      python3.10 build_pv11_state_bank.py --check  # recompute and diff
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BANK_PATH = HERE / "pv11_state_bank.json"
MANIFEST_PATH = HERE / "pv11_state_manifest.json"

# ───────────────────────────── frozen versions ──────────────────────────────
PROTOCOL_VERSION = "pv11"
STATE_BANK_VERSION = "pv11-states-v1"
ARM_LABELS = ("A", "B", "C", "D")
K = 4
N_PER_CELL = 20

# Deterministic construction: no RNG anywhere in this file. Every assignment
# below is an explicit table or a modular rotation, so the bank is reproducible
# without carrying a seed, and a diff in --check means a real edit.

# ────────────────────────── Commitment block counts ─────────────────────────
# Every arm carries 20 trials: sample size is held constant so the ONLY thing
# that varies is leader-challenger separation. n=20 is also the smallest n that
# can express a .05 gap exactly (11/20 vs 10/20); n=15 cannot.
#
# gap .05 matches PV10-A's median leader-challenger gap (.049); gap .20 sits
# inside the range spanned by PV10-A/B/C (.049/.238/.287). Both are reachable
# magnitudes, neither copies a skewed centre.
COMMITMENT_COUNTS = {
    # (successes, trials) by evidence rank 1..4
    "weak":   [(11, 20), (10, 20), (8, 20), (6, 20)],   # gap .05
    "strong": [(14, 20), (10, 20), (8, 20), (6, 20)],   # gap .20
}
COMMITMENT_HORIZONS = {"short": 5, "long": 20}

# ────────────────────────── Acquisition block counts ────────────────────────
# The three non-probe arms are IDENTICAL across all four cells, so any cell
# difference is attributable to the probe alone.
ACQUISITION_FIXED = [
    ("incumbent",   (11, 20)),   # .55
    ("alternative", (10, 20)),   # .50
    ("other",       (6, 20)),    # .30
]
ACQUISITION_PROBE = {
    ("low_n", "high_rate"):     (3, 5),     # .60
    ("low_n", "low_rate"):      (2, 5),     # .40
    ("matched_n", "high_rate"): (12, 20),   # .60
    ("matched_n", "low_rate"):  (8, 20),    # .40
}
ACQUISITION_HORIZON = 20

# Latent probabilities. The probe's TRUE rank is balanced 5/5/5/5 across the 20
# state_ids so the bank contains no learnable rule of the form "the low-count
# arm is always the best one". The ladder is PV10's frozen environment ladder,
# reused because it is a known-reachable spacing, not because the environment
# is shared.
LATENT_LADDER = (0.60, 0.50, 0.40, 0.30)


def _probe_rank_by_state() -> list[int]:
    """Probe's TRUE rank (1=best) per state_id, exactly 5 of each."""
    return [(i % 4) + 1 for i in range(N_PER_CELL)]


def _probe_label_by_state() -> list[str]:
    """Probe's LABEL per state_id, exactly 5 of each.

    Offset by i//4 against the rank cycle so label does not track rank; with
    20 states and 4 levels each this yields a rank x label crossing whose cell
    counts differ by at most 1.
    """
    return [ARM_LABELS[(i % 4 + i // 4) % 4] for i in range(N_PER_CELL)]


# Probe display row, as an EXPLICIT table rather than a formula.
#
# The first attempt used `((i % 4) + 2 * (i // 4)) % 4`. Its marginals were a
# perfect 5/5/5/5 and it still failed: an EVEN offset against the `i % 4` rank
# cycle keeps rank and row at the same parity, so 8 of the 16 rank x row cells
# were empty (observed spread 3, counts {(1,1):3, (1,3):2, ...}). Balanced
# margins with a degenerate crossing is precisely the confound this bank exists
# to avoid -- probe rank would have been recoverable from its display row.
#
# Rows here are one permutation of (1,2,3,4) per block of 4 state_ids, chosen
# by exhaustive search so that ALL THREE pairwise crossings (rank x label,
# rank x row, label x row) have a max-min cell-count difference of <= 1.
# `test_pv11_state_bank.py` re-verifies that property rather than trusting it.
_PROBE_ROW_BLOCKS = (
    (1, 2, 3, 4),
    (1, 2, 3, 4),
    (4, 1, 2, 3),
    (2, 3, 4, 1),
    (3, 4, 1, 2),
)


def _probe_row_by_state() -> list[int]:
    """Probe's DISPLAY ROW (1-indexed) per state_id, exactly 5 of each."""
    return [_PROBE_ROW_BLOCKS[i // 4][i % 4] for i in range(N_PER_CELL)]


def _display_order(probe_label: str, probe_row: int) -> list[str]:
    """Row order placing `probe_label` at `probe_row`, others in label order."""
    rest = [x for x in ARM_LABELS if x != probe_label]
    order = list(rest)
    order.insert(probe_row - 1, probe_label)
    return order


def build_commitment() -> list[dict]:
    """80 states: 2 evidence levels x 2 horizons x 20 matched state_ids.

    The four cells of one state_id share latent probabilities and the
    continuation tape, so evidence x horizon is a within-state_id contrast.
    """
    states = []
    for sid in range(N_PER_CELL):
        # Which LABEL holds evidence rank 1..4, rotated by state_id so the
        # leading label is balanced 5/5/5/5 across state_ids.
        rank_to_label = [ARM_LABELS[(r + sid) % 4] for r in range(4)]
        # Display row order rotated independently of the label rotation.
        display = [ARM_LABELS[(j + 2 * sid) % 4] for j in range(4)]
        # Latent: rank 1 by displayed evidence is genuinely best. Commitment
        # asks about the COMMIT THRESHOLD under given evidence, so displayed
        # and latent are aligned here -- the displayed/latent split is the
        # Acquisition block's instrument, not this one.
        latent = {rank_to_label[r]: LATENT_LADDER[r] for r in range(4)}
        for evidence, counts in COMMITMENT_COUNTS.items():
            displayed = {rank_to_label[r]: list(counts[r]) for r in range(4)}
            for horizon_name, horizon in COMMITMENT_HORIZONS.items():
                states.append({
                    "state_uid": f"C-{evidence}-{horizon_name}-{sid:02d}",
                    "block": "commitment",
                    "state_id": sid,
                    "cell": f"{evidence}/{horizon_name}",
                    "evidence": evidence,
                    "horizon_name": horizon_name,
                    "remaining_horizon": horizon,
                    "display_order": display,
                    "displayed_counts": displayed,
                    "latent_probs": latent,
                    "true_best": rank_to_label[0],
                    "displayed_best": rank_to_label[0],
                    "tape_key": f"C-{sid:02d}",
                })
    return states


def build_acquisition() -> list[dict]:
    """80 states: 2 sample sizes x 2 rates x 20 matched state_ids."""
    ranks = _probe_rank_by_state()
    labels = _probe_label_by_state()
    rows = _probe_row_by_state()
    states = []
    for sid in range(N_PER_CELL):
        probe_label = labels[sid]
        probe_rank = ranks[sid]
        probe_row = rows[sid]
        display = _display_order(probe_label, probe_row)
        others = [x for x in ARM_LABELS if x != probe_label]
        # Fixed arms take incumbent/alternative/other in label order; the probe
        # is spliced into the latent ladder at its assigned TRUE rank.
        fixed = {lab: list(c) for lab, (_, c)
                 in zip(others, ACQUISITION_FIXED)}
        other_ranks = [r for r in range(1, 5) if r != probe_rank]
        latent = {probe_label: LATENT_LADDER[probe_rank - 1]}
        for lab, r in zip(others, other_ranks):
            latent[lab] = LATENT_LADDER[r - 1]
        true_best = min(latent, key=lambda x: (-latent[x], x))
        for (size, rate), probe_counts in ACQUISITION_PROBE.items():
            displayed = dict(fixed)
            displayed[probe_label] = list(probe_counts)
            states.append({
                "state_uid": f"A-{size}-{rate}-{sid:02d}",
                "block": "acquisition",
                "state_id": sid,
                "cell": f"{size}/{rate}",
                "probe_sample_size": size,
                "probe_rate_level": rate,
                "probe_label": probe_label,
                "probe_true_rank": probe_rank,
                "probe_display_row": probe_row,
                "remaining_horizon": ACQUISITION_HORIZON,
                "display_order": display,
                "displayed_counts": displayed,
                "latent_probs": latent,
                "true_best": true_best,
                "displayed_best": min(
                    displayed,
                    key=lambda x: (-displayed[x][0] / displayed[x][1], x)),
                "tape_key": f"A-{sid:02d}",
            })
    return states


def build_bank() -> dict:
    commitment = build_commitment()
    acquisition = build_acquisition()
    return {
        "state_bank_version": STATE_BANK_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "k": K,
        "arm_labels": list(ARM_LABELS),
        "n_states": len(commitment) + len(acquisition),
        "states": commitment + acquisition,
    }


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


# ─────────────────────────────── manifest ───────────────────────────────────
# calibration_basis: PV10 alpha=0 cells ONLY. Steered cells are excluded so
# that the choice of synthetic magnitudes cannot be influenced by steered
# behaviour. These are DISTRIBUTIONS used to bound the synthetic values; no
# PV10 state is copied into the bank.
CALIBRATION_BASIS = {
    "policy": ("PV10-A/B/C alpha=0 cells ONLY. Steered cells excluded so the "
               "choice of synthetic magnitudes cannot be influenced by "
               "steered behaviour. Used to BOUND magnitudes, never to copy "
               "states -- see the module docstring on selection bias and "
               "non-orthogonality."),
    "sources": {
        "pv10a_v2_a0": {"sha256_prefix": "cfceec33d50c"},
        "pv10b_v2_a0": {"sha256_prefix": "bddae1fc7375"},
        "pv10c_a0":    {"sha256_prefix": "d9ac63de846c"},
    },
    "observed": {
        "exit_arm_trials_q25_median_q75": {
            "pv10a_v2_a0": [1, 4, 39],
            "pv10b_v2_a0": [1, 1, 3],
            "pv10c_a0":    [1, 1, 23.5],
        },
        "leader_challenger_gap_median": {
            "pv10a_v2_a0": 0.049,
            "pv10b_v2_a0": 0.238,
            "pv10c_a0":    0.287,
        },
        "max_arm_share_median": {
            "pv10a_v2_a0": 0.556,
            "pv10b_v2_a0": 0.571,
            "pv10c_a0":    0.960,
        },
        "remaining_budget_at_commit_median": {
            "pv10a_v2_a0": 0,
            "pv10b_v2_a0": 88,
            "pv10c_a0":    0,
        },
    },
    "derivation": {
        "commitment_gap_weak_0.05": ("matches PV10-A median gap .049; n=20 is "
                                     "the smallest n expressing .05 exactly"),
        "commitment_gap_strong_0.20": ("inside the .049-.287 range spanned by "
                                       "the three alpha=0 cells"),
        "trials_20": ("within the observed exit-arm trial range; PV10-A Q75 "
                      "is 39, so 20 is reachable and not extreme"),
        "probe_low_n_5": ("above the pervasive min_trials=1 floor but well "
                          "below matched-n, so under-sampling is unambiguous"),
        "horizon_5_and_20": ("PV10-B commits with a median 88 budget "
                             "remaining; 5 and 20 bracket a scarce and an "
                             "ample continuation without exposing a total"),
    },
    "caveat": ("These distributions are themselves products of PV10's failing "
               "policy (max_arm_share .96 in PV10-C is lock-in, not a target). "
               "They bound attainable magnitudes only; their skewed central "
               "values are deliberately NOT copied."),
}

MANIPULATION_GATE = {
    "unit_of_inference": "state (n=20 matched state_ids per cell)",
    "denominator_convention": "ITT",
    "M1_commitment_sensitivity": {
        "statistic": "P(first action = COMMIT)",
        "primary": ("P(COMMIT | strong) - P(COMMIT | weak) >= 0.15, pooled "
                    "over horizons"),
        "threshold": 0.15,
        "direction": ">=",
        "secondary": ("no pronounced reversal within either horizon level; "
                      "reported, human-read, NOT mechanized"),
        "rationale": ("if commitment does not respond to evidence strength at "
                      "alpha=0, a steered commitment contrast has no "
                      "behaviour to move"),
    },
    "M2_acquisition_sensitivity": {
        "statistic": "P(first action = SAMPLE probe_arm)",
        "primary": ("P(sample probe | low_n/low_rate) - P(sample probe | "
                    "matched_n/low_rate) >= 0.15"),
        "threshold": 0.15,
        "direction": ">=",
        "why_low_rate_only": ("the high_rate cells cannot separate "
                             "information-seeking from rate-chasing; that "
                             "conflation is exactly what invalidated PV10-C's "
                             ".774"),
        "why_not_an_absolute_floor": ("an absolute floor such as 4/20 = .20 "
                                      "sits at the K=4 random-sampling rate "
                                      "(~.25), so uniform random behaviour "
                                      "would PASS it. The cross-cell "
                                      "difference is ~0 under random "
                                      "sampling, so only the difference is "
                                      "gated."),
    },
    "interface_validity": {
        "statistic": "fraction of states with a parseable first action",
        "threshold": 0.90,
        "direction": ">=",
        "note": "interface check, NOT a capability criterion",
    },
    "on_failure": ("Manipulation-gate failure does not trigger prompt "
                   "iteration. It means that this controlled interface did "
                   "not elicit the prerequisite behavior needed for an "
                   "interpretable steering test; no steered cells are run."),
}

ANALYSIS_CONSTRAINTS = {
    "primary_estimand": ("the FIRST action only. It is the sole strictly "
                         "state-matched quantity: after it, action and reward "
                         "fork the trajectory. Everything downstream is "
                         "secondary/descriptive and must never be called a "
                         "state-matched alpha effect."),
    "itt_denominator": ("M1/M2 primaries use ITT: the denominator is all 20 "
                        "states of the cell; invalid counts in the "
                        "denominator and not in any numerator. valid-only "
                        "figures are reported ALONGSIDE with their explicit "
                        "denominator. invalid is its own first-action "
                        "category. Rationale: PV10-A's cross-alpha accuracy "
                        "became uninterpretable because each cell dropped a "
                        "different, alpha-dependent seed set."),
    "within_cell_reference_is_diagnostic": (
        "P(sample probe) > mean P(sample other non-leader arms) is a PER-CELL "
        "DIAGNOSTIC and is NOT cross-cell comparable, because it asks a "
        "different question in each: in low_n the probe is the only "
        "low-sample arm ('is the under-sampled arm preferred'), while in "
        "matched_n all arms are at 20 ('is the low-rate arm preferred at "
        "equal n'). It does not participate in PASS/FAIL."),
    "no_cross_block_pooling": (
        "Commitment and Acquisition are analysed separately and their horizon "
        "effects must never be merged. Their initial evidence totals differ "
        "(80 trials for Commitment; 65 or 80 for Acquisition), so '20 more "
        "samples' denotes a different fraction of accumulated evidence in "
        "each block."),
    "history_prohibition": (
        "PV11's main experiment must NOT include per-pull arm-label history. "
        "Current evidence is represented ONLY by explicit counts/rates. This "
        "avoids binding sample size to label token frequency, and structurally "
        "prevents reconstructing count-based metrics from a label history -- "
        "the degeneration that made empirical_top2_share banned in "
        "count-based form. A history-present presentation control may be run "
        "only AFTER the full +-4 main experiment."),
    "balance_claim": (
        "probe true rank, probe label and probe display row are each EXACTLY "
        "5/5/5/5 marginally. Their pairwise crossings are balanced only to "
        "within a max-min cell-count difference of 1; with 20 states and "
        "three 4-level factors exact joint orthogonality is unattainable and "
        "is NOT claimed."),
    "tape_indexing": (
        "Continuation reward tapes start at pull index 0 at the FIRST SAMPLE "
        "AFTER the synthetic state. They do not continue the synthetic "
        "counts' notional pull history -- those counts are a display, not a "
        "sampled history. All alpha cells share the tape for a given "
        "(tape_key, arm, pull_index)."),
    "shared_within_state_id": (
        "The four cells of one state_id share latent_probs and tape_key in "
        "BOTH blocks, so each block's 2x2 is a within-state_id contrast."),
}


def build_manifest(bank: dict) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "state_bank_version": STATE_BANK_VERSION,
        "builder_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "state_bank_sha256": sha256(bank),
        "k": K,
        "arm_labels": list(ARM_LABELS),
        "n_states": bank["n_states"],
        "blocks": {
            "commitment": {
                "n_states": 80,
                "design": "2 evidence x 2 horizon x 20 matched state_ids",
                "counts": {k: [list(c) for c in v]
                           for k, v in COMMITMENT_COUNTS.items()},
                "horizons": COMMITMENT_HORIZONS,
                "trials_per_arm": 20,
            },
            "acquisition": {
                "n_states": 80,
                "design": "2 sample size x 2 rate x 20 matched state_ids",
                "fixed_arms": {n: list(c) for n, c in ACQUISITION_FIXED},
                "probe_counts": {f"{a}/{b}": list(v)
                                 for (a, b), v in ACQUISITION_PROBE.items()},
                "horizon": ACQUISITION_HORIZON,
                "naming": ("the arm is `probe_arm`, never `challenger`: under "
                           "high_rate it IS the empirical leader, and a name "
                           "asserting otherwise is how .774 happened"),
            },
        },
        "latent_ladder": list(LATENT_LADDER),
        "calibration_basis": CALIBRATION_BASIS,
        "manipulation_gate": MANIPULATION_GATE,
        "analysis_constraints": ANALYSIS_CONSTRAINTS,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="recompute and diff against the stored files")
    args = ap.parse_args()

    bank = build_bank()
    manifest = build_manifest(bank)

    if args.check:
        ok = True
        for path, fresh in ((BANK_PATH, bank), (MANIFEST_PATH, manifest)):
            if not path.exists():
                print(f"MISSING {path.name}")
                ok = False
                continue
            stored = json.loads(path.read_text())
            if canonical(stored) != canonical(fresh):
                print(f"MISMATCH {path.name}")
                ok = False
            else:
                print(f"ok {path.name}")
        if ok:
            print(f"\nstate_bank_sha256 {manifest['state_bank_sha256']}")
        return 0 if ok else 1

    BANK_PATH.write_text(json.dumps(bank, indent=2, sort_keys=True) + "\n")
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {BANK_PATH.name} ({bank['n_states']} states)")
    print(f"wrote {MANIFEST_PATH.name}")
    print(f"state_bank_sha256 {manifest['state_bank_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
