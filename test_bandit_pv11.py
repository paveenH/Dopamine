#!/usr/bin/env python3.10
"""PV11 prompt-surface invariants. Uses the REAL Llama-3.1 tokenizer. No GPU.

Scope is the PROMPT layer only; state-bank structure is `test_pv11_state_bank`.
Kept to the mechanical failures that would void a whole batch:

  * the anchor moving off token 220 (pv7's `.rstrip()` trap -- silent, and it
    relocates the injection site to `:` (25))
  * a withheld field leaking into the prompt (history / totals / probe
    identity), which would dissolve the manipulation being measured
  * inheriting PV10's `tau - k*FORCED_INIT` fire formula, which under-counts by
    k here and, since attestation is fail-closed, raises on the first seed
  * the parser silently diverging from PV10-v2

If the tokenizer is not in the local HF cache the token checks SKIP with a
visible notice rather than passing vacuously.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bandit_pv10 as p10
import bandit_pv11 as p11

HERE = Path(__file__).resolve().parent
FAILURES: list[str] = []
SKIPPED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


def _load_states():
    bank = json.loads((HERE / "pv11_state_bank.json").read_text())
    return bank["states"]


def _prompt(state, remaining=None):
    counts = {k: tuple(v) for k, v in state["displayed_counts"].items()}
    h = state["remaining_horizon"] if remaining is None else remaining
    return p11.build_decision_prompt(state["display_order"], counts, h)


def main() -> int:
    states = _load_states()
    comm = [s for s in states if s["block"] == "commitment"]
    acq = [s for s in states if s["block"] == "acquisition"]

    # ── the anchor ends every prompt, at every horizon ──────────────────────
    for s in states:
        p = _prompt(s)
        check(p.endswith(p11.REASON_ANCHOR),
              f"{s['state_uid']}: prompt does not end at the anchor")
        check(not p.endswith("  "),
              f"{s['state_uid']}: prompt ends in a double space (token 256)")
    for h in (0, 1, 5, 20):
        p = _prompt(comm[0], remaining=h)
        check(p.endswith("Reason: "), f"H={h}: anchor lost")

    # ── real tokenizer: the tail must be token 220 ─────────────────────────
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct", local_files_only=True)
    except Exception as exc:                       # noqa: BLE001
        SKIPPED.append(f"tokenizer unavailable ({type(exc).__name__}); "
                       f"token-220 assertions SKIPPED")
        tok = None
    if tok is not None:
        for s in (comm[0], acq[0]):
            for h in (0, 1, 20):
                ids = tok(_prompt(s, remaining=h),
                          add_special_tokens=False)["input_ids"]
                check(ids[-1] == 220,
                      f"{s['state_uid']} H={h}: tail token {ids[-1]} != 220")
        # Bare candidate letters, as forced by the trailing space.
        for lab in p11.ARM_LABELS:
            ids = tok(lab, add_special_tokens=False)["input_ids"]
            check(len(ids) == 1 and ids[0] == ord(lab) - ord("A") + 32,
                  f"candidate {lab!r} is not the expected bare-letter token")

    # ── withheld fields must NOT appear ────────────────────────────────────
    banned = ("CHOICE HISTORY", "Samples used", "Samples remaining",
              "probe", "Probe", "latent", "Latent", "in total",
              "acquisition", "Acquisition", "commitment", "Commitment")
    for s in states:
        p = _prompt(s)
        for token in banned:
            check(token not in p,
                  f"{s['state_uid']}: withheld term {token!r} leaked into "
                  f"the prompt")
    # No integer in the prompt may equal the initial evidence total, which
    # differs across Acquisition cells (65 vs 80) and would re-expose the
    # sample-size manipulation as a budget difference.
    for s in acq:
        total = sum(t for _, t in s["displayed_counts"].values())
        p = _prompt(s)
        check(f" {total} " not in p and f" {total}\n" not in p,
              f"{s['state_uid']}: evidence total {total} appears in the prompt")

    # ── the four Acquisition cells differ ONLY in the probe row ────────────
    for sid in range(20):
        group = sorted((s for s in acq if s["state_id"] == sid),
                       key=lambda x: x["state_uid"])
        probe = group[0]["probe_label"]
        lines = [{ln for ln in _prompt(s).splitlines()
                  if ln.startswith("- Button ")} for s in group]
        common = set.intersection(*lines)
        check(len(common) == 3,
              f"A-{sid}: cells share {len(common)} OPTION lines, expected 3")
        check(all(f"Button {probe}:" not in ln for ln in common),
              f"A-{sid}: the probe line is identical across cells")
        # and the horizon line is the same in all four
        hz = {p11.format_horizon(s["remaining_horizon"]) for s in group}
        check(len(hz) == 1, f"A-{sid}: horizon differs across cells")

    # ── the two Commitment horizons differ ONLY in the horizon line ────────
    for sid in range(20):
        for ev in ("weak", "strong"):
            pair = [s for s in comm
                    if s["state_id"] == sid and s["evidence"] == ev]
            a, b = (_prompt(x) for x in sorted(
                pair, key=lambda x: x["remaining_horizon"]))
            da = [x for x in a.splitlines() if x not in b.splitlines()]
            db = [x for x in b.splitlines() if x not in a.splitlines()]
            check(len(da) == 1 and len(db) == 1,
                  f"C-{ev}-{sid}: horizons differ in more than one line: "
                  f"{da} vs {db}")
            check("sample" in da[0] and "sample" in db[0],
                  f"C-{ev}-{sid}: the differing line is not the horizon line")

    # ── terminal variant withdraws SAMPLE ──────────────────────────────────
    term = _prompt(comm[0], remaining=0)
    check("Policy: SAMPLE Button X" not in term,
          "terminal prompt still offers SAMPLE")
    check("Policy: COMMIT Button X" in term,
          "terminal prompt does not offer COMMIT")
    sampling = _prompt(comm[0], remaining=5)
    check("Policy: SAMPLE Button X" in sampling,
          "sampling prompt does not offer SAMPLE")

    # ── fire arithmetic is PV11's, not PV10's ──────────────────────────────
    check(p11.expected_fires_max(5) == 6 * 9,
          "expected_fires_max(5) != 54")
    check(p11.expected_fires_max(20) == 21 * 9,
          "expected_fires_max(20) != 189")
    check(p11.expected_fires_for_calls(1) == 9,
          "a single-call episode must be 9 sites")
    # PV10's formula would give (tau - 4) + 1; assert PV11 does NOT match it
    # for any plausible tau, i.e. that nobody re-imported it.
    check(p11.expected_fires_max(20) != p10.expected_fires(4, 20 - 4 + 1, 4),
          "PV11 fire arithmetic coincides with PV10's -- check the import")
    for bad in (-1,):
        try:
            p11.expected_fires_max(bad)
            check(False, f"expected_fires_max({bad}) did not raise")
        except ValueError:
            pass

    # ── parser and stop boundary are PV10-v2, shared not copied ────────────
    check(p11.parse_policy is p10.parse_policy,
          "parse_policy is not PV10's object")
    check(p11.apply_stop_boundary is p10.apply_stop_boundary,
          "apply_stop_boundary is not PV10's object")
    check(p11.POLICY_PARSER_VERSION == "pv10-strict-v2",
          f"parser version drifted: {p11.POLICY_PARSER_VERSION}")
    check(p11.STOP_STRINGS == ("#",), "stop strings drifted")
    # strictness spot-check: prose after the arm stays invalid
    arms = list(p11.ARM_LABELS)
    good = p11.parse_policy("Reason: A leads.\nPolicy: SAMPLE Button A", arms)
    check(good.valid and good.action == "SAMPLE" and good.arm == "A",
          "a canonical SAMPLE line failed to parse")
    bad = p11.parse_policy(
        "Reason: x\nPolicy: COMMIT Button A because it is best", arms)
    check(not bad.valid, "prose after the arm was accepted")
    term_bad = p11.parse_policy(
        "Reason: x\nPolicy: SAMPLE Button A", arms, terminal=True)
    check(not term_bad.valid, "SAMPLE was accepted at a terminal decision")

    # ── resume key cannot collide with PV10/PV10-C ─────────────────────────
    tag = p11.interface_tag(range(20), {5, 20})
    check(tag.startswith("pv11_p11_"), f"unexpected interface tag: {tag}")
    # NOTE the argument orders differ: PV10 is (k, seeds), PV11 is
    # (states, horizons). They are separate functions and neither delegates.
    check(tag != p10.interface_tag(4, list(range(20))),
          "PV11 interface tag collides with PV10's")
    check(p11.interface_tag(range(20), {5}) != tag,
          "interface tag ignores the horizon set")
    check(p11.interface_tag(range(19), {5, 20}) != tag,
          "interface tag ignores the state set")

    # ── malformed input raises rather than rendering ───────────────────────
    st = comm[0]
    counts = {k: tuple(v) for k, v in st["displayed_counts"].items()}
    for bad_call, label in (
            (lambda: p11.build_decision_prompt(["A", "B"], counts, 5),
             "display_order/counts mismatch"),
            (lambda: p11.build_decision_prompt(["A", "A", "B", "C"],
                                               counts, 5),
             "duplicate display row"),
            (lambda: p11.format_horizon(-1), "negative horizon")):
        try:
            bad_call()
            check(False, f"{label} did not raise")
        except (ValueError, KeyError):
            pass

    if FAILURES:
        print(f"FAIL ({len(FAILURES)})")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    for s in SKIPPED:
        print(f"SKIP  {s}")
    print("ok  test_bandit_pv11.py  (prompt surface, anchor, fires, parser)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
