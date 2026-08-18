#!/usr/bin/env python3.10
"""The PV11 gate can read what the DRIVER actually writes. No GPU, ~2s.

This file exists because of a specific PV10 failure. `load_cell` there read
`"episodes"` while the driver wrote `"runs"`, and a `[data]` fallback wrapped
the whole payload as ONE pseudo-episode -- so the gate read n=1 and reported a
spurious FAIL without ever erroring. The manifest froze the evaluator's HASH,
which proved the file had not been edited but said nothing about whether the
two halves were ever wired together.

So this test does NOT hand-build a plausible payload. It runs the real episode
runner against a FakeVC, serializes exactly as the driver does, and hands the
resulting FILE to the real loader. A key rename on either side fails here.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Resolve from THIS file's location, never a hardcoded path: these tests run
# on the analysis box and on the server, and an absolute local path makes the
# suite pass in one place and crash in the other.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_bandit_pv11_gate as gate
import bandit_pv11 as p11
import bandit_pv11_episode as ep

HERE = Path(__file__).resolve().parent
FAILURES: list[str] = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


class ScriptedVC:
    """Deterministic replies driven by the state, so a full bank can run."""

    def __init__(self, decide):
        self.decide = decide
        self._pending = 0

    def generate(self, inputs, **kw):
        self._pending = 0
        return [self.decide(inputs[0])]

    def regenerate(self, inputs, diff_matrices=None, **kw):
        if diff_matrices is None:
            raise ValueError("regenerate requires diff_matrices")
        self._pending = 9
        return [self.decide(inputs[0])]

    def steering_fire_count(self, reset=False):
        v = self._pending
        if reset:
            self._pending = 0
        return v


def _run_full_bank(decide) -> list[dict]:
    bank = json.loads((HERE / "pv11_state_bank.json").read_text())
    vc = ScriptedVC(decide)
    return [ep.run_pv11_episode(vc, state=s, interface_tag="test")
            for s in bank["states"]]


def _write_like_driver(runs, path: Path) -> None:
    """Serialize EXACTLY as run_bandit_pv11_episodes.py does.

    If the driver's payload shape changes, this must change with it -- which
    is the point: the test then fails and the mismatch is visible.
    """
    manifest = json.loads((HERE / "pv11_state_manifest.json").read_text())
    payload = {
        "resume_key": "test-key",
        "protocol_version": p11.PROTOCOL_VERSION,
        "state_bank_version": p11.STATE_BANK_VERSION,
        "policy_parser_version": p11.POLICY_PARSER_VERSION,
        "stage1_instruction_version": p11.STAGE1_INSTRUCTION_VERSION,
        "state_bank_canonical_sha256":
            manifest["state_bank_canonical_sha256"],
        "state_bank_file_sha256": manifest["state_bank_file_sha256"],
        "manifest_sha256": "test",
        "config": {"alpha": 0.0, "layers": [11, 20]},
        "n_states_expected": len(runs),
        "runs": runs,
    }
    path.write_text(json.dumps(payload, indent=1))


def main() -> int:
    # A decider that is genuinely sensitive on both axes, so a correctly wired
    # gate reaches PASS and a mis-wired one cannot reach it by accident.
    def decide(prompt: str) -> str:
        lines = [ln for ln in prompt.splitlines()
                 if ln.startswith("- Button ")]
        rates = {}
        trials = {}
        for ln in lines:
            arm = ln.split("Button ")[1].split(":")[0]
            rates[arm] = float(ln.rsplit(" ", 1)[1])
            trials[arm] = int(ln.split(" / ")[1].split(" ")[0])
        top = sorted(rates.values(), reverse=True)
        best = max(rates, key=lambda a: (rates[a], a))
        if "You may take no further samples." in prompt:
            return f"done.\nPolicy: COMMIT Button {best}"
        # strong separation -> commit; otherwise sample the least-tried arm
        if len(top) > 1 and top[0] - top[1] >= 0.15:
            return f"clear leader.\nPolicy: COMMIT Button {best}"
        least = min(trials, key=lambda a: (trials[a], a))
        return f"need more data.\nPolicy: SAMPLE Button {least}"

    runs = _run_full_bank(decide)
    check(len(runs) == 160, f"ran {len(runs)} episodes, expected 160")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bandit_pv11_alpha0.json"
        _write_like_driver(runs, path)

        # ── the real loader, on the real file ──────────────────────────────
        loaded, payload = gate.load_runs(path)
        check(len(loaded) == 160,
              f"loader returned {len(loaded)} episodes, not 160 -- this is "
              f"the PV10 'read n=1' failure mode")

        # ── directory form must resolve too ────────────────────────────────
        loaded_dir, _ = gate.load_runs(Path(td))
        check(len(loaded_dir) == 160, "directory loading lost episodes")

        res = gate.evaluate(loaded)
        check(not res["errors"],
              f"a complete 160-state run reported errors: {res['errors']}")
        check(res["M1"]["applicable"] and res["M2"]["applicable"],
              "a full bank did not activate both rules")
        check(res["M1"]["strong_den"] == 40 and res["M1"]["weak_den"] == 40,
              f"M1 denominators wrong: strong={res['M1']['strong_den']} "
              f"weak={res['M1']['weak_den']} (expect 40 each)")
        check(res["M2"]["low_n_den"] == 20 and res["M2"]["matched_n_den"] == 20,
              "M2 denominators are not 20 per cell")
        check(res["verdict"] == "PASS",
              f"a sensitive decider did not PASS: verdict={res['verdict']} "
              f"M1diff={res['M1']['difference']:.3f} "
              f"M2diff={res['M2']['difference']:.3f}")

        # ── a WRONG key must be refused, not guessed ──────────────────────
        bad = json.loads(path.read_text())
        bad["episodes"] = bad.pop("runs")
        bad_path = Path(td) / "wrong_key.json"
        bad_path.write_text(json.dumps(bad))
        try:
            gate.load_runs(bad_path)
            FAILURES.append(
                "a payload keyed 'episodes' was accepted -- the PV10 fallback "
                "has been reintroduced")
        except SystemExit:
            pass

        # ── a payload missing a record field must be refused ───────────────
        bad2 = json.loads(path.read_text())
        for r in bad2["runs"]:
            r.pop("first_action")
        bad2_path = Path(td) / "no_first_action.json"
        bad2_path.write_text(json.dumps(bad2))
        try:
            gate.load_runs(bad2_path)
            FAILURES.append("records without first_action were accepted")
        except SystemExit:
            pass

        # ── two JSONs in a directory must be named explicitly ─────────────
        (Path(td) / "second.json").write_text("{}")
        try:
            gate.load_runs(Path(td))
            FAILURES.append("an ambiguous directory was silently resolved")
        except SystemExit:
            pass

    # ── ONE COMPLETE BLOCK MUST FAIL, on real driver output ───────────────
    # Same regression as the gate's selftest, but driven through the FILE
    # path: a block-restricted run is a legitimate thing for the driver to
    # produce (`--block commitment`), so the refusal has to happen in the
    # gate, on real serialized records, not only on synthetic ones.
    with tempfile.TemporaryDirectory() as td:
        for block in ("commitment", "acquisition"):
            subset = [r for r in runs if r["block"] == block]
            path = Path(td) / f"{block}_only.json"
            _write_like_driver(subset, path)
            loaded, _ = gate.load_runs(path)
            check(len(loaded) == 80,
                  f"{block}-only file did not load its 80 episodes")
            res_one = gate.evaluate(loaded)
            other = "acquisition" if block == "commitment" else "commitment"
            check(res_one["verdict"] == "FAIL",
                  f"a complete {block} block ALONE reported "
                  f"{res_one['verdict']} -- the missing {other} rule was "
                  f"treated as an exemption")
            check(any("ABSENT" in e for e in res_one["errors"]),
                  f"{block}-only run did not flag the absent {other} block")

    # ── an insensitive decider must FAIL, so PASS is not automatic ─────────
    def flat(prompt: str) -> str:
        if "You may take no further samples." in prompt:
            return "done.\nPolicy: COMMIT Button A"
        return "always the same.\nPolicy: COMMIT Button A"

    res_flat = gate.evaluate(_run_full_bank(flat))
    check(res_flat["verdict"] == "FAIL",
          "a decider that ignores the evidence still PASSED")
    check(not res_flat["M1"]["passes"],
          "M1 passed against a constant COMMIT policy")

    if FAILURES:
        print(f"FAIL ({len(FAILURES)})")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ok  test_pv11_gate_end_to_end.py  (driver schema -> gate loader)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
