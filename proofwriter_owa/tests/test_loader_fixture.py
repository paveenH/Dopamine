#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loader fixture tests for proofwriter_owa/data_proofwriter_owa.py. Builds a
small SYNTHETIC zip archive in-memory (no network, no real ProofWriter data)
matching the official depth-N/OWA/meta-test.jsonl layout, and exercises
parse_depth_split + build_manifest + the manifest-determinism/hash guarantee
against it.

    python3 test_loader_fixture.py
"""

import io
import json
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_proofwriter_owa as dpo  # noqa: E402

FAILS = []
N = 0


def check(cond, msg):
    global N
    N += 1
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return bool(cond)


def make_theory(tid, n_facts, n_rules, questions):
    """questions: list of (qkey, question_text, answer, qdep)"""
    triples = {f"triple{i}": {"text": f"Fact{i} of theory {tid}."}
              for i in range(n_facts)}
    rules = {f"rule{i}": {"text": f"Rule{i} of theory {tid}."}
             for i in range(n_rules)}
    qs = {}
    for qkey, qtext, ans, qdep in questions:
        qs[qkey] = {"question": qtext, "answer": ans, "QDep": qdep,
                   "proofs": "" if ans == "Unknown" else "()"}
    return {"id": tid, "triples": triples, "rules": rules, "questions": qs}


def build_fixture_zip(n_true=3, n_false=3, n_unknown=3, target_depth=3,
                      depth_dirname="depth-3"):
    """Builds a minimal zip with exactly the shape parse_depth_split expects:
    <depth_dirname>/OWA/meta-test.jsonl, one theory per JSON line."""
    theories = []
    tid = 0
    for i in range(n_true):
        theories.append(make_theory(
            f"T{tid}", 3 + i, 2, [(f"Q{tid}", f"Question true {i}?", True, target_depth)]))
        tid += 1
    for i in range(n_false):
        theories.append(make_theory(
            f"T{tid}", 3 + i, 2, [(f"Q{tid}", f"Question false {i}?", False, target_depth)]))
        tid += 1
    for i in range(n_unknown):
        theories.append(make_theory(
            f"T{tid}", 3 + i, 2, [(f"Q{tid}", f"Question unknown {i}?", "Unknown", None)]))
        tid += 1

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        jsonl = "\n".join(json.dumps(t) for t in theories)
        zf.writestr(f"proofwriter-dataset-V2020.12.3/{depth_dirname}/OWA/meta-test.jsonl",
                   jsonl)
    buf.seek(0)
    return buf.read()


def write_temp_zip(data: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def test_parse_depth_split_basic():
    zdata = build_fixture_zip(n_true=5, n_false=4, n_unknown=6, target_depth=3)
    path = write_temp_zip(zdata)
    try:
        recs = dpo.parse_depth_split(path, "D3", "test")
        check(len(recs) == 15, f"expected 15 questions, got {len(recs)}")
        labels = [r["answer"] for r in recs]
        check(labels.count("True") == 5, f"got {labels.count('True')} True")
        check(labels.count("False") == 4, f"got {labels.count('False')} False")
        check(labels.count("Unknown") == 6, f"got {labels.count('Unknown')} Unknown")
        for r in recs:
            if r["answer"] in ("True", "False"):
                check(r["qdep"] == 3, f"expected qdep 3, got {r['qdep']}")
            else:
                check(r["qdep"] is None, f"Unknown item should have qdep None here, got {r['qdep']}")
    finally:
        os.unlink(path)


def test_parse_depth_split_missing_owa_hard_stops():
    # a zip with NO OWA subfolder at all under depth-3 should hard-stop
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("proofwriter-dataset-V2020.12.3/depth-3/CWA/meta-test.jsonl", "{}")
    buf.seek(0)
    path = write_temp_zip(buf.read())
    try:
        try:
            dpo.parse_depth_split(path, "D3", "test")
            check(False, "should have hard-stopped: no OWA subfolder present")
        except SystemExit as e:
            check(e.code == 2, f"expected exit code 2, got {e.code}")
    finally:
        os.unlink(path)


def test_parse_depth_split_missing_field_hard_stops():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # missing "rules" field entirely
        bad = json.dumps({"id": "T0", "triples": {}, "questions": {}})
        zf.writestr("proofwriter-dataset-V2020.12.3/depth-3/OWA/meta-test.jsonl", bad)
    buf.seek(0)
    path = write_temp_zip(buf.read())
    try:
        try:
            dpo.parse_depth_split(path, "D3", "test")
            check(False, "should have hard-stopped: missing 'rules' field")
        except SystemExit as e:
            check(e.code == 2, f"expected exit code 2, got {e.code}")
    finally:
        os.unlink(path)


def test_parse_depth_split_unrecognized_label_hard_stops():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        bad_theory = {
            "id": "T0", "triples": {"t0": {"text": "F0."}}, "rules": {},
            "questions": {"Q0": {"question": "Q0?", "answer": "maybe", "QDep": 1}},
        }
        zf.writestr("proofwriter-dataset-V2020.12.3/depth-3/OWA/meta-test.jsonl",
                   json.dumps(bad_theory))
    buf.seek(0)
    path = write_temp_zip(buf.read())
    try:
        try:
            dpo.parse_depth_split(path, "D3", "test")
            check(False, "should have hard-stopped: unrecognized label 'maybe'")
        except SystemExit as e:
            check(e.code == 2, f"expected exit code 2, got {e.code}")
    finally:
        os.unlink(path)


def test_build_manifest_shape():
    # Build both D3 (target depth 3) and D5 (target depth 5) fixtures with
    # comfortably more than N_PER_LABEL of each label so the manifest can
    # reach the frozen 50/50/50 shape without triggering a shortfall.
    recs_d3 = _synthetic_records("D3", target_depth=3, n_per_label=60)
    recs_d5 = _synthetic_records("D5", target_depth=5, n_per_label=60)
    rows, shortfalls = dpo.build_manifest({"D3": recs_d3, "D5": recs_d5}, seed=0)

    check(shortfalls == [], f"expected no shortfalls with ample pools, got {shortfalls}")
    check(len(rows) == dpo.N_TOTAL, f"expected {dpo.N_TOTAL} rows, got {len(rows)}")

    from collections import Counter
    by_ds = Counter(r["dataset"] for r in rows)
    check(by_ds["D3"] == dpo.N_PER_DATASET, f"got D3={by_ds['D3']}")
    check(by_ds["D5"] == dpo.N_PER_DATASET, f"got D5={by_ds['D5']}")

    for ds in ("D3", "D5"):
        by_label = Counter(r["answer"] for r in rows if r["dataset"] == ds)
        for lab in ("True", "False", "Unknown"):
            check(by_label[lab] == dpo.N_PER_LABEL,
                  f"{ds}/{lab}: expected {dpo.N_PER_LABEL}, got {by_label[lab]}")

    # True/False rows in the manifest must carry the dataset's target QDep
    # (this is the field that a REGRESSION of the closure-rebind bug would
    # silently violate, since the un-refilled small pool would still pass
    # through unchanged when pools are already large enough -- so this check
    # alone would not have caught THAT bug; test_build_manifest_shortfall_fix
    # below specifically targets it).
    for r in rows:
        if r["dataset"] == "D3" and r["answer"] in ("True", "False"):
            check(r["qdep"] == 3, f"D3 True/False row has qdep={r['qdep']}, expected 3")
        if r["dataset"] == "D5" and r["answer"] in ("True", "False"):
            check(r["qdep"] == 5, f"D5 True/False row has qdep={r['qdep']}, expected 5")
        if r["answer"] == "Unknown":
            check("depth_match_note" in r, "Unknown rows must carry depth_match_note")
            note = r["depth_match_note"]
            # The note is allowed to MENTION a depth number only inside the
            # explicit disclaimer "NOT described as having a <k>-step gold
            # proof" -- it must never assert the row HAS one.
            check("NOT described as having" in note,
                  f"depth_match_note must explicitly disclaim a fixed-depth "
                  f"gold proof claim, got: {note!r}")
            check("matched to the True/False pool" in note,
                  f"depth_match_note must explain the matching basis "
                  f"(dataset + theory size), got: {note!r}")


def test_build_manifest_shortfall_fix_regression():
    """Regression test for the closure-rebind bug: a dataset with FEWER than
    N_PER_LABEL True items at the target depth must end up with EXACTLY
    N_PER_LABEL True rows in the final manifest (filled from nearby depths),
    not a silently-truncated pool. Before the fix, reassigning the for-loop
    variable `pool` never wrote back into `true_pool`, so the manifest kept
    the SHORT pool despite `shortfalls` claiming it had been filled."""
    # Only 10 True-at-depth-3 items (< N_PER_LABEL=50), but plenty at other
    # depths and plenty False/Unknown, so the shortfall-fill path fires.
    recs = []
    for i in range(10):
        recs.append({"dataset": "D3", "answer": "True", "qdep": 3,
                     "official_theory_id": f"T{i}", "official_qid": "Q0",
                     "theory_text": f"theory {i}", "question_text": f"q{i}?",
                     "n_facts": 3, "n_rules": 2})
    for i in range(10, 200):
        # filler True items at OTHER depths, so the shortfall-fill has
        # somewhere to draw from
        recs.append({"dataset": "D3", "answer": "True", "qdep": (i % 5) + 1,
                     "official_theory_id": f"T{i}", "official_qid": "Q0",
                     "theory_text": f"theory {i}", "question_text": f"q{i}?",
                     "n_facts": 3, "n_rules": 2})
    recs.extend(_synthetic_records_partial("D3", "False", 3, 100))
    recs.extend(_synthetic_records_partial("D3", "Unknown", None, 100))
    recs.extend(_synthetic_records("D5", target_depth=5, n_per_label=60))

    rows, shortfalls = dpo.build_manifest({"D3": recs, "D5": []}, seed=0)
    # D5 is empty here on purpose to isolate the D3 shortfall path; build_manifest
    # itself does not require D5 to be non-empty for this unit-level check.
    from collections import Counter
    by_label_d3 = Counter(r["answer"] for r in rows if r["dataset"] == "D3")
    check(by_label_d3["True"] == dpo.N_PER_LABEL,
          f"REGRESSION: D3/True should be refilled to {dpo.N_PER_LABEL}, "
          f"got {by_label_d3['True']} (this is exactly the bug the fix addresses)")
    check(any("D3/True" in s for s in shortfalls),
          "a shortfall should have been reported for D3/True")


def _synthetic_records(dataset, target_depth, n_per_label):
    recs = []
    recs.extend(_synthetic_records_partial(dataset, "True", target_depth, n_per_label))
    recs.extend(_synthetic_records_partial(dataset, "False", target_depth, n_per_label))
    recs.extend(_synthetic_records_partial(dataset, "Unknown", None, n_per_label))
    return recs


def _synthetic_records_partial(dataset, label, qdep, n):
    out = []
    for i in range(n):
        out.append({
            "dataset": dataset, "answer": label, "qdep": qdep,
            "official_theory_id": f"{dataset}_{label}_T{i}", "official_qid": "Q0",
            "theory_text": f"theory {dataset} {label} {i}",
            "question_text": f"question {dataset} {label} {i}?",
            "n_facts": 3 + (i % 4), "n_rules": 2 + (i % 3),
        })
    return out


def test_manifest_determinism_same_seed():
    recs_d3 = _synthetic_records("D3", target_depth=3, n_per_label=60)
    recs_d5 = _synthetic_records("D5", target_depth=5, n_per_label=60)
    rows1, _ = dpo.build_manifest({"D3": recs_d3, "D5": recs_d5}, seed=0)
    rows2, _ = dpo.build_manifest({"D3": recs_d3, "D5": recs_d5}, seed=0)

    ids1 = [(r["dataset"], r["official_theory_id"], r["official_qid"]) for r in rows1]
    ids2 = [(r["dataset"], r["official_theory_id"], r["official_qid"]) for r in rows2]
    check(ids1 == ids2, "same seed, same input -> identical manifest row order")

    keys1 = [r["key"] for r in rows1]
    keys2 = [r["key"] for r in rows2]
    check(keys1 == keys2, "same seed -> identical salted keys")


def test_manifest_different_seed_differs():
    recs_d3 = _synthetic_records("D3", target_depth=3, n_per_label=60)
    recs_d5 = _synthetic_records("D5", target_depth=5, n_per_label=60)
    rows_a, _ = dpo.build_manifest({"D3": recs_d3, "D5": recs_d5}, seed=0)
    rows_b, _ = dpo.build_manifest({"D3": recs_d3, "D5": recs_d5}, seed=1)

    keys_a = [r["key"] for r in rows_a]
    keys_b = [r["key"] for r in rows_b]
    check(keys_a != keys_b, "different seeds must produce different salted keys")


def test_manifest_hash_stable_across_process_runs():
    """sha16() over the manifest's (sample_id,key,dataset,answer) projection
    must be identical every time this test runs -- it must NOT depend on
    Python's per-process string-hash randomization (the exact failure mode
    hash() has and sha256 does not)."""
    recs_d3 = _synthetic_records("D3", target_depth=3, n_per_label=60)
    recs_d5 = _synthetic_records("D5", target_depth=5, n_per_label=60)
    rows, _ = dpo.build_manifest({"D3": recs_d3, "D5": recs_d5}, seed=0)
    digest1 = dpo.sha16(json.dumps(
        [{"sample_id": r["sample_id"], "key": r["key"],
          "dataset": r["dataset"], "answer": r["answer"]} for r in rows],
        sort_keys=True))
    rows2, _ = dpo.build_manifest({"D3": recs_d3, "D5": recs_d5}, seed=0)
    digest2 = dpo.sha16(json.dumps(
        [{"sample_id": r["sample_id"], "key": r["key"],
          "dataset": r["dataset"], "answer": r["answer"]} for r in rows2],
        sort_keys=True))
    check(digest1 == digest2, f"manifest digest must be stable: {digest1} vs {digest2}")
    check(len(digest1) == 16, f"sha16 should be 16 hex chars, got {len(digest1)}")


def test_salted_key_not_process_salted():
    """salted_key uses sha256, so unlike hash(), it must be identical run to
    run and process to process -- this is a subprocess check for real
    guarantee (a single-process check cannot detect PYTHONHASHSEED effects)."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    script = (
        "import sys; sys.path.insert(0, %r); import data_proofwriter_owa as d; "
        "print(d.salted_key('a', 'b', 'c'))"
    ) % os.path.dirname(here)
    out1 = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, env={**os.environ, "PYTHONHASHSEED": "1"})
    out2 = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, env={**os.environ, "PYTHONHASHSEED": "2"})
    check(out1.returncode == 0 and out2.returncode == 0,
          f"subprocess failed: {out1.stderr}{out2.stderr}")
    check(out1.stdout.strip() == out2.stdout.strip(),
          f"salted_key must be identical across PYTHONHASHSEED values, got "
          f"{out1.stdout.strip()!r} vs {out2.stdout.strip()!r}")


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{N - len(FAILS)}/{N} checks passed")
    if FAILS:
        print(f"{len(FAILS)} FAILURES")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
