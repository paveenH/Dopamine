#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProofWriter OWA (D3/D5, Task 1 answer-prediction) loader.

Protocol: `proofwriter-owa-v0` (see PREREG_PROOFWRITER_OWA.md). This is
ProofWriter's OWN task-specific workpoint exploration, NOT a GSM8K
fixed-workpoint transfer test.

SOURCE POLICY (load-bearing, see PREREG S1.1)
----------------------------------------------
Primary source is the OFFICIAL AI2 release archive
(`proofwriter-dataset-V2020.12.3.zip`, default URL below, documented on the
ProofWriter project page). An HF mirror is used ONLY if --hf_mirror is passed
together with --verify_against_official, and only AFTER the mirror rows are
shown byte-identical (theory/question/answer/proof) to the official archive
already on disk for the sampled/full ID set. There is no silent mirror
fallback anywhere in this file.

ONLY OWA Task 1 (answer prediction / proof generation), D3 and D5. CWA, D0-D2,
abduction, implication-enumeration, Birds-Electricity and ParaRules are never
read by this loader -- the schema check below hard-stops on anything outside
the two depth-folder / OWA-subfolder / test-split shape it expects.

DEPTH FIELD (`QDep`) -- REPORTED, NOT ASSUMED (PREREG S1.2)
-------------------------------------------------------------
`QDep` is read per-question from the archive as the shortest-proof depth for
THAT question, not inferred from which depth-folder the theory came from. The
loader prints the observed QDep distribution split by label BEFORE building
any manifest, and refuses (hard stop, not silent reinterpretation) if the
field is absent or non-integer on a material fraction of parsed questions --
see the QDep handling inside `parse_depth_split`.

FAIL-CLOSED SCHEMA HANDLING (PREREG S1.4): every structural assumption below
is asserted, and a violation prints the REAL structure found and stops, rather
than guessing a mapping.

NO SAMPLE SELECTION BY MODEL OUTPUT ANYWHERE IN THIS FILE. This script only
downloads, parses, reports and (with --build_manifest) freezes a 300-item
sample by a salted hash of official IDs -- never by running any model.

Usage
-----
    # 1. Report-only pass (required before any manifest is built):
    python3 data_proofwriter_owa.py --archive_dir <dir> --report

    # 2. Build the frozen 300-item manifest (idempotent; refuses to overwrite):
    python3 data_proofwriter_owa.py --archive_dir <dir> --out_dir <dir> \\
        --build_manifest

    # 3. Verify a downloaded state reproduces the frozen digests, no writes:
    python3 data_proofwriter_owa.py --archive_dir <dir> --check

The archive itself is fetched by `--download <dir>` (separate step, network
required; not invoked by --report/--check/--build_manifest so those stay
offline-safe once the archive is already on disk).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import re
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict

PROTOCOL = "proofwriter-owa-v0"
SALT = "proofwriter-owa-v0"

OFFICIAL_URL = (
    "https://aristo-data-public.s3.amazonaws.com/proofwriter/"
    "proofwriter-dataset-V2020.12.3.zip"
)
ARCHIVE_BASENAME = "proofwriter-dataset-V2020.12.3.zip"

# Only these two depth families, only OWA, only the answer-prediction (Task 1)
# meta files. A rule-set folder outside this set (birds-electricity,
# AttNegNoRule, AttRulesNoNeg, ParaRules, ...) is never read.
DEPTH_TASKS = {"D3": "depth-3", "D5": "depth-5"}
ALLOWED_SPLITS = {"train", "dev", "test"}
FORMAL_SPLIT = "test"
EXEMPLAR_SPLIT = "train"

VALID_LABELS = {"True", "False", "Unknown"}

# Whitelist / forbidden-key sets for the blind manifest, matching the
# label-firewall pattern used by data_logiqa2.py / data_bbh_numeric.py /
# data_cruxeval.py / data_gsm_hard.py.
BLIND_ALLOWED = {
    "sample_id", "key", "dataset", "official_theory_id", "official_qid",
    "theory_text", "question_text", "content_sha256",
}
LABEL_FIELDS = {"answer", "label", "gold", "gold_answer", "correct",
                "accuracy", "proof", "proofs", "target"}

N_TOTAL = 300
N_PER_DATASET = 150
N_PER_LABEL = 50


def die(msg: str):
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def salted_key(*parts: str) -> str:
    payload = "\x1f".join(parts)
    return hashlib.sha256((SALT + ":" + payload).encode("utf-8")).hexdigest()


# ───────────────────────── download ─────────────────────────

def _verify_zip_integrity(path: str):
    """zipfile.testzip() reads every member's CRC -- catches a truncated or
    corrupted archive that would otherwise pass a bare os.path.exists()
    check and only fail much later, deep inside _find_meta_files with a
    confusing error. Returns the name of the first bad member, or None."""
    with zipfile.ZipFile(path) as zf:
        return zf.testzip()


def download_archive(dest_dir: str, url: str = OFFICIAL_URL,
                      expected_sha256: str | None = None,
                      chunk_size: int = 1 << 20,
                      progress_every_bytes: int = 20 << 20) -> str:
    """Fetch the official archive to dest_dir. Network required. Not called by
    --report/--check/--build_manifest, which assume the archive already
    exists locally (offline-safe once fetched once).

    Reviewed 2026-09-04: the previous implementation did `f.write(r.read())`
    -- one blocking read of the ENTIRE response body with no progress output
    (so a slow/stalled connection is indistinguishable from a hung process
    for the whole duration) and no interrupt safety (a Ctrl-C mid-download
    left a truncated file at the FINAL path; the next run's `os.path.exists`
    check would then treat that truncated file as "already downloaded" and
    never re-fetch it, since no integrity check ran on the cached-file path
    either). Fixed by: (1) streaming reads in `chunk_size` blocks with
    periodic progress printed to stdout, so activity is visible instead of
    silent "fetching..."; (2) writing to a `.part` sibling file and only
    os.replace()-ing it to the final path after the download completes AND
    a zip-integrity check passes -- an interrupted download can never leave
    a file at the final path; (3) verifying zip integrity (not just
    presence) on the ALREADY-PRESENT-file fast path too, so a corrupt cached
    archive from before this fix is caught here rather than surfacing later
    inside _find_meta_files with a confusing "structure differs" error.
    """
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, ARCHIVE_BASENAME)
    part_path = path + ".part"

    if os.path.exists(path):
        bad_member = _verify_zip_integrity(path)
        if bad_member is not None:
            die(f"{path} exists but failed zip integrity check (bad member: "
                f"{bad_member}); this looks like a truncated/corrupted "
                "download from before this script verified zip integrity on "
                "the already-present-file path. Delete it and re-run to "
                "re-download: rm " + path)
        got = sha256_file(path)
        if expected_sha256 and got != expected_sha256:
            die(f"{path} already exists but sha256={got} != expected "
                f"{expected_sha256}; refusing to silently re-download over a "
                "mismatched file. Delete it deliberately if this is intentional.")
        print(f"[download] archive already present and zip-integrity-verified: "
              f"{path} sha256={got}")
        return path

    if os.path.exists(part_path):
        print(f"[download] found a stale partial download {part_path} "
              "(likely from an interrupted previous run); removing it and "
              "starting over rather than trying to resume it.")
        os.remove(part_path)

    print(f"[download] fetching {url}")
    with urllib.request.urlopen(url, timeout=300) as r:
        total = r.getheader("Content-Length")
        total = int(total) if total is not None else None
        total_str = f"{total / (1 << 20):.1f} MB" if total else "unknown size"
        print(f"[download] response opened, {total_str}")
        written = 0
        next_report = progress_every_bytes
        try:
            with open(part_path, "wb") as f:
                while True:
                    chunk = r.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
                    if written >= next_report:
                        pct = f" ({100 * written / total:.1f}%)" if total else ""
                        print(f"[download] {written / (1 << 20):.1f} MB "
                              f"written{pct}")
                        next_report += progress_every_bytes
        except BaseException:
            # covers Ctrl-C (KeyboardInterrupt) as well as any I/O error --
            # in both cases the partial file must NOT be left where a future
            # run's os.path.exists(path) check could mistake it for a
            # complete download. It stays at `.part` so it is inspectable,
            # but it is never renamed to the final path.
            print(f"[download] interrupted/failed after {written} bytes; "
                  f"left the incomplete file at {part_path} (NOT at the "
                  "final path, so a future run will not mistake it for a "
                  "complete download). Delete it manually if you want to "
                  "clean up disk space before retrying.", file=sys.stderr)
            raise
    print(f"[download] download complete, {written} bytes written to "
          f"{part_path}; verifying before making it the final artifact")

    bad_member = _verify_zip_integrity(part_path)
    if bad_member is not None:
        die(f"downloaded file failed zip integrity check (bad member: "
            f"{bad_member}); the download is corrupt or was truncated by a "
            f"network issue. Left at {part_path} for inspection; delete it "
            "and re-run to retry.")

    got = sha256_file(part_path)
    if expected_sha256 and got != expected_sha256:
        die(f"downloaded sha256={got} != expected {expected_sha256}; the "
            "official archive content changed or the download is corrupt. "
            f"This is a hard stop, not a silent accept. Left at {part_path} "
            "for inspection.")

    # Atomic rename: only now does a file appear at the FINAL path, so a
    # concurrent or later run's os.path.exists(path) check can never see a
    # partial/unverified file there.
    os.replace(part_path, path)
    print(f"[download] wrote {path} sha256={got}")
    return path


# ───────────────────────── archive parsing ─────────────────────────

def _find_meta_files(zf: zipfile.ZipFile, depth_dirname: str, split: str):
    """Locate the OWA meta-<split>.jsonl for one depth family inside the zip.

    FAILS CLOSED: prints the actual paths under the depth folder and stops if
    the expected OWA/meta-<split>.jsonl shape is not found, rather than
    guessing an alternate layout.
    """
    names = zf.namelist()
    depth_matches = sorted({n for n in names
                             if f"/{depth_dirname}" in n or n.startswith(depth_dirname)})
    if not depth_matches:
        die(f"no path under the archive matches depth folder {depth_dirname!r}. "
            f"First 20 archive entries for inspection: {names[:20]}")

    owa_candidates = sorted({n for n in depth_matches if "/OWA/" in n or "OWA/" in n})
    if not owa_candidates:
        die(f"depth folder {depth_dirname!r} was found but no OWA subfolder "
            f"inside it. Matches under that folder: {depth_matches[:20]}")

    target_name = f"meta-{split}.jsonl"
    exact = [n for n in owa_candidates if n.endswith(target_name)]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        die(f"multiple candidates for {depth_dirname}/OWA/{target_name}: {exact}. "
            "Archive layout is ambiguous; refusing to silently pick one.")
    die(f"expected a file ending in {target_name!r} under an OWA subfolder of "
        f"{depth_dirname!r}; found instead: {owa_candidates[:20]}. The archive "
        "schema differs from what this loader expects -- reporting the real "
        "structure rather than guessing a mapping.")


def _normalize_label(raw) -> str:
    """AI2's `answer` field arrives as either a JSON bool or the literal
    string "Unknown" (sometimes lowercase in various dumps). Reported
    verbatim before normalization by the --report pass; here we normalize to
    exactly {"True","False","Unknown"} and hard-stop on anything else."""
    if isinstance(raw, bool):
        return "True" if raw else "False"
    if isinstance(raw, str):
        s = raw.strip()
        if s.lower() == "true":
            return "True"
        if s.lower() == "false":
            return "False"
        if s.lower() == "unknown":
            return "Unknown"
    die(f"unrecognized gold-label token {raw!r}; expected a JSON bool or one "
        "of the strings true/false/unknown (any case). Reporting the raw "
        "token rather than guessing its meaning.")


def parse_depth_split(zip_path: str, dataset: str, split: str) -> list[dict]:
    """Parse one (dataset in {D3,D5}, split) meta-<split>.jsonl into a flat
    list of question-level records. FAILS CLOSED on any schema surprise."""
    if dataset not in DEPTH_TASKS:
        die(f"dataset must be one of {sorted(DEPTH_TASKS)}, got {dataset!r}")
    if split not in ALLOWED_SPLITS:
        die(f"split must be one of {sorted(ALLOWED_SPLITS)}, got {split!r}")
    depth_dirname = DEPTH_TASKS[dataset]

    with zipfile.ZipFile(zip_path) as zf:
        meta_path = _find_meta_files(zf, depth_dirname, split)
        raw = zf.read(meta_path).decode("utf-8")

    records = []
    missing_qdep = 0
    n_theories = 0
    n_missing_fields = 0
    seen_theory_ids = set()
    dup_theory_ids = 0

    for lineno, line in enumerate(raw.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            theory = json.loads(line)
        except json.JSONDecodeError as e:
            die(f"{meta_path}:{lineno}: not valid JSON ({e}); "
                "refusing to skip a malformed line silently.")

        n_theories += 1
        for f in ("id", "triples", "rules", "questions"):
            if f not in theory:
                n_missing_fields += 1
                die(f"{meta_path}:{lineno}: theory object missing required "
                    f"field {f!r}; keys present: {sorted(theory.keys())}. "
                    "Archive schema differs from expectation.")

        tid = theory["id"]
        if tid in seen_theory_ids:
            dup_theory_ids += 1
        seen_theory_ids.add(tid)

        # theory text: prefer an explicit English rendering if the release
        # ships one (varies by dump); else synthesize from triple/rule
        # `text` fields, which the release always carries per-triple/rule.
        theory_text = theory.get("theory") or theory.get("text")
        if not theory_text:
            parts = []
            for t in theory["triples"].values() if isinstance(theory["triples"], dict) else theory["triples"]:
                if isinstance(t, dict) and "text" in t:
                    parts.append(t["text"])
            for rul in theory["rules"].values() if isinstance(theory["rules"], dict) else theory["rules"]:
                if isinstance(rul, dict) and "text" in rul:
                    parts.append(rul["text"])
            if not parts:
                die(f"{meta_path}: theory {tid!r} has no renderable English "
                    "text in triples/rules and no 'theory'/'text' field. "
                    "Reporting the raw theory keys: "
                    f"{sorted(theory.keys())}")
            theory_text = " ".join(parts)

        n_facts = len(theory["triples"]) if hasattr(theory["triples"], "__len__") else None
        n_rules = len(theory["rules"]) if hasattr(theory["rules"], "__len__") else None

        questions = theory["questions"]
        if not isinstance(questions, dict):
            die(f"{meta_path}: theory {tid!r} 'questions' is not a dict "
                f"(got {type(questions)}); archive schema differs from "
                "expectation.")

        for qkey, q in questions.items():
            for f in ("question", "answer"):
                if f not in q:
                    die(f"{meta_path}: theory {tid!r} question {qkey!r} "
                        f"missing field {f!r}; keys present: "
                        f"{sorted(q.keys())}")
            label = _normalize_label(q["answer"])
            qdep = q.get("QDep")
            if qdep is None:
                missing_qdep += 1
                qdep_int = None
            else:
                try:
                    qdep_int = int(qdep)
                except (TypeError, ValueError):
                    die(f"{meta_path}: theory {tid!r} question {qkey!r} has "
                        f"non-integer QDep={qdep!r}")
            proof_text = q.get("proofs") or q.get("proof") or ""

            records.append({
                "dataset": dataset,
                "split": split,
                "official_theory_id": tid,
                "official_qid": qkey,
                "theory_text": theory_text,
                "question_text": q["question"],
                "answer": label,
                "qdep": qdep_int,
                "proof_text": proof_text,
                "n_facts": n_facts,
                "n_rules": n_rules,
            })

    if dup_theory_ids:
        die(f"{meta_path}: {dup_theory_ids} duplicate theory id(s) found; "
            "refusing to proceed with an ambiguous id space.")

    frac_missing_qdep = missing_qdep / len(records) if records else 1.0
    if frac_missing_qdep > 0.0:
        # Reported unconditionally per PREREG S1.4; treated as a hard stop
        # only past a material fraction so a handful of legitimately-absent
        # QDep values (documented in the official release for some Unknown
        # items) do not block the whole loader, but a wholesale absence does.
        print(f"[schema] {meta_path}: {missing_qdep}/{len(records)} "
              f"({frac_missing_qdep:.1%}) questions have no QDep field.")
        if frac_missing_qdep > 0.5:
            die(f"{meta_path}: QDep is missing on {frac_missing_qdep:.1%} of "
                "questions -- more than half. The depth-stratification "
                "assumption in PREREG S1.2 does not hold for this file; "
                "reporting rather than silently proceeding.")

    return records


# ───────────────────────── report (S2) ─────────────────────────

def report(records_by_dataset: dict[str, list[dict]]):
    for ds in ("D3", "D5"):
        recs = records_by_dataset.get(ds, [])
        theories = {r["official_theory_id"] for r in recs}
        labels = Counter(r["answer"] for r in recs)
        qdep_by_label = defaultdict(Counter)
        for r in recs:
            qdep_by_label[r["answer"]][r["qdep"]] += 1
        n_missing_id = sum(1 for r in recs
                            if not r["official_theory_id"] or not r["official_qid"])
        seen_keys = Counter((r["official_theory_id"], r["official_qid"]) for r in recs)
        n_dup = sum(1 for c in seen_keys.values() if c > 1)
        n_unparseable = sum(1 for r in recs if r["answer"] not in VALID_LABELS)

        print(f"\n=== {ds} (split={FORMAL_SPLIT if recs and recs[0]['split']==FORMAL_SPLIT else recs[0]['split'] if recs else '?'}) ===")
        print(f"  theories             : {len(theories)}")
        print(f"  questions            : {len(recs)}")
        print(f"  True / False / Unknown: {labels.get('True',0)} / "
              f"{labels.get('False',0)} / {labels.get('Unknown',0)}")
        print(f"  missing official id  : {n_missing_id}")
        print(f"  duplicate (tid,qid)  : {n_dup}")
        print(f"  unparseable gold     : {n_unparseable}")
        print(f"  QDep distribution by label (gold query/proof depth):")
        for lab in ("True", "False", "Unknown"):
            hist = qdep_by_label.get(lab, Counter())
            hist_str = ", ".join(f"{k}:{v}" for k, v in sorted(
                hist.items(), key=lambda kv: (kv[0] is None, kv[0])))
            print(f"    {lab:8s}: {hist_str}")


# ───────────────────────── manifest (S3) ─────────────────────────

def _pick_depth_pool(recs: list[dict], target_qdep: int, label: str, n: int,
                      rng_key_fn):
    pool = [r for r in recs if r["answer"] == label and r["qdep"] == target_qdep]
    pool.sort(key=lambda r: rng_key_fn(r))
    return pool[:n], pool[n:]


def _pick_unknown_matched(recs: list[dict], n: int, true_false_pool: list[dict],
                           rng_key_fn):
    """Match Unknown items to the True/False pool on dataset + theory size
    (n_facts + n_rules), by nearest total-size match, tie-broken by the
    frozen hash order. Never described as having a 3/5-step gold proof."""
    unk = [r for r in recs if r["answer"] == "Unknown"]
    if not unk:
        return [], []
    target_sizes = sorted(
        (r["n_facts"] or 0) + (r["n_rules"] or 0) for r in true_false_pool)
    med_size = target_sizes[len(target_sizes) // 2] if target_sizes else None

    def dist(r):
        sz = (r["n_facts"] or 0) + (r["n_rules"] or 0)
        d = abs(sz - med_size) if med_size is not None else 0
        return (d, rng_key_fn(r))

    unk_sorted = sorted(unk, key=dist)
    return unk_sorted[:n], unk_sorted[n:]


def build_manifest(records_by_dataset: dict[str, list[dict]], seed: int = 0,
                    _assert_final_shape: bool = True):
    """`_assert_final_shape` (default True, keyword-only in spirit -- kept
    positional-callable only for existing test call sites): every REAL caller
    must leave this True, which is what turns "the manifest silently came out
    smaller than 300" into a hard stop (review finding #4, 2026-09-04). The
    one legitimate reason to pass False is a unit test that deliberately
    starves one dataset (e.g. an empty D5) to isolate the shortfall-refill
    logic for ONE label in ONE dataset in ONE call, where reaching a full
    300-row manifest is not the point of that test -- see
    test_loader_fixture.py's test_build_manifest_shortfall_fix_regression."""
    manifest_rows = []
    shortfalls = []

    for ds in ("D3", "D5"):
        recs = records_by_dataset[ds]
        target_qdep = 3 if ds == "D3" else 5

        def rng_key(r):
            return salted_key(str(seed), ds, r["official_theory_id"], r["official_qid"])

        true_pool, _ = _pick_depth_pool(recs, target_qdep, "True", N_PER_LABEL, rng_key)
        false_pool, _ = _pick_depth_pool(recs, target_qdep, "False", N_PER_LABEL, rng_key)

        # BUG FIX (found in review, 2026-09-04): the refill below used to
        # reassign the FOR-LOOP VARIABLE `pool` (`for lab, pool in
        # (("True", true_pool), ("False", false_pool)): ... pool = pool +
        # remaining[:need]`), which only rebinds that local name for the rest
        # of ONE loop iteration -- it never wrote back into `true_pool` /
        # `false_pool` themselves. Every consumer downstream (tf_pool,
        # unk_pool's size-matching, and the final row-building loop) kept
        # reading the ORIGINAL, un-refilled pools, so a genuine depth
        # shortfall silently produced a manifest with fewer than
        # N_PER_LABEL*2 True/False rows (and a doubly-wrong Unknown match,
        # since unk_pool was matched against the un-refilled tf_pool too) --
        # while `shortfalls` still claimed the gap had been filled. Fixed by
        # writing the refilled pool into a NEW variable per label and using
        # those variables (refilled_true / refilled_false) for everything
        # downstream instead of reusing the loop variable's name.
        refilled = {"True": true_pool, "False": false_pool}
        for lab, pool in (("True", true_pool), ("False", false_pool)):
            if len(pool) < N_PER_LABEL:
                shortfalls.append(
                    f"{ds}/{lab}: only {len(pool)}/{N_PER_LABEL} items with "
                    f"QDep=={target_qdep}; filling from nearest available "
                    "depth for this label.")
                # fill from nearest depth for this label, frozen-hash ordered
                have_ids = {(r["official_theory_id"], r["official_qid"]) for r in pool}
                remaining = [r for r in recs
                             if r["answer"] == lab
                             and (r["official_theory_id"], r["official_qid"]) not in have_ids
                             and r["qdep"] is not None]
                remaining.sort(key=lambda r: (abs(r["qdep"] - target_qdep), rng_key(r)))
                need = N_PER_LABEL - len(pool)
                refilled[lab] = pool + remaining[:need]
        true_pool, false_pool = refilled["True"], refilled["False"]

        tf_pool = true_pool + false_pool
        unk_pool, _ = _pick_unknown_matched(recs, N_PER_LABEL, tf_pool, rng_key)
        if len(unk_pool) < N_PER_LABEL:
            shortfalls.append(
                f"{ds}/Unknown: only {len(unk_pool)}/{N_PER_LABEL} matched items available.")

        for lab, pool, target_qdep_for_row in (
            ("True", true_pool, target_qdep),
            ("False", false_pool, target_qdep),
            ("Unknown", unk_pool, None),
        ):
            for r in pool:
                row = dict(r)
                row["target_qdep"] = target_qdep_for_row
                if lab == "Unknown":
                    row["depth_match_note"] = (
                        "Unknown items generally carry no single gold proof "
                        "depth under OWA; matched to the True/False pool on "
                        "dataset + theory size (n_facts+n_rules), not on "
                        "QDep. This row is NOT described as having a "
                        f"{target_qdep}-step gold proof.")
                manifest_rows.append(row)

    # deterministic final order: dataset, label, then frozen hash
    label_order = {"True": 0, "False": 1, "Unknown": 2}
    manifest_rows.sort(key=lambda r: (
        r["dataset"], label_order[r["answer"]],
        salted_key(str(seed), r["dataset"], r["official_theory_id"], r["official_qid"])))

    for i, r in enumerate(manifest_rows):
        r["sample_id"] = i
        r["key"] = salted_key(str(seed), r["dataset"], r["official_theory_id"], r["official_qid"])
        r["content_sha256"] = sha16(r["theory_text"] + "\x1f" + r["question_text"])

    # HARD ASSERTION (review finding #4, 2026-09-04): the shortfall-refill
    # logic above can legitimately under-fill a pool when the real data is
    # short (that is what `shortfalls` reports), but nothing previously
    # verified the manifest ACTUALLY reached its target shape before it was
    # written to disk and frozen. A silent 280-item or 90/60/60-label
    # manifest would pass through untouched, and the 300/150/100 numbers
    # baked into the pre-registration and every downstream test would then
    # be describing a manifest that does not exist. Fail closed rather than
    # let a shortfall-heavy real run silently freeze a smaller manifest.
    # Gated on `_assert_final_shape` only so a unit test isolating the
    # per-label refill logic on a deliberately-starved dataset (e.g. an
    # empty D5) can opt out; every real call site leaves this True.
    if not _assert_final_shape:
        return manifest_rows, shortfalls
    if len(manifest_rows) != N_TOTAL:
        raise SystemExit(
            f"[FATAL] manifest has {len(manifest_rows)} rows, expected "
            f"exactly {N_TOTAL}. shortfalls recorded: {shortfalls}")
    by_ds = collections.Counter(r["dataset"] for r in manifest_rows)
    for ds in ("D3", "D5"):
        if by_ds.get(ds) != N_PER_DATASET:
            raise SystemExit(
                f"[FATAL] manifest dataset={ds} has {by_ds.get(ds, 0)} rows, "
                f"expected exactly {N_PER_DATASET}. shortfalls: {shortfalls}")
    by_ds_lab = collections.Counter((r["dataset"], r["answer"]) for r in manifest_rows)
    for ds in ("D3", "D5"):
        for lab in ("True", "False", "Unknown"):
            n = by_ds_lab.get((ds, lab), 0)
            if n != N_PER_LABEL:
                raise SystemExit(
                    f"[FATAL] manifest {ds}/{lab} has {n} rows, expected "
                    f"exactly {N_PER_LABEL}. shortfalls: {shortfalls}")

    return manifest_rows, shortfalls


def write_manifest_files(manifest_rows: list[dict], out_dir: str, seed: int,
                          archive_sha256: str, source: str, revision_note: str):
    os.makedirs(out_dir, exist_ok=True)

    blind = []
    for r in manifest_rows:
        rec = {
            "sample_id": r["sample_id"], "key": r["key"],
            "dataset": r["dataset"],
            "official_theory_id": r["official_theory_id"],
            "official_qid": r["official_qid"],
            "theory_text": r["theory_text"],
            "question_text": r["question_text"],
            "content_sha256": r["content_sha256"],
        }
        extra = set(rec) - BLIND_ALLOWED
        if extra:
            die(f"blind record carries non-whitelisted key(s) {sorted(extra)}")
        leaked = {k for k in rec if k.lower() in LABEL_FIELDS}
        if leaked:
            die(f"label field {sorted(leaked)} reached the blind record")
        blind.append(rec)

    digest = sha16(json.dumps(
        [{"sample_id": r["sample_id"], "key": r["key"],
          "dataset": r["dataset"], "answer": r["answer"]} for r in manifest_rows],
        sort_keys=True))

    meta = {
        "protocol": PROTOCOL, "seed": seed, "n_total": len(manifest_rows),
        "n_per_dataset": {"D3": sum(1 for r in manifest_rows if r["dataset"] == "D3"),
                          "D5": sum(1 for r in manifest_rows if r["dataset"] == "D5")},
        "label_counts": dict(Counter(r["answer"] for r in manifest_rows)),
        "archive_sha256": archive_sha256, "source": source,
        "revision_note": revision_note,
        "manifest_sha256_16": digest,
        "owa_semantics": ("query provable -> True; explicit negation of the "
                          "query provable -> False; neither provable -> "
                          "Unknown. Failing to prove the query does NOT by "
                          "itself mean False."),
        "cot_note": ("Explicit CoT prompting is this project's own "
                     "construction, NOT an official ProofWriter LLM prompt."),
        "not_a_transfer_test": ("This is ProofWriter's own task-specific "
                                "workpoint exploration, not a GSM8K "
                                "fixed-workpoint transfer test."),
    }

    bpath = os.path.join(out_dir, "manifest_blind.json")
    gpath = os.path.join(out_dir, "manifest_gold.json")
    mpath = os.path.join(out_dir, "manifest_meta.json")
    for p in (bpath, gpath, mpath):
        if os.path.exists(p):
            die(f"{p} exists; refusing to overwrite a frozen manifest. "
                "Delete it deliberately if this is a real re-freeze.")

    json.dump({"meta": {**meta, "contains_labels": False}, "data": blind},
              open(bpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump({"meta": {**meta, "contains_labels": True}, "data": manifest_rows},
              open(gpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(meta, open(mpath, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"\nblind    -> {bpath}   (labels: NO -- the generator reads this)")
    print(f"gold     -> {gpath}   (labels: YES -- scorer/audit reads this)")
    print(f"meta     -> {mpath}")
    print(f"manifest_sha256_16 = {digest}")
    return meta


# ───────────────────────── CLI ─────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", metavar="DIR",
                    help="fetch the official archive into DIR (network required)")
    ap.add_argument("--url", default=OFFICIAL_URL)
    ap.add_argument("--expected_sha256", default=None,
                    help="pin the archive's sha256; hard stop on mismatch")
    ap.add_argument("--archive_dir", default=None,
                    help="directory already containing the official zip")
    ap.add_argument("--archive_path", default=None,
                    help="explicit path to the zip, overrides --archive_dir")
    ap.add_argument("--report", action="store_true",
                    help="print D3/D5 theory/question/label/QDep report; no writes")
    ap.add_argument("--build_manifest", action="store_true",
                    help="freeze the 300-item manifest; requires --out_dir")
    ap.add_argument("--check", action="store_true",
                    help="re-run report + manifest build in-memory and print "
                         "digests without writing (idempotence check)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf_mirror", default=None,
                    help="opt-in HF mirror repo id; requires "
                         "--verify_against_official and is NOT used otherwise")
    ap.add_argument("--verify_against_official", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()

    if a.download:
        download_archive(a.download, url=a.url, expected_sha256=a.expected_sha256)
        if not (a.report or a.build_manifest or a.check):
            return

    if a.hf_mirror and not a.verify_against_official:
        die("--hf_mirror requires --verify_against_official; there is no "
            "silent-fallback path to an unverified mirror in this loader.")
    if a.hf_mirror:
        die("HF-mirror verification is not implemented in this offline pass "
            "(no network/datasets library available in this environment). "
            "Run this loader's official-archive path on the server, or "
            "implement+run mirror verification there deliberately before "
            "using --hf_mirror for anything.")

    zip_path = a.archive_path
    if not zip_path and a.archive_dir:
        zip_path = os.path.join(a.archive_dir, ARCHIVE_BASENAME)
    if not zip_path or not os.path.exists(zip_path):
        die(f"official archive not found (looked for {zip_path!r}). Run "
            "--download <dir> first, or pass --archive_path/--archive_dir "
            "pointing at an already-downloaded proofwriter-dataset-V2020.12.3.zip.")

    archive_sha = sha256_file(zip_path)
    print(f"[archive] {zip_path}  sha256={archive_sha}")

    records_by_dataset = {}
    for ds in ("D3", "D5"):
        records_by_dataset[ds] = parse_depth_split(zip_path, ds, FORMAL_SPLIT)

    if a.report or a.check:
        report(records_by_dataset)

    if a.build_manifest or a.check:
        rows, shortfalls = build_manifest(records_by_dataset, seed=a.seed)
        if shortfalls:
            print("\n[manifest] shortfalls (reported, not silently forced):")
            for s in shortfalls:
                print(f"  - {s}")
        n_by_ds = Counter(r["dataset"] for r in rows)
        n_by_label = Counter(r["answer"] for r in rows)
        print(f"\n[manifest] n_total={len(rows)}  by_dataset={dict(n_by_ds)}  "
              f"by_label={dict(n_by_label)}")

        if a.check:
            digest = sha16(json.dumps(
                [{"sample_id": r["sample_id"], "key": r["key"],
                  "dataset": r["dataset"], "answer": r["answer"]} for r in rows],
                sort_keys=True))
            print(f"[check] manifest_sha256_16 = {digest}  (nothing written)")
            return

        if not a.out_dir:
            die("--build_manifest requires --out_dir")
        write_manifest_files(rows, a.out_dir, seed=a.seed,
                             archive_sha256=archive_sha,
                             source=f"official:{a.url}",
                             revision_note=("official AI2 release archive "
                                            "V2020.12.3, OWA D3/D5 test split"))


if __name__ == "__main__":
    main()
