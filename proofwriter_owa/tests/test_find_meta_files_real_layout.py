#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression test for _find_meta_files against the REAL official archive
layout, captured from an actual server run against
proofwriter-dataset-V2020.12.3.zip (2026-09-04):

    [FATAL] multiple candidates for depth-3/OWA/meta-test.jsonl:
    ['proofwriter-dataset-V2020.12.3/OWA/depth-3/meta-test.jsonl',
     'proofwriter-dataset-V2020.12.3/OWA/depth-3ext-NatLang/meta-test.jsonl',
     'proofwriter-dataset-V2020.12.3/OWA/depth-3ext/meta-test.jsonl']

Two real-layout facts an earlier version of _find_meta_files got wrong:
  1. the path order is OWA/<depth-N>/..., not <depth-N>/OWA/... as an
     earlier version assumed (both orders are now accepted, since the
     match no longer depends on segment order).
  2. depth-N is a PREFIX shared by "depth-N", "depth-Next" and
     "depth-Next-NatLang" -- a substring/prefix match on the folder name
     makes all three look like valid candidates for plain "depth-N", which
     is a real ambiguity bug (fail-closed correctly refused to guess, but
     the correct fix is precise matching, not accepting the ambiguity).
     PREREG_PROOFWRITER_OWA.md scopes this loader to the official OWA
     depth-3/depth-5 Task 1 files only -- the ext/ext-NatLang variants are
     a different task family and must never be silently substituted or
     merged in.

No network, no real ProofWriter data -- synthetic zip only. Run with:
    python3 test_find_meta_files_real_layout.py
"""

from __future__ import annotations

import io
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_proofwriter_owa as dpo  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


def _real_layout_zip_bytes():
    """Reproduces the exact path set from the real server error message,
    for both depth-3 and depth-5 families, plus the OWA/depth-N/... file
    that must actually be selected."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        root = "proofwriter-dataset-V2020.12.3/OWA"
        zf.writestr(f"{root}/depth-3/meta-test.jsonl", "{}")
        zf.writestr(f"{root}/depth-3ext/meta-test.jsonl", "{}")
        zf.writestr(f"{root}/depth-3ext-NatLang/meta-test.jsonl", "{}")
        zf.writestr(f"{root}/depth-5/meta-test.jsonl", "{}")
        zf.writestr(f"{root}/depth-5ext/meta-test.jsonl", "{}")
        zf.writestr(f"{root}/depth-5ext-NatLang/meta-test.jsonl", "{}")
    return buf.getvalue()


def test_exact_depth3_selected_over_ext_variants():
    with zipfile.ZipFile(io.BytesIO(_real_layout_zip_bytes())) as zf:
        got = dpo._find_meta_files(zf, "depth-3", "test")
    check("selects exactly OWA/depth-3/meta-test.jsonl, not an ext variant",
          got == "proofwriter-dataset-V2020.12.3/OWA/depth-3/meta-test.jsonl",
          f"got {got!r}")


def test_exact_depth5_selected_over_ext_variants():
    with zipfile.ZipFile(io.BytesIO(_real_layout_zip_bytes())) as zf:
        got = dpo._find_meta_files(zf, "depth-5", "test")
    check("selects exactly OWA/depth-5/meta-test.jsonl, not an ext variant",
          got == "proofwriter-dataset-V2020.12.3/OWA/depth-5/meta-test.jsonl",
          f"got {got!r}")


def test_owa_before_depth_order_also_works():
    """The real archive puts OWA before depth-N (OWA/depth-3/...); an
    earlier version of this function assumed the reverse order
    (depth-3/OWA/...) and both must work since path-segment matching does
    not depend on ordering."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("root/depth-3/OWA/meta-test.jsonl", "{}")  # depth-first
    with zipfile.ZipFile(buf) as zf:
        got = dpo._find_meta_files(zf, "depth-3", "test")
    check("depth-N-before-OWA ordering still resolves correctly",
          got == "root/depth-3/OWA/meta-test.jsonl", f"got {got!r}")


def test_ext_only_archive_is_not_silently_substituted():
    """If an archive somehow shipped ONLY the ext variant (no plain
    depth-3), asking for depth-3 must still fail closed rather than
    silently returning the ext file as if it were the requested task."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("root/OWA/depth-3ext/meta-test.jsonl", "{}")
    died = False
    try:
        with zipfile.ZipFile(buf) as zf:
            dpo._find_meta_files(zf, "depth-3", "test")
    except SystemExit:
        died = True
    check("an ext-only archive hard-stops for a plain depth-3 request "
          "rather than silently substituting the ext variant", died)


def main():
    print("== exact depth-3 selected over ext variants (real layout) ==")
    test_exact_depth3_selected_over_ext_variants()
    print("== exact depth-5 selected over ext variants (real layout) ==")
    test_exact_depth5_selected_over_ext_variants()
    print("== OWA-before-depth ordering also resolves ==")
    test_owa_before_depth_order_also_works()
    print("== ext-only archive is not silently substituted ==")
    test_ext_only_archive_is_not_silently_substituted()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
