#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression test for the max_new_tokens hard-freeze (review finding,
2026-09-04): a comment-only constraint on the argparse default left
--max_new_tokens freely overridable from the command line with no
enforcement, so "raise once to 1024 and never again" was not actually
binding.

get_answer_proofwriter_owa.py imports numpy/tqdm unconditionally at module
level (a repo-wide convention for GPU-dependent scripts -- see e.g.
get_answer_bbh_numeric.py), so it cannot be imported OR run as a subprocess
in this numpy-less local environment; even --help fails with
ModuleNotFoundError before argparse ever runs. That means this test cannot
drive the real CLI end-to-end here (a real server run, where numpy/torch are
installed, is a separate live check). Instead it extracts the SOURCE of
main() with `ast` and asserts the hard-rejection guard is present and
correctly wired -- structurally equivalent to what a mutation of this file
would need to defeat, without needing numpy/torch/a GPU.

No GPU, no network, no real ProofWriter data. Run with:
    python3 test_max_new_tokens_frozen.py
"""

from __future__ import annotations

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PW_DIR = os.path.dirname(HERE)
SCRIPT_PATH = os.path.join(PW_DIR, "get_answer_proofwriter_owa.py")

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


def _parse_module():
    src = open(SCRIPT_PATH, encoding="utf-8").read()
    return src, ast.parse(src, filename=SCRIPT_PATH)


def _find_assign(tree, name):
    """Top-level `name = <literal>` assignment; returns the literal value."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    return None


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def test_max_new_tokens_frozen_constant_is_1024():
    _, tree = _parse_module()
    val = _find_assign(tree, "MAX_NEW_TOKENS_FROZEN")
    check("MAX_NEW_TOKENS_FROZEN is defined as a module-level literal",
          val is not None)
    check("MAX_NEW_TOKENS_FROZEN == 1024 (the decided, frozen budget)",
          val == 1024, f"got {val!r}")


def test_argparse_default_uses_the_frozen_constant():
    """The --max_new_tokens argparse default must reference the SAME
    constant that main() checks against, not a separately hardcoded 1024 --
    otherwise the two could silently drift apart on a future edit."""
    src, _ = _parse_module()
    check("the --max_new_tokens argparse default references "
          "MAX_NEW_TOKENS_FROZEN (not a separately hardcoded literal)",
          'default=MAX_NEW_TOKENS_FROZEN' in src.replace(" ", ""),
          "expected 'default=MAX_NEW_TOKENS_FROZEN' (whitespace-"
          "insensitive) somewhere in the --max_new_tokens argument "
          "definition")


def test_main_hard_rejects_non_frozen_value_before_manifest_load():
    """Structural check on main()'s AST: an `if args.max_new_tokens !=
    MAX_NEW_TOKENS_FROZEN:` guard (or equivalent) must exist, and it must
    appear BEFORE the first call to load_manifest(...) in main()'s body --
    otherwise a wrong value could reach manifest/model loading before being
    rejected, which would misreport the real problem as a manifest error
    when debugging a failed launch."""
    _, tree = _parse_module()
    main_fn = _find_function(tree, "main")
    check("main() function exists", main_fn is not None)
    if main_fn is None:
        return

    guard_index = None
    load_manifest_index = None
    for i, stmt in enumerate(main_fn.body):
        stmt_src = ast.dump(stmt)
        if (guard_index is None and isinstance(stmt, ast.If)
                and "max_new_tokens" in ast.dump(stmt.test)
                and "MAX_NEW_TOKENS_FROZEN" in ast.dump(stmt.test)
                and "NotEq" in ast.dump(stmt.test)):
            guard_index = i
        if load_manifest_index is None and "load_manifest" in stmt_src:
            load_manifest_index = i

    check("main() contains an `if args.max_new_tokens != "
          "MAX_NEW_TOKENS_FROZEN:` guard", guard_index is not None)
    check("load_manifest(...) is called somewhere in main()",
          load_manifest_index is not None)
    if guard_index is not None and load_manifest_index is not None:
        check("the max_new_tokens guard appears BEFORE load_manifest() is "
              "called (not after)", guard_index < load_manifest_index,
              f"guard at stmt {guard_index}, load_manifest at "
              f"{load_manifest_index}")


def test_guard_body_calls_die_not_a_silent_pass():
    """The guard's body must actually call die() (which raises SystemExit)
    -- not just log a warning and continue, which would defeat the whole
    point of a hard rejection."""
    _, tree = _parse_module()
    main_fn = _find_function(tree, "main")
    guard = None
    for stmt in main_fn.body:
        if (isinstance(stmt, ast.If) and "max_new_tokens" in ast.dump(stmt.test)
                and "MAX_NEW_TOKENS_FROZEN" in ast.dump(stmt.test)):
            guard = stmt
            break
    check("found the max_new_tokens guard's If node", guard is not None)
    if guard is None:
        return
    # guard.body is a LIST of statements, not a single AST node -- ast.dump
    # needs one node at a time.
    body_src = " ".join(ast.dump(stmt) for stmt in guard.body)
    check("the guard's body calls die(...) (raises SystemExit), not a "
          "silent pass/continue", "die" in body_src)


def main():
    print("== MAX_NEW_TOKENS_FROZEN constant is 1024 ==")
    test_max_new_tokens_frozen_constant_is_1024()
    print("== argparse default references the frozen constant ==")
    test_argparse_default_uses_the_frozen_constant()
    print("== main() hard-rejects a non-frozen value before load_manifest ==")
    test_main_hard_rejects_non_frozen_value_before_manifest_load()
    print("== the guard actually calls die(), not a silent pass ==")
    test_guard_body_calls_die_not_a_silent_pass()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
