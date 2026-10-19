#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrity/provenance check of a `gsm8k-role-correspondence-v1` Qwen cell. Stdlib only, read-only.

Used (a) by the launcher before skipping an existing cell and (b) before offline analysis.
Re-renders the prompts from the test file with the repo's own template and compares sha256
with run_meta, so a stale or differently-routed cell fails instead of being skipped.
Exit 0 = passed, 1 = failed (every problem printed).
"""
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from template import select_templates_gsm8k  # noqa: E402

ROLES = ["an expert", "a non expert", "neutral"]
KEYS = {"an expert": "an_expert", "a non expert": "a_non_expert", "neutral": "neutral"}


def verify(cell_dir, test_file, expect_n=300, quiet=False):
    prob = []
    outs = [f for f in os.listdir(cell_dir) if f.startswith("gsm8k_") and f.endswith(".json")] if os.path.isdir(cell_dir) else []
    metas = [f for f in os.listdir(cell_dir) if f.startswith("run_meta_gsm8k_")] if os.path.isdir(cell_dir) else []
    if len(outs) != 1 or len(metas) != 1:
        return [f"expected exactly 1 answers json + 1 run_meta in {cell_dir}, got {outs} / {metas}"]
    d = json.load(open(os.path.join(cell_dir, outs[0]), encoding="utf-8"))
    m = json.load(open(os.path.join(cell_dir, metas[0]), encoding="utf-8"))
    test = json.load(open(test_file, encoding="utf-8"))
    data = d["data"]
    if len(data) != expect_n or len(test) != expect_n:
        prob.append(f"n data={len(data)} test={len(test)} expect {expect_n}")
    if [x["question"] for x in data] != [x["question"] for x in test]:
        prob.append("question order differs from the test file")
    if [str(x.get("answer")) for x in data] != [str(x.get("answer")) for x in test]:
        prob.append("per-item gold answer differs from the test file")
    for r in ROLES:
        k = f"generated_{KEYS[r]}"
        if any(not isinstance(x.get(k), str) for x in data):
            prob.append(f"{k} missing/non-string in some samples")
    if m.get("roles") != ROLES:
        prob.append(f"roles {m.get('roles')} != {ROLES}")
    if m.get("alpha") != 0 or m.get("steering_fires") != 0 or m.get("steering_fires_expected") != 0:
        prob.append(f"alpha/fires: {m.get('alpha')} {m.get('steering_fires')} {m.get('steering_fires_expected')} (need 0/0/0)")
    if m.get("n_generated") != expect_n * len(ROLES):
        prob.append(f"n_generated {m.get('n_generated')} != {expect_n * len(ROLES)}")
    if not m.get("checks_passed"):
        prob.append(f"run_meta checks_passed false: {m.get('problems')}")
    g = m.get("generation", {})
    if (g.get("max_new_tokens"), g.get("temperature"), g.get("batch_size")) != (768, 0.0, 24):
        prob.append(f"generation params {g}")
    if (m.get("suite"), m.get("fmt_wording"), m.get("cot")) != ("default", "plain", False):
        prob.append(f"suite/wording/cot {m.get('suite')} {m.get('fmt_wording')} {m.get('cot')}")
    if m.get("model_dir") != "Qwen/Qwen2.5-7B-Instruct":
        prob.append(f"model_dir {m.get('model_dir')}")
    if m.get("test_file_sha256") != hashlib.sha256(open(test_file, "rb").read()).hexdigest():
        prob.append("test_file_sha256 differs from the supplied test file")
    # Re-render with the repo template: routing + hash must reproduce.
    tpl = select_templates_gsm8k(suite="default", cot=False, wording="plain")
    for r in ROLES:
        ps = [tpl["neutral"].format(context=s["question"]) if r == "neutral"
              else tpl["neg"].format(character=r, context=s["question"]) for s in test]
        h = hashlib.sha256("\x1f".join(ps).encode("utf-8")).hexdigest()
        a = m.get("prompt_audit", {}).get(r)
        if not a or a.get("prompts_sha256") != h:
            prob.append(f"role {r!r}: stored prompt hash != re-rendered hash")
        if "honest" in ps[0] or "step by step" in ps[0]:
            prob.append(f"role {r!r}: prompt contains honest/CoT wording")
    for r in ("an expert", "a non expert"):
        if "Now you are " + r + "." not in tpl["neg"].format(character=r, context="Q"):
            prob.append(f"neg template does not render role sentence for {r!r}")
    if "Now you are" in tpl["neutral"]:
        prob.append("neutral template carries a role sentence")
    if not quiet:
        print("[verify] " + ("PASS" if not prob else "FAIL") + f" {cell_dir}")
        for p in prob:
            print("  -", p)
    return prob


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell_dir", required=True)
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--expect_n", type=int, default=300)
    a = ap.parse_args()
    sys.exit(1 if verify(a.cell_dir, a.test_file, a.expect_n) else 0)
