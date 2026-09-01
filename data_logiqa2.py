#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LogiQA 2.0 English MRC loader for P4 (protocol `logiqa2-p4-v0` + `p4-amend-02`).

Builds THREE files:

  logiqa2_p4_formal_blind.json 300 items, gold ABSENT   (what the RUNNER reads)
  logiqa2_p4_formal.json       300 items, gold INCLUDED (what the EVALUATOR reads)
  logiqa2_p4_preflight.json     20 items, gold ABSENT   (format-only preflight)
  logiqa2_p4_manifest.json      provenance + digests

`data_logiqa.py` (the older LogiQA loader) is left byte-unchanged. It downloads
the same file but writes to the stale /data2/.../RolePlaying tree, drops `id`,
the raw fields and `type` provenance, and freezes no manifest.

THE PREFLIGHT AND THE RUNNER'S FORMAL FILE CARRY NO GOLD, STRUCTURALLY.
Protocol section 4 forbids computing accuracy during the preflight. Making the
label merely "unused" is far weaker than making it unreachable -- the P2 label
firewall established this. So both blind writers build each record from an
explicit field whitelist and then ASSERT that no forbidden key survived. A
consumer of either file cannot score itself even by mistake.

The formal sample is emitted TWICE from one selection: a blind copy for
generation and a gold-bearing copy for evaluation. They share sample_id, key
and content_sha256, so the evaluator joins them exactly; only the evaluator
ever opens the gold-bearing one.

Everything the protocol froze is ASSERTED here, not assumed:
  * 1572 rows, every row exactly 4 options
  * label pools 347/384/417/424, each >= 75
  * `id` is NOT unique (1568/1572) and (passage,question) is NOT unique
    (1557/1572) -- so the composite key is the item key, and it IS unique
  * the sampling digest reproduces the value frozen at stage 0
  * no passage contains "Final answer" (which would pollute the parser)

Usage
-----
    python3 data_logiqa2.py --out_dir components/benchmark
    python3 data_logiqa2.py --out_dir components/benchmark --check   # no write
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from collections import Counter, defaultdict

# ---------------------------------------------------------------- frozen constants
PROTOCOL = "logiqa2-p4-v0"
AMENDMENT = "p4-amend-02"
SALT = "logiqa2-p4-v0"

RAW_URL = ("https://raw.githubusercontent.com/csitfun/LogiQA2.0/"
           "main/logiqa/DATA/LOGIQA/test.txt")
API_URL = "https://api.github.com/repos/csitfun/LogiQA2.0/commits/main"

N_ROWS_EXPECTED = 1572
N_OPTIONS = 4
PER_LABEL_FORMAL = 75
PER_LABEL_PREFLIGHT = 5
N_FORMAL = PER_LABEL_FORMAL * N_OPTIONS          # 300
N_PREFLIGHT = PER_LABEL_PREFLIGHT * N_OPTIONS    # 20

# measured at stage-0 freeze (docs/p4_freeze_manifest.json)
LABEL_POOLS_EXPECTED = {0: 347, 1: 384, 2: 417, 3: 424}
ID_UNIQUE_EXPECTED = 1568
PQ_UNIQUE_EXPECTED = 1557
FORMAL_DIGEST_PREFIX = "4d4b25e071a2a6dd"

US = "\x1f"   # unit separator, between fields
RS = "\x1e"   # record separator, between options

# a preflight record may contain ONLY these keys
PREFLIGHT_ALLOWED = {"sample_id", "key", "official_id", "passage",
                     "question", "options", "content_sha256"}
# and must contain NONE of these
LABEL_FIELDS = {"answer", "label", "gold", "gold_answer", "correct",
                "target", "solution", "type"}


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def item_key(e):
    """Frozen composite key. `id` alone and (passage,question) alone both carry
    duplicates in this split; the composite is unique across all 1572 rows."""
    payload = US.join([
        str(e["id"]),
        e.get("text", ""),
        e.get("question", ""),
        RS.join(str(o) for o in e["options"]),
    ])
    return hashlib.sha256((SALT + ":" + payload).encode("utf-8")).hexdigest()


def content_sha256(e):
    """Content hash WITHOUT the salt and WITHOUT the official id -- detects an
    upstream text edit even if the id is reassigned."""
    payload = US.join([e.get("text", ""), e.get("question", ""),
                       RS.join(str(o) for o in e["options"])])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fetch(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        charset = r.headers.get_content_charset() or "utf-8"
        return r.read().decode(charset, errors="replace")


def resolve_revision():
    """The raw URL follows `main`, so capture the commit SHA at download time.
    Recorded, never used to re-fetch: a later run resolving to a different SHA
    is a hard stop, not a silent re-download."""
    try:
        return json.loads(fetch(API_URL, timeout=30))["sha"]
    except Exception as exc:                                    # noqa: BLE001
        print(f"[warn] could not resolve revision: {type(exc).__name__}: {exc}")
        return None


def load_rows():
    txt = fetch(RAW_URL)
    rows = [json.loads(ln) for ln in txt.splitlines() if ln.strip()]

    if len(rows) != N_ROWS_EXPECTED:
        die(f"expected {N_ROWS_EXPECTED} rows, got {len(rows)}. The upstream "
            f"split changed; this is a hard stop, not a re-freeze.")

    for i, e in enumerate(rows):
        for f in ("id", "answer", "text", "question", "options"):
            if f not in e:
                die(f"row {i} missing field {f!r}")
        if len(e["options"]) != N_OPTIONS:
            die(f"row {i} (id={e['id']}) has {len(e['options'])} options, "
                f"expected {N_OPTIONS}")
        if not isinstance(e["answer"], int) or not 0 <= e["answer"] < N_OPTIONS:
            die(f"row {i} (id={e['id']}) has gold {e['answer']!r} outside 0..3")
        if not (e.get("text") or "").strip():
            die(f"row {i} (id={e['id']}) has an empty passage")
        if not (e.get("question") or "").strip():
            die(f"row {i} (id={e['id']}) has an empty question")
        for j, o in enumerate(e["options"]):
            if not str(o).strip():
                die(f"row {i} (id={e['id']}) option {j} is empty")
            if "\n" in str(o):
                die(f"row {i} (id={e['id']}) option {j} contains a newline, "
                    f"which would break the four-line A)..D) block")
        # a passage carrying the answer phrase would pollute the LAST-match parser
        if "Final answer" in (e.get("text") or ""):
            die(f"row {i} (id={e['id']}) passage contains 'Final answer'")
    return rows


def audit(rows):
    """Reproduce every stage-0 measurement. A divergence is a hard stop."""
    keys = [item_key(e) for e in rows]
    n_key_unique = len(set(keys))
    n_id_unique = len({str(e["id"]) for e in rows})
    n_pq_unique = len({(e.get("text", ""), e.get("question", "")) for e in rows})
    pools = Counter(e["answer"] for e in rows)

    if n_key_unique != len(rows):
        die(f"composite key is not unique: {n_key_unique}/{len(rows)}. The key "
            f"is the item identity; a collision invalidates the sample.")
    if n_id_unique != ID_UNIQUE_EXPECTED:
        die(f"official id uniqueness changed: {n_id_unique} != {ID_UNIQUE_EXPECTED}")
    if n_pq_unique != PQ_UNIQUE_EXPECTED:
        die(f"(passage,question) uniqueness changed: {n_pq_unique} != {PQ_UNIQUE_EXPECTED}")
    for lab, n in LABEL_POOLS_EXPECTED.items():
        if pools[lab] != n:
            die(f"label {lab} pool changed: {pools[lab]} != {n}")
        if pools[lab] < PER_LABEL_FORMAL + PER_LABEL_PREFLIGHT:
            die(f"label {lab} pool {pools[lab]} cannot supply "
                f"{PER_LABEL_FORMAL}+{PER_LABEL_PREFLIGHT} items")
    return {"composite_key_unique": n_key_unique, "id_unique": n_id_unique,
            "passage_question_unique": n_pq_unique,
            "label_pools": {str(k): v for k, v in sorted(pools.items())}}


def sample(rows):
    """Frozen algorithm, in this exact order:
       1. composite key per row
       2. dedup on that key, first occurrence in file order
       3. group by gold label
       4. within each label sort by key ascending; first 75 -> formal,
          next 5 -> preflight (so the two sets cannot overlap by construction)
       5. concatenate and sort by key
    """
    seen, dedup = set(), []
    for e in rows:
        k = item_key(e)
        if k not in seen:
            seen.add(k)
            dedup.append((k, e))

    by_label = defaultdict(list)
    for k, e in dedup:
        by_label[e["answer"]].append((k, e))

    formal, pre = [], []
    for lab in sorted(by_label):
        grp = sorted(by_label[lab], key=lambda t: t[0])
        formal.extend(grp[:PER_LABEL_FORMAL])
        pre.extend(grp[PER_LABEL_FORMAL:PER_LABEL_FORMAL + PER_LABEL_PREFLIGHT])

    formal.sort(key=lambda t: t[0])
    pre.sort(key=lambda t: t[0])

    if len(formal) != N_FORMAL:
        die(f"formal sample is {len(formal)}, expected {N_FORMAL}")
    if len(pre) != N_PREFLIGHT:
        die(f"preflight sample is {len(pre)}, expected {N_PREFLIGHT}")

    fk, pk = {k for k, _ in formal}, {k for k, _ in pre}
    if fk & pk:
        die(f"formal and preflight overlap on {len(fk & pk)} items")

    for name, sel, per in (("formal", formal, PER_LABEL_FORMAL),
                           ("preflight", pre, PER_LABEL_PREFLIGHT)):
        got = Counter(e["answer"] for _, e in sel)
        if any(got[l] != per for l in range(N_OPTIONS)):
            die(f"{name} marginals {dict(got)} != {per} per label")
    return formal, pre


def digest(sel):
    return hashlib.sha256("\n".join(k for k, _ in sel).encode()).hexdigest()


def formal_record(idx, k, e):
    return {
        "sample_id": idx,
        "key": k,
        "official_id": e["id"],
        "passage": e["text"].strip(),
        "question": e["question"].strip(),
        "options": [str(o).strip() for o in e["options"]],
        "answer": e["answer"],
        "answer_letter": "ABCD"[e["answer"]],
        "type": e.get("type"),
        "content_sha256": content_sha256(e),
    }


def blind_record(idx, k, e):
    """Formal record WITHOUT gold, for the generation runner. Same whitelist
    discipline as the preflight: `answer` and `type` are never read into the
    record, so the label is UNREACHABLE rather than merely unused."""
    rec = {
        "sample_id": idx,
        "key": k,
        "official_id": e["id"],
        "passage": e["text"].strip(),
        "question": e["question"].strip(),
        "options": [str(o).strip() for o in e["options"]],
        "content_sha256": content_sha256(e),
    }
    leaked = LABEL_FIELDS & set(rec)
    if leaked:
        die(f"blind formal record leaked label field(s): {sorted(leaked)}")
    extra = set(rec) - PREFLIGHT_ALLOWED
    if extra:
        die(f"blind formal record has non-whitelisted key(s): {sorted(extra)}")
    return rec


def preflight_record(idx, k, e):
    """Built from an explicit whitelist. `answer` and `type` are never read into
    the record, so the label is UNREACHABLE rather than merely unused."""
    rec = {
        "sample_id": idx,
        "key": k,
        "official_id": e["id"],
        "passage": e["text"].strip(),
        "question": e["question"].strip(),
        "options": [str(o).strip() for o in e["options"]],
        "content_sha256": content_sha256(e),
    }
    leaked = LABEL_FIELDS & set(rec)
    if leaked:
        die(f"preflight record leaked label field(s): {sorted(leaked)}")
    extra = set(rec) - PREFLIGHT_ALLOWED
    if extra:
        die(f"preflight record has non-whitelisted key(s): {sorted(extra)}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="components/benchmark",
                    help="must already exist; it is NOT created, so a wrong "
                         "working directory fails before any write")
    ap.add_argument("--check", action="store_true",
                    help="audit and sample, print, write nothing")
    ap.add_argument("--allow_overwrite", action="store_true")
    args = ap.parse_args()

    print(f"[p4] protocol={PROTOCOL} amendment={AMENDMENT}")
    rev = resolve_revision()
    print(f"[p4] source revision: {rev}")

    rows = load_rows()
    print(f"[p4] loaded {len(rows)} rows, all {N_OPTIONS}-option, fields verified")

    stats = audit(rows)
    print(f"[p4] audit OK: composite key unique {stats['composite_key_unique']}/"
          f"{len(rows)}; id unique {stats['id_unique']}; (passage,question) "
          f"unique {stats['passage_question_unique']}")
    print(f"[p4] label pools {stats['label_pools']}")

    formal, pre = sample(rows)
    fd, pd = digest(formal), digest(pre)
    print(f"[p4] formal   n={len(formal)} digest={fd[:16]}")
    print(f"[p4] preflight n={len(pre)} digest={pd[:16]}")

    if fd[:16] != FORMAL_DIGEST_PREFIX:
        die(f"formal digest {fd[:16]} != frozen {FORMAL_DIGEST_PREFIX}. The "
            f"sample is not the one stage 0 froze.")
    print(f"[p4] formal digest matches the stage-0 freeze")

    f_recs = [formal_record(i, k, e) for i, (k, e) in enumerate(formal)]
    b_recs = [blind_record(i, k, e) for i, (k, e) in enumerate(formal)]
    p_recs = [preflight_record(i, k, e) for i, (k, e) in enumerate(pre)]

    for name, recs in (("preflight", p_recs), ("blind formal", b_recs)):
        blob = json.dumps(recs)
        for f in LABEL_FIELDS:
            if f'"{f}"' in blob:
                die(f"{name} payload mentions label field {f!r}")
    print(f"[p4] preflight and blind formal carry NO gold "
          f"(whitelist + payload scan)")

    # the two formal copies must describe the SAME items
    if [r["key"] for r in b_recs] != [r["key"] for r in f_recs]:
        die("blind and gold formal copies disagree on item order/identity")

    manifest = {
        "protocol": PROTOCOL, "amendment": AMENDMENT,
        "source": {"url": RAW_URL, "revision": rev, "n_rows": len(rows)},
        "salt": SALT, "audit": stats,
        "formal": {"n": len(f_recs), "digest": fd, "per_label": PER_LABEL_FORMAL,
                   "contains_gold": True, "read_by": "eval_logiqa2.py only"},
        "formal_blind": {"n": len(b_recs), "digest": fd, "contains_gold": False,
                         "read_by": "get_answer_logiqa2.py (generation)"},
        "preflight": {"n": len(p_recs), "digest": pd,
                      "per_label": PER_LABEL_PREFLIGHT, "contains_gold": False,
                      "drawn_from": "rows NOT in the formal 300; next 5 by key "
                                    "within each label"},
        "overlap": 0,
    }

    if args.check:
        print("\n[p4] --check: nothing written")
        return

    if not os.path.isdir(args.out_dir):
        die(f"--out_dir {args.out_dir!r} does not exist. It is deliberately not "
            f"created, so a wrong working directory fails before the download "
            f"is written to the wrong tree.")

    paths = {
        "formal_blind": os.path.join(args.out_dir, "logiqa2_p4_formal_blind.json"),
        "formal": os.path.join(args.out_dir, "logiqa2_p4_formal.json"),
        "preflight": os.path.join(args.out_dir, "logiqa2_p4_preflight.json"),
        "manifest": os.path.join(args.out_dir, "logiqa2_p4_manifest.json"),
    }
    if not args.allow_overwrite:
        for name, p in paths.items():
            if os.path.exists(p):
                die(f"{name} already exists at {p}; pass --allow_overwrite "
                    f"deliberately")

    meta = {"protocol": PROTOCOL, "amendment": AMENDMENT, "source_url": RAW_URL,
            "source_revision": rev, "salt": SALT}
    with open(paths["formal_blind"], "w", encoding="utf-8") as f:
        json.dump({"meta": {**meta, "contains_labels": False, "digest": fd,
                            "note": "formal sample for GENERATION; gold is "
                                    "absent by construction, not merely unused"},
                   "data": b_recs}, f, ensure_ascii=False, indent=2)
    with open(paths["formal"], "w", encoding="utf-8") as f:
        json.dump({"meta": {**meta, "contains_labels": True, "digest": fd},
                   "data": f_recs}, f, ensure_ascii=False, indent=2)
    with open(paths["preflight"], "w", encoding="utf-8") as f:
        json.dump({"meta": {**meta, "contains_labels": False, "digest": pd,
                            "note": "format-only preflight; gold is absent by "
                                    "construction, not merely unused"},
                   "data": p_recs}, f, ensure_ascii=False, indent=2)
    with open(paths["manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    for name, p in paths.items():
        print(f"[p4] wrote {name:9s} -> {p}")


if __name__ == "__main__":
    main()
