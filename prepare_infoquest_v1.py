#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_infoquest_v1.py -- data/scenario preparation for `infoquest-v1`
(first-turn information-gap selectivity; see CLAUDE.md "InfoQuest" entry).

Subcommands (all offline once the three source files are cached):
  sample  Download seed_messages.jsonl + settings.jsonl at the PINNED revision,
          apply the structural filter, draw 130 distinct requests with
          random.Random(42), fix 30 as pilot / 100 as formal, pick one hidden
          setting per request by a hash rule. Writes infoquest_v1/sampling_v1.json.
          Refuses to overwrite.
  items   Parse the authored items_src_part*.txt into scenario_items_v1.json.
  build   Merge sampling_v1.json with the AUTHORED items file
          infoquest_v1/scenario_items_v1.json -> infoquest_v1/scenarios_v1.json
          (Complete = Incomplete + " " + supplement). Validates every scenario.
  freeze  Only if EVERY scenario is review_status == "reviewed": write
          infoquest_v1/scenarios_v1.freeze.json (sha256 of scenarios_v1.json).
          get_answer_infoquest_v1.py refuses to run without a matching freeze.
  check   Re-derive sampling from the source files and compare to the stored
          sampling_v1.json; re-validate scenarios_v1.json. Writes nothing.

No scenario is ever replaced because of model behaviour: the only filter is the
structural one below and it runs BEFORE the draw, so no sampled item is swapped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys

REPO_ID = "bryanlincoln/infoquest"
REVISION = "1f54a770a8bed73edcab86411254fff64ed53d25"
SOURCE_FILES = ("seed_messages.jsonl", "settings.jsonl")
SEED = 42
N_PILOT, N_FORMAL = 30, 100
RULE_VERSION = "infoquest-v1-sampling-1"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "infoquest_v1")
SAMPLING = os.path.join(DATA_DIR, "sampling_v1.json")
ITEMS = os.path.join(DATA_DIR, "scenario_items_v1.json")
SCENARIOS = os.path.join(DATA_DIR, "scenarios_v1.json")
FREEZE = os.path.join(DATA_DIR, "scenarios_v1.freeze.json")

# Instruction-like wording that must never appear in a Complete supplement
# (the Complete condition must differ from Incomplete ONLY by facts).
LEAK_PATTERNS = [
    r"\bno need to ask\b", r"\bdon'?t ask\b", r"\bdo not ask\b", r"\bwithout asking\b",
    r"\bjust (answer|give|tell)\b", r"\bdirectly\b", r"\balready (told|gave|provided|know)\b",
    r"\benough (information|detail|context)\b", r"\bsufficient\b", r"\bfollow-?up\b",
    r"\bclarif", r"\bchecklist\b", r"\bplease (answer|respond)\b",
]
REVIEW_STATES = ("draft_unreviewed", "reviewed")

# Content-safety rule, defined BEFORE any model run and applied uniformly (it
# never looks at model output). A sampled scenario whose request/setting text
# matches is replaced by the next passing candidate of a SECOND deterministic
# queue (random.Random(43) over the not-drawn candidates); every replacement is
# recorded in sampling_v1.json["safety_replacements"].
SAFETY_RE = re.compile(
    r"\b(drug dealer|dealer|illegal substances?|narcotics?|smuggl\w*|launder\w*|"
    r"evade (the )?(police|law)|avoid(ing)? (police|detection))\b", re.I)
QUEUE_SEED = 43


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: str) -> str:
    with open(p, "rb") as f:
        return sha256_bytes(f.read())


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def fetch_source(src_dir: str | None):
    """Return {filename: path}. Uses --src_dir if given (offline/tests),
    else downloads the pinned revision."""
    paths = {}
    if src_dir:
        for fn in SOURCE_FILES:
            p = os.path.join(src_dir, fn)
            if not os.path.isfile(p):
                die(f"{p} missing")
            paths[fn] = p
        return paths
    from huggingface_hub import hf_hub_download
    for fn in SOURCE_FILES:
        paths[fn] = hf_hub_download(REPO_ID, fn, repo_type="dataset", revision=REVISION)
    return paths


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def pick_setting_key(request_id: int) -> str:
    h = hashlib.sha256(f"infoquest-v1:setting:{request_id}".encode()).hexdigest()
    return "setting1" if int(h, 16) % 2 == 0 else "setting2"


def derive_sampling(paths: dict) -> dict:
    seeds = {r["id"]: r for r in load_jsonl(paths["seed_messages.jsonl"])}
    sets = {r["id"]: r for r in load_jsonl(paths["settings.jsonl"])}

    # (1) structural filter -- runs BEFORE the draw, never after.
    excluded = {}
    ok = {}
    for rid in sorted(seeds):
        msg = (seeds[rid].get("seed_message") or "").strip()
        st = sets.get(rid)
        reason = None
        if not msg or "\n" in msg:
            reason = "empty_or_multiline_request"
        elif st is None:
            reason = "no_settings_row"
        elif (st.get("seed_message") or "").strip() != msg:
            reason = "request_text_mismatch_between_files"
        else:
            sk = pick_setting_key(rid)
            s = st.get(sk) or {}
            if not s.get("constraints") or not s.get("checklist") or not s.get("description"):
                reason = "chosen_setting_missing_fields"
        if reason:
            excluded[rid] = reason
        else:
            ok[rid] = msg

    # (2) distinct initial requests: among identical normalised texts keep the
    # smallest id. The rule is purely textual -- no model output involved.
    first_by_text = {}
    for rid in sorted(ok):
        t = norm_text(ok[rid])
        if t in first_by_text:
            excluded[rid] = f"duplicate_request_text_of_id_{first_by_text[t]}"
        else:
            first_by_text[t] = rid
    cands = sorted(first_by_text.values())

    def flagged(rid):
        s_ = sets[rid][pick_setting_key(rid)]
        blob = " ".join([seeds[rid]["seed_message"], s_.get("persona") or "", s_.get("goal") or "",
                         s_["description"], " ".join(s_["constraints"])])
        return bool(SAFETY_RE.search(blob))

    rng = random.Random(SEED)
    drawn = rng.sample(cands, N_PILOT + N_FORMAL)
    replacements = []
    queue = random.Random(QUEUE_SEED).sample([c for c in cands if c not in set(drawn)], len(cands) - len(drawn))
    qi = 0
    for pos, rid in enumerate(list(drawn)):
        if flagged(rid):
            while flagged(queue[qi]):
                qi += 1
            replacements.append({"position": pos, "dropped_request_id": rid,
                                 "replacement_request_id": queue[qi], "reason": "safety_regex"})
            drawn[pos] = queue[qi]
            qi += 1
    split = {rid: "pilot" for rid in drawn[:N_PILOT]}
    split.update({rid: "formal" for rid in drawn[N_PILOT:]})

    scen = []
    for rid in drawn:
        sk = pick_setting_key(rid)
        s = sets[rid][sk]
        scen.append({
            "scenario_id": f"iq{rid:03d}",
            "split": split[rid],
            "request_id": rid,
            "setting_key": sk,
            "incomplete_text": ints(seeds[rid]["seed_message"]),
            "source": {
                "persona": s.get("persona"),
                "description": s["description"],
                "goal": s.get("goal"),
                "obstacle": s.get("obstacle"),
                "constraints": s["constraints"],
                "checklist": s["checklist"],
                "solution": s.get("solution"),
            },
        })
    return {
        "protocol": "infoquest-v1",
        "rule_version": RULE_VERSION,
        "source": {"repo": REPO_ID, "revision": REVISION,
                   "file_sha256": {fn: sha256_file(paths[fn]) for fn in SOURCE_FILES}},
        "rule": {
            "seed": SEED, "rng": "random.Random(42).sample(sorted candidate ids, 130)",
            "n_pilot": N_PILOT, "n_formal": N_FORMAL,
            "pilot": "first 30 of the draw", "formal": "next 100 of the draw",
            "safety_rule": "scenarios matching safety_regex are replaced in place from Random(43)'s queue of undrawn candidates",
            "structural_filter": "non-empty single-line request; settings row present with identical "
                                 "request text; chosen setting has description/constraints/checklist",
            "dedup": "identical whitespace/case-normalised request text -> keep smallest id",
            "setting_choice": "setting1 if int(sha256('infoquest-v1:setting:<id>'),16) even else setting2",
        },
        "n_candidates": len(cands),
        "excluded": {str(k): v for k, v in sorted(excluded.items())},
        "safety_replacements": replacements,
        "safety_regex": SAFETY_RE.pattern,
        "scenarios": scen,
        "request_id_digest": sha256_bytes(canon([s["request_id"] for s in scen]).encode()),
    }


def ints(s):  # request text: strip only; wording is otherwise untouched
    return s.strip()


def cmd_sample(a):
    if os.path.exists(SAMPLING):
        die(f"{SAMPLING} exists; refusing to overwrite a frozen sampling manifest")
    os.makedirs(DATA_DIR, exist_ok=True)
    out = derive_sampling(fetch_source(a.src_dir))
    with open(SAMPLING, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    p = sum(s["split"] == "pilot" for s in out["scenarios"])
    print(f"wrote {SAMPLING}: pilot={p} formal={len(out['scenarios']) - p} "
          f"candidates={out['n_candidates']} excluded={len(out['excluded'])} "
          f"digest={out['request_id_digest'][:16]}")


def validate_items(sc: dict, it: dict) -> list:
    """Return list of problems for one scenario (empty = ok)."""
    errs = []
    sup = (it.get("supplement") or "").strip()
    if not sup:
        errs.append("empty supplement")
    for pat in LEAK_PATTERNS:
        if re.search(pat, sup, re.I):
            errs.append(f"supplement contains instruction-like wording /{pat}/")
    if it.get("review_status") not in REVIEW_STATES:
        errs.append(f"bad review_status {it.get('review_status')!r}")
    items = it.get("items") or []
    targets = [x for x in items if x.get("role", "target") == "target"]
    if not 2 <= len(targets) <= 5:
        errs.append(f"{len(targets)} target items (need 2..5)")
    ids = [x.get("item_id") for x in items]
    if len(set(ids)) != len(ids) or any(not i for i in ids):
        errs.append("item_id missing/duplicated")
    need = {"item_id", "information_need", "fact", "why_relevant",
            "provided_in_incomplete", "provided_in_complete", "acceptable_question_examples"}
    inc = sc["incomplete_text"]
    for x in items:
        miss = need - set(x)
        if miss:
            errs.append(f"{x.get('item_id')}: missing fields {sorted(miss)}")
            continue
        if x.get("role", "target") == "target":
            if x["provided_in_incomplete"] or not x["provided_in_complete"]:
                errs.append(f"{x['item_id']}: target must be absent in Incomplete, present in Complete")
            span = x.get("complete_span", "")
            if not span or span not in sup:
                errs.append(f"{x['item_id']}: complete_span not a substring of supplement")
            if span and span in inc:
                errs.append(f"{x['item_id']}: complete_span already in Incomplete")
        else:  # auxiliary: given in the original request
            if not x["provided_in_incomplete"]:
                errs.append(f"{x['item_id']}: auxiliary item must be provided in Incomplete")
        if not x["acceptable_question_examples"]:
            errs.append(f"{x['item_id']}: no acceptable_question_examples")
    return errs


def cmd_items(a):
    """items_src_part*.txt (pipe-delimited, authored) -> scenario_items_v1.json.
    supplement = the spans joined in order; fact = the span itself."""
    import glob
    if os.path.exists(ITEMS) and not a.rebuild_draft:
        die(f"{ITEMS} exists; pass --rebuild_draft to regenerate (reviewed status would be reset)")
    if os.path.exists(FREEZE):
        die("a freeze manifest exists; scenarios are frozen")
    by = {}
    for fp in sorted(glob.glob(os.path.join(DATA_DIR, "items_src_part*.txt"))):
        for ln, line in enumerate(open(fp, encoding="utf-8"), 1):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            f = line.split("|")
            if len(f) != 5:
                die(f"{fp}:{ln}: expected 5 pipe-separated fields, got {len(f)}")
            sid, need, span, why, ex = [x.strip() for x in f]
            by.setdefault(sid, []).append((need, span, why, ex))
    out = {}
    for sid, rows in by.items():
        items = [{"item_id": f"{sid}_i{k}", "role": "target", "information_need": need, "fact": span,
                  "complete_span": span, "why_relevant": why, "provided_in_incomplete": False,
                  "provided_in_complete": True, "acceptable_question_examples": [ex]}
                 for k, (need, span, why, ex) in enumerate(rows, 1)]
        out[sid] = {"supplement": " ".join(r[1] for r in rows), "items": items,
                    "review_status": "draft_unreviewed"}
    with open(ITEMS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {ITEMS}: {len(out)} scenarios, {sum(len(v['items']) for v in out.values())} items")


def cmd_build(a):
    if not os.path.isfile(SAMPLING) or not os.path.isfile(ITEMS):
        die("need sampling_v1.json and scenario_items_v1.json")
    samp = json.load(open(SAMPLING, encoding="utf-8"))
    items = json.load(open(ITEMS, encoding="utf-8"))
    if os.path.exists(SCENARIOS) and not a.rebuild_draft:
        die(f"{SCENARIOS} exists; pass --rebuild_draft to regenerate (refused if a freeze exists)")
    if os.path.exists(FREEZE):
        die("a freeze manifest exists; scenarios are frozen and cannot be rebuilt")
    problems, out = [], []
    ids = {s["scenario_id"] for s in samp["scenarios"]}
    extra = set(items) - ids
    if extra:
        problems.append(f"items for unknown scenarios: {sorted(extra)[:5]}")
    for sc in samp["scenarios"]:
        it = items.get(sc["scenario_id"])
        if it is None:
            problems.append(f"{sc['scenario_id']}: no authored items")
            continue
        errs = validate_items(sc, it)
        problems += [f"{sc['scenario_id']}: {e}" for e in errs]
        sup = it["supplement"].strip()
        out.append({
            **{k: sc[k] for k in ("scenario_id", "split", "request_id", "setting_key",
                                  "incomplete_text", "source")},
            "complete_text": f"{sc['incomplete_text']} {sup}",
            "supplement": sup,
            "added_words": len(sup.split()),
            "added_chars": len(sup),
            "items": it["items"],
            "target_item_ids": [x["item_id"] for x in it["items"] if x.get("role", "target") == "target"],
            "review_status": it["review_status"],
        })
    if problems:
        print("\n".join(problems[:60]))
        die(f"{len(problems)} validation problem(s); nothing written")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SCENARIOS, "w", encoding="utf-8") as f:
        json.dump({"protocol": "infoquest-v1", "sampling_digest": samp["request_id_digest"],
                   "scenarios": out}, f, ensure_ascii=False, indent=1)
    rv = sum(s["review_status"] == "reviewed" for s in out)
    print(f"wrote {SCENARIOS}: {len(out)} scenarios, reviewed={rv}, draft={len(out) - rv}")


def cmd_freeze(a):
    if os.path.exists(FREEZE):
        die(f"{FREEZE} exists; refusing to overwrite")
    d = json.load(open(SCENARIOS, encoding="utf-8"))
    bad = [s["scenario_id"] for s in d["scenarios"] if s["review_status"] != "reviewed"]
    if bad:
        die(f"{len(bad)} scenario(s) not reviewed (e.g. {bad[:5]}); cannot freeze")
    man = {"protocol": "infoquest-v1", "scenarios_sha256": sha256_file(SCENARIOS),
           "sampling_sha256": sha256_file(SAMPLING), "n": len(d["scenarios"])}
    with open(FREEZE, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    print(f"wrote {FREEZE}: scenarios_sha256={man['scenarios_sha256'][:16]}")


def cmd_check(a):
    samp = json.load(open(SAMPLING, encoding="utf-8"))
    fresh = derive_sampling(fetch_source(a.src_dir))
    if canon(fresh) != canon(samp):
        die("stored sampling_v1.json differs from a fresh derivation")
    pil = {s["request_id"] for s in samp["scenarios"] if s["split"] == "pilot"}
    frm = {s["request_id"] for s in samp["scenarios"] if s["split"] == "formal"}
    assert len(pil) == N_PILOT and len(frm) == N_FORMAL and not (pil & frm)
    print(f"[ok] sampling reproducible; pilot/formal disjoint ({len(pil)}/{len(frm)}); "
          f"digest {samp['request_id_digest'][:16]}")
    if os.path.isfile(SCENARIOS):
        d = json.load(open(SCENARIOS, encoding="utf-8"))
        n = 0
        for s in d["scenarios"]:
            errs = validate_items(s, {"supplement": s["supplement"], "items": s["items"],
                                      "review_status": s["review_status"]})
            if errs or s["complete_text"] != f"{s['incomplete_text']} {s['supplement']}":
                die(f"{s['scenario_id']}: {errs or 'complete_text inconsistent'}")
            n += 1
        print(f"[ok] {n} scenarios valid; reviewed={sum(s['review_status']=='reviewed' for s in d['scenarios'])}")
    if os.path.isfile(FREEZE):
        m = json.load(open(FREEZE))
        assert m["scenarios_sha256"] == sha256_file(SCENARIOS), "freeze/scenarios mismatch"
        print("[ok] freeze manifest matches scenarios_v1.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, fn in (("sample", cmd_sample), ("items", cmd_items), ("build", cmd_build), ("freeze", cmd_freeze), ("check", cmd_check)):
        p = sub.add_parser(n)
        p.add_argument("--src_dir", default=None, help="directory holding the source jsonl (offline)")
        if n in ("build", "items"):
            p.add_argument("--rebuild_draft", action="store_true")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
