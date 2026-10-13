#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_infoquest_v1.py -- data/scenario preparation for `infoquest-v1`
(first-turn information-gap selectivity; see CLAUDE.md "InfoQuest" entry).

Server workflow (everything under /data1/paveen/Dopamine, no local preparation):
  python prepare_infoquest_v1.py prepare     # download pinned source, sample, build drafts, validate, report
  python prepare_infoquest_v1.py migrate     # ONLY on a server that already holds the pilot-time (r1) work dir/benchmark:
                                             # backs everything up, keeps the old benchmark, moves to the r2 materials
  (review/edit components/benchmark/infoquest_v1_work/scenario_items_v1.json)
  python prepare_infoquest_v1.py review --confirm-reviewed --reviewer NAME   # explicit; never automatic
  python prepare_infoquest_v1.py export      # rebuild + validate + write components/benchmark/infoquest_v1.json

The only shipped inputs are infoquest_v1/items_src_r2_part*.txt + r2_review.json (authored by us; the public data
supplies constraints/checklist, not our facts or scoring items). infoquest_v1/r1/ keeps the pilot-time materials
(used only by `migrate` to detect hand edits). Revision infoquest-v1-r2 = post-pilot, pre-formal material revision.

Subcommands (all offline once the source files are cached):
  sample  Download seed_messages.jsonl + settings.jsonl at the PINNED revision,
          apply the structural filter, draw 130 distinct requests with
          random.Random(42), fix 30 as pilot / 100 as formal, pick one hidden
          setting per request by a hash rule. Writes infoquest_v1/sampling_v1.json.
          Refuses to overwrite.
  items   Parse the authored items_src_part*.txt into scenario_items_v1.json.
  build   Merge sampling_v1.json with the AUTHORED items file
          infoquest_v1/scenario_items_v1.json -> infoquest_v1/scenarios_v1.json
          (Complete = Incomplete + " " + supplement). Validates every scenario.
  freeze  Validate everything; only if EVERY scenario is "reviewed" write the single runtime
          file components/benchmark/infoquest_v1.json (scenarios + sampling provenance).
          Refuses to overwrite; writes nothing on any failure.
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
BENCH_REVISION = "infoquest-v1-r2"      # material revision (post-pilot, pre-formal); see infoquest_v1/r2_review.json
CATEGORIES = ("task_clarification", "constraint")      # scored target categories
BACKGROUND = "background"                               # in the Complete text, never scored
KNOWN_FLAGS = ("object_unspecified_in_incomplete", "object_unresolved_in_complete",
               "object_weakly_identified", "request_infeasible")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "infoquest_v1")        # shipped build resources (authored items_src_part*.txt)
WORK_DIR = os.path.join(HERE, "components", "benchmark", "infoquest_v1_work")   # derived/edited files
SAMPLING = os.path.join(WORK_DIR, "sampling_v1.json")
ITEMS = os.path.join(WORK_DIR, "scenario_items_v1.json")      # the file a reviewer edits
SCENARIOS = os.path.join(WORK_DIR, "scenarios_v1.json")
STATE = os.path.join(WORK_DIR, "prepare_state.json")
BENCH = os.path.join(HERE, "components", "benchmark", "infoquest_v1.json")  # the ONLY runtime file

# Instruction-like wording that must never appear in a Complete supplement
# (the Complete condition must differ from Incomplete ONLY by facts).
LEAK_PATTERNS = [
    r"\bno need to ask\b", r"\bdon'?t ask\b", r"\bdo not ask\b", r"\bwithout asking\b",
    r"\bjust (answer|give|tell)\b", r"\bdirectly\b", r"\balready (told|gave|provided|know)\b",
    r"\benough (information|detail|context)\b", r"\bsufficient\b", r"\bfollow-?up\b",
    r"\bclarif", r"\bchecklist\b", r"\bplease (answer|respond)\b",
]
REVIEW_STATES = ("draft_unreviewed", "reviewed")
R2_REVIEW = os.path.join(DATA_DIR, "r2_review.json")   # shipped: per-scenario flags, edit basis, predecessor hash
R1_DIR = os.path.join(DATA_DIR, "r1")                  # shipped: the original (pilot-time) materials, for migration
BACKUP_DIR = os.path.join(HERE, "components", "benchmark", "infoquest_v1_r1_backup")

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
    os.makedirs(WORK_DIR, exist_ok=True)
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
    for fl in it.get("flags") or []:
        if fl not in KNOWN_FLAGS:
            errs.append(f"unknown flag {fl!r}")
    items = it.get("items") or []
    targets = [x for x in items if x.get("role", "target") == "target"]
    if not 1 <= len(targets) <= 5:
        errs.append(f"{len(targets)} target items (need 1..5; a scenario with none is a material problem)")
    ids = [x.get("item_id") for x in items]
    if len(set(ids)) != len(ids) or any(not i for i in ids):
        errs.append("item_id missing/duplicated")
    need = {"item_id", "information_need", "fact", "why_relevant", "category",
            "provided_in_incomplete", "provided_in_complete", "acceptable_question_examples"}
    inc = sc["incomplete_text"]
    for x in items:
        miss = need - set(x)
        if miss:
            errs.append(f"{x.get('item_id')}: missing fields {sorted(miss)}")
            continue
        role = x.get("role", "target")
        if role == "target" and x["category"] not in CATEGORIES:
            errs.append(f"{x['item_id']}: target category {x['category']!r} not in {CATEGORIES}")
        if role == "background" and x["category"] != BACKGROUND:
            errs.append(f"{x['item_id']}: background item must have category 'background'")
        if role in ("target", "background"):
            if x["provided_in_incomplete"] or not x["provided_in_complete"]:
                errs.append(f"{x['item_id']}: must be absent in Incomplete, present in Complete")
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


def _read_rows(files, nfields):
    by = {}
    for fp in files:
        for ln, line in enumerate(open(fp, encoding="utf-8"), 1):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            f = line.split("|")
            if len(f) != nfields:
                die(f"{fp}:{ln}: expected {nfields} pipe-separated fields, got {len(f)}")
            by.setdefault(f[0].strip(), []).append([x.strip() for x in f[1:]])
    return by


def derive_items_r2():
    """items_src_r2_part*.txt (authored; id|category|need|span|why|example) + r2_review.json -> items dict.
    supplement = ALL spans joined in order (background rows included); only non-background rows are targets."""
    import glob
    by = _read_rows(sorted(glob.glob(os.path.join(DATA_DIR, "items_src_r2_part*.txt"))), 6)
    rv = json.load(open(R2_REVIEW, encoding="utf-8"))
    out = {}
    for sid, rows in by.items():
        items = []
        for k, (cat, need, span, why, ex) in enumerate(rows, 1):
            if cat not in CATEGORIES + (BACKGROUND,):
                die(f"{sid}: bad category {cat!r}")
            items.append({"item_id": f"{sid}_i{k}", "role": "background" if cat == BACKGROUND else "target",
                          "category": cat, "information_need": need, "fact": span, "complete_span": span,
                          "why_relevant": why, "provided_in_incomplete": False, "provided_in_complete": True,
                          "acceptable_question_examples": [ex]})
        out[sid] = {"supplement": " ".join(r[2] for r in rows), "items": items,
                    "flags": sorted(rv["flags"].get(sid, [])), "revision_note": rv["notes"].get(sid),
                    "review_status": "draft_unreviewed"}
    return out


def derive_items_r1():
    """The ORIGINAL pilot-time materials (infoquest_v1/r1/), in the r1 items-file shape; used only by `migrate`."""
    import glob
    by = _read_rows(sorted(glob.glob(os.path.join(R1_DIR, "items_src_part*.txt"))), 5)
    out = {}
    for sid, rows in by.items():
        items = [{"item_id": f"{sid}_i{k}", "role": "target", "information_need": need, "fact": span,
                  "complete_span": span, "why_relevant": why, "provided_in_incomplete": False,
                  "provided_in_complete": True, "acceptable_question_examples": [ex]}
                 for k, (need, span, why, ex) in enumerate(rows, 1)]
        out[sid] = {"supplement": " ".join(r[1] for r in rows), "items": items, "review_status": "draft_unreviewed"}
    return out


def cmd_items(a):
    if os.path.exists(ITEMS) and not a.rebuild_draft:
        die(f"{ITEMS} exists; pass --rebuild_draft to regenerate (reviewed status would be reset)")
    out = derive_items_r2()
    with open(ITEMS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {ITEMS}: {len(out)} scenarios, {sum(len(v['items']) for v in out.values())} items")


def cmd_build(a):
    if not os.path.isfile(SAMPLING) or not os.path.isfile(ITEMS):
        die("need sampling_v1.json and scenario_items_v1.json")
    samp = json.load(open(SAMPLING, encoding="utf-8"))
    items = json.load(open(ITEMS, encoding="utf-8"))
    if os.path.exists(SCENARIOS) and not a.rebuild_draft:
        die(f"{SCENARIOS} exists; pass --rebuild_draft to regenerate ")
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
            "flags": it.get("flags") or [],
            "revision_note": it.get("revision_note"),
            "review_status": it["review_status"],
        })
    if problems:
        print("\n".join(problems[:60]))
        die(f"{len(problems)} validation problem(s); nothing written")
    os.makedirs(WORK_DIR, exist_ok=True)
    with open(SCENARIOS, "w", encoding="utf-8") as f:
        json.dump({"protocol": "infoquest-v1", "revision": BENCH_REVISION, "sampling_digest": samp["request_id_digest"],
                   "scenarios": out}, f, ensure_ascii=False, indent=1)
    rv = sum(s["review_status"] == "reviewed" for s in out)
    print(f"wrote {SCENARIOS}: {len(out)} scenarios, reviewed={rv}, draft={len(out) - rv}")


def cmd_freeze(a):
    if os.path.exists(BENCH):
        die(f"{BENCH} exists; refusing to overwrite (move it away deliberately to re-export)")
    d = json.load(open(SCENARIOS, encoding="utf-8"))
    samp = json.load(open(SAMPLING, encoding="utf-8"))
    if d.get("sampling_digest") != samp["request_id_digest"]:
        die("scenarios_v1.json was built from a different sampling manifest")
    if d.get("revision") != BENCH_REVISION:
        die(f"scenarios_v1.json is revision {d.get('revision')!r}, expected {BENCH_REVISION!r} (run `migrate` on an old work dir)")
    if [s["scenario_id"] for s in d["scenarios"]] != [s["scenario_id"] for s in samp["scenarios"]]:
        die("scenario ids/order differ from sampling_v1.json")
    probs = []
    for sc, ref in zip(d["scenarios"], samp["scenarios"]):
        for k in ("split", "request_id", "setting_key", "incomplete_text", "source"):
            if sc[k] != ref[k]:
                probs.append(f"{sc['scenario_id']}: {k} differs from sampling")
        if sc["complete_text"] != f"{sc['incomplete_text']} {sc['supplement']}":
            probs.append(f"{sc['scenario_id']}: complete_text != incomplete + supplement")
        if "flags" not in sc:
            probs.append(f"{sc['scenario_id']}: no flags field (stale r1 build?)")
        tids = [x["item_id"] for x in sc["items"] if x.get("role", "target") == "target"]
        if sc.get("target_item_ids") != tids:
            probs.append(f"{sc['scenario_id']}: target_item_ids {sc.get('target_item_ids')} != target items {tids}")
        probs += [f"{sc['scenario_id']}: {e}" for e in validate_items(
            sc, {"supplement": sc["supplement"], "items": sc["items"], "review_status": sc["review_status"],
                 "flags": sc.get("flags")})]
    if probs:
        print("\n".join(probs[:40]))
        die(f"{len(probs)} validation problem(s); cannot freeze")
    bad = [s["scenario_id"] for s in d["scenarios"] if s["review_status"] != "reviewed"]
    if bad:
        die(f"{len(bad)} scenario(s) not reviewed (e.g. {bad[:5]}); cannot freeze")
    meta = {k: v for k, v in samp.items() if k != "scenarios"}
    meta["selected_request_ids_in_order"] = [x["request_id"] for x in samp["scenarios"]]
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    review = {k: st.get(k) for k in ("reviewed_by", "reviewer_kind", "reviewed_at", "reviewed_content_sha256")}
    rv = json.load(open(R2_REVIEW, encoding="utf-8"))
    material = {k: rv[k] for k in ("revision", "timing", "predecessor_benchmark_sha256", "agent_review_note")}
    out = {"protocol": "infoquest-v1", "revision": BENCH_REVISION, "status": "reviewed_export", "review": review,
           "material_revision": material, "n_pilot": N_PILOT, "n_formal": N_FORMAL,
           "sampling": meta, "scenarios": d["scenarios"]}
    os.makedirs(os.path.dirname(BENCH), exist_ok=True)
    with open(BENCH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {BENCH}: {len(d['scenarios'])} reviewed scenarios, file sha256={sha256_file(BENCH)[:16]}")


def resources():
    import glob
    fps = sorted(glob.glob(os.path.join(DATA_DIR, "items_src_r2_part*.txt"))) + [R2_REVIEW]
    if len(fps) < 2 or not all(os.path.isfile(f) for f in fps):
        die(f"build resources missing: need {DATA_DIR}/items_src_r2_part*.txt and r2_review.json (sync the code AND infoquest_v1/ resources)")
    return sha256_bytes(b"".join(open(f, "rb").read() for f in fps))


def cmd_prepare(a):
    rsha = resources()
    os.makedirs(WORK_DIR, exist_ok=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else None
    if os.path.exists(SAMPLING):
        fresh = derive_sampling(fetch_source(a.src_dir))
        if canon(fresh) != canon(json.load(open(SAMPLING, encoding="utf-8"))):
            die(f"{SAMPLING} conflicts with a fresh derivation from the pinned source; not overwriting")
        print("[reuse] sampling_v1.json matches the pinned source")
    else:
        cmd_sample(a)
    if os.path.exists(ITEMS):
        if state is None:
            die(f"{ITEMS} exists but {STATE} is missing; cannot tell which resources it came from")
        if state.get("revision") != BENCH_REVISION:
            die(f"{WORK_DIR} holds pilot-time (r1) materials; run `python prepare_infoquest_v1.py migrate` "
                "(it backs everything up first and keeps the old benchmark)")
        if state["resources_sha256"] != rsha:
            die("shipped build resources changed since this draft was created. Your edits in "
                f"{ITEMS} are kept untouched; move that file aside deliberately to re-derive it")
        print("[reuse] scenario_items_v1.json kept (manual edits are never overwritten)")
    else:
        cmd_items(argparse.Namespace(rebuild_draft=False))
        state = {"resources_sha256": rsha, "revision": BENCH_REVISION,
                 "sampling_digest": json.load(open(SAMPLING))["request_id_digest"]}
        json.dump(state, open(STATE, "w"), indent=1)
    cmd_build(argparse.Namespace(rebuild_draft=True))
    d = json.load(open(SCENARIOS, encoding="utf-8"))["scenarios"]
    n_rev = sum(s["review_status"] == "reviewed" for s in d)
    print(f"\n[status] {len(d)} scenarios, reviewed={n_rev}, unreviewed={len(d) - n_rev}")
    print(f"[drafts] edit: {ITEMS}\n         built: {SCENARIOS}")
    if n_rev < len(d):
        print("[next]  review/edit the items file, then:\n"
              "        python prepare_infoquest_v1.py review --confirm-reviewed --reviewer <name>\n"
              "        python prepare_infoquest_v1.py export\n"
              "        (nothing is marked reviewed automatically; pilot cannot start before export)")
    else:
        print("[next]  python prepare_infoquest_v1.py export")


def cmd_migrate(a):
    """Move an existing pilot-time (r1) work dir + benchmark to the r2 materials WITHOUT losing anything.
    Everything is checked before anything is touched; old files are copied to BACKUP_DIR first."""
    import datetime
    import shutil
    if not os.path.exists(ITEMS) or not os.path.exists(STATE):
        die(f"no old work dir at {WORK_DIR}; use `prepare` on a fresh server instead")
    state = json.load(open(STATE))
    if state.get("revision") == BENCH_REVISION:
        die("work dir is already revision " + BENCH_REVISION)
    if os.path.exists(BACKUP_DIR):
        die(f"{BACKUP_DIR} exists; refusing to overwrite an earlier backup")
    if os.path.abspath(BACKUP_DIR).startswith(os.path.abspath(WORK_DIR) + os.sep):
        die("backup directory must not be inside the work directory")
    rv = json.load(open(R2_REVIEW, encoding="utf-8"))
    old_bench_sha = sha256_file(BENCH) if os.path.exists(BENCH) else None
    if old_bench_sha is None:
        print("[warn] no components/benchmark/infoquest_v1.json found: cannot verify it against the pilot cells' benchmark_sha256")
    elif old_bench_sha != rv["predecessor_benchmark_sha256"]:
        die(f"existing benchmark sha256 {old_bench_sha} != the one the pilot cells record "
            f"({rv['predecessor_benchmark_sha256']}); not touching anything. Find the file the pilot actually ran with.")
    fresh = derive_sampling(fetch_source(a.src_dir))
    if canon(fresh) != canon(json.load(open(SAMPLING, encoding="utf-8"))):
        die(f"{SAMPLING} does not match a fresh derivation from the pinned source; not migrating")
    old = json.load(open(ITEMS, encoding="utf-8"))
    r1, r2 = derive_items_r1(), derive_items_r2()
    strip = lambda v: canon({k: x for k, x in v.items() if k != "review_status"})
    edited = sorted(sid for sid in r1 if sid in old and strip(old[sid]) != strip(r1[sid]))
    missing = sorted(set(r2) ^ set(old))
    if missing:
        die(f"scenario ids differ between the old work file and the r2 materials: {missing[:6]}")
    if edited and a.on_conflict != "r2":
        rep = os.path.join(WORK_DIR, "migration_conflicts.json")
        json.dump({sid: {"server_version": old[sid], "r2_version": r2[sid]} for sid in edited},
                  open(rep, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        die(f"{len(edited)} scenario(s) were edited by hand on the server (e.g. {edited[:5]}). Both versions are in {rep}. "
            "Re-run with `--on-conflict r2` to take the r2 material (the hand-edited version stays in the backup), "
            "after porting by hand any edit you want to keep into infoquest_v1/items_src_r2_part*.txt")
    os.makedirs(BACKUP_DIR)
    shutil.copytree(WORK_DIR, os.path.join(BACKUP_DIR, "work"))
    if old_bench_sha:
        shutil.copy2(BENCH, os.path.join(BACKUP_DIR, "infoquest_v1.json"))
        if sha256_file(os.path.join(BACKUP_DIR, "infoquest_v1.json")) != old_bench_sha:
            die("backup copy of the benchmark does not match; aborting before any change")
    json.dump({"migrated_at": datetime.datetime.now().isoformat(timespec="seconds"), "old_benchmark_sha256": old_bench_sha,
               "old_state": state, "hand_edited_scenarios": edited, "on_conflict": a.on_conflict,
               "old_items_sha256": sha256_file(ITEMS)}, open(os.path.join(BACKUP_DIR, "migration_report.json"), "w"), indent=1)
    if old_bench_sha:
        os.remove(BENCH)          # the backup holds it; export refuses to overwrite, so it must be gone
    with open(ITEMS, "w", encoding="utf-8") as f:
        json.dump(r2, f, ensure_ascii=False, indent=1)
    new_state = {"resources_sha256": resources(), "revision": BENCH_REVISION, "sampling_digest": fresh["request_id_digest"],
                 "migrated_from": {"old_benchmark_sha256": old_bench_sha, "old_reviewed_by": state.get("reviewed_by"),
                                   "old_reviewed_at": state.get("reviewed_at"), "backup": BACKUP_DIR}}
    json.dump(new_state, open(STATE, "w"), indent=1)
    cmd_build(argparse.Namespace(rebuild_draft=True))
    print(f"[migrated] old benchmark/work saved under {BACKUP_DIR}; the old sign-off is void (materials changed).\n"
          "[next]  review the r2 materials, then: review --confirm-reviewed --reviewer <name>; export")


def items_content_sha():
    """Hash of everything a reviewer can change, EXCLUDING the review_status flag itself."""
    items = json.load(open(ITEMS, encoding="utf-8"))
    return sha256_bytes(canon({k: {kk: vv for kk, vv in v.items() if kk != "review_status"}
                               for k, v in items.items()}).encode())


def cmd_review(a):
    if not a.confirm_reviewed:
        die("refusing: pass --confirm-reviewed (and --reviewer) only after you have read the scenarios")
    items = json.load(open(ITEMS, encoding="utf-8"))
    for v in items.values():
        v["review_status"] = "reviewed"
    json.dump(items, open(ITEMS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    import datetime
    st = json.load(open(STATE)); st["reviewed_by"] = a.reviewer
    st["reviewer_kind"] = a.reviewer_kind     # "human" | "automated": recorded as given, never inferred
    st["reviewed_content_sha256"] = items_content_sha()
    st["reviewed_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    json.dump(st, open(STATE, "w"), indent=1)
    print(f"marked {len(items)} scenarios reviewed by {a.reviewer} ({a.reviewer_kind}{', NOT human-verified' if a.reviewer_kind != 'human' else ''})")


def cmd_export(a):
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    if not st.get("reviewed_by") or st.get("reviewed_content_sha256") != items_content_sha():
        die("items were not signed off, or were modified after the sign-off; re-run "
            "`review --confirm-reviewed --reviewer NAME` after finishing your edits")
    fresh = derive_sampling(fetch_source(a.src_dir))
    if canon(fresh) != canon(json.load(open(SAMPLING, encoding="utf-8"))):
        die(f"{SAMPLING} does not match a fresh derivation from the pinned source; not exporting")
    cmd_build(argparse.Namespace(rebuild_draft=True))     # derived file is always re-derived from the edited items
    cmd_freeze(a)


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
                                      "review_status": s["review_status"], "flags": s.get("flags")})
            if errs or s["complete_text"] != f"{s['incomplete_text']} {s['supplement']}":
                die(f"{s['scenario_id']}: {errs or 'complete_text inconsistent'}")
            n += 1
        print(f"[ok] {n} scenarios valid; reviewed={sum(s['review_status']=='reviewed' for s in d['scenarios'])}")
    if os.path.isfile(BENCH):
        m = json.load(open(BENCH, encoding="utf-8"))
        assert m["protocol"] == "infoquest-v1" and m["status"] == "reviewed_export"
        assert m["sampling"]["selected_request_ids_in_order"] == [x["request_id"] for x in samp["scenarios"]], \
            "benchmark selection differs from sampling"
        assert m["scenarios"] == json.load(open(SCENARIOS, encoding="utf-8"))["scenarios"], "benchmark != scenarios_v1.json"
        print(f"[ok] runtime benchmark matches local scenarios (sha256 {sha256_file(BENCH)[:16]})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, fn in (("prepare", cmd_prepare), ("migrate", cmd_migrate), ("review", cmd_review), ("export", cmd_export), ("sample", cmd_sample), ("items", cmd_items), ("build", cmd_build), ("freeze", cmd_freeze), ("check", cmd_check)):
        p = sub.add_parser(n)
        p.add_argument("--src_dir", default=None, help="directory holding the source jsonl (offline)")
        if n == "review":
            p.add_argument("--confirm-reviewed", dest="confirm_reviewed", action="store_true")
            p.add_argument("--reviewer", required=True, help="name, or the actual model id for an automated review")
            p.add_argument("--reviewer-kind", dest="reviewer_kind", choices=("human", "automated"), default="human")
        if n == "migrate":
            p.add_argument("--on-conflict", dest="on_conflict", choices=("abort", "r2"), default="abort")
        if n in ("build", "items"):
            p.add_argument("--rebuild_draft", action="store_true")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
