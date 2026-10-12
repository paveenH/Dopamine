#!/usr/bin/env python3.10
"""Local CPU checks for infoquest-v1 (no GPU, no real model, no network needed once the
source jsonl is in the HF cache). Exit code != 0 on any failure."""
import copy
import hashlib
import json
import os
import sys
import tempfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prepare_infoquest_v1 as P
import get_answer_infoquest_v1 as G

N_OK = 0


def ok(cond, msg):
    global N_OK
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    N_OK += 1
    print(f"ok   {msg}")


def raises_exit(fn, msg):
    try:
        fn()
    except SystemExit as e:
        ok(e.code == 2, msg)
        return
    ok(False, msg + " (no SystemExit)")


# ----------------------------------------------------------------- prepare (server workflow, temp WORK dir)
import argparse
TW = tempfile.mkdtemp()
P.WORK_DIR = TW
P.SAMPLING, P.ITEMS, P.SCENARIOS, P.STATE = (os.path.join(TW, f) for f in (
    "sampling_v1.json", "scenario_items_v1.json", "scenarios_v1.json", "prepare_state.json"))
P.BENCH = os.path.join(TW, "bench", "infoquest_v1.json")
NSP = argparse.Namespace(src_dir=None)
real_data_dir = P.DATA_DIR
P.DATA_DIR = os.path.join(TW, "no_resources")
raises_exit(lambda: P.cmd_prepare(NSP), "missing shipped build resources -> clear error")
P.DATA_DIR = real_data_dir
P.cmd_prepare(NSP)
ok(all(os.path.exists(f) for f in (P.SAMPLING, P.ITEMS, P.SCENARIOS, P.STATE)), "prepare writes sampling/items/scenarios/state in the work dir")
ok(not os.path.exists(P.BENCH), "prepare does not export a benchmark")
ok(all(v["review_status"] == "draft_unreviewed" for v in json.load(open(P.ITEMS)).values()), "prepare leaves everything unreviewed")
raises_exit(lambda: P.cmd_export(NSP), "export refuses unreviewed material")
ok(not os.path.exists(P.BENCH), "no benchmark after refused export")
# manual edits survive a repeated prepare
it = json.load(open(P.ITEMS)); it["iq329"]["items"][0]["information_need"] = "EDITED BY REVIEWER"
json.dump(it, open(P.ITEMS, "w"), ensure_ascii=False)
P.cmd_prepare(NSP)
ok(json.load(open(P.ITEMS))["iq329"]["items"][0]["information_need"] == "EDITED BY REVIEWER", "repeat prepare keeps manual edits")
ok(next(x for x in json.load(open(P.SCENARIOS))["scenarios"] if x["scenario_id"] == "iq329")["items"][0]["information_need"]
   == "EDITED BY REVIEWER", "built scenarios pick the edit up")
sam0 = open(P.SAMPLING).read()
sd = json.load(open(P.SAMPLING)); sd["rule"]["seed"] = 7
json.dump(sd, open(P.SAMPLING, "w"))
raises_exit(lambda: P.cmd_prepare(NSP), "sampling conflicting with the pinned source -> error")
open(P.SAMPLING, "w").write(sam0)
st = json.load(open(P.STATE)); st["resources_sha256"] = "x"
json.dump(st, open(P.STATE, "w"))
raises_exit(lambda: P.cmd_prepare(NSP), "changed shipped resources vs existing draft -> error, edits untouched")
st["resources_sha256"] = P.resources(); json.dump(st, open(P.STATE, "w"))
raises_exit(lambda: P.cmd_review(argparse.Namespace(confirm_reviewed=False, reviewer="x")), "review refuses without --confirm-reviewed")
P.cmd_review(argparse.Namespace(confirm_reviewed=True, reviewer="tester"))
ok(json.load(open(P.STATE))["reviewed_by"] == "tester", "explicit review recorded")
P.cmd_export(NSP)
ok(os.path.exists(P.BENCH) and json.load(open(P.BENCH))["status"] == "reviewed_export", "export writes the single benchmark file")
G.load_benchmark(P.BENCH, "pilot"); G.load_benchmark(P.BENCH, "formal")
ok(True, "exported benchmark is accepted by the generator loader")
os.remove(P.BENCH)
it = json.load(open(P.ITEMS))
for v in it.values():
    v["review_status"] = "draft_unreviewed"
json.dump(it, open(P.ITEMS, "w"), ensure_ascii=False)

# ----------------------------------------------------------------- sampling / materials
samp = json.load(open(P.SAMPLING, encoding="utf-8"))
sc = samp["scenarios"]
pil = [s for s in sc if s["split"] == "pilot"]
frm = [s for s in sc if s["split"] == "formal"]
ok(len(pil) == 30 and len(frm) == 100, "30 pilot / 100 formal")
ok(not ({s["request_id"] for s in pil} & {s["request_id"] for s in frm}), "pilot/formal request ids disjoint")
ok(len({P.norm_text(s["incomplete_text"]) for s in sc}) == 130, "130 distinct normalised requests")
ok(len({s["scenario_id"] for s in sc}) == 130, "scenario ids unique")
paths = P.fetch_source(None)
fresh = P.derive_sampling(paths)
ok(P.canon(fresh) == P.canon(samp), "sampling reproduces byte-for-byte from the pinned source")
ok(all(s["setting_key"] == P.pick_setting_key(s["request_id"]) for s in sc), "setting rule is the hash rule")
ok(all(not P.SAFETY_RE.search(s["incomplete_text"] + " " + s["source"]["description"]) for s in sc),
   "no sampled scenario matches the safety regex")

scen = json.load(open(P.SCENARIOS, encoding="utf-8"))["scenarios"]
by = {s["scenario_id"]: s for s in scen}
ok(len(scen) == 130, "130 scenarios built")
ok(all(s["complete_text"].startswith(s["incomplete_text"] + " ") for s in scen), "Complete = Incomplete + supplement")
ok(all(len(s["target_item_ids"]) >= 2 for s in scen), "each scenario has >=2 target items (K_i fixed)")
ok(all(not P.validate_items(s, {"supplement": s["supplement"], "items": s["items"],
                                "review_status": s["review_status"]}) for s in scen), "all scenarios validate")
bad = copy.deepcopy(scen[0])
bad["supplement"] += " No need to ask me anything."
ok(P.validate_items(bad, {"supplement": bad["supplement"], "items": bad["items"],
                          "review_status": "reviewed"}), "instruction-like wording in Complete is rejected")
bad = copy.deepcopy(scen[0])
bad["items"][0]["complete_span"] = "text that is not in the supplement"
ok(P.validate_items(bad, {"supplement": bad["supplement"], "items": bad["items"],
                          "review_status": "reviewed"}), "complete_span must be in the supplement")
bad = copy.deepcopy(scen[0])
bad["items"][0]["provided_in_incomplete"] = True
ok(P.validate_items(bad, {"supplement": bad["supplement"], "items": bad["items"],
                          "review_status": "reviewed"}), "target item present in Incomplete is rejected")
import re
ok(all(not re.search(r"\b(checklist|persona)\b", s["supplement"], re.I) for s in scen),
   "no persona/checklist wording copied into supplements")

# ------------------------------------------------------------------- benchmark loading
tmp = tempfile.mkdtemp()
rev = [dict(s, review_status="reviewed") for s in scen]
bp = os.path.join(tmp, "infoquest_v1.json")
good = {"protocol": "infoquest-v1", "status": "reviewed_export", "scenarios": rev}
json.dump(good, open(bp, "w"))
raises_exit(lambda: G.load_benchmark(os.path.join(tmp, "missing.json"), "pilot"), "missing benchmark -> refuse, exact path")
got, bsha = G.load_benchmark(bp, "pilot")
ok(len(got) == 30 and bsha == G.sha256_file(bp), "reviewed export loads (30 pilot), sha = whole-file sha256")
ok(len(G.load_benchmark(bp, "formal")[0]) == 100, "formal loads (100)")
for name, d in (("draft status", dict(good, status="draft")),
                ("wrong protocol", dict(good, protocol="other")),
                ("unreviewed scenario", dict(good, scenarios=[dict(rev[0], review_status="draft_unreviewed")] + rev[1:])),
                ("wrong pilot count", dict(good, scenarios=rev[1:]))):
    json.dump(d, open(bp, "w"))
    raises_exit(lambda: G.load_benchmark(bp, "pilot"), f"benchmark refused: {name}")
json.dump(good, open(bp, "w"))
fake_all = good
sp = bp


# ------------------------------------------------------------------- fake model
class FakeTok:
    padding_side = "left"
    bos_token = "<s>"
    bos_token_id = 1
    chat_template = "fake-template-v1"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True and len(msgs) == 1 and msgs[0]["role"] == "user"
        return "<s><|sys|>You are helpful.<|u|>" + msgs[0]["content"].strip() + "<|a|>\n\n"

    def __call__(self, text, add_special_tokens=True):
        ids = [1] if add_special_tokens else []
        if text.startswith("<s>"):
            ids.append(1)
            text = text[3:]
        ids += [100 + (ord(c) % 50) for c in text]
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(str(i) for i in ids)


class FakeModel:
    def parameters(self):
        return iter(())


class FakeVC:
    def __init__(self, n_dec=32):
        self.tokenizer = FakeTok()
        self.model = FakeModel()
        self.n_dec = n_dec
        self.terminators = [2]
        self._fires = 0
        self.calls = []

    def _find_decoder_layers(self):
        return list(range(self.n_dec))

    def steering_fire_count(self, reset=False):
        n = self._fires
        if reset:
            self._fires = 0
        return n

    def regenerate(self, inputs, max_new_tokens, temperature, top_p, diff_matrices, prefill_only,
                   prefill_tail_len, batch_size, return_metadata):
        self.calls.append({"prefill_only": prefill_only, "tail": prefill_tail_len, "temp": temperature,
                           "mnt": max_new_tokens, "n": len(inputs)})
        assert len(diff_matrices) == self.n_dec
        steered = sum(1 for r in diff_matrices if np.any(r != 0))
        self._fires += steered * len(inputs) * prefill_tail_len
        scale = float(sum(np.abs(r).sum() for r in diff_matrices))
        out = []
        for t in inputs:
            h = int(hashlib.sha256(t.encode()).hexdigest(), 16)
            out.append({"text": f"reply {h % 997} alpha-norm {round(scale, 1)}?", "generated_token_count": 7,
                        "stop_reason": "natural_eos" if h % 5 else "budget_exhausted"})
        return out


def make_mask(band=(10, 19), n=32, seed=0):
    m = np.zeros((n, 16))
    m[band[0]:band[1]] = np.random.default_rng(seed).normal(size=(band[1] - band[0], 16))
    return m


mask = make_mask()
out_dir = os.path.join(tmp, "out")
pilot = [s for s in fake_all["scenarios"] if s["split"] == "pilot"]
formal = [s for s in fake_all["scenarios"] if s["split"] == "formal"][:6]   # small formal for speed
dlr = lambda a, b: list(range(a - 1, b - 1))

vc = FakeVC()
w = G.run_cells(vc, "llama3", "pilot", pilot, "s" * 64, mask, out_dir, "m" * 64, "mask.npy", decoder_layer_range=dlr)
ok(len(w) == 1 and "_a0.json" in w[0], "pilot writes exactly one cell (alpha=0)")
c0 = json.load(open(w[0]))
ok(c0["meta"]["steering_fires"] == 0 and c0["meta"]["alpha"] == 0, "alpha=0 -> zero injections")
ok(len(c0["data"]) == 60 and c0["meta"]["n_rows"] == 60, "30 requests x 2 conditions = 60 replies")
ok(all(c["prefill_only"] is True and c["tail"] == 1 and c["temp"] == 0.0 and c["mnt"] == 512 for c in vc.calls),
   "regenerate always prefill_only, tail=1, greedy, 512 tokens")
ok(c0["meta"]["extra_anchor_added"] is False and c0["meta"]["chat_template_applied"], "no extra anchor, chat template applied")
ok(c0["data"][0]["messages"] == [{"role": "user", "content": pilot[0]["incomplete_text"]}], "single user message = Incomplete text")
ok(c0["data"][1]["messages"][0]["content"] == pilot[0]["complete_text"], "Complete row follows Incomplete row")

# formal: 3 alphas, same inputs, injection counts L*n
vc = FakeVC()
w = G.run_cells(vc, "llama3", "formal", formal, "s" * 64, mask, out_dir, "m" * 64, "mask.npy", decoder_layer_range=dlr)
ok(len(w) == 3, "formal writes three cells")
cells = {json.load(open(p))["meta"]["alpha"]: json.load(open(p)) for p in w}
ok(set(cells) == {-4, 0, 4}, "alpha set {-4,0,+4}")
n = 2 * len(formal)
ok(cells[0]["meta"]["steering_fires"] == 0 and cells[4]["meta"]["steering_fires"] == 9 * n
   and cells[-4]["meta"]["steering_fires"] == 9 * n, "fires = 0 at alpha=0, L*n (L=9) otherwise")
ok(len({c["meta"]["prompt_ids_sha256"] for c in cells.values()}) == 1, "tokenised inputs identical across alpha")
ok(all([r["prompt_sha256"] for r in cells[a]["data"]] == [r["prompt_sha256"] for r in cells[0]["data"]] for a in (-4, 4)),
   "per-row prompt hashes identical across alpha")
ok(cells[4]["data"][0]["generated"] != cells[0]["data"][0]["generated"], "steered output differs from alpha=0 (fake model sees the mask)")
ok(cells[0]["meta"]["injection_last_token_ids_all_prompts"] == [int(100 + (ord("\n") % 50))],
   "last prefill token is the native assistant header (no anchor added)")

# resume / conflicts
before = {p: (os.path.getmtime(p), G.sha256_file(p)) for p in w}
w2 = G.run_cells(FakeVC(), "llama3", "formal", formal, "s" * 64, mask, out_dir, "m" * 64, "mask.npy", decoder_layer_range=dlr)
ok(w2 == [] and all((os.path.getmtime(p), G.sha256_file(p)) == before[p] for p in w), "identical re-run resumes, rewrites nothing")
raises_exit(lambda: G.run_cells(FakeVC(), "llama3", "formal", formal, "x" * 64, mask, out_dir, "m" * 64, "mask.npy",
                                decoder_layer_range=dlr), "conflicting scenario source -> refuse")
raises_exit(lambda: G.run_cells(FakeVC(), "llama3", "formal", formal, "s" * 64, mask, out_dir, "z" * 64, "mask.npy",
                                decoder_layer_range=dlr), "conflicting mask hash -> refuse")
ok(all(G.sha256_file(p) == before[p][1] for p in w), "conflicting re-runs left existing cells untouched")
ok(not [f for r, _, fs in os.walk(out_dir) for f in fs if f.endswith(".tmp")], "no stray .tmp files")

raises_exit(lambda: G.run_cells(FakeVC(), "llama3", "formal", formal, "s" * 64, mask, out_dir, "m" * 64, "mask.npy",
                                decoder_layer_range=dlr, model_dir="other/weights"), "different actual model_dir -> refuse resume")
ok(cells[0]["meta"]["model_dir"] == "meta-llama/Llama-3.1-8B-Instruct", "actual model_dir recorded")

# freeze/export re-validation (monkeypatched paths, temp copies)
import shutil
fz = os.path.join(tmp, "fz"); os.makedirs(fz)
for fn in ("sampling_v1.json", "scenarios_v1.json"):
    shutil.copy(os.path.join(TW, fn), fz)
P.SAMPLING, P.SCENARIOS, P.BENCH = (os.path.join(fz, "sampling_v1.json"), os.path.join(fz, "scenarios_v1.json"),
                                   os.path.join(fz, "bench", "infoquest_v1.json"))
dd = json.load(open(P.SCENARIOS))
for x in dd["scenarios"]:
    x["review_status"] = "reviewed"
json.dump(dd, open(P.SCENARIOS, "w"))
P.cmd_freeze(None)
ex_ = json.load(open(P.BENCH))
ok(ex_["status"] == "reviewed_export" and len(ex_["scenarios"]) == 130, "valid reviewed material exports one benchmark file")
ok(ex_["sampling"]["seed"] == 42 if "seed" in ex_["sampling"] else ex_["sampling"]["rule"]["seed"] == 42, "sampling provenance kept")
ok(len(ex_["sampling"]["selected_request_ids_in_order"]) == 130 and "safety_replacements" in ex_["sampling"]
   and "source" in ex_["sampling"], "selection order, replacements and source hashes kept")
raises_exit(lambda: P.cmd_freeze(None), "existing benchmark file is not overwritten")
G.load_benchmark(P.BENCH, "pilot")
os.remove(P.BENCH)
for name, mut in (("edited Complete text", lambda d: d["scenarios"][3].__setitem__("complete_text", "tampered")),
                  ("changed split", lambda d: d["scenarios"][3].__setitem__("split", "pilot" if d["scenarios"][3]["split"] == "formal" else "formal")),
                  ("changed request text", lambda d: d["scenarios"][3].__setitem__("incomplete_text", "other request")),
                  ("target_item_ids out of sync with items", lambda d: d["scenarios"][3]["items"].append(
                      dict(d["scenarios"][3]["items"][0], item_id="extra_i9"))),
                  ("leak wording added", lambda d: d["scenarios"][3].__setitem__("supplement", d["scenarios"][3]["supplement"] + " No need to ask."))):
    d2 = copy.deepcopy(dd); mut(d2); json.dump(d2, open(P.SCENARIOS, "w"))
    raises_exit(lambda: P.cmd_freeze(None), f"export refuses: {name}")
    ok(not os.path.exists(P.BENCH), f"no benchmark written: {name}")
json.dump(dict(dd, scenarios=[dict(dd["scenarios"][0], review_status="draft_unreviewed")] + dd["scenarios"][1:]), open(P.SCENARIOS, "w"))
raises_exit(lambda: P.cmd_freeze(None), "export refuses unreviewed scenario")

# guards
raises_exit(lambda: G.run_cells(FakeVC(), "llama3", "pilot", pilot, "s" * 64, make_mask((11, 20)),
                                os.path.join(tmp, "o2"), "m" * 64, "mask.npy", decoder_layer_range=dlr),
            "mask band shifted by one layer -> refuse")
raises_exit(lambda: G.run_cells(FakeVC(n_dec=28), "llama3", "pilot", pilot, "s" * 64, mask,
                                os.path.join(tmp, "o3"), "m" * 64, "mask.npy", decoder_layer_range=dlr),
            "mask rows != decoder layers -> refuse")
orig = G.strip_leading_bos
G.strip_leading_bos = lambda tok, t: t
raises_exit(lambda: G.run_cells(FakeVC(), "llama3", "pilot", pilot, "s" * 64, mask,
                                os.path.join(tmp, "o4"), "m" * 64, "mask.npy", decoder_layer_range=dlr),
            "double BOS is detected")
G.strip_leading_bos = orig

# qwen config: own band
qm = make_mask((15, 21), n=28)
vq = FakeVC(n_dec=28)
wq = G.run_cells(vq, "qwen2.5", "formal", formal, "s" * 64, qm, os.path.join(tmp, "oq"), "q" * 64, "qmask.npy",
                 decoder_layer_range=dlr)
cq = json.load(open([p for p in wq if "_a4" in p][0]))
ok(cq["meta"]["L"] == 6 and cq["meta"]["steering_fires"] == 6 * n and cq["meta"]["layer_start"] == 16,
   "qwen: band [16,22), L=6, fires 6*n")
raises_exit(lambda: G.run_cells(FakeVC(n_dec=32), "qwen2.5", "pilot", pilot, "s" * 64, mask, os.path.join(tmp, "oq2"),
                                "q" * 64, "m.npy", decoder_layer_range=dlr), "llama mask used for qwen -> refuse")

print(f"\nALL {N_OK} CHECKS PASSED")
