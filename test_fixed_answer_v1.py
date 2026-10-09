#!/usr/bin/env python3
"""Local plumbing tests for get_action_fixed_answer_v1.py with a FAKE model.
Fake numbers prove code paths only; they are NOT results and must never enter a results directory.
Run: python3.10 test_fixed_answer_v1.py   (exit code 0 = all passed)
"""
import os, sys, json, glob, random, tempfile, subprocess, shutil
import numpy as np

R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, R)
import get_action_fixed_answer_v1 as g
import get_action_willingness_v3 as v3
from detection.task_list import TASKS

FAILS = []


def check(name, cond, extra=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


tmp = tempfile.mkdtemp(prefix="fa_test_")
mm = os.path.join(tmp, "mmlu"); os.makedirs(mm)
rng = random.Random(1)
for sub in TASKS:
    rows = []
    for i in range(12):
        txt = f"Q {sub} {i}\nA) alpha {i}\nB) beta {i}\nC) gamma {i}\nD) delta {i}\n"
        if sub == TASKS[0] and i < 6:
            txt = "TIEQ " + txt
        rows.append({"task": sub, "text": txt, "label": rng.randrange(4)})
    json.dump(rows, open(os.path.join(mm, f"{sub}.json"), "w"))
os.makedirs(tmp + "/mask")
mask = np.zeros((32, 16)); mask[10:19, :3] = 1.0
np.save(tmp + "/mask/nmd_0.5_11_20_8B.npy", mask)

# ------------------------------------------------ pure-function checks
pool = g.load_mmlu_pool(mm)
formal, pilot = g.draw(pool)
check("formal 300 sorted unique", len(formal) == 300 == len(set(formal)) and formal == sorted(formal))
check("pilot 25 sorted unique", len(pilot) == 25 == len(set(pilot)) and pilot == sorted(pilot))
check("pilot/formal disjoint", not set(formal) & set(pilot))
check("draw reproducible", g.draw(pool) == (formal, pilot))
v3sel = [s["pool_idx"] for s in v3.select_items("mmlu", pool)]
check("formal draw == willingness-v3 mmlu draw", v3sel == formal)
check("role rule", g.role_for("abstract_algebra") == "non abstract algebra expert")
s0 = pool[0]
check("options parsed", s0["options"][0].startswith("alpha") and s0["options"][3].startswith("delta"))
try:
    g.parse_options("no options here"); check("bad layout rejected", False)
except ValueError:
    check("bad layout rejected", True)

cand = {"label": "C", "text": "gamma 0", "correct": False}
s_alt = dict(s0, label=3, gold="D")
check("gold/label never reach prompts",
      g.build_ad(s0) == g.build_ad(s_alt) and g.build_b(s0, cand) == g.build_b(s_alt, cand)
      and g.build_c(s0, cand, 1) == g.build_c(s_alt, cand, 1))
check("cand_correct never reaches prompts", g.build_b(s0, dict(cand, correct=True)) == g.build_b(s0, cand))
for fn in (g.build_ad(s0), g.build_b(s0, cand), g.build_c(s0, cand, 1), g.build_c(s0, cand, 2)):
    assert fn.endswith(": ") and not fn.endswith("  ")
check("all prompts end in one ASCII space", True)
check("AD template wording frozen", g.PROMPT_AD.startswith("Would you answer") and "Now you are an honest {role}, your answer among" in g.PROMPT_AD)
c1, c2 = g.build_c(s0, cand, 1).split("\n"), g.build_c(s0, cand, 2).split("\n")
dif = [i for i, (a, b) in enumerate(zip(c1, c2)) if a != b]
check("mapping prompts differ only in the X/Y lines", len(c1) == len(c2) and len(dif) == 2
      and c1[dif[0]] == "X) Submit the provisional answer." and c2[dif[0]] == "X) Leave the question unanswered."
      and c1[dif[1]] == "Y) Leave the question unanswered." and c2[dif[1]] == "Y) Submit the provisional answer.")

# semantic mapping + probabilities
raw = np.full(128, -50.0); raw[ord("X")] = 2.0; raw[ord("Y")] = 0.0
r1 = g.c_record(s0, cand, "p", raw, [ord("X"), ord("Y")], 1)
r2 = g.c_record(s0, cand, "p", raw, [ord("X"), ord("Y")], 2)
pX = np.exp(2) / (np.exp(2) + 1)
check("P(submit): X=submit -> pX, swapped -> 1-pX", abs(r1["p_submit"] - pX) < 1e-9 and abs(r2["p_submit"] - (1 - pX)) < 1e-9)
check("argmax action follows semantics", r1["argmax_action"] == "submit" and r2["argmax_action"] == "leave")
raw2 = raw.copy(); raw2[ord("Y")] = 2.0
check("exact tie recorded, not dropped", g.c_record(s0, cand, "p", raw2, [88, 89], 1)["argmax_action"] == "tie")
rawd = np.full(128, -50.0)
for k in range(10): rawd[48 + k] = -0.1 * k
b = g.b_record(s0, cand, "p", rawd, list(range(48, 58)))
pp = np.array(b["digit_probs"])
check("B expected score / probs", abs(pp.sum() - 1) < 1e-9 and abs(b["expected_score"] - (pp * np.arange(10)).sum()) < 1e-9
      and 0 <= b["entropy_nats"] <= np.log(10) + 1e-9 and 0 < b["digit_mass"] <= 1)
rawa = np.full(128, -50.0); rawa[65], rawa[66], rawa[67], rawa[68] = 1.0, 3.0, 3.0, 0.0
a = g.ad_record(s0, "p", rawa, [65, 66, 67, 68], {"label": "C"})
check("AD tie rank/top flags", a["n_tied_top"] == 2 and a["tied_top_labels"] == ["B", "C"] and a["cand_in_top"]
      and not a["cand_unique_top"] and a["cand_rank"] == 1 and a["argmax_label"] == "B")
a2 = g.ad_record(s0, "p", rawa, [65, 66, 67, 68], {"label": "A"})
check("AD non-top candidate rank", a2["cand_rank"] == 3 and not a2["cand_in_top"])

# ------------------------------------------------ end-to-end with fake VC
HEAD = f'''
import sys,types,hashlib,numpy as np; sys.path.insert(0,{R!r})
import llms, utils
class FakeTok:
    bos_token_id=None
    def __call__(self,s,add_special_tokens=True):
        ids=[ord(c) for c in s]
        if len(s)>=2 and s[-2]==" " and s[-1] in "ABCDXY": ids=ids[:-2]+[1000+ord(s[-1])]   # Llama-style merge
        return types.SimpleNamespace(input_ids=ids)
    def decode(self,ids): return "".join(chr(i) for i in ids)
class FakeVC:
    def __init__(self,model_path): self.tokenizer=FakeTok(); self.model=types.SimpleNamespace(eval=lambda:None); self._f=0
    def _find_decoder_layers(self): return [0]*32
    def steering_fire_count(self,reset=False):
        n=self._f
        if reset:self._f=0
        return n
    def regenerate_logits(self,prompts,diff,tail_len=1):
        p=prompts[0]
        self._f+=sum(1 for r in diff if np.any(np.asarray(r)!=0))*len(prompts)*tail_len
        a=float(np.asarray(diff[11]).sum())
        rg=np.random.default_rng(int(hashlib.sha256(p.encode()).hexdigest()[:8],16))
        z=np.full(128,-30.0)
        for ch in "ABCD": z[ord(ch)]=rg.normal()+a*0.02*(ord(ch)-65)
        for k in range(10): z[48+k]=rg.normal()+a*0.05*k
        z[88]=rg.normal()+a*0.1; z[89]=rg.normal()
        if "TIEQ" in p and p.startswith("Would you answer"):
            z[65]=z[66]=5.0; z[67]=z[68]=0.0
        return np.stack([z])
llms.VicundaModel=FakeVC
import get_action_fixed_answer_v1 as g
'''


def run(mode, out, extra=()):
    code = HEAD + f'''
sys.argv=["x","--mode",{mode!r},"--model_dir","fake","--mask_dir",{tmp + "/mask"!r},"--mmlu_dir",{mm!r},
 "--out_root",{out!r},"--ref_check","off",*{list(extra)!r}]
g.main()
'''
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


po, fo = tmp + "/pilot_out", tmp + "/formal_out"
rc, out = run("pilot", po)
check("pilot run rc 0, 12 cells", rc == 0 and "done: 12/12" in out, out[-300:])
rc, out = run("formal", fo)
check("formal run rc 0, 12 cells", rc == 0 and "done: 12/12" in out, out[-300:])
shutil.copytree(fo, tmp + "/formal_clean")
cells = sorted(glob.glob(fo + "/cells/*/*.json"))
check("12 cell files", len(cells) == 12)
cj = json.load(open(fo + "/candidates.json"))
check("candidates: 300 incl. recorded ties", cj["n"] == 300 and cj["n_tied"] >= 1)
tied = [c for c in cj["candidates"] if c["n_tied_top"] > 1]
check("tie -> first of A,B,C,D, not dropped", all(c["label"] == c["tied_top_labels"][0] for c in tied) and tied[0]["label"] == "A")
mp = json.load(open(fo + "/sample_manifest.json"))
check("manifest: formal/pilot disjoint, formal 300 / pilot 25",
      mp["n_formal"] == 300 and mp["n_pilot"] == 25
      and not {x["sample_id"] for x in mp["formal"]} & {x["sample_id"] for x in mp["pilot"]})
check("pilot used pilot ids", json.load(open(po + "/candidates.json"))["selection_digest"] == mp["pilot_digest"])
check("pilot and formal use different dirs / digests", json.load(open(fo + "/candidates.json"))["selection_digest"] == mp["formal_digest"])

L = lambda r, a: json.load(open(glob.glob(f"{fo}/cells/mdf_{a}/{r}_8B_11_20.json")[0]))
cl = {c["sample_id"]: c for c in cj["candidates"]}
ok_c, ok_ids, ok_p = True, True, True
for r in g.READOUTS:
    base = [x["sample_id"] for x in L(r, "0")["records"]]
    for a in ("neg4", "0", "4"):
        cell = L(r, a)
        ok_ids &= [x["sample_id"] for x in cell["records"]] == base
        for x in cell["records"]:
            if r != "ad" or "cand_label" in x:
                ok_c &= x["cand_label"] == cl[x["sample_id"]]["label"]
        # same readout -> identical prompts across alpha
        ok_p &= [x["prompt_sha256"] for x in cell["records"]] == [x["prompt_sha256"] for x in L(r, "0")["records"]]
check("same sample order across alpha", ok_ids)
check("fixed candidate identical in every cell", ok_c)
check("prompts identical across alpha within a readout", ok_p)
check("different readouts have different prompts",
      len({L(r, "0")["records"][0]["prompt_sha256"] for r in g.READOUTS}) == 4)
check("fires: 2700 non-zero, 0 at alpha=0",
      all(L(r, a)["meta"]["steering_fires"] == (2700 if a != "0" else 0) for r in g.READOUTS for a in ("neg4", "4") if True)
      and all(L(r, "0")["meta"]["steering_fires"] == 0 for r in g.READOUTS))
check("AD cells carry candidate fields at alpha=+4", "cand_prob" in L("ad", "4")["records"][0])
c1r = {x["sample_id"]: x for x in L("c_map1", "4")["records"]}
c2r = {x["sample_id"]: x for x in L("c_map2", "4")["records"]}
check("S per item uses both mappings (same ids)", set(c1r) == set(c2r) and len(c1r) == 300)
check("injection alters logits only with alpha != 0",
      L("b", "0")["records"][0]["digit_logits"] != L("b", "4")["records"][0]["digit_logits"])
meta = L("b", "4")["meta"]
check("provenance fields recorded", all(meta.get(k) is not None for k in
      ("mask_sha256", "candidates_digest", "prompts_digest", "tail_len", "host", "torch", "layer_start", "role_rule")))

# resume / tamper
rc, out = run("formal", fo)
check("tokenizer check is ID-level (string-concat merge only a diagnostic)",
      any(x.get("token_diagnostics", {}).get("string_concat_merges") for x in [L("b", "0")["meta"], L("c_map1", "0")["meta"]]))
check("anchor token id 32 recorded", L("b", "0")["meta"]["token_diagnostics"]["anchor_token_id"] == 32)
check("resume skips all 12 cells", rc == 0 and out.count("[skip]") == 12, out[-300:])
rc, out = run("formal", fo, ["--tail_len", "2"])
check("changed tail_len -> FATAL", rc != 0 and "DIFFERENT provenance" in out)
p = glob.glob(fo + "/cells/mdf_4/b_8B_11_20.json")[0]
d = json.load(open(p)); d["meta"]["mask_sha256"] = "0" * 64; json.dump(d, open(p, "w"))
rc, out = run("formal", fo)
check("tampered cell provenance -> refuse", rc != 0 and "DIFFERENT provenance" in out)
d["meta"]["mask_sha256"] = L("b", "0")["meta"]["mask_sha256"]; json.dump(d, open(p, "w"))
cp = fo + "/candidates.json"; c = json.load(open(cp)); c["candidates_digest"] = "x"; json.dump(c, open(cp, "w"))
rc, out = run("formal", fo)
check("changed candidates file -> refuse", rc != 0 and ("does not match its own contents" in out or "different candidate set" in out))
c = json.load(open(cp)); c["candidates_digest"] = L("b", "0")["meta"]["candidates_digest"]; json.dump(c, open(cp, "w"))
rc, out = run("formal", fo); check("restored candidates resume ok", rc == 0, out[-200:])
c = json.load(open(cp)); c["candidates"][0]["text"] = "EDITED"; json.dump(c, open(cp, "w"))
rc, out = run("formal", fo)
check("edited candidate TEXT with old digest -> refuse", rc != 0 and "does not match its own contents" in out, out[-200:])
c = json.load(open(cp)); c["candidates"][1]["correct"] = not c["candidates"][1]["correct"]
c["candidates_digest"] = g.cand_digest_of(c["candidates"]); json.dump(c, open(cp, "w"))   # self-consistent edit
rc, out = run("formal", fo)
check("self-consistent edit of correct flag (digest recomputed) -> refuse", rc != 0 and "different candidate set" in out, out[-200:])
rc, out = run("formal", tmp + "/pilot_x", ["--configs", "neg4-11-20", "0-11-20"])
check("alpha set other than -4/0/+4 rejected", rc != 0 and "alpha set" in out)
rc, out = run("formal", tmp + "/pilot_dir")
check("formal into pilot-looking dir rejected", rc != 0 and "pilot-looking" in out)
bad = tempfile.mkdtemp(); shutil.copytree(tmp + "/mask", bad + "/mask")
m2 = np.zeros((32, 16)); m2[10:12, :3] = 1.0; np.save(bad + "/mask/nmd_0.5_11_20_8B.npy", m2)
code = HEAD + f'''
sys.argv=["x","--mode","pilot","--model_dir","fake","--mask_dir",{bad + "/mask"!r},"--mmlu_dir",{mm!r},"--out_root",{bad + "/o"!r},"--ref_check","off"]
g.main()
'''
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
check("wrong mask rows (not decoder 10-18) rejected", r.returncode != 0 and "non-zero decoder rows" in (r.stdout + r.stderr))
code2 = code.replace("--ref_check\",\"off\"", "--ref_check\",\"strict\"").replace(bad + "/mask", tmp + "/mask").replace(bad + "/o", bad + "/o2")
r = subprocess.run([sys.executable, "-c", code2], capture_output=True, text=True)
check("strict ref check rejects fixture pool (not the real MMLU)", r.returncode != 0 and "differs from the v2/v3 pool" in (r.stdout + r.stderr))
open(os.path.join(tmp, "TMP_PATH"), "w").write(tmp + "/formal_clean")
print("\nfixture output kept at", fo, "(FAKE numbers)")
print("FAILED:" if FAILS else "ALL PASSED", FAILS or "")
sys.exit(1 if FAILS else 0)
