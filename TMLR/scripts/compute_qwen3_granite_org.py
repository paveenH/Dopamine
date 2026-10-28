import sys, json, csv
from pathlib import Path
RA = Path("/Users/paveenhuang/Documents/RSNResult/RoleAnswer")
sys.path.insert(0, str(RA))
from unified_behavior_gsm8k_math import compute_row, load_cell
cells = []
for a in [-8,-6,-4,0,4,6,8,10]:
    cells.append(("qwen3","768",a,RA/f"AdaResult/qwen3/gsm8k/bare/mdf_{a}/gsm8k_qwen3_bare_8B_answers_17_26.json"))
for p in sorted((RA/"AdaResult/qwen3/gsm8k/bare_2048").rglob("*.json")):
    a = int(p.parent.name.replace("mdf_","").replace("neg","-"))
    cells.append(("qwen3","2048",a,p))
for a in [-10,-8,-6,0,6,8,10]:
    cells.append(("granite","768",a,RA/f"AdaResult/granite/gsm8k/bare/mdf_{a}/gsm8k_granite33_bare_8B_answers_11_20.json"))
keys=["first_acc","last_acc","valid_submission_rate","early_candidate_rate","candidate_coverage","reason_first_rate","pre_candidate_chars_median"]
out=[]
for m,b,a,p in cells:
    r = compute_row(m,"gsm8k",False,a,load_cell(p))
    row={"model":m,"budget":b,"alpha":a,"n":r["n"],**{k:r[k] for k in keys}}
    out.append(row); print(row)
with open(sys.argv[1],"w") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
