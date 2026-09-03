"""Parse each wps cell's REAL argv with the generator's OWN parser.

bash -n cannot catch an unrecognized argument; only argparse can. This is the
check that would have caught --n_samples on the GSM8K generator.
"""
import subprocess, sys, argparse, io, contextlib, re, os

CELLS = {"gsm8k_cot_neg2":("run_wps_llama3.sh","get_answer_regenerate_gsm8k.py"),
         "math_neg8":("run_wps_llama3.sh","get_answer_regenerate_math.py"),
         "math_cot_neg8":("run_wps_llama3.sh","get_answer_regenerate_math.py"),
         "llama_cot_neg4":("run_wps_gsm_hard.sh","get_answer_gsm_hard_blind.py"),
         "qwen_cot_6":("run_wps_gsm_hard.sh","get_answer_gsm_hard_blind.py"),
         "qwen_cot_10":("run_wps_gsm_hard.sh","get_answer_gsm_hard_blind.py"),
         "qwen_nocot_10":("run_wps_gsm_hard.sh","get_answer_gsm_hard_blind.py")}

def build_parser(path):
    """Re-execute only the argparse block, wherever the generator builds it.

    Two shapes exist in this repo: built inline under __main__ (the regenerate
    family, variable `parser`) or inside a function (the blind script, `p`).
    """
    src = open(path).read()
    i = src.find("argparse.ArgumentParser")
    start = src.rfind("\n", 0, src.rfind("\n", 0, i) + 1)
    blk = src[src.rindex("\n", 0, i) + 1:]
    lines = blk.splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    keep = []
    for l in lines:
        st = l.strip()
        if not st:
            continue
        cur = len(l) - len(l.lstrip())
        if cur < indent:
            break
        if re.match(r"(args|a)\s*=\s*(p|parser)\.parse_args|return\s+(p|parser)\.parse_args", st):
            break
        keep.append(l[indent:])
        if st.startswith("return ") or re.match(r"(args|a)\s*=", st):
            break
    ns = {"argparse": argparse}
    exec("\n".join(keep), ns)
    for name in ("parser", "p"):
        if name in ns and isinstance(ns[name], argparse.ArgumentParser):
            return ns[name]
    raise KeyError(f"no ArgumentParser found in {path}")

with open("/tmp/echoargs","w") as f:
    f.write('#!/bin/bash\n[ "$1" = "-c" ] && exit 0\necho "ARGV:$@"\n')
os.chmod("/tmp/echoargs", 0o755)

import tempfile
STUB = tempfile.mkdtemp()
for d, f in [("components/mask/llama3_non_logits","nmd_0.5_11_20_8B.npy"),
             ("components/benchmark","gsm8k_test_sample.json"),
             ("components/benchmark","math_test_sample.json"),
             ("components/mask/qwen2.5_non_logits","nmd_0.5_16_22_7B.npy")]:
    os.makedirs(os.path.join(STUB,d), exist_ok=True)
    open(os.path.join(STUB,d,f),"w").close()
import json
json.dump({"meta":{"contains_labels":False,"questions_sha256":"0"*64},
           "data":[{"question":"q","sample_id":0}]},
          open(os.path.join(STUB,"components/benchmark/gsm_hard_p3_questions.json"),"w"))

fail = 0
for cell, (launcher, gen) in CELLS.items():
    # WORK_DIR/BASE_DIR/MASK are server paths; stub them so the launcher
    # reaches the generator call. We are testing argv construction, not I/O.
    r = subprocess.run(["bash","-c",
        f"sed -e 's#^WORK_DIR=.*#WORK_DIR={STUB}#' {launcher} > {STUB}/L.sh; "
        f"bash {STUB}/L.sh {cell}"],
        env={"PATH":"/usr/bin:/bin","CUDA_VISIBLE_DEVICES":"0","PY":"/tmp/echoargs"},
        capture_output=True, text=True, cwd=os.getcwd())
    argv = None
    for l in r.stdout.splitlines():
        if l.startswith("ARGV:") and gen in l:
            argv = l[5:].split()[1:]
    if argv is None:
        print(f"[x] {cell}: could not capture argv"); fail += 1; continue
    p = build_parser(gen)
    try:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            a = p.parse_args(argv)
        print(f"[v] {cell:16s} -> {gen}\n     configs={a.configs} cot={a.cot} "
              f"mnt={a.max_new_tokens} bs={a.batch_size} "
              f"{'ans=' + a.ans_file if hasattr(a,'ans_file') else 'out=' + a.out_dir.split('/')[-1]}"
              f"{' n=' + str(a.n_samples) if hasattr(a,'n_samples') else ''}")
    except SystemExit:
        print(f"[x] {cell:16s} -> {gen} REJECTED:\n     {err.getvalue().strip().splitlines()[-1]}")
        fail += 1
sys.exit(fail)
