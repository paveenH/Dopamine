#!/usr/bin/env python3
"""Llama answer-first pattern across prompt conditions -- EXPLORATORY.

Builds Table 5.11e of AdaptiveThinking.md: the four llama GSM-Hard cells
(No-CoT / CoT x alpha 0 / -6), each with accuracy, committed n, answer-first
count and posN median.

STATUS. The two No-CoT cells were unsealed on 2026-08-30 and are permanently
EXPLORATORY. The two CoT cells inherit p3-supp-v1's locked status for their
ACCURACY (frozen prediction, stage-2 commitment sealed before unsealing), but
the answer-first readout itself was defined after seeing them, so the whole
table is labelled exploratory. It exists to CORRECT the interpretation of posN,
not to supply new performance evidence.

FROZEN DEFINITION -- answer-first:

    the first non-whitespace content of the generation is a parseable
    `#### <number>`

Anchored on all_hash's regex, so "parseable" means exactly what every published
commit number means. It is deliberately NOT "the first token": `#### 42` spans
several tokenizer tokens, and a token-level rule would additionally depend on
which tokenizer is loaded.

Accuracy is recomputed here through the SAME frozen extractor chain the
evaluator uses (all_hash first, fallback_gsm8k second, norm_gsm8k compare), so
these cells must reproduce p3_evaluation.json / p3_supp_evaluation.json exactly.
That reproduction is asserted, not assumed -- it is the acceptance check for
any future edit to this file.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "thinking_curve"))

from analyze_first_last_acc import all_hash, norm_gsm8k, fallback_gsm8k
from extract_metrics import per_sample

GOLD = os.path.join(ROOT, "gsm_hard_p3_gold.SEALED.json")
FNAME = "gsm_hard_8B_11_20.json"
SUB = "llama3/gsm_hard"

# (label, alpha, directory tag, protocol the cell must declare)
CELLS = [("No-CoT", 0,  "mdf_0",        "p3-v1"),
         ("No-CoT", -6, "mdf_neg6",     "p3-v1"),
         ("CoT",    0,  "mdf_0_cot",    "p3-supp-v1"),
         ("CoT",    -6, "mdf_neg6_cot", "p3-supp-v1")]

# Expected accuracy, read from the two frozen evaluation artifacts. A mismatch
# means this file's extractor chain has drifted from the evaluator's.
EXPECT_ACC = {("No-CoT", 0): .1800, ("No-CoT", -6): .2433,
              ("CoT", 0): .2000, ("CoT", -6): .2600}

# The first non-whitespace content is a parseable #### <number>.
ANSWER_FIRST = re.compile(r"^\s*####\s*([+-]?[\d,]+\.?\d*)")


def med(v):
    v = sorted(x for x in v if x == x)
    if not v:
        return float("nan")
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def load(tag, alpha, protocol):
    p = os.path.join(ROOT, SUB, tag, FNAME)
    d = json.load(open(p, encoding="utf-8"))
    meta = d["meta"]
    if meta.get("protocol") != protocol:
        sys.exit(f"FAIL [{tag}]: protocol {meta.get('protocol')!r}, expected {protocol!r}")
    if meta.get("alpha") != alpha:
        sys.exit(f"FAIL [{tag}]: meta alpha {meta.get('alpha')}, expected {alpha}")
    if len(d["data"]) != 300:
        sys.exit(f"FAIL [{tag}]: n={len(d['data'])}, expected 300")
    return d["data"]


gold = {int(s["sample_id"]): str(s["gold"])
        for s in json.load(open(GOLD, encoding="utf-8"))["data"]}

print("=" * 96)
print("Table 5.11e  Llama answer-first pattern across prompt conditions   [EXPLORATORY]")
print("=" * 96)
print(f"{'condition':>9} {'a':>4} {'acc':>7} {'committed':>10} "
      f"{'answer-first':>21} {'posN_med':>9}")
print("-" * 96)

rows_out = []
for label, alpha, tag, protocol in CELLS:
    data = load(tag, alpha, protocol)
    n_corr = n_af = 0
    committed, posN = [], []
    for s in data:
        g = norm_gsm8k(gold[int(s["sample_id"])])
        txt = s["generated"]
        hits = all_hash(txt)
        if hits:
            n_corr += int(norm_gsm8k(hits[0]) == g)
        else:
            fb = fallback_gsm8k(txt)
            n_corr += int(fb is not None and norm_gsm8k(fb) == g)
        row = per_sample({"generated": txt, "question": s["question"],
                          "correct": 0, "x_prefill": 0.0, "x_decode": []})
        if row["commit_state"] == "committed":
            committed.append(row)
            posN.append(row["posN"])
            if ANSWER_FIRST.match(txt):
                n_af += 1

    acc = n_corr / len(data)
    nc = len(committed)
    exp = EXPECT_ACC[(label, alpha)]
    if abs(acc - exp) > 5e-4:
        sys.exit(f"FAIL [{label} a={alpha}]: acc {acc:.4f} does not reproduce the "
                 f"frozen evaluation value {exp:.4f}")
    pm = med(posN)
    print(f"{label:>9} {alpha:>+4d} {acc:>7.4f} {nc:>10d} "
          f"{n_af:>13d} ({100*n_af/nc:>4.1f}%) {pm:>9.4f}")
    rows_out.append(dict(condition=label, alpha=alpha, accuracy=round(acc, 4),
                         committed_n=nc, answer_first_n=n_af,
                         answer_first_pct=round(100 * n_af / nc, 1),
                         posN_median=round(pm, 4)))

print("-" * 96)
print("accuracy reproduces p3_evaluation.json / p3_supp_evaluation.json in all four cells")
print("\nanswer-first := the first non-whitespace content of the generation is a")
print("parseable '#### <number>'. Denominator is the committed subset.")
print("\nEXPLORATORY. Corrects the interpretation of posN; not new performance evidence.")

json.dump({"table": "5.11e", "status": "exploratory", "model": "llama3",
           "answer_first_definition": ("the first non-whitespace content of the "
                                       "generation is a parseable '#### <number>'"),
           "denominator": "committed subset",
           "rows": rows_out},
          open(os.path.join(HERE, "answer_first_panel.json"), "w"), indent=2)
