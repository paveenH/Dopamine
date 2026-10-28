"""Fig 2: GSM8K Bare No-CoT dose curves for four models.

Per panel: first_acc, early_candidate_rate (EC) and reason_first_rate on one
0-100% axis; the fixed GSM8K workpoint is shaded. Inputs are the frozen
offline metrics:
  data/gsm8k_org_llama_qwen25.csv  <- RoleAnswer/unified_behavior_gsm8k_math.csv
  data/gsm8k_org_qwen3_granite.csv <- scripts/compute_qwen3_granite_org.py
Qwen3 uses the 768-token sweep (same budget as the other three models).
reason_first_rate is computed on the candidate-covered subset (>=93% of 300 in
every cell); the other two use all 300 items.

Run: ~/.venvs/rsn-analysis/bin/python TMLR/scripts/fig2_dose_organization.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Figures" / "fig2_dose_organization"

PANELS = [  # (key, title, workpoint)
    ("llama3", "Llama-3.1-8B", -6),
    ("granite", "Granite-3.3-8B", -6),
    ("qwen2.5", "Qwen2.5-7B", 8),
    ("qwen3", "Qwen3-8B", 10),
]
SERIES = [  # (column, label, color, marker, linestyle)
    ("first_acc", "Accuracy (first_acc)", "#2a78d6", "o", "-"),
    ("early_candidate_rate", "Early candidate (EC)", "#eb6834", "s", "--"),
    ("reason_first_rate", "Reason first", "#1baf7a", "^", ":"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e6e5e0"


def load():
    rows = {}
    with open(ROOT / "data" / "gsm8k_org_llama_qwen25.csv") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["model"], []).append(r)
    with open(ROOT / "data" / "gsm8k_org_qwen3_granite.csv") as f:
        for r in csv.DictReader(f):
            if r["budget"] == "768":
                rows.setdefault(r["model"], []).append(r)
    for k in rows:
        rows[k].sort(key=lambda r: float(r["alpha"]))
        assert all(int(r["n"]) == 300 for r in rows[k]), k
    return rows


def main():
    rows = load()
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0), sharey=True)
    for ax, (key, title, wp) in zip(axes, PANELS):
        data = rows[key]
        x = [float(r["alpha"]) for r in data]
        ax.axvspan(wp - 0.8, wp + 0.8, color="#f1efe8", zorder=0)
        ax.axvline(0, color=GRID, lw=1, zorder=0)
        for col, label, color, mk, ls in SERIES:
            y = [float(r[col]) for r in data]
            ax.plot(x, y, color=color, marker=mk, ms=4.5, lw=1.8, ls=ls,
                    label=label, zorder=3)
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(v):+d}" if v else "0" for v in x], fontsize=8)
        ax.set_xlabel(r"$\alpha$ (model-specific scale)")
        ax.set_ylim(-3, 103)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(wp, 101, "workpoint", ha="center", va="bottom", fontsize=7,
                color=MUTED)
    axes[0].set_ylabel("% of items")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
