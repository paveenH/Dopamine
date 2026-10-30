"""Fig 4: net change in correct answers by scoring path (baseline -> workpoint).

Each item is placed on one path by whether its answer was scored from a
parseable `####` marker (M) or by the fallback extractor (F) at alpha=0 and at
the workpoint. Values are net corrections (wrong->correct minus
correct->wrong) per path, copied from:
  - Llama / Qwen2.5: RoleAnswer/reasoningTaskAnalysis/score_path_audit_v1/
    review_20261028/REPORT.md (768 tokens)
  - Qwen3 GSM8K 2048: Note/ReasoningBare.md Table 9.4
  - Qwen3 GSM8K 768: recomputed from AdaResult/qwen3/gsm8k/bare/mdf_{0,10}
    with the frozen first-marker + fallback scorer (148 -> 219 correct)

Run: ~/.venvs/rsn-analysis/bin/python TMLR/scripts/fig4_score_path.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Figures" / "fig4_score_path"

PATHS = ["M → M", "F → M", "M → F", "F → F"]
# config: ([net per path in PATHS order], total net)
DATA = {
    "Llama\nGSM8K": ([19, 17, 6, 12], 54),
    "Llama\nGSM-Hard": ([3, 4, 7, 5], 19),
    "Qwen2.5\nGSM8K": ([49, 5, 0, 0], 54),
    "Qwen2.5\nGSM-Hard": ([33, 16, 0, 0], 49),
    "Qwen3 GSM8K\n768 tokens": ([2, 78, -24, 15], 71),
    "Qwen3 GSM8K\n2048 tokens": ([4, 86, -11, 6], 85),
}
COLORS = ["#2a78d6", "#1baf7a", "#eb6834", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e6e5e0"


def main():
    for name, (vals, total) in DATA.items():
        assert sum(vals) == total, name
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    xs = range(len(DATA))
    for x, (name, (vals, total)) in zip(xs, DATA.items()):
        pos, neg = 0, 0
        for i, v in enumerate(vals):
            if v == 0:
                continue
            base = pos if v > 0 else neg
            ax.bar(x, v, bottom=base, width=0.56, color=COLORS[i],
                   edgecolor="white", linewidth=1.0, zorder=3)
            if v > 0:
                pos += v
            else:
                neg += v
        ax.scatter(x, total, s=26, color=INK, zorder=5)
        ax.annotate(f"+{total}", (x, max(pos, total)), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK)
    for i, p in enumerate(PATHS):
        ax.bar(0, 0, color=COLORS[i], label=p)
    ax.scatter([], [], s=26, color=INK, label="Total net change")
    ax.axhline(0, color=MUTED, lw=0.8, zorder=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(DATA.keys(), color=INK)
    ax.set_ylabel("Net change in correct items")
    ax.set_ylim(-30, 105)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
