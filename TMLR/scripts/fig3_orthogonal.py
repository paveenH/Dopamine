"""Fig 3: RSN versus norm-matched orthogonal injections (GSM8K Bare No-CoT).

Each point is one 300-item condition: x = early-candidate rate, y = first_acc.
Values are copied from Note/ReasoningBare.md Table 7.3 (on-support and
off-support controls; Llama alpha=-6, Qwen2.5 alpha=+8, same support/norm per
layer for on-support, disjoint support for off-support).

Run: ~/.venvs/rsn-analysis/bin/python TMLR/scripts/fig3_orthogonal.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Figures" / "fig3_orthogonal"

# condition: (first_acc, EC)
DATA = {
    "Llama-3.1-8B (α = −6)": {
        "Baseline": (60.00, 48.00), "RSN": (78.00, 18.67),
        "On 1": (71.00, 31.67), "On 2": (66.00, 46.67), "On 3": (56.67, 52.00),
        "Off 1": (75.33, 23.67), "Off 2": (63.00, 37.00), "Off 3": (62.67, 37.67),
    },
    "Qwen2.5-7B (α = +8)": {
        "Baseline": (69.33, 96.67), "RSN": (85.00, 5.00),
        "On 1": (69.67, 97.67), "On 2": (68.33, 96.33), "On 3": (69.67, 95.67),
        "Off 1": (72.33, 89.00), "Off 2": (68.33, 93.00), "Off 3": (72.00, 88.33),
    },
}
STYLE = {  # kind: (color, marker, size, label)
    "Baseline": ("#6b6a65", "D", 46, "Baseline (α = 0)"),
    "RSN": ("#2a78d6", "o", 70, "RSN"),
    "On": ("#eb6834", "s", 40, "On-support orthogonal (3 seeds)"),
    "Off": ("#1baf7a", "^", 44, "Off-support orthogonal (3 seeds)"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e6e5e0"


def main():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    seen = set()
    for ax, (title, conds) in zip(axes, DATA.items()):
        for name, (acc, ec) in conds.items():
            kind = name.split()[0]
            color, mk, size, label = STYLE[kind]
            ax.scatter(ec, acc, s=size, marker=mk, color=color,
                       edgecolor="white", linewidth=1.2,
                       zorder=5 if kind in ("Baseline", "RSN") else 3,
                       label=None if kind in seen else label)
            seen.add(kind)
            if kind in ("Baseline", "RSN"):
                ax.annotate(name, (ec, acc), xytext=(6, 4),
                            textcoords="offset points", fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("Early-candidate rate (%)")
        ax.set_xlim(-3, 103)
        ax.grid(color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Accuracy, first_acc (%)")
    axes[0].set_ylim(50, 90)
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
