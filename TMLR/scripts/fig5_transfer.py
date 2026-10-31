"""Fig 5 and Table 6: fixed-workpoint transfer (top) and task boundaries (bottom).

Reads TMLR/data/transfer_boundary.csv (compute_transfer_table.py). Points are
accuracy differences (workpoint - alpha=0, percentage points) with paired
bootstrap 95% CIs; filled markers pass Holm in the chosen family, hollow ones
do not. FAMILY selects the transfer family ("per_task4" or "all8"); boundary
rows use one family per condition (m=6).

Run: ~/.venvs/rsn-analysis/bin/python TMLR/scripts/fig5_transfer.py [--family all8]
"""
import csv, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transfer_boundary.csv"
FIG = ROOT / "Figures" / "fig5_transfer"
TABLE = ROOT / "data" / "table6_transfer.md"
FAMILY = sys.argv[sys.argv.index("--family") + 1] if "--family" in sys.argv else "per_task4"
COL = {"per_task4": "holm_per_task4", "all8": "holm_all8"}[FAMILY]

MODELS = {"Llama": "#2a78d6", "Qwen2.5": "#eb6834", "Qwen3": "#1baf7a", "Granite": "#eda100"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e6e5e0"
TASK_ORDER = ["GSM-Hard", "MATH", "LogiQA 2.0", "BBH", "CRUXEval-O"]


def load():
    rows = list(csv.DictReader(open(DATA)))
    for r in rows:
        for k in ("alpha", "n", "w2c", "c2w", "holm_boundary_m"):
            r[k] = int(r[k]) if r[k] else None
        for k in ("acc0", "acc1", "diff", "ci_lo", "ci_hi", "p", "holm_all8", "holm_per_task4", "holm_boundary"):
            r[k] = float(r[k]) if r[k] else None
        r["padj"] = r[COL] if r["section"] == "transfer" else r["holm_boundary"]
    key = lambda r: (TASK_ORDER.index(r["task"]), r["condition"] != "No-CoT", list(MODELS).index(r["model"]))
    return ([r for r in sorted(rows, key=key) if r["section"] == "transfer"],
            [r for r in sorted(rows, key=key) if r["section"] == "boundary"])


def a(x):
    return f"{x:+d}".replace("-", "−")


def pp(x):
    return f"{x:+.2f}".replace("-", "−")


def pfmt(p):
    if p >= 0.001:
        return f"{p:.3f}".lstrip("0") if p < 1 else "1.000"
    m, e = f"{p:.1e}".split("e")
    return f"{m}e−{int(e[1:])}" if e.startswith("-") else f"{m}e{int(e)}"


def panel(ax, rows, title, show_cond):
    ys = list(range(len(rows)))[::-1]
    prev = None
    for y, r in zip(ys, rows):
        c = MODELS[r["model"]]
        sig = r["padj"] < 0.05
        mk = "D" if r["condition"] == "CoT" else "o"
        ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=c, lw=2, solid_capstyle="round", zorder=3)
        ax.scatter(r["diff"], y, s=46, marker=mk, color=c if sig else "white", edgecolor=c,
                   linewidth=1.6, zorder=4)
        ax.text(1.02, y, f"{pp(r['diff'])}  [{pp(r['ci_lo'])}, {pp(r['ci_hi'])}]   {pfmt(r['padj'])}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=INK, family="DejaVu Sans")
        group = (r["task"], r["condition"]) if show_cond else r["task"]
        if group != prev and prev is not None:
            ax.axhline(y + 0.5, color=GRID, lw=0.8, zorder=0)
        prev = group
    labels = []
    for r in rows:
        cond = f" · {r['condition']}" if show_cond else ""
        labels.append(f"{r['task']}{cond} · {r['model']} ({a(r['alpha'])})")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, color=INK, fontsize=8)
    ax.axvline(0, color=MUTED, lw=0.9, zorder=1)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax.text(1.02, len(rows) - 0.3, "Δ (pp)  [95% CI]   Holm p", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color=MUTED, va="bottom")


def figure(tr, bd):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 7.0), sharex=True,
                                 gridspec_kw={"height_ratios": [len(tr), len(bd)], "hspace": 0.18})
    panel(a1, tr, "A. Fixed-workpoint transfer", False)
    panel(a2, bd, "B. Task boundaries", True)
    a2.set_xlabel("Accuracy change at the fixed workpoint (percentage points)", color=INK)
    a2.set_xlim(-12, 27)
    handles = [Line2D([], [], color=c, marker="o", lw=2, label=m) for m, c in MODELS.items()]
    handles += [Line2D([], [], color=MUTED, marker="o", ls="", markerfacecolor=MUTED, label="Holm p < .05"),
                Line2D([], [], color=MUTED, marker="o", ls="", markerfacecolor="white", label="Holm p ≥ .05"),
                Line2D([], [], color=MUTED, marker="D", ls="", markerfacecolor="white", label="CoT")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.45, -0.01))
    fig.subplots_adjust(left=0.30, right=0.70, bottom=0.11, top=0.96)
    fig.savefig(FIG.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(FIG.with_suffix(".pdf"), bbox_inches="tight")


def table(tr, bd):
    def line(r, cond):
        c = f" | {r['condition']}" if cond else ""
        return (f"| {r['task']}{c} | {r['model']} | {a(r['alpha'])} | {r['acc0']:.2f} | {r['acc1']:.2f} | "
                f"{pp(r['diff'])} [{pp(r['ci_lo'])}, {pp(r['ci_hi'])}] | {r['w2c']}/{r['c2w']} | {pfmt(r['p'])} | "
                f"{pfmt(r['padj'])} |")
    out = [f"<!-- generated by TMLR/scripts/fig5_transfer.py; transfer family = {FAMILY} -->",
           "**Fixed-workpoint transfer (No-CoT)**", "",
           "| Task | Model | α | Acc. α=0 (%) | Acc. α (%) | Δ pp [95% CI] | W→C / C→W | p | Holm p |",
           "|---|---|---:|---:|---:|---|---:|---:|---:|"]
    out += [line(r, False) for r in tr]
    out += ["", "**Task boundaries**", "",
            "| Task | Condition | Model | α | Acc. α=0 (%) | Acc. α (%) | Δ pp [95% CI] | W→C / C→W | p | Holm p |",
            "|---|---|---|---:|---:|---:|---|---:|---:|---:|"]
    out += [line(r, True) for r in bd]
    TABLE.write_text("\n".join(out) + "\n")


def main():
    tr, bd = load()
    figure(tr, bd)
    table(tr, bd)
    print("wrote", FIG.with_suffix(".png"), "and", TABLE)


if __name__ == "__main__":
    main()
