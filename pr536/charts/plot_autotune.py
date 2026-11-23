"""Tuned vs fixed launch time for the 52 #536 performance cases (B200, BF16).

Point = median over 5 process groups of tuned / fixed; bar = [tuned_min / fixed_max,
tuned_max / fixed_min]. A bar entirely left of 1 is the PR's "confirmed faster"
(non-overlapping process-median ranges); entirely right of 1 is "tuned slower".
"""

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FASTER, SLOWER, NEUTRAL = "#2a78d6", "#eb6834", "#9a9893"


def verdict(lo, hi):
    if hi < 1.0:
        return "faster"
    if lo > 1.0:
        return "slower"
    return "overlap"


STYLE = {
    "faster": dict(color=FASTER, marker="o", fill=True),
    "slower": dict(color=SLOWER, marker="^", fill=True),
    "overlap": dict(color=NEUTRAL, marker="o", fill=False),
}

labels = [f"{r['operator']} · {r['preset']} · {r['phase']}" for r in rows]
n = len(rows)
fig, axes = plt.subplots(1, 2, figsize=(11, 0.21 * n + 1.9), sharey=True, facecolor=SURFACE)
counts = {}
for ax, mode, title in ((axes[0], "eager", "Eager wall time"), (axes[1], "graph", "CUDA graph device time")):
    ax.set_facecolor(SURFACE)
    c = {"faster": 0, "slower": 0, "overlap": 0}
    for i, r in enumerate(rows):
        f, fmin, fmax = (float(r[f"{mode}_fixed{s}_us"]) for s in ("", "_min", "_max"))
        t, tmin, tmax = (float(r[f"{mode}_tuned{s}_us"]) for s in ("", "_min", "_max"))
        ratio, lo, hi = t / f, tmin / fmax, tmax / fmin
        v = verdict(lo, hi)
        expected = {"faster": "confirmed"}.get(v)
        if expected and r[f"{mode}_benefit"] != expected:
            raise SystemExit(f"verdict mismatch at {labels[i]} ({mode})")
        c[v] += 1
        st = STYLE[v]
        y = n - 1 - i
        ax.plot([lo, hi], [y, y], color=st["color"], lw=2 if st["fill"] else 1.2,
                alpha=1 if st["fill"] else 0.7, solid_capstyle="round", zorder=2)
        ax.plot(
            ratio, y, marker=st["marker"], ms=6.5, mew=1.6, zorder=3, ls="none",
            mfc=st["color"] if st["fill"] else SURFACE, mec=st["color"] if not st["fill"] else SURFACE,
        )
    counts[mode] = c
    ax.axvline(1.0, color=INK2, lw=1, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.6, 2.4)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0])
    ax.set_xticklabels(["0.6", "0.7", "0.8", "0.9", "1", "1.25", "1.5", "2"])
    ax.minorticks_off()
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_title(
        f"{title}\n{c['faster']} faster · {c['slower']} slower · {c['overlap']} overlapping",
        color=INK, fontsize=10, loc="left",
    )
    ax.set_xlabel("tuned / fixed launch time (log scale, < 1 = tuning faster)", color=INK2, fontsize=8.5)

axes[0].set_yticks(range(n))
axes[0].set_yticklabels(list(reversed(labels)), fontsize=7.5, color=INK)
axes[0].set_ylim(-0.8, n - 0.2)
legend = [
    Line2D([], [], color=FASTER, marker="o", lw=2, mec=SURFACE, label="tuned faster (process-median ranges do not overlap)"),
    Line2D([], [], color=SLOWER, marker="^", lw=2, mec=SURFACE, label="tuned slower (ranges do not overlap)"),
    Line2D([], [], color=NEUTRAL, marker="o", lw=2, mfc=SURFACE, label="ranges overlap (no claim)"),
]
fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.6, 0.0), ncol=3, frameon=False, fontsize=8.5, labelcolor=INK)
fig.text(0.01, 0.993, "Opt-in auto-tuning on B200, BF16: 52 operator / preset / phase cases, 5 process groups each",
         color=INK, fontsize=11.5, ha="left", va="top")
fig.text(0.01, 0.976, "point = median of tuned / fixed; bar = [tuned min / fixed max, tuned max / fixed min], "
         "clipped to the axis; 1 = same time",
         color=INK2, fontsize=8.5, ha="left", va="top")
fig.subplots_adjust(left=0.215, right=0.985, top=0.925, bottom=0.075, wspace=0.06)
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(counts)
