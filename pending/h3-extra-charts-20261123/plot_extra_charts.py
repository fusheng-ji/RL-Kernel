#!/usr/bin/env python3
"""Charts added to #732, #735 and #743 (2026-11-23): numbers from their published tables and evidence."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1f9e89", "#9aa6b2"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OUT = Path(sys.argv[1])


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)


def label(ax, bars, fmt):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt(b.get_height()), ha="center", va="bottom", fontsize=7.5, color=INK)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", metadata={"Software": None}); plt.close(fig)


# #732: weights held and peak per rank, B200 vs H100
layouts = ["TP2\n40 rows", "TP4 + SP\n928", "TP4×CP2\nring 925", "TP4×CP2 + SP\nring 928", "TP4×CP2 + SP\nUlysses 928", "TP8 + SP\n928"]
resident = [43.78, 34.81, 34.81, 34.81, 34.81, 30.33]
peak_b200 = [45.35, 37.78, 36.72, 37.26, 37.26, 33.03]
peak_h100 = [45.5, 37.8, 36.7, 37.3, 37.3, 33.1]
fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
x = np.arange(len(layouts)); w = 0.27
label(ax, ax.bar(x - w, resident, w - 0.02, color=S4, label="weights held per rank"), lambda v: f"{v:.1f}")
label(ax, ax.bar(x, peak_b200, w - 0.02, color=S1, label="peak, B200 (provider boundaries)"), lambda v: f"{v:.1f}")
label(ax, ax.bar(x + w, peak_h100, w - 0.02, color=S2, label="peak, H100 (strict boundaries)"), lambda v: f"{v:.1f}")
ax.axhline(61.73, color=MUTED, ls="--", lw=1)
ax.text(len(layouts) - 0.5, 61.73, "full model on every rank (#701 harness): 61.73", ha="right", va="bottom", fontsize=8, color=MUTED)
ax.set_xticks(x, layouts, fontsize=8.5, color=INK); ax.set_ylim(0, 70); style(ax)
ax.set_ylabel("GiB per rank", fontsize=9, color=INK)
fig.suptitle("Weight-sharded H3: memory per rank by layout (every layout bitwise against Grid(1, 1))", fontsize=12, color=INK, x=0.0, ha="left")
fig.legend(frameon=False, fontsize=9, loc="outside lower center", ncols=3)
save(fig, "pr732-memory-per-rank-b200-h100.png")

# #735: request wall time on the bound TP4 engine, B200 vs H100
reqs = ["A alone\n(first request)", "B alone", "A + B together"]
data = {"B200, B random": [9.02, 3.86, 6.82], "B200, B = 0": [9.29, 3.72, 6.77],
        "H100, B random": [15.33, 4.78, 9.24], "H100, B = 0": [8.51, 4.79, 9.19]}
fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
x = np.arange(len(reqs)); w = 0.2
for i, ((name, vals), c) in enumerate(zip(data.items(), (S1, "#7fb0ea", S2, "#f4ab83"))):
    label(ax, ax.bar(x - 1.5 * w + i * w, vals, w - 0.02, color=c, label=name), lambda v: f"{v:.1f}")
ax.set_xticks(x, reqs, fontsize=9, color=INK); style(ax)
ax.set_ylabel("seconds per request (2 steps, 448×256, 4 s)", fontsize=9, color=INK)
fig.suptitle("vLLM-Omni TP4 with the bound H3 DiT: request time", fontsize=12, color=INK, x=0.0, ha="left")
fig.legend(frameon=False, fontsize=9, loc="outside lower center", ncols=4)
save(fig, "pr735-request-time-b200-h100.png")

# #743: the two rows against the provider's expressions (one B200)
bench = json.loads(Path(sys.argv[2]).read_text())["pack_noise"]
rows = [r["row"] for r in bench]
fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
x = np.arange(len(rows)); w = 0.36
label(ax, ax.bar(x - w / 2, [r["provider_ms"] for r in bench], w - 0.03, color=S1, label="provider (index_copy ×3 / eager expression)"), lambda v: f"{v:.3f}")
label(ax, ax.bar(x + w / 2, [r["row_ms"] for r in bench], w - 0.03, color=S2, label="RL-Kernel row (CUDA)"), lambda v: f"{v:.3f}")
ax.set_xticks(x, [f"{r['row']}\n{tuple(r['shape'])}" for r in bench], fontsize=9, color=INK); style(ax)
ax.set_ylabel("ms per call (median of 25)", fontsize=9, color=INK)
fig.suptitle("Packed-sequence and noise-mix rows on one B200 (outputs bitwise equal)", fontsize=12, color=INK, x=0.0, ha="left")
fig.legend(frameon=False, fontsize=9, loc="outside lower center", ncols=2)
save(fig, "pr743-row-time-b200.png")
print("ok")
