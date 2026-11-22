#!/usr/bin/env python3
"""G6 step time per phase: P/P (stock verl-omni) vs R/R (RL-Kernel strict), mean of steps 2-8."""
import json, re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S1, S2 = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
KEYS = ["timing_s/gen", "timing_s/old_log_prob", "timing_s/update_actor", "timing_s/update_weights", "timing_s/step"]


def means(path):
    seen = {}
    for line in Path(path).read_text().splitlines():
        m = re.search(r"step:(\d+)", line)
        if m:
            seen[int(m.group(1))] = {k: float(re.search(re.escape(k) + r":([0-9.e-]+)", line).group(1)) for k in KEYS}
    rows = [seen[s] for s in sorted(seen) if s > 1]
    return {k: sum(r[k] for r in rows) / len(rows) for k in KEYS}


pp, rr, out = means(sys.argv[1]), means(sys.argv[2]), Path(sys.argv[3])
fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
labels = ["rollout (gen)", "old log-prob", "actor update", "weight sync", "step"]
x = np.arange(len(labels)); w = 0.38
for off, d, c, n in ((-w / 2, pp, S1, "P/P: stock verl-omni"), (w / 2, rr, S2, "R/R: RL-Kernel strict")):
    bars = ax.bar(x + off, [d[k] for k in KEYS], w - 0.03, color=c, label=n)
    for b, k in zip(bars, KEYS):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{d[k]:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
ax.set_xticks(x, labels, fontsize=9, color=INK)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(colors=MUTED, labelsize=9)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_ylabel("seconds (mean of steps 2–8)", fontsize=9, color=INK)
fig.suptitle("G6, TP4×CP2 on 8×B200: step time per phase", fontsize=12, color=INK, x=0.0, ha="left")
fig.legend(frameon=False, fontsize=9, loc="outside upper right", ncols=2)
fig.savefig(out, dpi=150, facecolor="white", metadata={"Software": None})
json.dump({"pp": pp, "rr": rr}, open(out.with_suffix(".json"), "w"), indent=1)
