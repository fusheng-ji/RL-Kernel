#!/usr/bin/env python3
"""final_e2e_diffusers_cuda charts from compare/report.json: P-vs-R drift per step, time split."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

S1, S2 = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
report = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
cases = report["cases"]
steps = cases[0]["provider_vs_candidate"]["entries"][-1]["step"] + 1


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# 1. drift per step: max |latent_R - latent_P| per step, median and range over the cases
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
for ax, modality, color in zip(axes, ("video", "audio"), (S1, S2)):
    per = np.full((len(cases), steps), np.nan)
    for i, c in enumerate(cases):
        for e in c["provider_vs_candidate"]["entries"]:
            if e["kind"] == "latent" and e["modality"] == modality:
                per[i, e["step"]] = e["max_abs_diff"]
    x = np.arange(steps)
    ax.fill_between(x, per.min(0), per.max(0), color=color, alpha=0.18, lw=0, label="range, 8 cases")
    ax.plot(x, np.median(per, 0), color=color, lw=2, marker="o", ms=4, label="median")
    ax.set_title(f"{modality} latent", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("scheduler step", fontsize=9, color=INK)
    ax.set_xticks(x)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
axes[0].set_ylabel("max |R - P| after the step", fontsize=9, color=INK)
fig.suptitle(
    "Pinned Diffusers (P) vs RL-Kernel strict (R), T2VA on 1 x H100: drift per step",
    fontsize=11, color=INK, x=0.0, ha="left",
)
fig.savefig(out / "h3-e2e-t2va-drift-h100.png", dpi=150, facecolor="white", metadata={"Software": None})

# 2. time per run, by phase (mean over the 8 cases' first runs; transformer = steady calls)
s = report["summary"]["time_mean_first_run"]
phases = [
    ("text encoding", "text_encoder"),
    ("transformer, per call", "transformer_steady_per_call"),
    ("transformer, 8 steady calls", None),
    ("scheduler, 18 steps", "scheduler"),
    ("decode (video + audio)", "decode"),
]
vals = {}
for arm in ("provider", "candidate"):
    t = s[arm]
    vals[arm] = [
        t["text_encoder"],
        t["transformer_steady_per_call"],
        8 * t["transformer_steady_per_call"],
        t["scheduler"],
        t["decode"],
    ]
fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
y = np.arange(len(phases))
h = 0.38
for off, arm, color, label in ((-h / 2, "provider", S1, "P: pinned Diffusers"), (h / 2, "candidate", S2, "R: RL-Kernel strict")):
    bars = ax.barh(y + off, vals[arm], h - 0.04, color=color, label=label)
    for b, v in zip(bars, vals[arm]):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:.2f} s", va="center", fontsize=8, color=INK)
ax.set_yticks(y, [p for p, _ in phases], fontsize=9, color=INK)
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(colors=MUTED, labelsize=9)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_xlabel(
    "seconds, mean of the 8 cases' first runs (transformer: steady calls; "
    "decode can include the VAE's offload move)",
    fontsize=9,
    color=INK,
)
fig.suptitle("T2VA on 1 x H100: time per phase", fontsize=11, color=INK, x=0.0, ha="left")
fig.legend(frameon=False, fontsize=9, loc="outside upper right", ncols=2)
fig.savefig(out / "h3-e2e-t2va-time-h100.png", dpi=150, facecolor="white", metadata={"Software": None})
