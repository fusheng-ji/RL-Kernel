"""Chart for PR C from lora-vs-strict.json: LoRA gradient error against FP64, per target.

    python plot_prc.py <lora-vs-strict.json> <out dir>
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OURS32, STRICT, OURS16 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1, 2, 3
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
                     "ytick.color": TEXT2, "text.color": TEXT, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
SERIES = [("ours_fp32_params", "h3-parallel-lora-v1, FP32 adapters", OURS32),
          ("strict", "#574 strict_lora_linear (FP32 adapters)", STRICT),
          ("ours_bf16_params", "h3-parallel-lora-v1, BF16 adapters (recipe)", OURS16)]


def main():
    rep = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    rows = rep["targets"]; names = [r["target"] for r in rows][::-1]; rows = rows[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    h = 0.26
    for ax, grad in zip(axes, ("dA", "dB")):
        for j, (key, label, color) in enumerate(SERIES):
            vals = [r[f"{grad}_{key}"] for r in rows]
            ys = [i + (1 - j) * h for i in range(len(rows))]
            ax.barh(ys, vals, height=h - 0.03, color=color, label=label)
            for y, v in zip(ys, vals):
                ax.text(v * 1.15, y, f"{v:.1e}", va="center", fontsize=8, color=TEXT)
        ax.set_xscale("log"); ax.set_xlim(1e-7, 2e-2)
        ax.set_title(f"{grad}", loc="left", fontsize=11, color=TEXT)
        ax.set_xlabel("relative error against FP64 (log scale)")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(names)), names)
    fig.suptitle("LoRA gradients on real block-0 weights: canonical h3-parallel-lora-v1 vs #574", x=0.01, ha="left",
                 fontsize=12, color=TEXT)
    fig.text(0.01, 0.905, f"r = {rep['lora']['rank']}, alpha = {rep['lora']['alpha']}, B random; "
             f"{rep['rows_per_case']} BF16 rows per target; torch {rep['torch']}, {rep['device']}; RL-Kernel {rep['rl_kernel_commit']}",
             fontsize=9, color=TEXT2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, fontsize=9); fig.savefig(out / "prc-lora-accuracy.png", dpi=160); plt.close(fig)
    print("chart written")


if __name__ == "__main__":
    main()
