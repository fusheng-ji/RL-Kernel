"""Charts for the row-mean PR: one PNG per Results table.

    python plot_row_mean.py <row-mean.json> <logprob-vs-verl-omni.json> <out dir>
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NEW, OLD, REF = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
                     "ytick.color": TEXT2, "text.color": TEXT, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def base(ax, title, sub):
    ax.set_title(title, loc="left", fontsize=12, pad=22); ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=TEXT2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def main():
    rm, lp, out = json.loads(Path(sys.argv[1]).read_text()), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    # timing
    rows = [(r["modality"], b) for r in rm["rows"] if r["mean"] == "row_mean_v1" for b in ("ms_B1", "ms_B8")]
    get = lambda mod, name, b: next(r for r in rm["rows"] if r["modality"] == mod and r["mean"] == name)[b]["median_ms"]
    fig, ax = plt.subplots(figsize=(8, 3.6)); h = 0.36
    for i, (mod, b) in enumerate(rows[::-1]):
        o, n = get(mod, "gemm_h3_projection_v1", b), get(mod, "row_mean_v1", b)
        ax.barh(i + h / 2, o, height=h, color=OLD, label="before: fixed-order GEMM [B, K] x [K, 1]" if i == 0 else None)
        ax.barh(i - h / 2, n, height=h, color=NEW, label="after: h3-row-mean-v1" if i == 0 else None)
        ax.text(o * 1.1, i + h / 2, f"{o:.3f}", va="center", fontsize=9); ax.text(n * 1.1, i - h / 2, f"{n:.3f}", va="center", fontsize=9)
    ax.set_xscale("log"); ax.set_xlim(0.01, 2000)
    ax.set_yticks(range(len(rows)), [f"{m}, batch {b[4:]}" for m, b in rows[::-1]])
    ax.set_xlabel("median ms per fixed_order_mean call (log scale)")
    base(ax, "fixed_order_mean, before and after", f"video K = 3,096,576, audio K = 6,624 per sample; {rm['device']}, torch {rm['torch']}")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "row-mean-timing.png", dpi=160); plt.close(fig)
    # accuracy of the log-prob
    rs = lp["rows"]
    fig, ax = plt.subplots(figsize=(8, 5.4)); h = 0.27
    for i, r in enumerate(rs[::-1]):
        for j, (key, col, lab) in enumerate((("old_mean_rel_err_vs_fp64", OLD, "before: fixed-order GEMM mean"),
                                             ("new_mean_rel_err_vs_fp64", NEW, "after: h3-row-mean-v1"),
                                             ("verl_omni_rel_err_vs_fp64", REF, "verl-omni (Tensor.mean, batch dependent)"))):
            y = i + (1 - j) * h
            ax.barh(y, r[key], height=h, color=col, label=lab if i == 0 else None)
            ax.text(r[key] * 1.15, y, f"{r[key]:.1e}", va="center", fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(1e-8, 0.3)
    ax.set_yticks(range(len(rs)), [f"{r['modality']}, step {r['step']}" for r in rs[::-1]])
    ax.set_xlabel("max relative error of the CPS log-prob against an FP64 mean (batch of 8, log scale)")
    base(ax, "CPS log-prob accuracy, scoring path", f"H3 latent sizes; RL-Kernel {lp['rl_kernel_commit']}, {lp['device']}, torch {lp['torch']}")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "row-mean-logprob-accuracy.png", dpi=160); plt.close(fig)
    print("charts written")


if __name__ == "__main__":
    main()
