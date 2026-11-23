"""#548 item 4 chart: trainer memory per probe row, and observed vs AdamW-predicted update differences.

Inputs (all published on pr_assets next to this script, under data/):
  h100-round1/, h100-round2/, b200-precheck/  probe report.json files, one per row
  update-sensitivity.json                     the H100 diagnosis (round 2)
Usage: python plot_pr548_item4.py <data dir> <out.png>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROUND1, ROUND2, B200 = "#eb6834", "#2a78d6", "#1baf7a"  # reference palette slots 2, 1, 3
INK, MUTED, GRID, CRITICAL = "#1f1f1e", "#6b6a63", "#e4e3dd", "#d03b3b"
LIMIT_GIB = 88.0
ROWS = [
    ("megatron", "fp32", "actor"),
    ("megatron", "fp32", "critic"),
    ("megatron", "bf16", "actor"),
    ("megatron", "bf16", "critic"),
    ("deepspeed", "fp32", "actor"),
    ("deepspeed", "fp32", "critic"),
    ("deepspeed", "bf16", "actor"),
    ("deepspeed", "bf16", "critic"),
]
RUNS = [
    ("actor-fp32-impl", "HF implementation control", "#6b6a63", "s"),
    ("megatron-fp32-actor-tp2", "Megatron FP32 actor, TP2", "#2a78d6", "o"),
    ("megatron-fp32-actor-tp4", "Megatron FP32 actor, TP4", "#eb6834", "D"),
]


def load_round(directory):
    rows = {}
    for path in sorted(Path(directory).glob("*.json")):
        report = json.loads(path.read_text())
        key = (report["framework"], report["precision"], report["role"])
        rows[key] = report
    return rows


def memory_panel(ax, data):
    rounds = [
        ("H100 round 1 (1bba6d19)", load_round(data / "h100-round1"), ROUND1),
        ("H100 round 2 (19b62661)", load_round(data / "h100-round2"), ROUND2),
        ("B200 pre-check (19b62661)", load_round(data / "b200-precheck"), B200),
    ]
    width = 0.26
    for offset, (name, rows, color) in enumerate(rounds):
        for i, key in enumerate(ROWS):
            report = rows[key]
            gib = (report.get("peak_allocated_bytes") or 0) / 2**30
            x = i + (offset - 1) * (width + 0.02)
            ok = report["status"] == "probe_complete"
            ax.bar(x, gib, width, color=color if ok else "white", edgecolor=color, linewidth=1.2,
                   hatch=None if ok else "////", label=name if i == 0 else None)
            ax.text(x, gib + 1.2, f"TP{report['tp']}", ha="center", fontsize=6, color=MUTED)
            if not ok:
                reason = "OOM" if gib > LIMIT_GIB else "failed"
                ax.text(x, gib + 4.5, reason, ha="center", fontsize=7, color=CRITICAL, fontweight="bold")
    ax.axhline(LIMIT_GIB, color=CRITICAL, linewidth=1, linestyle="--")
    ax.text(len(ROWS) - 0.5, LIMIT_GIB + 1.5, "88 GiB limit per rank", ha="right", fontsize=8, color=CRITICAL)
    ax.set_xticks(range(len(ROWS)))
    ax.set_xticklabels([f"{f.replace('deepspeed', 'DeepSpeed').replace('megatron', 'Megatron')}\n{p.upper()}\n{r}"
                        for f, p, r in ROWS], fontsize=7.5, color=INK)
    ax.set_ylim(0, 105)
    ax.set_ylabel("peak allocated GiB per rank", color=MUTED, fontsize=9)
    ax.set_title("Trainer memory probe, 8 rows", loc="left", fontsize=11, color=INK, pad=26)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, labelcolor=INK,
              borderaxespad=0.2)
    style(ax, "y")
    ax.text(0, -0.24, "Hatched: the row did not complete. Round 1 DeepSpeed BF16 stopped on an unsupported dtype mix;\n"
            "round 2 runs it at ZeRO-1, and both BF16 actors at TP4.",
            transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")


def update_panel(ax, data):
    report = json.loads((data / "update-sensitivity.json").read_text())
    runs = report["runs"]
    low, high = 1e-8, 2e-6
    ax.plot([low, high], [low, high], color=MUTED, linewidth=1, linestyle=":")
    ax.text(2.2e-8, 3.3e-8, "observed = predicted", fontsize=7.5, color=MUTED, rotation=45, ha="center", va="bottom")
    for key, name, color, marker in RUNS:
        tensors = runs[key]["tensors"]
        xs = [t["max_abs_predicted"] for t in tensors.values()]
        ys = [t["max_abs_observed"] for t in tensors.values()]
        ax.scatter(xs, ys, s=60, color=color, marker=marker, edgecolor="white", linewidth=1.5, label=name, zorder=3)
    flagged = runs["megatron-fp32-actor-tp4"]["tensors"]["model.layers.17.post_attention_layernorm.weight"]
    ax.annotate(
        "layer 17 post-attention norm, TP4: the tensor\nover its FP32 bound (2.29×) — observed 7.75e-7,\n"
        f"AdamW predicts {flagged['max_abs_predicted']:.3g}",
        (flagged["max_abs_predicted"], flagged["max_abs_observed"]),
        xytext=(-200, 8),
        textcoords="offset points",
        fontsize=7.5,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("AdamW-predicted |Δ update| from the gradient difference (FP64)", color=MUTED, fontsize=9)
    ax.set_ylabel("observed |Δ update| against the HF reference", color=MUTED, fontsize=9)
    ax.set_title("Update differences, 1 × H100 diagnosis", loc="left", fontsize=11, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK, borderaxespad=0.6)
    style(ax, "both")
    ax.text(0, -0.17, "Three near-eps norm tensors per run, max over elements.\nHF repeat: 0 observed, 0 predicted (not on a log axis).", transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")


def style(ax, grid):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    data, out = Path(sys.argv[1]), sys.argv[2]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [6, 4]})
    fig.patch.set_facecolor("white")
    memory_panel(axes[0], data)
    update_panel(axes[1], data)
    fig.suptitle("#548 item 4: Qwen3-8B fixed-rollout qualification", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160)


if __name__ == "__main__":
    main()
