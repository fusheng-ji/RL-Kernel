"""Before/after chart for #744 from the evidence files in the release ZIP.

Usage: python plot_gdn_one_launch.py <unzipped qwen3-next-gdn-one-launch dir> <out.png>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BEFORE, AFTER = "#eb6834", "#2a78d6"  # reference palette slots 2 and 1
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def probe_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def label_for(lengths):
    if len(lengths) == 1:
        return f"{lengths[0]} tokens"
    if len(set(lengths)) == 1:
        return f"{len(lengths)} × {lengths[0]}"
    return "4 mixed\n(64-1500)"


def bars(ax, labels, before, after, names, title, unit, note):
    x = range(len(labels))
    width = 0.38
    b = ax.bar([i - width / 2 - 0.01 for i in x], before, width, color=BEFORE, label=names[0])
    a = ax.bar([i + width / 2 + 0.01 for i in x], after, width, color=AFTER, label=names[1])
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_ylabel(unit, color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=26)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, before_value, after_value in zip(a, before, after):
        ax.annotate(
            f"{before_value / after_value:.0f}× faster",
            (bar.get_x() + bar.get_width() / 2, after_value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=INK, borderaxespad=0.2)
    ax.text(0, -0.2, note, transform=ax.transAxes, fontsize=8, color=MUTED, va="top")
    return b, a


def main():
    root, out = Path(sys.argv[1]), sys.argv[2]
    h100 = probe_rows(root / "h100-213a9bcd/A/probe.jsonl")
    if not all(r["output_equal"] and r["state_equal"] and r["output_max_abs"] == 0.0 for r in h100):
        raise SystemExit("probe rows are not all bit-identical")
    before = json.loads((root / "b200-before-fix-f5100979/bench/report.json").read_text())
    after = json.loads((root / "b200-tp4-213a9bcd/bench/report.json").read_text())
    shapes = [
        ("certified", "p64-b1-r16", "64, b1"),
        ("certified", "p256-b1-r16", "256, b1"),
        ("certified", "p1024-b1-r16", "1024, b1"),
        ("long", "p512-b1-r512", "512, b1"),
        ("long", "p512-b16-r512", "512, b16"),
        ("long", "p512-b64-r512", "512, b64"),
    ]

    def prefill(report, workload, shape):
        return report["phases"]["rollout"]["results"]["strict"][workload][shape]["prefill_s"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [4, 6]})
    fig.patch.set_facecolor("white")
    bars(
        axes[0],
        [label_for(r["lengths"]) for r in h100],
        [r["decode_loop_ms"] for r in h100],
        [r["one_launch_ms"] for r in h100],
        ("per-token decode loop (before)", "one launch (after)"),
        "One GDN layer, 1 × H100",
        "milliseconds (log scale)",
        "4 / 4 cases bit-identical to the loop: output, final state, cache slots (max |Δ| = 0).",
    )
    bars(
        axes[1],
        [label for _, _, label in shapes],
        [prefill(before, w, s) for w, s, _ in shapes],
        [prefill(after, w, s) for w, s, _ in shapes],
        (f"before ({before['rl_kernel_commit'][:8]})", f"after ({after['rl_kernel_commit'][:8]})"),
        "Strict rollout prefill (time to first token), TP4 on 4 × B200",
        "seconds (log scale)",
        "Prompt tokens, batch. Medians over 2 rounds. After: TP4 chain l2 + bi 97 / 97 passed at the same commit.",
    )
    fig.suptitle(
        "Qwen3-Next strict GDN recurrence: per-token loop → one launch per layer",
        x=0.01,
        ha="left",
        fontsize=13,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160)


if __name__ == "__main__":
    main()
