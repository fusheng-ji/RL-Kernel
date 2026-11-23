"""Before/after outcome grid for #740 (no timing in this PR; the comparison is the outcome).

Every cell is from a file: H100 results 55c25dca / 76079edc (before), #740 Results at 8284ce2b (after).
Usage: python plot_followup_outcomes.py <out.png>
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

GOOD, CRITICAL = "#0ca30c", "#d03b3b"
INK, MUTED, CELL = "#1f1f1e", "#6b6a63", "#f4f3ef"

ROWS = [
    (
        "User site supplies a different\napache-tvm-ffi (H100, item 4)",
        (False, "rollout smoke aborts while FlashInfer loads:\n6 / 6 probe cases rc 1 (`__ffi_repr__`)"),
        (True, "refused up front: identity and preflight exit 1,\n`environment_mismatch`, before any GPU work"),
    ),
    (
        "`l2` when the caller does not set\nCUDA_DEVICE_MAX_CONNECTIONS\n(H100, item 5)",
        (False, "`l2` fails in Megatron `validate_args`\n(#543 and #544 chains)"),
        (True, "`l2` passes: the stage environment sets it;\n`bi` 97 / 97 afterwards"),
    ),
    (
        "Four-GPU CI step where\nevery check skips (item 1)",
        (False, "step exits 0: an all-skipped pytest run\ncounts as success"),
        (True, "step fails: any skip, or no pass, is non-zero\n(provider step's guard)"),
    ),
    (
        "Mandatory case that xpasses\n(item 2)",
        (False, "recorded as `skipped`"),
        (True, "recorded as `xpass` (verdict unchanged:\nthe run fails)"),
    ),
]


def cell(ax, x, y, w, h, ok, text):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=CELL, ec="none"))
    ax.add_patch(FancyBboxPatch((x, y), 0.07, h, boxstyle="square,pad=0", fc=GOOD if ok else CRITICAL, ec="none"))
    label = "✓ passes / refuses cleanly" if ok else "✗ wrong outcome"
    ax.text(x + 0.2, y + h - 0.16, label, fontsize=9, fontweight="bold", color=INK, va="top")
    ax.text(x + 0.2, y + h - 0.5, text.replace("`", ""), fontsize=8.5, color=INK, va="top", linespacing=1.4)


def main():
    out = sys.argv[1]
    row_h, gap = 1.25, 0.14
    height = len(ROWS) * (row_h + gap) + 0.55
    fig, ax = plt.subplots(figsize=(12.5, height * 0.78))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, height)
    ax.axis("off")
    top = height - 0.45
    ax.text(3.3, top + 0.2, "Before: the stack's code (H100: #543 / #544 heads)", fontsize=10, color=MUTED)
    ax.text(8.2, top + 0.2, "After: #740 (8284ce2b), 4 × H100 and 4 × B200", fontsize=10, color=MUTED)
    for i, (name, before, after) in enumerate(ROWS):
        y = top - (i + 1) * (row_h + gap) + gap
        ax.text(0, y + row_h / 2, name.replace("`", ""), fontsize=9, color=INK, va="center", linespacing=1.4)
        cell(ax, 3.3, y, 4.75, row_h, *before)
        cell(ax, 8.2, y, 4.75, row_h, *after)
    fig.suptitle(
        "Qwen3-Next follow-up: what each fix changes, before and after",
        x=0.01,
        ha="left",
        fontsize=13,
        color=INK,
    )
    fig.text(
        0.01,
        0.01,
        "Before, H100 rows: the 2026-11-18 and 2026-11-19 H100 runs of the stack. Items 1 and 2: the step's script with stubbed pytest"
        " output, and the case recorder's tests.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(out, dpi=160)


if __name__ == "__main__":
    main()
