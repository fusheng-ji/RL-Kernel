#!/usr/bin/env python3
"""Charts for the GEMM-row PRs from the staged evidence (pr_assets layout under --root)."""
import json, re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S1, S2 = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ROOT = Path(sys.argv[1]); P1 = ROOT / "pending/h3-strict-gemm-rows"; P2 = ROOT / "pending/h3-parallel-v2-gemm-rows"
PI = ROOT / "pending/h3-gemm-rows-integration"
NODES = {"o_proj": "O (h3_attention_o_gemm)", "ffn_gate_up": "gate/up (h3_ffn_gate_up_gemm)", "ffn_down": "down (h3_ffn_down_gemm)"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)


def bars(ax, labels, a, b, names, fmt):
    import numpy as np
    x = np.arange(len(labels)); w = 0.38
    for off, vals, c, n in ((-w / 2, a, S1, names[0]), (w / 2, b, S2, names[1])):
        r = ax.bar(x + off, vals, w - 0.03, color=c, label=n)
        for rect, v in zip(r, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), fmt(v), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, labels, fontsize=9, color=INK)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white", metadata={"Software": None}); plt.close(fig)


def pr1():
    rep = json.loads((P1 / "h100/H1/block-replay.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained", sharey=True)
    for ax, (node, title) in zip(axes, NODES.items()):
        seqs, cand, prov = [], [], []
        for c in rep["forward_cases"]:
            n = next(n for n in c["nodes"] if n["node"] == node)
            seqs.append(f"S={c['seq_len']}"); cand.append(n["candidate_rel_err_vs_golden"] * 1e3); prov.append(n["provider_rel_err_vs_golden"] * 1e3)
        bars(ax, seqs, cand, prov, ("RL-Kernel row operator", "Diffusers (provider)"), lambda v: f"{v:.2f}")
        ax.set_title(title, fontsize=10, color=INK, loc="left"); style(ax)
    axes[0].set_ylabel("relative L2 error vs FP32 golden (×1e-3)", fontsize=9, color=INK)
    fig.suptitle("WS1 block forward on H100: the three GEMM rows on their operators", fontsize=12, color=INK, x=0.0, ha="left")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=9, loc="outside upper right", ncols=2)
    save(fig, P1 / "h3-strict-gemm-rows-h100-error.png")


def pr2_crosscheck():
    rep = json.loads((P2 / "h100/H2/crosscheck.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    names = {"h3_attention_o_gemm": ("O forward", "forward"), "h3_ffn_down_gemm": ("down forward", "forward"), "h3_ffn_gate_up_gemm": ("gate/up dX", "dx")}
    for ax, (row, (title, piece)) in zip(axes, names.items()):
        rs = [r for r in rep["results"] if r["row"] == row]
        tps = ["1", "2", "4", "8"]
        import numpy as np
        x = np.arange(len(tps)); w = 0.8 / len(rs)
        palette = [S1, S2, "#1f9e89"]
        for i, r in enumerate(rs):
            vals = [r["tp"][t]["reported"][piece]["max_abs"] for t in tps]
            frac = [r["tp"][t]["reported"][piece]["differing"] for t in tps]
            ax.bar(x - 0.4 + w * (i + 0.5), vals, w - 0.02, color=palette[i], label=f"{r['rows']} rows")
        ax.set_xticks(x, [f"TP{t}" for t in tps], fontsize=9, color=INK)
        ax.set_title(title + " (8-unit tree vs the operator)", fontsize=10, color=INK, loc="left"); style(ax)
    axes[0].set_ylabel("max |Δ| (BF16 output)", fontsize=9, color=INK)
    fig.suptitle("h3-parallel-v2 vs the rows' TP=1 operators on H100 (gate/up forward, O/down dX, all dW: bitwise)", fontsize=12, color=INK, x=0.0, ha="left")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=9, loc="outside upper right", ncols=3)
    save(fig, P2 / "h3-parallel-v2-crosscheck-h100.png")


def g6():
    def avg(path):
        seen = {}
        for line in (path / "gate-lines.txt").read_text().splitlines():
            m = re.search(r"step:(\d+)", line)
            if m:
                seen[int(m.group(1))] = {k: float(re.search(re.escape(k) + r":([0-9.e-]+)", line).group(1)) for k in
                                         ("timing_s/gen", "timing_s/old_log_prob", "timing_s/update_actor", "timing_s/step")}
        st = [seen[s] for s in sorted(seen) if s > 1]
        return {k: sum(r[k] for r in st) / len(st) for k in st[0]}
    before, after = avg(PI / "b200/g6-rr-before"), avg(PI / "b200/g6-rr-after")
    keys = ["timing_s/gen", "timing_s/old_log_prob", "timing_s/update_actor", "timing_s/step"]
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    bars(ax, ["rollout (gen)", "old log-prob", "actor update", "step"], [before[k] for k in keys], [after[k] for k in keys],
         ("h3-parallel-v1 GEMMs", "v2: the rows' fixed FP32-tile GEMM"), lambda v: f"{v:.1f}")
    style(ax); ax.set_ylabel("seconds (mean of steps 2–8)", fontsize=9, color=INK)
    fig.suptitle("G6 R/R, TP4×CP2 on 8×B200: step time", fontsize=12, color=INK, x=0.0, ha="left")
    fig.legend(frameon=False, fontsize=9, loc="outside upper right", ncols=2)
    save(fig, PI / "h3-gemm-rows-g6-step-time-b200.png")
    (PI / "g6-step-time.json").write_text(json.dumps({"before": before, "after": after}, indent=1) + "\n")


def bench():
    f = P2 / "b200/bench/bench.json"
    if not f.exists():
        return
    rep = json.loads(f.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained", sharey=False)
    for ax, row in zip(axes, ("h3_attention_o_gemm", "h3_ffn_gate_up_gemm", "h3_ffn_down_gemm")):
        rs = [r for r in rep["gemm"] if r["row"] == row and r["m"] == 4296]
        tps = sorted({r["tp"] for r in rs})
        tot = lambda k, t: next(r for r in rs if r["kernel"] == k and r["tp"] == t)
        v1 = [sum(tot("v1_h3_projection", t)[p] for p in ("forward", "dx", "dw")) for t in tps]
        v2 = [sum(tot("v2_fixed_fp32", t)[p] for p in ("forward", "dx", "dw")) for t in tps]
        bars(ax, [f"TP{t}" for t in tps], v1, v2, ("v1: h3_projection.gemm", "v2: fixed FP32-tile GEMM"), lambda v: f"{v:.2f}")
        ax.set_title(row + " (4296 rows)", fontsize=10, color=INK, loc="left"); style(ax)
    axes[0].set_ylabel("ms per rank: forward + dX + dW", fontsize=9, color=INK)
    fig.suptitle(f"Per-rank GEMM time at the TP shard shapes, one B200", fontsize=12, color=INK, x=0.0, ha="left")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=9, loc="outside upper right", ncols=2)
    save(fig, P2 / "h3-parallel-v2-gemm-time-b200.png")


pr1(); pr2_crosscheck(); g6(); bench()
print("ok")
