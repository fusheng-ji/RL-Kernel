"""Charts for PR A from the evidence reports. One PNG per Results table.

    python plot_pra.py <evidence dir> <out dir>
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OURS, REF = "#2a78d6", "#eb6834"  # categorical slots 1 and 2: RL-Kernel, the reference implementation
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
                     "ytick.color": TEXT2, "text.color": TEXT, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def base(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def legend(ax):
    ax.legend(frameon=False, loc="lower right", fontsize=9)


def main():
    ev, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    # 1. verl-omni equality
    vc = json.loads((ev / "verl-omni-compare.json").read_text())
    names = ["prev_sample", "prev_sample_mean", "std_dev_t", "sqrt_dt", "log_prob"]
    eq = [int(vc["bitwise_equal"][n].split("/")[0]) for n in names]; tot = vc["cases"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = range(len(names))[::-1]
    ax.barh(list(y), eq, height=0.45, color=OURS)
    for yi, e, n in zip(y, eq, names):
        note = f"{e} / {tot}" + ("  (fixed-order mean, by design; max |diff| %.1e)" % vc["log_prob_max_abs"] if n == "log_prob" else "")
        ax.text(e + 6, yi, note, va="center", fontsize=9, color=TEXT)
    ax.set_yticks(list(y), names); ax.set_xlim(0, tot * 1.75); ax.set_xticks([0, tot // 4, tot // 2, 3 * tot // 4, tot]); ax.set_xlabel("cases bitwise equal to verl-omni (of %d)" % tot)
    base(ax, "h3_sample_previous_step vs verl-omni sample_h3_transition (CPS)",
         f"verl-omni {vc['verl_omni_commit']}, Diffusers {vc['diffusers']}, torch {vc['torch']}, {vc['device']}; RL-Kernel {vc['rl_kernel_commit']}")
    fig.tight_layout(); fig.savefig(out / "pra-verl-omni-equality.png", dpi=160); plt.close(fig)
    # 2. log_prob batch dependence
    lb = json.loads((ev / "logprob-batch.json").read_text())
    keys = [(r["modality"], r["step"]) for r in lb["rows"] if r["implementation"] == "rl_kernel"]
    get = lambda impl, k: next(r for r in lb["rows"] if r["implementation"] == impl and (r["modality"], r["step"]) == k)
    fig, ax = plt.subplots(figsize=(8, 3.8)); h = 0.36
    for i, k in enumerate(keys[::-1]):
        a, b = get("verl_omni", k), get("rl_kernel", k)
        ax.barh(i + h / 2, a["differ_from_alone"], height=h, color=REF, label="verl-omni (Tensor.mean)" if i == 0 else None)
        ax.barh(i - h / 2, b["differ_from_alone"], height=h, color=OURS, label="RL-Kernel (fixed-order mean)" if i == 0 else None)
        ax.text(a["differ_from_alone"] + 0.3, i + h / 2, f"{a['differ_from_alone']} / {a['comparisons']}", va="center", fontsize=9)
        ax.text(0.3, i - h / 2, "= 0, bitwise" if b["differ_from_alone"] == 0 else str(b["differ_from_alone"]), va="center", fontsize=9)
    ax.set_yticks(range(len(keys)), [f"{m}, step {s}" for m, s in keys[::-1]]); ax.set_xlim(0, 24)
    ax.set_xlabel("samples whose log_prob differs from their batch-of-1 value (of 24)")
    base(ax, "A sample's log_prob scored in batches of 2, 4 and 8 vs alone", f"scoring path, CPS, H3 latent sizes; torch {lb['torch']}, {lb['device']}")
    legend(ax); fig.tight_layout(); fig.savefig(out / "pra-logprob-batch.png", dpi=160); plt.close(fig)
    # 3. norm across layouts
    npart = json.loads((ev / "norm-partition.json").read_text())
    real = [l for n in (2, 4, 8) for l in json.loads((ev / f"multirank-{n}.json").read_text())["layouts"]]
    groups = [("one process,\n16 partitions / orders", npart["torch_bitwise_equal"], npart["exact_bitwise_equal"], npart["layouts"]),
              ("multi-GPU,\n" + " / ".join(l["layout"] for l in real), sum(l["torch_norm_equal_all_ranks"] for l in real), sum(l["exact_norm_equal_all_ranks"] for l in real), len(real))]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (lab, t, e, n) in enumerate(groups[::-1]):
        ax.barh(i + h / 2, t / n, height=h, color=REF, label="torch (get_total_norm; DTensor clip_grad_norm_)" if i == 0 else None)
        ax.barh(i - h / 2, e / n, height=h, color=OURS, label="RL-Kernel exact_norm" if i == 0 else None)
        ax.text(t / n + 0.01, i + h / 2, f"{t} / {n}", va="center", fontsize=9); ax.text(e / n + 0.01, i - h / 2, f"{e} / {n}", va="center", fontsize=9)
    ax.set_yticks(range(len(groups)), [g[0] for g in groups[::-1]]); ax.set_xlim(0, 1.25); ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
    ax.set_xlabel("layouts whose norm equals the one-GPU norm, bitwise")
    base(ax, "Gradient norm: same LoRA gradients, other layouts", f"700 FP32 tensors, {npart['elements']/1e6:.1f} M elements; torch {npart['torch']}, B200")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=9); fig.tight_layout(); fig.savefig(out / "pra-norm-layouts.png", dpi=160); plt.close(fig)
    # 4. timing
    tm = json.loads((ev / "timing.json").read_text())["rows"]
    rows = [("norm", "norm_torch", "norm_exact"), ("norm + clip", "clip_torch", "clip_exact"),
            ("transition, video, sampling", "transition_video_sample_verl_omni", "transition_video_sample_rl_kernel"),
            ("transition, video, scoring", "transition_video_score_verl_omni", "transition_video_score_rl_kernel"),
            ("transition, audio, sampling", "transition_audio_sample_verl_omni", "transition_audio_sample_rl_kernel"),
            ("transition, audio, scoring", "transition_audio_score_verl_omni", "transition_audio_score_rl_kernel")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (lab, r, o) in enumerate(rows[::-1]):
        ax.barh(i + h / 2, tm[r]["median_ms"], height=h, color=REF, label="torch / verl-omni" if i == 0 else None)
        ax.barh(i - h / 2, tm[o]["median_ms"], height=h, color=OURS, label="RL-Kernel" if i == 0 else None)
        ax.text(tm[r]["median_ms"] * 1.08, i + h / 2, f"{tm[r]['median_ms']:.2f}", va="center", fontsize=9)
        ax.text(tm[o]["median_ms"] * 1.08, i - h / 2, f"{tm[o]['median_ms']:.2f}", va="center", fontsize=9)
    ax.set_xscale("log"); ax.set_xlim(0.1, max(v["median_ms"] for v in tm.values()) * 4); ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set_xlabel("median ms per call (log scale, 30 calls after 5 warm-up)")
    base(ax, "Cost per call, one B200 with no other job", "norm: 700 LoRA gradients; transition: video (2, 24, 32, 48, 84), audio (2, 207, 32)")
    legend(ax); fig.tight_layout(); fig.savefig(out / "pra-timing.png", dpi=160); plt.close(fig)
    print("charts written")


if __name__ == "__main__":
    main()
