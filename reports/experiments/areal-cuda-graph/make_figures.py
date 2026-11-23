# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Regenerate figures/*.png from the run artefacts under runs/.

    python reports/experiments/areal-cuda-graph/make_figures.py

Only reads runs/; needs matplotlib.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIGURES = HERE / "figures"

NATIVE = "#7f7f7f"
BEFORE = "#ff7f0e"
ROUND1 = "#1f77b4"
COMPONENTS = (
    ("rollout", "rollout", "#9ecae1"),
    ("recompute_logp", "recompute logp", "#fdae6b"),
    ("train_step", "actor train", "#e6550d"),
    ("update_weights", "weight update", "#74c476"),
)


def _json(path: Path):
    return json.loads(path.read_text())


def _perf(stage: str, arm: str) -> dict | None:
    path = RUNS / stage / "perf.json"
    return _json(path).get(arm) if path.is_file() else None


def _stats(stage: str, arm: str) -> list[dict]:
    path = RUNS / stage / arm / "stats.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line)["stats"] for line in path.read_text().splitlines() if line.strip()]


def _save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / name)


def _stacked(ax, arms: list[tuple[str, dict]], components, title: str) -> None:
    labels = [label for label, _ in arms]
    bottoms = [0.0] * len(arms)
    for key, name, color in components:
        values = [perf["means"].get(key) or 0.0 for _, perf in arms]
        ax.bar(labels, values, bottom=bottoms, color=color, label=name, width=0.6)
        bottoms = [b + v for b, v in zip(bottoms, values, strict=True)]
    other = [max(perf["means"]["step_seconds"] - b, 0.0)
             for (_, perf), b in zip(arms, bottoms, strict=True)]
    ax.bar(labels, other, bottom=bottoms, color="#d9d9d9", label="other", width=0.6)
    for i, (_, perf) in enumerate(arms):
        total = perf["means"]["step_seconds"]
        ax.text(i, total + 1.5, f"{total:.1f} s", ha="center", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("seconds per step (steady-state mean)")
    ax.tick_params(axis="x", labelsize=8)


def step_breakdown() -> None:
    sync = [
        ("native", _perf("baseline/S3-sync-graph", "native")),
        ("RL-Kernel\nbefore", _perf("baseline/S3-sync-graph", "rlkernel")),
        ("RL-Kernel\nround 1", _perf("round1/S3-sync-graph", "rlkernel")),
    ]
    asyn = [
        ("native\n(baseline run)", _perf("baseline/S4-async-graph", "native")),
        ("RL-Kernel\nbefore", _perf("baseline/S4-async-graph", "rlkernel")),
        ("native\n(round 1 run)", _perf("round1/S4-async-graph", "native")),
        ("RL-Kernel\nround 1", _perf("round1/S4-async-graph", "rlkernel")),
    ]
    sync = [item for item in sync if item[1]]
    asyn = [item for item in asyn if item[1]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    _stacked(axes[0], sync, COMPONENTS, "sync, vLLM CUDA graphs")
    _stacked(axes[1], asyn, COMPONENTS[1:], "async (max off-policyness 1), CUDA graphs\n"
             "rollout overlaps training")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Qwen3-8B, 2 x 4 H100: time per RL step", y=1.09)
    _save(fig, "step_breakdown.png")


def per_step_times() -> None:
    arms = [
        ("native", _perf("baseline/S3-sync-graph", "native"), NATIVE),
        ("RL-Kernel before", _perf("baseline/S3-sync-graph", "rlkernel"), BEFORE),
        ("RL-Kernel round 1", _perf("round1/S3-sync-graph", "rlkernel"), ROUND1),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, title in ((axes[0], "timeperf/train_step", "actor train"),
                           (axes[1], "timeperf/recompute_logp", "recompute logp")):
        for label, perf, color in arms:
            if not perf:
                continue
            steps = [row["global_step"] for row in perf["per_step"]]
            values = [row[key] for row in perf["per_step"]]
            ax.plot(steps, values, marker="o", color=color, label=label)
        ax.axvspan(-0.5, 0.5, color="#eeeeee", zorder=0)
        ax.text(0, ax.get_ylim()[1] * 0.97, "warm-up\n(skipped)", ha="center", va="top", fontsize=8)
        ax.set_title(f"{title}, sync + CUDA graphs")
        ax.set_xlabel("step")
        ax.set_ylabel("seconds")
        ax.set_xticks(range(0, 10))
    axes[0].legend()
    _save(fig, "per_step_times.png")


def _fraction(stats: list[dict], num: str, den: str) -> list[float]:
    prefix = "compute_logp/consistency/"
    return [100.0 * row.get(prefix + num, 0.0) / row[prefix + den] if row.get(prefix + den)
            else 0.0 for row in stats]


def consistency_sync() -> None:
    native = _stats("round1/S3-sync-observe", "native")
    rlk = _stats("round1/S3-sync-graph", "rlkernel")
    native_r3 = native or _stats("baseline/S3-sync-graph", "native")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    top = 1.0
    if native:
        values = _fraction(native, "mismatch_tokens", "n_valid_tokens")
        top = max(values)
        axes[0].plot(range(len(native)), values, marker="o", color=NATIVE,
                     label="native (observe)")
        for i, value in enumerate(values):
            axes[0].text(i, value + top * 0.03, f"{value:.0f}%", ha="center", fontsize=7)
    axes[0].plot(range(len(rlk)), _fraction(rlk, "mismatch_tokens", "n_valid_tokens"),
                 marker="s", color=ROUND1, label="RL-Kernel round 1 (0 every step)")
    axes[0].set_title("tokens whose train and rollout logp differ\n(bitwise, delta != 0)")
    axes[0].set_ylabel("% of compared tokens")
    axes[0].set_ylim(-top * 0.05, top * 1.15)
    key = "compute_logp/r3/rollout_train_logp_abs_diff/"
    for stats, label, color, marker in ((native_r3, "native", NATIVE, "o"),
                                        (rlk, "RL-Kernel round 1", ROUND1, "s")):
        axes[1].plot(range(len(stats)), [row[key + "max"] for row in stats], marker=marker,
                     color=color, label=f"{label}: max")
        axes[1].plot(range(len(stats)), [row[key + "avg"] for row in stats], marker=marker,
                     color=color, linestyle="--", label=f"{label}: mean")
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[1].set_title("|train logp - rollout logp| per step")
    axes[1].set_ylabel("absolute difference (symlog)")
    for ax in axes:
        ax.set_xlabel("step")
        ax.set_xticks(range(0, 10))
    axes[0].legend(fontsize=8, loc="center right")
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.suptitle("sync + CUDA graphs: train/rollout consistency", y=1.06)
    _save(fig, "consistency_sync.png")


def consistency_async() -> None:
    arms = [
        ("native\n(baseline run)", _stats("baseline/S4-async-graph", "native")),
        ("RL-Kernel\nbefore", _stats("baseline/S4-async-graph", "rlkernel")),
        ("native\n(round 1 run)", _stats("round1/S4-async-graph", "native")),
        ("RL-Kernel\nround 1", _stats("round1/S4-async-graph", "rlkernel")),
    ]
    arms = [(label, stats) for label, stats in arms if stats]
    prefix = "compute_logp/consistency/"
    parts = (
        ("same-version, bitwise equal", "#31a354", lambda t: t["n_same_version_tokens"]
         - t["same_version_mismatch_tokens"]),
        ("same-version, differs", "#de2d26", lambda t: t["same_version_mismatch_tokens"]),
        ("stale (older weights), equal", "#a1d99b", lambda t: t["n_stale_tokens"]
         - t["stale_mismatch_tokens"]),
        ("stale (older weights), differs", "#fc9272", lambda t: t["stale_mismatch_tokens"]),
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [label for label, _ in arms]
    bottoms = [0.0] * len(arms)
    totals = []
    for _, stats in arms:
        total = {key: sum(row.get(prefix + key, 0.0) for row in stats) for key in (
            "n_same_version_tokens", "same_version_mismatch_tokens", "n_stale_tokens",
            "stale_mismatch_tokens", "n_valid_tokens")}
        totals.append(total)
    for name, color, getter in parts:
        values = [100.0 * getter(t) / t["n_valid_tokens"] for t in totals]
        ax.bar(labels, values, bottom=bottoms, color=color, label=name, width=0.6)
        bottoms = [b + v for b, v in zip(bottoms, values, strict=True)]
    for i, t in enumerate(totals):
        same = t["n_same_version_tokens"]
        bad = t["same_version_mismatch_tokens"]
        ax.text(i, 101, f"same-version\n{int(bad):,} / {int(same):,} differ", ha="center",
                fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("% of compared tokens (10 steps)")
    ax.set_title("async + CUDA graphs: only same-version tokens must match bitwise")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _save(fig, "consistency_async.png")


RANGES = (("attention", "attention"), ("rope", "RoPE"), ("mlp", "MLP"), ("lm_head", "LM head"),
          ("det_collective.all_reduce", "det. all-reduce"))


def _profile(stage: str, arm: str, kind: str) -> dict | None:
    files = sorted((RUNS / stage / arm / "profile").glob(f"{kind}-v*-*.json"))
    return _json(files[0]) if files else None


def attribution() -> None:
    arms = [("native", "round1/S3-profile", "native", NATIVE),
            ("RL-Kernel before", "round1/S3-profile-base", "rlkernel", BEFORE),
            ("RL-Kernel round 1", "round1/S3-profile", "rlkernel", ROUND1)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, kind, title in ((axes[0], "forward_batch", "recompute logp (forward_batch)"),
                            (axes[1], "train_batch", "actor train (train_batch)")):
        names = ["GPU kernels\n(total)"] + [label for _, label in RANGES]
        width = 0.27
        for j, (label, stage, arm, color) in enumerate(arms):
            data = _profile(stage, arm, kind)
            if data is None:
                continue
            values = [data["device_kernel_s"]] + [
                data["ranges"].get(key, {}).get("device_s", 0.0) for key, _ in RANGES]
            xs = [i + (j - 1) * width for i in range(len(names))]
            ax.bar(xs, values, width=width, color=color, label=label)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("GPU seconds, rank 0, one profiled step")
    axes[0].legend()
    fig.suptitle("torch.profiler attribution (forward ranges; CUPTI inflates many small kernels)",
                 y=1.02)
    _save(fig, "attribution.png")


def vllm_probe() -> None:
    probes = RUNS / "round1" / "probes"
    capture = [
        ("native", _perf("baseline/S3-sync-graph", "native"), NATIVE),
        ("RL-Kernel before", _perf("baseline/S3-sync-graph", "rlkernel"), BEFORE),
        ("RL-Kernel round 1", _perf("round1/S3-sync-graph", "rlkernel"), ROUND1),
    ]
    decode = [("native", "native", NATIVE), ("RL-Kernel before", "rlkernel-before", BEFORE),
              ("RL-Kernel round 1", "rlkernel-round1", ROUND1)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = [label for label, _, _ in capture]
    axes[0].bar(labels, [perf["vllm"]["capture_seconds"] for _, perf, _ in capture],
                color=[color for _, _, color in capture])
    axes[0].set_title("vLLM graph capture, sizes 1..128\n(TP4, from rollout.log)")
    axes[0].set_ylabel("seconds")
    for ax, mode in ((axes[1], "graph"), (axes[2], "eager")):
        values = [_json(probes / f"{name}-{mode}.json")["decode_step_ms"] for _, name, _ in decode]
        ax.bar([label for label, _, _ in decode], values, color=[c for _, _, c in decode])
        for i, value in enumerate(values):
            ax.text(i, value * 1.01, f"{value:.1f}", ha="center", fontsize=9)
        ax.set_title(f"decode step, {mode}\n(single GPU, TP1, batch 128)")
        ax.set_ylabel("ms per step")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    _save(fig, "vllm_probe.png")


def main() -> None:
    step_breakdown()
    per_step_times()
    consistency_sync()
    consistency_async()
    attribution()
    vllm_probe()


if __name__ == "__main__":
    main()
