#!/usr/bin/env python
"""Charts (PNG) and their matching tables (tables.md) for the RFC #738 PRs.

    python plot_charts.py --data data-b200-104eb599f.json --evidence <job results dir> \
        --probe-b200 <probe-b200.json> --probe-h100 <probe-h100.json> \
        --text-b200 <text-evidence-b200.json> --text-h100 <text-evidence-h100.json> --out <dir>

Palette: the validated default categorical order (blue, orange, aqua, yellow) on the light
surface; every chart is direct-labelled and has a table, since aqua/yellow sit below 3:1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
    }
)

TABLES: list[str] = []


def table(title: str, header: list[str], rows: list[list[object]]) -> None:
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    TABLES.append("\n".join(lines) + "\n")


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    print("wrote", name)


def _fmt_params(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return str(int(n))


# --- per-PR charts ---------------------------------------------------------------------------


def fingerprint(d, out):
    g = d["fingerprint"]["groups"]
    names, vals = list(g)[::-1], [g[k] for k in list(g)[::-1]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(names, vals, color=S1, height=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (log scale)")
    for y, v in enumerate(vals):
        ax.text(v * 1.15, y, _fmt_params(v), va="center", color=INK2, fontsize=9)
    ax.set_xlim(right=max(vals) * 12)
    ax.set_title(
        f"MiniMax-H3-Image transformer: {d['fingerprint']['tensors']} tensors, "
        f"{_fmt_params(d['fingerprint']['parameters'])} parameters"
    )
    save(fig, out, "h3-image-fingerprint-params.png")
    table(
        "Parameters by top-level module (analytic table = released headers)",
        ["module", "parameters"],
        [[k, f"{v:,}"] for k, v in g.items()],
    )


def position_grid(d, out):
    rows = d["position_grid"]["rows"]
    eq = [r for r in rows if r["rows_fp64"] == 0]
    f64 = [r for r in rows if r["rows_fp64"] > 0 and r["rows_fp32"] == 0]
    f32 = [r for r in rows if r["rows_fp32"] > 0]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for group, color, label, size in (
        (eq, GRID, f"Torch linspace equal ({len(eq)})", 10),
        (f64, S1, f"differs in FP64 only ({len(f64)})", 10),
        (f32, S2, f"difference survives the FP32 cast ({len(f32)})", 30),
    ):
        ax.scatter([r["w"] for r in group], [r["h"] for r in group], s=size, color=color, label=label)
    ax.set_xlabel("latent width")
    ax.set_ylabel("latent height")
    ax.set_title("Probe I8: Torch linspace vs the released NumPy grid, 1,419 T2I canvases")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9)
    save(fig, out, "h3-image-position-grid-i8.png")
    table(
        "Probe I8 (Torch linspace) vs the released grid",
        ["canvases", "differ in FP64", "differ after FP32 cast", "max abs FP64"],
        [[len(rows), len(f64) + len(f32), len(f32), f"{d['position_grid']['max_abs_fp64']:.2e}"]],
    )


def canvas(d, out):
    grid = d["canvas"]["grid"]
    sizes = sorted({g["h"] for g in grid})
    m = np.zeros((len(sizes), len(sizes)))
    for g in grid:
        m[sizes.index(g["h"]), sizes.index(g["w"])] = g["area_ratio"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(np.log2(m), cmap="Blues", origin="lower")
    ax.set_xticks(range(len(sizes)), sizes, rotation=45)
    ax.set_yticks(range(len(sizes)), sizes)
    ax.grid(False)
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            v = m[i, j]
            ax.text(j, i, f"{v:.1f}" if v >= 1.05 else "1", ha="center", va="center", fontsize=7,
                    color="white" if v > 6 else INK)
    ax.set_xlabel("requested width")
    ax.set_ylabel("requested height")
    ax.set_title("Canvas area / requested area (min area 2048², 32 px rounding)")
    fig.colorbar(im, ax=ax, label="log2 area ratio", shrink=0.8)
    save(fig, out, "h3-image-canvas-plan.png")
    table(
        "Canvas plans (requested -> canvas)",
        ["requested", "canvas", "scale", "resize back"],
        [[f"{g['h']}x{g['w']}", "x".join(map(str, g["canvas"])), f"{g['scale']:.4f}", g["resize"]]
         for g in grid if g["h"] == g["w"] or g["h"] in (768, 4096)][:16],
    )


def packed_layout(pb, ph, out):
    cases = [("float64", "fp64"), ("float32", "fp32")]
    b = [pb["permutation_drift"][k]["max_abs"] for k, _ in cases]
    h = [ph["permutation_drift"][k]["max_abs"] for k, _ in cases]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(x - 0.18, b, 0.34, color=S1, label="B200")
    ax.bar(x + 0.18, h, 0.34, color=S2, label="H100")
    ax.set_yscale("log")
    ax.set_xticks(x, ["float64", "float32"])
    ax.set_ylabel("max |output diff| (log)")
    for xs, vals in ((x - 0.18, b), (x + 0.18, h)):
        for xi, v in zip(xs, vals):
            ax.text(xi, v * 1.3, f"{v:.1e}", ha="center", fontsize=9, color=INK2)
    ax.set_title("Tiny transformer, 20 consistent permutations: SDPA order drift")
    ax.legend()
    save(fig, out, "h3-image-packed-layout-permutation.png")
    table(
        "Consistent-permutation drift (row identity preserved; repeat bitwise on both GPUs)",
        ["dtype", "B200 max abs", "H100 max abs", "bitwise permutations (of 20)"],
        [[n, f"{bv:.2e}", f"{hv:.2e}", f"{pb['permutation_drift'][k]['bitwise_equal']} / "
          f"{ph['permutation_drift'][k]['bitwise_equal']}"] for (k, n), bv, hv in zip(cases, b, h)],
    )


def patch_pack(d, out):
    c = d["patch_pack"]["cases"]
    labels = [f"{x['latent'][0]}x{x['latent'][1]}\n{x['rows']:,} rows" for x in c]
    x = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    p = [v["pack_ms"] * 1e3 for v in c]
    u = [v["unpack_ms"] * 1e3 for v in c]
    ax.bar(x - 0.18, p, 0.34, color=S1, label="pack")
    ax.bar(x + 0.18, u, 0.34, color=S2, label="unpack")
    for xs, vals in ((x - 0.18, p), (x + 0.18, u)):
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 3, f"{v:.0f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("µs per call (median of 20)")
    ax.set_title(f"Image patch pack / unpack on {d['environment'].get('gpu', 'GPU')}")
    ax.legend()
    save(fig, out, "h3-image-patch-pack.png")
    table(
        "Patch pack / unpack (CUDA, FP32, batch 1)",
        ["latent", "rows", "pack µs", "unpack µs", "round trip bitwise"],
        [[f"{v['latent'][0]}x{v['latent'][1]}", v["rows"], f"{v['pack_ms'] * 1e3:.1f}",
          f"{v['unpack_ms'] * 1e3:.1f}", v["round_trip_bitwise"]] for v in c],
    )


def refcond(ev, out):
    p = ev["probes"]
    arms = [
        ("I12 noise before target", p["I12"]),
        ("I13 clean (t = 1)", p["I13_clean"]),
        ("I13 schedule t", p["I13_schedule_t"]),
        ("Python-float level", p["python_float_level"]),
    ]
    names = [a for a, _ in arms]
    vals = [r["max_abs"] for _, r in arms]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(names[::-1], vals[::-1], color=S1, height=0.55)
    ax.set_xscale("log")
    ax.set_xlabel("max |reference-row diff| vs released (log)")
    for y, (_, r) in enumerate(arms[::-1]):
        share = 100 * r["elements_changed"] / r["elements"]
        ax.text(r["max_abs"] * 1.4, y, f"{r['max_abs']:.1e} · {share:.0f}% of elements",
                va="center", fontsize=9, color=INK2)
    ax.set_xlim(right=max(vals) * 3000)
    ax.set_title("Reference conditioning probe arms (edit3, 747 reference rows)")
    save(fig, out, "h3-image-reference-conditioning-probes.png")
    rows = [[a, f"{r['max_abs']:.2e}", f"{r['rows_changed']} / {r['rows']}",
             f"{r['elements_changed']:,} / {r['elements']:,}", r.get("ledger", "digest differs")]
            for a, r in arms]
    rows += [["I13 resample at step 3", "—", "—", "—", f"check_fixed trips at step {p['I13_resample']['step']}"],
             ["I14 write every row", "—", "—", "—", f"check_fixed trips at step {p['I14']['step']}"]]
    table("Reference conditioning probe arms", ["arm", "max abs", "rows changed",
          "elements changed", "guard"], rows)


def text(tb, th, out):
    cases = ["t2i", "t2i_truncated", "edit", "edit_swapped"]
    b = [tb["cases"][c]["contract_50_layer_stack"]["ms"] for c in cases]
    h = [th["cases"][c]["contract_50_layer_stack"]["ms"] for c in cases]
    x = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(x - 0.18, b, 0.34, color=S1, label="B200")
    ax.bar(x + 0.18, h, 0.34, color=S2, label="H100")
    for xs, vals in ((x - 0.18, b), (x + 0.18, h)):
        for xi, v in zip(xs, vals):
            ax.text(xi, v * 1.02, f"{v:.0f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x, [f"{c}\n{tb['cases'][c]['num_rows']:,} rows" for c in cases])
    ax.set_ylabel("ms, 50-layer stack")
    ax.set_title("Layer-50 text feature: contract latency (each equals Diffusers bitwise)")
    ax.legend()
    save(fig, out, "h3-image-text-presentation.png")
    table(
        "Text presentation: equality and latency (50-layer stack)",
        ["case", "rows", "B200 ms", "H100 ms", "equal Diffusers B200 / H100"],
        [[c, tb["cases"][c]["num_rows"], f"{bv:.1f}", f"{hv:.1f}",
          f"{tb['cases'][c]['contract_full_model'].get('bitwise_equal', '?')} / "
          f"{th['cases'][c]['contract_full_model'].get('bitwise_equal', '?')}"]
         for c, bv, hv in zip(cases, b, h)],
    )


def pdd(d, out):
    p = d["pdd_schedule"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for key, color, label in (
        ("released", S1, "released PDD grid (H3-Image)"),
        ("h3_shift12_linspace7", S2, "H3 shifted linspace, 7 points, shift 12"),
        ("shift_provided_sigmas", S3, "PDD grid + shift_provided_sigmas=True"),
    ):
        ax.plot(range(7), p[key], color=color, linewidth=2, marker="o", markersize=5, label=label)
    for i, s in enumerate(p["released"]):
        ax.text(i + 0.08, s + 0.025, f"{s:.4f}", fontsize=8, color=INK2)
    ax.set_xlabel("grid index (6 model steps + terminal 0)")
    ax.set_ylabel("sigma")
    ax.set_title("Seven-point sampling grids: only the released one is the PDD grid")
    ax.legend(fontsize=9, loc="lower left")
    save(fig, out, "h3-image-pdd-schedule.png")
    table(
        "Sigma grids (FP32)",
        ["index", "released", "H3 shift-12 linspace", "shift_provided_sigmas", "timestep (released)"],
        [[i, f"{a:.6f}", f"{b:.6f}", f"{c:.6f}", f"{p['timesteps'][i]:.6f}" if i < 6 else "—"]
         for i, (a, b, c) in enumerate(zip(p["released"], p["h3_shift12_linspace7"],
                                           p["shift_provided_sigmas"]))],
    )


def velocity(d, out):
    st = d["velocity"]["steps"]
    x = np.arange(len(st))
    frac = [100 * s["blend_vs_velocity_differing"] / s["elements"] for s in st]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(x, frac, 0.55, color=S2)
    for xi, f in zip(x, frac):
        ax.text(xi, f + 1, f"{f:.0f}%", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x, [f"step {s['step']}\nσ {s['sigma']:.3f}→{s['sigma_next']:.3f}" for s in st],
                  fontsize=8)
    ax.set_ylabel("% elements that differ")
    ax.set_ylim(0, 100)
    ax.set_title("H3 blend step vs velocity step on the same inputs")
    save(fig, out, "h3-image-velocity-step-programs.png")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ref = [s["reference_ms"] * 1e3 for s in st]
    tri = [s["triton_ms"] * 1e3 for s in st]
    ax.bar(x - 0.18, ref, 0.34, color=S1, label="PyTorch reference")
    ax.bar(x + 0.18, tri, 0.34, color=S3, label="Triton")
    ax.set_xticks(x, [f"step {s['step']}" for s in st])
    ax.set_ylabel("µs per update (median)")
    ax.set_title(f"Velocity update, {d['velocity']['rows']:,} x 96 rows (launch-bound)")
    ax.legend(loc="lower right")
    save(fig, out, "h3-image-velocity-step-latency.png")
    table(
        "Velocity step: programs and latency (16,384 x 96 rows, B200)",
        ["step", "sigma -> next", "Triton != reference", "blend != velocity", "max abs (blend)",
         "reference µs", "Triton µs"],
        [[s["step"], f"{s['sigma']:.4f} -> {s['sigma_next']:.4f}", s["triton_vs_reference_differing"],
          f"{s['blend_vs_velocity_differing']:,} ({100 * s['blend_vs_velocity_differing'] / s['elements']:.0f}%)",
          f"{s['blend_vs_velocity_max_abs']:.1e}", f"{s['reference_ms'] * 1e3:.0f}",
          f"{s['triton_ms'] * 1e3:.0f}"] for s in st],
    )


def phase_b(summary, chain_dir: Path, out):
    from PIL import Image

    fx = summary["fixtures"]
    names = list(fx)
    fig, axes = plt.subplots(len(names), 3, figsize=(9, 3 * len(names)))
    for i, n in enumerate(names):
        a = np.asarray(Image.open(chain_dir / f"{n}.rl-kernel.png").convert("RGB"))
        b = np.asarray(Image.open(chain_dir / f"{n}.diffusers.png").convert("RGB"))
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
        for j, (img, title) in enumerate(((a, "RL-Kernel chain"), (b, "pinned Diffusers"),
                                          (diff, f"|diff| max = {int(diff.max())}"))):
            ax = axes[i, j]
            step = max(1, img.shape[0] // 400)
            ax.imshow(img[::step, ::step] if img.ndim == 3 else img[::step, ::step],
                      cmap=None if img.ndim == 3 else "gray", vmin=0, vmax=255 if img.ndim == 2 else None)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(f"{n}: {title}" if j == 0 else title, fontsize=9, loc="left")
    save(fig, out, "h3-image-phase-b-outputs.png")
    x = np.arange(len(names))
    c = [fx[n]["chain_seconds"]["contract"] for n in names]
    d = [fx[n]["chain_seconds"]["diffusers"] for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(x - 0.18, c, 0.34, color=S1, label="RL-Kernel chain")
    ax.bar(x + 0.18, d, 0.34, color=S2, label="pinned Diffusers")
    for xs, vals in ((x - 0.18, c), (x + 0.18, d)):
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x, [f"{n}\n{fx[n]['size'][1]}x{fx[n]['size'][0]}" for n in names], fontsize=8)
    ax.set_ylabel("seconds, text excluded")
    ax.set_title("Phase B chain on the released weights: both arms byte-equal")
    ax.legend()
    save(fig, out, "h3-image-phase-b-timing.png")
    table(
        "Phase B chain, released weights",
        ["fixture", "request", "canvas", "sequence (ref rows)", "gate", "output bytes", "run to run",
         "RL-Kernel s", "Diffusers s", "peak GiB"],
        [[n, f"{fx[n]['size'][1]}x{fx[n]['size'][0]}", "x".join(map(str, fx[n]["canvas"][::-1])),
          f"{fx[n]['sequence_length']:,} ({fx[n]['num_reference_rows']:,})", fx[n]["gate"],
          fx[n]["secondary_equal"], fx[n].get("run_to_run_equal"), c[i], d[i], fx[n]["peak_memory_gib"]]
         for i, n in enumerate(names)],
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--probe-b200", required=True)
    p.add_argument("--probe-h100", required=True)
    p.add_argument("--text-b200", required=True)
    p.add_argument("--text-h100", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = json.loads(Path(a.data).read_text())
    ev = Path(a.evidence)
    fingerprint(d, out)
    position_grid(d, out)
    canvas(d, out)
    packed_layout(json.loads(Path(a.probe_b200).read_text()), json.loads(Path(a.probe_h100).read_text()), out)
    patch_pack(d, out)
    refcond(json.loads((ev / "refcond-evidence.json").read_text()), out)
    text(json.loads(Path(a.text_b200).read_text()), json.loads(Path(a.text_h100).read_text()), out)
    pdd(d, out)
    velocity(d, out)
    phase_b(json.loads((ev / "chain" / "summary.json").read_text()), ev / "chain", out)
    (out / "tables.md").write_text("\n".join(TABLES))
    print("wrote tables.md")


if __name__ == "__main__":
    main()
