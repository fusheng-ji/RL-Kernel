"""Render PR figures from recorded acceptance results, without running CUDA tests."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
PRECISIONS = {
    "torch.float32": "FP32",
    "torch.float16": "FP16",
    "torch.bfloat16": "BF16",
    "torch.float64": "FP64",
}
COLORS = {
    "pass": "#168067",
    "skip": "#D19A26",
    "reverse": "#2867B2",
    "jvp": "#138B78",
    "tolerance": "#CB6E22",
    "muted": "#607080",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#314252",
        "text.color": "#213242",
        "svg.fonttype": "none",
        "svg.hashsalt": "deterministic-logp-rebased",
        "figure.facecolor": "white",
    }
)


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, facecolor="white")
    svg = output / f"{name}.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def extract(report):
    rows = []
    maxima = {name: {"reverse/mixed": {}, "nested JVP": {}} for name in PRECISIONS.values()}
    for suite in ("recursive", "recursive-order10"):
        for case in report["suites"][suite]["cases"]:
            properties = case["properties"]
            if "orders" not in properties:
                continue
            assert case["status"] == "passed"
            name = PRECISIONS[properties["input_dtype"]]
            mode = "nested JVP" if "test_nested_forward_orders" in case["test"] else "reverse/mixed"
            for entry in json.loads(properties["orders"]):
                assert np.isfinite(entry["max_error"]) and 0 <= entry["max_error"] <= entry["atol"]
                assert isinstance(entry["order"], int) and entry["order"] > 0
                assert entry["rtol"] == 0
                row = {
                    "suite": suite,
                    "test": case["test"],
                    "precision": name,
                    "mode": mode,
                    **entry,
                }
                rows.append(row)
                order = entry["order"]
                maxima[name][mode][order] = max(
                    maxima[name][mode].get(order, 0), entry["max_error"]
                )
    for dtype, name in PRECISIONS.items():
        for order, value in report["recursive_max_errors_by_order"][dtype].items():
            assert value == max(mode.get(int(order), 0) for mode in maxima[name].values())
        expected_common = report["tested_orders"]["all_precisions_reverse_mixed_and_nested_JVP"]
        expected_extra = report["tested_orders"]["FP64_additional_reverse_and_mixed_only"]
        assert sorted(maxima[name]["nested JVP"]) == expected_common
        assert sorted(maxima[name]["reverse/mixed"]) == sorted(
            set(expected_common + (expected_extra if name == "FP64" else []))
        )
        assert (
            sorted(set().union(*[set(values) for values in maxima[name].values()]))
            == report["actual_tested_orders_by_precision"][dtype]
        )
    return rows, maxima


def overview(report, output):
    suites = [
        ("regression", "Legacy / onboarding regression"),
        ("higher-order", "FP64 / out / second-order regression"),
        ("third-order", "Third-order regression"),
        ("recursive", "Recursive derivatives + transforms"),
        ("acceptance", "Supplemental acceptance"),
        ("recursive-order10", "FP64 orders 9–10 (additional)"),
        ("layout", "Layout CI (separate; overlapping checks)"),
    ]
    totals = report["pytest_totals"]
    main = totals["passed"]
    additional = report["extended_order10_totals"]["passed"]
    layout = report["layout_totals"]["passed"]
    max_cases = max(sum(report["suites"][key]["totals"].values()) for key, _ in suites)
    fig, ax = plt.subplots(figsize=(11.6, 8.6))
    fig.subplots_adjust(left=0.38, right=0.88, bottom=0.25, top=0.68)
    fig.suptitle("Deterministic CUDA LogP · acceptance results", fontsize=21, x=0.5, y=0.97)
    fig.text(
        0.5,
        0.922,
        f"Single NVIDIA B200  |  rebased source {report['source_commit'][:7]}",
        ha="center",
        color=COLORS["muted"],
    )
    for x, number, label, color in [
        (0.21, str(main), "MAIN SUITE PASSED", COLORS["pass"]),
        (0.5, str(totals["skipped"]), "SKIPPED, NOT PASSED", COLORS["skip"]),
        (0.79, str(totals["failed"] + totals.get("error", 0)), "FAILED", "#314252"),
    ]:
        fig.text(x, 0.815, number, ha="center", fontsize=36, fontweight="bold", color=color)
        fig.text(x, 0.772, label, ha="center", fontsize=10, color=COLORS["muted"])
    for y, (key, label) in enumerate(suites):
        suite = report["suites"][key]["totals"]
        passed, skipped = suite.get("passed", 0), suite.get("skipped", 0)
        ax.barh(y, passed, color=COLORS["pass"], height=0.58)
        if skipped:
            ax.barh(y, skipped, left=passed, color=COLORS["skip"], height=0.58)
        text = f"{passed} passed" + (f" + {skipped} skipped" if skipped else "")
        ax.text(passed + skipped + max_cases * 0.018, y, text, va="center", fontsize=10)
    ax.set_yticks(range(len(suites)), [s[1] for s in suites])
    ax.invert_yaxis()
    ax.set_xlim(0, max_cases * 1.23)
    ax.set_xlabel("Test cases")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.16)
    fig.text(
        0.5,
        0.183,
        f"Additional FP64 orders 9–10: {additional} passed · Layout CI: {layout} passed",
        ha="center",
        fontsize=11.5,
    )
    fig.text(
        0.5,
        0.147,
        f"Both shown separately from the main {main}; layout includes overlapping checks.",
        ha="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    comparisons = report["old_binary_forward_comparison"]
    profiles = report["native_provenance"]["cases"]
    steps = report["strict_grpo_steps"]
    fig.text(
        0.5,
        0.105,
        f"{sum(c['bitwise_equal'] for c in comparisons)} / {len(comparisons)} forward bitwise comparisons  ·  "
        f"{sum(c['status'] == 'passed' for c in profiles)} / {len(profiles)} backend profiles  ·  {len(steps)} strict GRPO steps",
        ha="center",
        fontsize=10.5,
    )
    skip_groups = {}
    for reason, count in report["suites"]["regression"]["skip_reasons"].items():
        label = (
            "SM90/Hopper-only"
            if "sm90" in reason.lower() or "hopper" in reason.lower()
            else "Ascend-only" if "ascend" in reason.lower() else "other hardware/setup"
        )
        skip_groups[label] = skip_groups.get(label, 0) + count
    skip_text = " + ".join(f"{count} {label}" for label, count in skip_groups.items())
    fig.text(
        0.5,
        0.062,
        "Regression skips: " + (skip_text or "none") + "; excluded from passes.",
        ha="center",
        color=COLORS["muted"],
        fontsize=9.5,
    )
    save(fig, output, "validation-overview")


def errors(report, maxima, rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.4))
    fig.subplots_adjust(left=0.095, right=0.97, bottom=0.16, top=0.87, hspace=0.48, wspace=0.26)
    fig.suptitle("Measured derivative error by order", fontsize=22, y=0.97)
    fig.text(
        0.5,
        0.927,
        f"Source {report['source_commit'][:7]} · maximum absolute error against independent PyTorch reference · log axes",
        ha="center",
        color=COLORS["muted"],
    )
    for ax, name in zip(axes.flat, PRECISIONS.values()):
        ax.set_title(name, loc="left", fontsize=15, fontweight="bold")
        values = []
        zero_notes = []
        for mode, color, marker in [
            ("reverse/mixed", COLORS["reverse"], "o"),
            ("nested JVP", COLORS["jvp"], "s"),
        ]:
            data = maxima[name][mode]
            x = sorted(data)
            y = [data[i] if data[i] > 0 else np.nan for i in x]
            ax.plot(
                x,
                y,
                marker=marker,
                markersize=5,
                color=color,
                lw=1.9,
                label="Reverse / mixed" if mode == "reverse/mixed" else "Nested JVP",
            )
            values.extend(v for v in data.values() if v > 0)
            zeros = [str(i) for i in x if data[i] == 0]
            if zeros:
                zero_notes.append(f"{mode}: exact zero at " + ", ".join(zeros))
        max_order = max(maxima[name]["reverse/mixed"])
        high_tolerances = {r["atol"] for r in rows if r["precision"] == name and r["order"] >= 4}
        assert len(high_tolerances) == 1
        tolerance = high_tolerances.pop()
        ax.plot(
            [4, max_order],
            [tolerance, tolerance],
            color=COLORS["tolerance"],
            ls="--",
            lw=1.5,
            label=f"Order ≥4 atol: {tolerance:g}",
        )
        ax.set_yscale("log")
        ax.set_ylim(min(values) / 4 if values else tolerance / 1e6, tolerance * 6)
        ax.set_xlim(0.7, max_order + 0.35)
        ax.set_xticks(range(1, max_order + 1))
        ax.set_xlabel("Derivative order")
        ax.set_ylabel("Maximum absolute error")
        ax.grid(alpha=0.17)
        ax.legend(
            loc="upper right",
            fontsize=8.5,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.95,
        )
        if zero_notes:
            ax.text(
                0.02,
                0.035,
                "\n".join(zero_notes) + "\nExact zeros are omitted from log axes.",
                transform=ax.transAxes,
                fontsize=8,
                color=COLORS["muted"],
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
        if name == "FP64" and max_order > max(maxima[name]["nested JVP"]):
            ax.axvspan(
                max(maxima[name]["nested JVP"]) + 0.5,
                max_order + 0.35,
                color="#DBE7F2",
                alpha=0.45,
                zorder=-1,
            )
    fig.text(
        0.5,
        0.08,
        f"FP64 reverse / mixed through order {max(maxima['FP64']['reverse/mixed'])}; nested JVP through order {max(maxima['FP64']['nested JVP'])}.",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.046,
        "Bounded test fixtures; errors are not a forecast for larger models or higher orders. Per-case tolerances are in the CSV.",
        ha="center",
        fontsize=9.5,
        color=COLORS["muted"],
    )
    save(fig, output, "derivative-errors")


def scope(report, maxima, output):
    labels, cells = [], []
    max_order = max(max(data) for modes in maxima.values() for data in modes.values())
    common = set(report["tested_orders"]["all_precisions_reverse_mixed_and_nested_JVP"])
    for mode in ("reverse/mixed", "nested JVP"):
        for name in PRECISIONS.values():
            labels.append(
                f"{'Reverse / mixed' if mode == 'reverse/mixed' else 'Nested JVP'} · {name}"
            )
            measured = maxima[name][mode]
            cells.append(
                [
                    1 if order in common else 2 if order in measured else 0
                    for order in range(1, max_order + 1)
                ]
            )
    fig, ax = plt.subplots(figsize=(12, 8.2))
    fig.subplots_adjust(left=0.28, right=0.96, top=0.80, bottom=0.29)
    fig.suptitle("Algorithm has no hard-coded derivative-order limit", fontsize=21, y=0.965)
    fig.text(
        0.5,
        0.895,
        f"Source {report['source_commit'][:7]} · finite validation coverage, not an algorithmic cap.",
        ha="center",
        fontsize=12,
        color=COLORS["muted"],
    )
    ax.imshow(
        cells, cmap=ListedColormap(["#E7EBEF", "#168067", "#2867B2"]), vmin=0, vmax=2, aspect="auto"
    )
    ax.set_xticks(range(max_order), range(1, max_order + 1))
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Derivative order")
    ax.tick_params(length=0, pad=9)
    for row in range(len(labels)):
        for col in range(max_order):
            value = cells[row][col]
            ax.text(
                col,
                row,
                "PASS" if value else "—",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value else "#687786",
            )
    ax.set_xticks(np.arange(-0.5, max_order, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.legend(
        handles=[
            Patch(
                color="#168067",
                label=f"Four-precision orders {min(common)}–{max(common)} validation",
            ),
            Patch(color="#2867B2", label="Additional FP64 reverse / mixed validation"),
            Patch(color="#E7EBEF", label="Not tested"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=1,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.037,
        "Higher orders use the same primitive rules; runtime, memory and floating-point error remain workload dependent.",
        ha="center",
        fontsize=10,
        color=COLORS["muted"],
    )
    save(fig, output, "tested-order-scope")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=HERE
        / (
            "validation-report.json"
            if (HERE / "validation-report.json").exists()
            else "recursive-validation.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "passed" and not report["evidence_failures"]
    assert len(report["source_commit"]) == 40
    assert report["algorithm_order_limit"] is None
    for totals in (
        report["pytest_totals"],
        report["extended_order10_totals"],
        report["layout_totals"],
    ):
        assert totals["failed"] == 0 and totals.get("error", 0) == 0 and totals["passed"] > 0
    expected_main = {
        key: sum(
            report["suites"][suite]["totals"].get(key, 0)
            for suite in ("regression", "higher-order", "third-order", "recursive", "acceptance")
        )
        for key in ("passed", "skipped", "failed", "error")
    }
    assert expected_main == report["pytest_totals"]
    assert (
        report["suites"]["recursive-order10"]["totals"].get("passed")
        == report["extended_order10_totals"]["passed"]
    )
    assert report["suites"]["layout"]["totals"].get("passed") == report["layout_totals"]["passed"]
    assert len(report["native_provenance"]["cases"]) == 48
    assert all(case["status"] == "passed" for case in report["native_provenance"]["cases"])
    assert report["native_provenance"]["source_commit"] == report["source_commit"]
    plt.rcParams["svg.hashsalt"] = "deterministic-logp-" + report["source_commit"]
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / "validation-report.json"
    if snapshot.resolve() != args.report.resolve():
        shutil.copyfile(args.report, snapshot)
    script_copy = args.output / "generate_figures.py"
    if script_copy.resolve() != Path(__file__).resolve():
        shutil.copyfile(Path(__file__), script_copy)
    rows, maxima = extract(report)
    with (args.output / "derivative-errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    data = {
        "source_commit": report["source_commit"],
        "report_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "extension_sha256": report["native_provenance"]["extension_sha256"],
        "algorithm_order_limit": None,
        "tested_orders": report["tested_orders"],
        "main_totals": report["pytest_totals"],
        "additional_FP64_totals": report["extended_order10_totals"],
        "separate_layout_totals": report["layout_totals"],
        "max_errors": maxima,
        "matplotlib": matplotlib.__version__,
        "note": "Maximum error is aggregated within each precision/order/mode; exact zeros are not moved to a log-axis floor.",
    }
    (args.output / "figure-data.json").write_text(json.dumps(data, indent=2) + "\n")
    overview(report, args.output)
    errors(report, maxima, rows, args.output)
    scope(report, maxima, args.output)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }
    (args.output / "manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(
        json.dumps(
            {
                "figures": 3,
                "formats": ["PNG", "SVG"],
                "recorded_error_rows": len(rows),
                "source_commit": report["source_commit"],
            }
        )
    )


if __name__ == "__main__":
    main()
