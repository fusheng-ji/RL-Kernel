"""Record actual post-rebase outcomes and fail closed on incomplete evidence."""

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "source"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(SOURCE), *args], text=True).strip()


def main():
    metadata = json.loads((OUT / "run-metadata.json").read_text())
    failures = []

    def check(condition, reason):
        if not condition:
            failures.append(reason)

    for name, digest in metadata["preserved_original_evidence_sha256"].items():
        check(sha(OUT.parent / name) == digest, f"Historical evidence was overwritten: {name}")
    commit = git("rev-parse", "HEAD")
    check(commit == metadata["source_commit"] == metadata["source_commit_after"], "Source commit changed during validation")
    check(not git("status", "--porcelain") and not metadata["source_dirty_after"], "Source tree must be clean")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": metadata["upstream_main_commit"],
        "source_commit": commit,
        "worktree": str(SOURCE),
        "branch": metadata["branch"],
        "run_metadata": metadata,
        "original_evidence": {"path": str(OUT.parent / "recursive-validation.json"), "sha256": sha(OUT.parent / "recursive-validation.json"), "note": "Historical 0658d71 evidence is preserved unchanged; the current report records the rebased source separately."},
        "scope": {
            "devices": "single NVIDIA B200",
            "algorithm": "recursive differentiable primitives without a hard-coded derivative-order limit",
            "precision": ["FP32", "FP16", "BF16", "explicit FP64"],
            "out": "preserved storage identity; non-grad, non-aliasing buffers; detach after backward before reuse",
            "indexed": "fixed full-row output shape; duplicate rows idempotent; invalid rows ignored; invalid tokens have zero derivatives",
            "not_validated": ["multi-GPU training", "production-model training", "performance", "SM90", "Ascend", "torch.compile", "functionalize"],
        },
        "algorithm_order_limit": None,
        "suites": {},
    }
    main_totals, extended_totals, layout_totals = Counter(), Counter(), Counter()
    skip_totals = Counter()
    all_errors = defaultdict(lambda: defaultdict(float))
    orders_by_precision = defaultdict(set)
    for label in ("regression", "higher-order", "third-order", "recursive", "recursive-order10", "acceptance", "layout"):
        filename = OUT / (label + ".xml")
        totals, groups, skips, cases = Counter(), defaultdict(Counter), Counter(), []
        if not filename.exists():
            failures.append(f"Missing suite XML: {filename.name}")
            continue
        for case in ET.parse(filename).iter("testcase"):
            props = {p.attrib["name"]: p.attrib["value"] for p in case.findall("./properties/property")}
            status, reason, details = "passed", "All assertions completed", None
            for tag, state in (("skipped", "skipped"), ("failure", "failed"), ("error", "error")):
                node = case.find(tag)
                if node is not None:
                    status, reason, details = state, node.attrib.get("message", ""), node.text
                    break
            name = case.attrib.get("classname", "") + "::" + case.attrib["name"]
            totals[status] += 1
            groups[case.attrib.get("classname", "")][status] += 1
            if status == "skipped":
                skips[reason] += 1
                skip_totals[reason] += 1
            for key in ("orders", "nested_jvp_orders"):
                if key in props:
                    rows = json.loads(props[key])
                    precision = props.get("input_dtype", "unknown")
                    for row in rows:
                        all_errors[precision][str(row["order"])] = max(all_errors[precision][str(row["order"])], row["max_error"])
                        if status == "passed":
                            orders_by_precision[precision].add(row["order"])
            cases.append({"test": name, "status": status, "reason": reason, "properties": props, **({"details": details} if details else {})})
        check(not totals["failed"] and not totals["error"], f"Suite {label} has failures/errors")
        # Existing upstream suites can add coverage; totals are measured rather than hard-coded.
        expected_min = {"higher-order": 138, "third-order": 117, "recursive": 89, "recursive-order10": 8, "acceptance": 135}.get(label)
        if expected_min:
            check(totals["passed"] >= expected_min, f"Suite {label} lost required pass coverage ({totals['passed']} < {expected_min})")
        check(bool(cases), f"Suite {label} collected no tests")
        report["suites"][label] = {"totals": dict(totals), "by_module": {k: dict(v) for k, v in groups.items()}, "skip_reasons": dict(skips), "cases": cases}
        target = extended_totals if label == "recursive-order10" else layout_totals if label == "layout" else main_totals
        target.update(totals)
    report.update(
        pytest_totals={key: main_totals[key] for key in ("passed", "failed", "error", "skipped")},
        extended_order10_totals={key: extended_totals[key] for key in ("passed", "failed", "error", "skipped")},
        layout_totals={key: layout_totals[key] for key in ("passed", "failed", "error", "skipped")},
        skip_reasons=dict(skip_totals),
        actual_tested_orders_by_precision={k: sorted(v) for k, v in orders_by_precision.items()},
        tested_orders={"all_precisions_reverse_mixed_and_nested_JVP": list(range(1, 9)), "FP64_additional_reverse_and_mixed_only": [9, 10]},
        recursive_max_errors_by_order={k: dict(v) for k, v in all_errors.items()},
        transforms=["grad", "vjp", "jvp", "jacrev", "jacfwd", "hessian", "vmap", "vmap(grad)", "vmap(jvp)", "nested vmap", "forward-over-reverse", "reverse-over-forward", "forward-over-forward", "reverse-over-reverse", "no_grad JVP", "forward_ad dual tensors"],
    )
    for key in ("recursive-backends", "logp-backends", "forward-compatibility"):
        filename = OUT / (key + ".json")
        if not filename.exists():
            failures.append(f"Missing {filename.name}")
            continue
        item = json.loads(filename.read_text())
        if key == "forward-compatibility":
            check(len(item) == 21 and all(c["bitwise_equal"] for c in item), "Legacy binary forward mismatch or missing cases")
            report["old_binary_forward_comparison"] = item
            continue
        check(item["source_commit"] == commit, f"{key} source mismatch")
        extension = Path(item["extension"]).resolve()
        check(extension.parent == SOURCE / "rl_engine", f"{key} extension loaded from another checkout")
        check(extension.is_file() and sha(extension) == item["extension_sha256"], f"{key} extension hash mismatch")
        if key == "recursive-backends":
            check(not item["source_dirty"], "Recursive probe recorded dirty source")
            check(len(item["cases"]) == 48 and all(c["status"] == "passed" for c in item["cases"]), "Recursive backend proof is incomplete")
            check(item["algorithm_order_limit"] is None and item["tested_derivative_order"] == 8, "Recursive probe order metadata changed")
            report["native_provenance"] = item
        else:
            report["legacy_backend_provenance"] = item
    if "native_provenance" in report and "legacy_backend_provenance" in report:
        check(report["native_provenance"]["extension_sha256"] == report["legacy_backend_provenance"]["extension_sha256"], "Backend probes loaded different binaries")
    steps = [line for line in (OUT / "grpo.log").read_text().splitlines() if line.startswith("step=")]
    check(len(steps) == 3 and all("train_logp_source=DeterministicLogpCUDAOp" in s for s in steps), "Strict GRPO training source mismatch or missing steps")
    report["strict_grpo_steps"] = steps
    docs = (OUT / "docs.log").read_text()
    check("Documentation built" in docs and "WARNING -" not in docs, "Strict documentation build failed")
    report["docs_strict_build"] = "passed" if "Documentation built" in docs and "WARNING -" not in docs else "failed"
    for command in metadata["commands"]:
        check(command["returncode"] == 0, f"Command {command['label']} exited {command['returncode']}")
    report["artifact_sha256"] = {
        str(path.relative_to(OUT)): sha(path)
        for path in sorted(OUT.iterdir())
        if path.is_file() and path.name not in {"recursive-validation.json", "summary.log"}
    }
    report["artifact_sha256"]["../forward-baseline.pt"] = sha(OUT.parent / "forward-baseline.pt")
    report["rerun"] = [f"bash {OUT}/run-validation.sh"]
    report["evidence_failures"] = failures
    report["status"] = "passed" if not failures else "failed"
    (OUT / "recursive-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {k: report[k] for k in ("source_commit", "status", "pytest_totals", "extended_order10_totals", "layout_totals", "evidence_failures")}
    text = json.dumps(summary, indent=2) + "\n"
    (OUT / "summary.log").write_text(text)
    print(text, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
