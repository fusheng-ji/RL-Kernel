"""Run post-rebase acceptance while preserving the original 0658d71 evidence."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "source"
UPSTREAM = "16b81175"


def git(*args):
    return subprocess.check_output(["git", "-C", str(SOURCE), *args], text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if git("status", "--porcelain"):
        raise RuntimeError("Commit post-rebase implementation before recording final evidence.")
    subprocess.run(["git", "-C", str(SOURCE), "merge-base", "--is-ancestor", UPSTREAM, "HEAD"], check=True)
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git("rev-parse", "HEAD"),
        "source_dirty": False,
        "upstream_main_commit": git("rev-parse", UPSTREAM),
        "branch": git("branch", "--show-current"),
        "source": str(SOURCE),
        "output_directory": str(OUT),
        "original_evidence": str(OUT.parent / "recursive-validation.json"),
        "preserved_original_evidence_sha256": {name: sha(OUT.parent / name) for name in ("recursive-validation.json", "recursive-backends.json", "logp-backends.json", "forward-compatibility.json", "regression-v4.xml", "higher-order-v4.xml", "third-order-v4.xml", "recursive.xml", "recursive-order10.xml", "acceptance-v4.xml")},
        "commands": [],
    }
    py = sys.executable
    pytest_base = [py, "-m", "pytest"]

    def pytest(label, *paths, extra=()):
        return (label, [*pytest_base, *paths, *extra, "-v", "-rs", "-o", "junit_family=legacy", f"--junitxml={OUT / (label + '.xml')}"], {})

    commands = [
        pytest(
            "regression",
            "tests/ops/logprob/test_deterministic_logp_autograd.py",
            "tests/ops/logprob/test_deterministic_logp.py",
            "tests/ops/logprob/test_batch_invariant_logp.py",
            "tests/build/test_extension_smoke.py",
            "tests/runtime/test_dispatch.py",
            "tests/benchmarks/test_profiler.py",
            "tests/e2e/test_grpo_single_gpu_example.py",
        ),
        pytest("higher-order", "tests/ops/logprob/test_deterministic_logp_higher_order.py"),
        pytest("third-order", "tests/ops/logprob/test_deterministic_logp_third_order.py"),
        pytest("recursive", "tests/ops/logprob/test_deterministic_logp_recursive.py"),
        ("recursive-order10", pytest("recursive-order10", "tests/ops/logprob/test_deterministic_logp_recursive.py", extra=("-k", "recursive_orders and dtype3"))[1], {"RL_KERNEL_LOGP_MAX_ORDER": "10"}),
        pytest("acceptance", str(OUT / "test_logp_acceptance.py")),
        ("forward-compatibility", [py, str(OUT / "compare_forward_baseline.py")], {}),
        ("grpo", [py, "examples/post_training/grpo_single_gpu.py", "--device", "cuda", "--logp-backend", "deterministic", "--require-fused-logp", "--require-batch-invariant-logp", "--steps", "3", "--num-prompts", "1", "--samples-per-prompt", "2", "--prompt-len", "2", "--completion-len", "3", "--vocab-size", "257", "--hidden-dim", "8"], {}),
        ("logp-backends", [py, str(OUT / "probe_logp_backends.py")], {}),
        ("recursive-backends", [py, "tools/validation/operators/probe_deterministic_logp_autograd.py", "--output", str(OUT / "recursive-backends.json")], {}),
        ("docs", [py, "-m", "mkdocs", "build", "--strict", "-f", "mkdocs.yaml", "--site-dir", str(OUT / "site")], {}),
        ("layout", [py, "ci/run.py", "layout", "-v", "-rs", "-o", "junit_family=legacy", f"--junitxml={OUT / 'layout.xml'}"], {}),
    ]
    import os

    for label, command, overrides in commands:
        print(f"Running {label}; log: {OUT / (label + '.log')}", flush=True)
        record = {"label": label, "command": command, "environment_overrides": overrides, "started_at_utc": datetime.now(timezone.utc).isoformat()}
        with (OUT / (label + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=SOURCE, env={**os.environ, **overrides}, stdout=log, stderr=subprocess.STDOUT, check=False)
        record.update(returncode=result.returncode, finished_at_utc=datetime.now(timezone.utc).isoformat())
        metadata["commands"].append(record)
        (OUT / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    metadata.update(
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        source_commit_after=git("rev-parse", "HEAD"),
        source_dirty_after=bool(git("status", "--porcelain")),
        runner_sha256=sha(Path(__file__)),
    )
    (OUT / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result = subprocess.run([py, str(OUT / "summarize-validation.py")], cwd=SOURCE, check=False)
    if any(c["returncode"] for c in metadata["commands"]) or result.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
