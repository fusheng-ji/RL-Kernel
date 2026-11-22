# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Run an AReaL training script and tee every committed step to JSONL.

Usage::

    RL_KERNEL_AREAL_STATS_JSONL=/path/stats.jsonl \\
    python -m rl_engine.integrations.orchestrators.areal.run_trainer \\
        examples/math/gsm8k_rl.py --config my.yaml

Only ``StatsLogger.commit`` is wrapped: the values AReaL already logs are copied
verbatim, plus the wall-clock time of each commit. Training behaviour is unchanged.

If ``RL_KERNEL_AREAL_VLLM_SERVER_ARGS`` holds a JSON object, its keys are added
to the vLLM server arguments AReaL builds (``vLLMConfig.build_args``). This is
how the verifier sets vLLM options AReaL's config has no field for, such as the
CUDA graph capture sizes. Both arms receive the same value.
"""

from __future__ import annotations

import json
import math
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Any

STATS_ENV = "RL_KERNEL_AREAL_STATS_JSONL"
VLLM_SERVER_ARGS_ENV = "RL_KERNEL_AREAL_VLLM_SERVER_ARGS"
# Not RL_KERNEL_-prefixed on purpose: it is a launch-environment setting, not a numeric
# switch, so the verifier does not scrub it and both arms get the same value.
WORKERS_READY_TIMEOUT_ENV = "RLK_AREAL_WORKERS_READY_TIMEOUT"


def _plain(value: Any) -> Any:
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _plain(item())
        except (TypeError, ValueError, RuntimeError):
            pass
    try:
        return _plain(float(value))
    except (TypeError, ValueError):
        return repr(value)


def install_stats_tee(path: str | os.PathLike[str]) -> None:
    from areal.utils.stats_logger import StatsLogger

    if getattr(StatsLogger.commit, "__rl_kernel_tee__", False):
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    original = StatsLogger.commit
    started = time.time()

    def commit(self: Any, epoch: int, step: int, global_step: int, data: Any) -> Any:
        items = data if isinstance(data, list) else [data]
        with target.open("a", encoding="utf-8") as handle:
            for index, item in enumerate(items):
                row = {
                    "wall_time": time.time(),
                    "trainer_started": started,
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "index": index,
                    "stats": {str(key): _plain(value) for key, value in dict(item).items()},
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return original(self, epoch, step, global_step, data)

    commit.__rl_kernel_tee__ = True  # type: ignore[attr-defined]
    StatsLogger.commit = commit


def merge_server_args(args: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Add ``overrides`` to AReaL's vLLM server arguments, refusing silent conflicts."""

    if args.get("enforce_eager") and "compilation_config" in overrides:
        raise RuntimeError("vLLM CUDA graph arguments requested while vllm.enforce_eager=True")
    conflicts = sorted(
        key for key, value in overrides.items()
        if args.get(key) not in (None, False, "") and args[key] != value
    )
    if conflicts:
        raise RuntimeError(f"vLLM server arguments already set by AReaL config: {conflicts}")
    merged = dict(args)
    merged.update(overrides)
    return merged


def install_vllm_server_overrides(raw: str) -> dict[str, Any]:
    overrides = json.loads(raw)
    if not isinstance(overrides, dict) or not overrides:
        raise SystemExit(f"{VLLM_SERVER_ARGS_ENV} must be a non-empty JSON object")
    from areal.api.cli_args import vLLMConfig

    original = vLLMConfig.build_args
    if getattr(original, "__rl_kernel_overrides__", None) is not None:
        return overrides

    def build_args(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return merge_server_args(original(*args, **kwargs), overrides)

    build_args.__rl_kernel_overrides__ = overrides  # type: ignore[attr-defined]
    vLLMConfig.build_args = staticmethod(build_args)
    print(f"[RL-Kernel][areal] vLLM server overrides: {json.dumps(overrides)}", flush=True)
    return overrides


def install_workers_ready_timeout(raw: str) -> float:
    """Raise AReaL's fixed 30 s data-service worker start-up wait.

    On a multi-node scheduler the first worker process imports AReaL from a shared
    file system on a remote node, which can take longer than that.
    """

    seconds = float(raw)
    if seconds <= 0:
        raise SystemExit(f"{WORKERS_READY_TIMEOUT_ENV} must be positive")
    from areal.infra.data_service.controller.controller import DataController

    DataController._WORKERS_READY_TIMEOUT = seconds
    print(f"[RL-Kernel][areal] data-service worker start-up timeout: {seconds}s", flush=True)
    return seconds


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or not args[0].endswith(".py"):
        raise SystemExit("usage: run_trainer.py <areal_training_script.py> [args...]")
    stats_path = os.getenv(STATS_ENV, "").strip()
    if not stats_path:
        raise SystemExit(f"{STATS_ENV} must point to the JSONL output file")
    install_stats_tee(stats_path)
    overrides = os.getenv(VLLM_SERVER_ARGS_ENV, "").strip()
    if overrides:
        install_vllm_server_overrides(overrides)
    timeout = os.getenv(WORKERS_READY_TIMEOUT_ENV, "").strip()
    if timeout:
        install_workers_ready_timeout(timeout)
    script = args[0]
    sys.argv = [script, *args[1:]]
    sys.path.insert(0, str(Path(script).resolve().parent))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
