# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the vLLM CUDA graph option and the performance report."""

import json
import sys
import types

import pytest

from rl_engine.integrations.orchestrators.areal import launch, perf, run_trainer, verify

pytestmark = pytest.mark.unit

BASE = {
    "cluster": {"n_gpus_per_node": 8},
    "rollout": {"backend": "sglang:d4p1t1"},
    "actor": {"backend": "megatron:d4p1t1", "scheduling_spec": [{"cmd": "x", "env_vars": {}}]},
}


def _spec(**overrides):
    values = dict(model="/m", real_vocab_size=151936, padded_vocab_size=151936,
                  max_num_seqs=4)
    values.update(overrides)
    return launch.ExperimentSpec(**values)


def test_eager_is_default_and_adds_no_server_arguments(tmp_path):
    spec = _spec()
    assert spec.vllm_execution == launch.VLLM_EAGER and spec.vllm_enforce_eager
    assert launch.vllm_server_overrides(spec) == {}
    for arm in launch.ARMS:
        env = launch.arm_environment(spec, arm, readback_dir=tmp_path, stats_path=tmp_path / "s")
        assert launch.VLLM_SERVER_ARGS_ENV not in env


def test_graph_gives_both_arms_the_same_pr377_style_arguments(tmp_path):
    spec = _spec(vllm_execution=launch.VLLM_GRAPH)
    overrides = launch.vllm_server_overrides(spec)
    assert overrides["optimization_level"] == 0
    compilation = json.loads(overrides["compilation_config"])
    assert compilation == {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 3, 4],
        "max_cudagraph_capture_size": 4,
    }
    envs = {
        arm: launch.arm_environment(spec, arm, readback_dir=tmp_path, stats_path=tmp_path / "s")
        for arm in launch.ARMS
    }
    assert envs["native"][launch.VLLM_SERVER_ARGS_ENV] == envs["rlkernel"][
        launch.VLLM_SERVER_ARGS_ENV
    ]
    assert json.loads(envs["native"][launch.VLLM_SERVER_ARGS_ENV]) == overrides
    assert envs["rlkernel"]["RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE"] == "4"
    config = launch.build_config(BASE, spec, "native", arm_dir=tmp_path,
                                 env_vars=envs["native"], trial_name="t")
    assert config["vllm"]["enforce_eager"] is False


def test_spec_rejects_unknown_execution():
    with pytest.raises(ValueError, match="vllm_execution"):
        _spec(vllm_execution="cuda").validate()


def test_merge_server_args_refuses_conflicts():
    overrides = {"optimization_level": 0, "compilation_config": "{}"}
    merged = run_trainer.merge_server_args({"port": 1, "enforce_eager": False}, overrides)
    assert merged == {"port": 1, "enforce_eager": False, **overrides}
    with pytest.raises(RuntimeError, match="enforce_eager"):
        run_trainer.merge_server_args({"enforce_eager": True}, overrides)
    with pytest.raises(RuntimeError, match="already set"):
        run_trainer.merge_server_args({"optimization_level": 3}, overrides)


def test_install_overrides_wraps_areal_build_args(monkeypatch):
    class FakeConfig:
        @staticmethod
        def build_args(vllm_config, tp_size, pp_size, host=None, port=None):
            return {"model": vllm_config, "tensor_parallel_size": tp_size}

    cli_args = types.ModuleType("areal.api.cli_args")
    cli_args.vLLMConfig = FakeConfig
    monkeypatch.setitem(sys.modules, "areal", types.ModuleType("areal"))
    monkeypatch.setitem(sys.modules, "areal.api", types.ModuleType("areal.api"))
    monkeypatch.setitem(sys.modules, "areal.api.cli_args", cli_args)
    run_trainer.install_vllm_server_overrides('{"optimization_level": 0}')
    run_trainer.install_vllm_server_overrides('{"optimization_level": 0}')  # idempotent
    assert FakeConfig.build_args(vllm_config="m", tp_size=4, pp_size=1) == {
        "model": "m", "tensor_parallel_size": 4, "optimization_level": 0,
    }


def test_parse_role_resources():
    assert run_trainer.parse_role_resources(" actor=a_node , rollout=r_node,") == {
        "actor": "a_node", "rollout": "r_node",
    }
    for raw in ("actor", "=a_node", "actor=", " , "):
        with pytest.raises(SystemExit):
            run_trainer.parse_role_resources(raw)


def test_install_ray_role_resources_pins_only_the_named_role(monkeypatch):
    calls = []

    class FakeScheduler:
        def _create_placement_group(self, role, bundles, timeout):
            calls.append((role, bundles, timeout))

    ray_module = types.ModuleType("areal.infra.scheduler.ray")
    ray_module.RayScheduler = FakeScheduler
    for name in ("areal", "areal.infra", "areal.infra.scheduler"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "areal.infra.scheduler.ray", ray_module)
    run_trainer.install_ray_role_resources("actor=a_node")
    run_trainer.install_ray_role_resources("actor=a_node")  # idempotent
    bundles = [{"GPU": 1, "CPU": 4}]
    FakeScheduler()._create_placement_group("actor", bundles, 30.0)
    FakeScheduler()._create_placement_group("rollout", bundles, 30.0)
    assert calls == [
        ("actor", [{"GPU": 1, "CPU": 4, "a_node": 0.001}], 30.0),
        ("rollout", [{"GPU": 1, "CPU": 4}], 30.0),
    ]
    assert bundles == [{"GPU": 1, "CPU": 4}]


def _write_arm(arm_dir, *, arm, execution, rollout, log_lines, mode="sync"):
    arm_dir.mkdir(parents=True)
    rows = []
    for step, seconds in enumerate(rollout):
        rows.append({
            "global_step": step, "wall_time": 1000.0 + 50.0 * (step + 1) * (seconds / 10),
            "trainer_started": 900.0,
            "stats": {"timeperf/rollout": seconds, "timeperf/train_step": 5.0,
                      "timeperf/recompute_logp": 2.0, "timeperf/update_weights": 1.0,
                      "ppo_actor/n_valid_tokens": 4000.0},
        })
    (arm_dir / "stats.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    spec = {"mode": mode, "rollout_tp": 4, "rollout_dp": 1, "vllm_execution": execution}
    run = {"arm": arm, "spec": spec, "trial_name": f"{arm}-t", "elapsed_seconds": 300.0,
           "returncode": 0}
    (arm_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    log = arm_dir / "fileroot" / "logs" / "root" / "exp" / f"{arm}-t" / "rollout.log"
    log.parent.mkdir(parents=True)
    log.write_text("\n".join(log_lines), encoding="utf-8")
    return spec


GRAPH_LOG = [
    "INFO compilation_config={'mode': <CompilationMode.NONE: 0>, "
    "'cudagraph_mode': <CUDAGraphMode.FULL_DECODE_ONLY: (2, 0)>}",
    "INFO [gpu_model_runner.py:1] Graph capturing finished in 31 secs, took 0.62 GiB",
    "INFO Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: "
    "4000.0 tokens/s, Running: 128 reqs, Waiting: 0 reqs",
    "INFO Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: "
    "0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs",
]
EAGER_LOG = [
    "WARNING Enforce eager set, disabling torch.compile and CUDAGraphs.",
    "INFO Cudagraph is disabled under eager mode",
    "INFO Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: "
    "1000.0 tokens/s, Running: 64 reqs, Waiting: 0 reqs",
]


def test_vllm_log_evidence():
    graph = perf.VllmLogEvidence()
    assert graph.eager is None
    lines = "\n".join(GRAPH_LOG)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rollout.log"
        path.write_text(lines, encoding="utf-8")
        graph = perf.parse_vllm_log(path)
        path.write_text("\n".join(EAGER_LOG), encoding="utf-8")
        eager = perf.parse_vllm_log(path)
    assert graph.eager is False and graph.cudagraph_mode == "FULL_DECODE_ONLY"
    assert graph.capture_seconds == 31 and graph.capture_gib == pytest.approx(0.62)
    assert graph.gen_throughput_samples == 1 and graph.gen_throughput_median == 4000.0
    assert eager.eager is True and eager.gen_throughput_median == 1000.0


def test_summarize_arm_and_comparison(tmp_path):
    native = tmp_path / "S3" / "native"
    rlk = tmp_path / "S3" / "rlkernel"
    _write_arm(native, arm="native", execution="graph", rollout=[30, 10, 10, 10],
               log_lines=GRAPH_LOG)
    _write_arm(rlk, arm="rlkernel", execution="graph", rollout=[40, 20, 20, 20],
               log_lines=GRAPH_LOG)
    base = perf.summarize_arm(native, label="native-graph")
    cand = perf.summarize_arm(rlk, label="rlkernel-graph")
    assert base.means["rollout"] == 10 and cand.means["rollout"] == 20
    assert base.means["rollout_tokens_per_gpu_s"] == pytest.approx(100.0)
    assert base.startup_seconds == pytest.approx(1000 + 150 - 900)
    assert base.warnings == []
    table = "\n".join(perf.comparison_table(base, cand))
    assert "| Rollout time (s/step) | 10.00 | 20.00 | 2.00x (100.0% slower) |" in table


def test_graph_requested_but_eager_logged_is_flagged(tmp_path):
    arm = tmp_path / "native"
    _write_arm(arm, arm="native", execution="graph", rollout=[10, 10], log_lines=EAGER_LOG)
    assert any("graphs are disabled" in w for w in perf.summarize_arm(arm).warnings)


def test_old_results_without_wall_time_or_execution(tmp_path):
    arm = tmp_path / "rlkernel"
    arm.mkdir()
    (arm / "stats.jsonl").write_text(json.dumps(
        {"global_step": 0, "stats": {"timeperf/rollout": 5.0}}), encoding="utf-8")
    (arm / "run.json").write_text(json.dumps(
        {"arm": "rlkernel", "spec": {"mode": "sync", "vllm_enforce_eager": True}}),
        encoding="utf-8")
    item = perf.summarize_arm(arm)
    assert item.vllm_execution == "eager"
    assert item.per_step[0]["step_seconds"] == 5.0  # sync: sum of timeperf phases
    assert any("sum of timeperf" in w for w in item.warnings)


def test_async_rollout_timer_is_not_compared(tmp_path):
    for arm in ("native", "rlkernel"):
        _write_arm(tmp_path / arm, arm=arm, execution="graph", rollout=[5, 0.01, 0.01],
                   log_lines=GRAPH_LOG, mode="async")
    base = perf.summarize_arm(tmp_path / "native")
    cand = perf.summarize_arm(tmp_path / "rlkernel")
    assert base.means["rollout_tokens_per_gpu_s"] is None
    table = "\n".join(perf.comparison_table(base, cand))
    assert "| Rollout time (s/step) | - | - | - | - |" in table


def test_paired_ratio_ci():
    ratio, low, high = perf.paired_ratio_ci([10, 10, 10, 10], [20, 20, 20, 20])
    assert ratio == low == high == 2.0
    assert perf.paired_ratio_ci([1], [2]) == (2.0, None, None)
    assert perf.paired_ratio_ci([], []) == (None, None, None)


def test_perf_report_over_stages(tmp_path):
    for stage, execution in (("S3-sync-graph", "graph"), ("S5-sync-eager", "eager")):
        log = GRAPH_LOG if execution == "graph" else EAGER_LOG
        for arm, rollout in (("native", [20, 8, 8, 8]), ("rlkernel", [30, 12, 12, 12])):
            scale = 1 if execution == "graph" else 3
            spec = _write_arm(tmp_path / stage / arm, arm=arm, execution=execution,
                              rollout=[r * scale for r in rollout], log_lines=log)
        (tmp_path / stage / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        (tmp_path / stage / "summary.json").write_text(json.dumps({
            "rlkernel": {"verdict": {"verdict": "PASS",
                                     "totals": {"n_valid": 16000, "mismatch": 0}}},
        }), encoding="utf-8")
    report, payload = verify.render_perf_report(tmp_path, skip=1)
    assert "### sync: RL-Kernel cost with CUDA graphs on both arms" in report
    assert "### sync: CUDA graph gain on the RL-Kernel arm" in report
    assert "| S3-sync-graph | rlkernel-graph | PASS | 4 | 16000 | 0 |" in report
    assert set(payload) == {"S3-sync-graph/native-graph", "S3-sync-graph/rlkernel-graph",
                            "S5-sync-eager/native-eager", "S5-sync-eager/rlkernel-eager"}
    assert verify.main(["perf", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "PERF.md").is_file() and (tmp_path / "perf.json").is_file()
