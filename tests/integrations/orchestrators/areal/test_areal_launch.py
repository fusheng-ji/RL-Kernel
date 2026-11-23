# SPDX-License-Identifier: Apache-2.0
"""CPU checks for verification config and environment generation."""

import json

import pytest

from rl_engine.integrations.orchestrators.areal import launch

pytestmark = pytest.mark.unit

BASE = {
    "experiment_name": "gsm8k-grpo-megatron",
    "cluster": {"n_gpus_per_node": 8, "fileroot": "/tmp/x"},
    "rollout": {"backend": "sglang:d4p1t1", "max_head_offpolicyness": 2,
                "scheduling_spec": "${actor.scheduling_spec}"},
    "actor": {
        "backend": "megatron:d4p1t1",
        "path": "Qwen/Qwen2.5-1.5B-Instruct",
        "kl_ctl": 0.0,
        "scheduling_spec": [
            {"task_type": "worker", "port_count": 2, "gpu": 1, "mem": 32,
             "cmd": "python3 -m areal.infra.rpc.rpc_server", "env_vars": {}}
        ],
    },
    "ref": {"backend": "${actor.backend}"},
    "gconfig": {"n_samples": 4, "temperature": 1.0},
    "train_dataset": {"batch_size": 256},
}


def _spec(**overrides):
    values = dict(model="/models/qwen3-8b", real_vocab_size=151936, padded_vocab_size=151936)
    values.update(overrides)
    return launch.ExperimentSpec(**values)


def test_rlkernel_arm_swaps_only_the_actor_worker_command(tmp_path):
    spec = _spec()
    env = launch.arm_environment(spec, "rlkernel", readback_dir=tmp_path / "rb",
                                 stats_path=tmp_path / "s.jsonl")
    config = launch.build_config(BASE, spec, "rlkernel", arm_dir=tmp_path,
                                 env_vars=env, trial_name="t")
    assert config["actor"]["scheduling_spec"][0]["cmd"] == launch.WORKER_CMD
    assert config["rollout"]["scheduling_spec"][0]["cmd"] == launch.DEFAULT_RPC_CMD
    assert config["actor"]["scheduling_spec"][0]["env_vars"]["RL_KERNEL_AREAL_MODE"] == "strict"
    assert config["actor"]["backend"] == "megatron:d1p1t4"
    assert config["rollout"]["backend"] == "vllm:d1p1t4"
    assert config["rollout"]["max_head_offpolicyness"] == 0
    assert config["actor"]["megatron"]["enable_chunked_logits"] is True
    assert config["actor"]["kl_ctl"] == 0.0
    assert "ref" not in config
    assert BASE["actor"]["scheduling_spec"][0]["cmd"].endswith("rpc_server")  # input untouched
    assert env["VLLM_PLUGINS"] == "rl_kernel"
    assert env["RL_KERNEL_LOGP_CASE"] == "R/R"
    assert env["RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE"] == str(spec.max_num_seqs)


def test_fa4_jit_cache_is_persistent_on_the_rlkernel_arm_only(tmp_path, monkeypatch):
    monkeypatch.delenv("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", raising=False)
    kwargs = dict(readback_dir=tmp_path / "rb", stats_path=tmp_path / "s.jsonl")
    env = launch.arm_environment(_spec(), "rlkernel", **kwargs)
    assert env["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] == "1"
    assert "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR" not in env
    native = launch.arm_environment(_spec(), "native", **kwargs)
    assert not any(key.startswith("FLASH_ATTENTION_CUTE_DSL") for key in native)

    monkeypatch.setenv("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", str(tmp_path / "fa4"))
    env = launch.arm_environment(_spec(), "rlkernel", **kwargs)
    assert env["FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"] == str(tmp_path / "fa4")


def test_native_sync_arm_is_stock_and_async_native_only_observes(tmp_path):
    sync_env = launch.arm_environment(_spec(), "native", readback_dir=tmp_path,
                                      stats_path=tmp_path / "s")
    sync = launch.build_config(BASE, _spec(), "native", arm_dir=tmp_path,
                               env_vars=sync_env, trial_name="t")
    assert sync["actor"]["scheduling_spec"][0]["cmd"] == launch.DEFAULT_RPC_CMD
    assert "RL_KERNEL_AREAL_MODE" not in sync_env
    assert not any(key.startswith("RL_KERNEL_") and key != "RL_KERNEL_AREAL_STATS_JSONL"
                   for key in sync_env)
    assert sync["vllm"]["enforce_eager"] is True

    spec = _spec(mode="async", max_offpolicyness=2)
    env = launch.arm_environment(spec, "native", readback_dir=tmp_path, stats_path=tmp_path / "s")
    config = launch.build_config(BASE, spec, "native", arm_dir=tmp_path, env_vars=env,
                                 trial_name="t")
    assert env["RL_KERNEL_AREAL_MODE"] == "observe"
    assert config["actor"]["scheduling_spec"][0]["cmd"] == launch.WORKER_CMD
    assert config["rollout"]["max_head_offpolicyness"] == 2


def test_native_rope_fusion_uses_worker_and_env_only_on_native(tmp_path):
    spec = _spec(native_rope_fusion=True)
    env = launch.arm_environment(spec, "native", readback_dir=tmp_path, stats_path=tmp_path / "s")
    config = launch.build_config(BASE, spec, "native", arm_dir=tmp_path, env_vars=env,
                                 trial_name="t")
    assert env[launch.NATIVE_ROPE_FUSION_ENV] == "1"
    assert config["actor"]["scheduling_spec"][0]["cmd"] == launch.WORKER_CMD
    rlkernel = launch.arm_environment(spec, "rlkernel", readback_dir=tmp_path,
                                      stats_path=tmp_path / "s")
    assert launch.NATIVE_ROPE_FUSION_ENV not in rlkernel


def test_native_observe_adds_metrics_to_the_sync_native_arm_only(tmp_path):
    spec = _spec(native_observe=True)
    env = launch.arm_environment(spec, "native", readback_dir=tmp_path, stats_path=tmp_path / "s")
    config = launch.build_config(BASE, spec, "native", arm_dir=tmp_path, env_vars=env,
                                 trial_name="t")
    assert env["RL_KERNEL_AREAL_MODE"] == "observe"
    assert config["actor"]["scheduling_spec"][0]["cmd"] == launch.WORKER_CMD
    assert config["rollout"]["max_head_offpolicyness"] == 0
    assert not any(key.startswith("RL_KERNEL_")
                   and key not in ("RL_KERNEL_AREAL_STATS_JSONL", "RL_KERNEL_AREAL_MODE")
                   for key in env)
    rlkernel = launch.arm_environment(spec, "rlkernel", readback_dir=tmp_path,
                                      stats_path=tmp_path / "s")
    assert rlkernel["RL_KERNEL_AREAL_MODE"] == "strict"


def test_spec_validation_and_cp_backend():
    assert _spec(train_cp=2, train_tp=4, rollout_tp=0, rollout_dp=0).actor_backend() == (
        "megatron:d1p1t4c2"
    )
    with pytest.raises(ValueError, match="exceeds"):
        _spec(train_tp=8, rollout_tp=4).validate()
    with pytest.raises(ValueError, match="async"):
        _spec(mode="async", max_offpolicyness=0).validate()


def test_default_config_stays_single_node_local(tmp_path):
    spec = _spec()
    env = launch.arm_environment(spec, "native", readback_dir=tmp_path, stats_path=tmp_path / "s")
    config = launch.build_config(BASE, spec, "native", arm_dir=tmp_path, env_vars=env,
                                 trial_name="t")
    assert config["cluster"]["n_nodes"] == 1
    assert config["cluster"]["n_gpus_per_node"] == 8
    assert config["scheduler"]["type"] == "local"


def test_two_nodes_of_four_gpus_use_ray(tmp_path):
    spec = _spec(n_nodes=2, n_gpus_per_node=4, scheduler="ray")
    spec.validate()
    env = launch.arm_environment(spec, "rlkernel", readback_dir=tmp_path / "rb",
                                 stats_path=tmp_path / "s.jsonl")
    config = launch.build_config(BASE, spec, "rlkernel", arm_dir=tmp_path, env_vars=env,
                                 trial_name="t")
    assert config["cluster"]["n_nodes"] == 2
    assert config["cluster"]["n_gpus_per_node"] == 4
    assert config["scheduler"]["type"] == "ray"
    assert config["actor"]["backend"] == "megatron:d1p1t4"
    assert config["rollout"]["backend"] == "vllm:d1p1t4"
    assert config["actor"]["scheduling_spec"][0]["cmd"] == launch.WORKER_CMD


def test_multi_node_validation():
    with pytest.raises(ValueError, match="ray"):
        _spec(n_nodes=2, n_gpus_per_node=4, scheduler="local").validate()
    with pytest.raises(ValueError, match="scheduler"):
        _spec(scheduler="slurm").validate()
    with pytest.raises(ValueError, match="one node"):
        _spec(n_nodes=2, n_gpus_per_node=4, scheduler="ray", train_tp=8, rollout_tp=4).validate()
    with pytest.raises(ValueError, match="one node"):
        _spec(n_nodes=2, n_gpus_per_node=2, scheduler="ray", train_tp=2, rollout_tp=4).validate()
    with pytest.raises(ValueError, match="exceeds 8 GPUs"):
        _spec(n_nodes=2, n_gpus_per_node=4, scheduler="ray", train_tp=4, rollout_tp=4,
              rollout_dp=2).validate()
    with pytest.raises(ValueError, match="positive"):
        _spec(n_nodes=0).validate()


def test_profile_steps_reach_both_arms_through_the_worker(tmp_path):
    spec = _spec(steps=4, profile_steps=(3,))
    spec.validate()
    for arm in launch.ARMS:
        env = launch.arm_environment(spec, arm, readback_dir=tmp_path / "rb",
                                     stats_path=tmp_path / arm / "stats.jsonl")
        config = launch.build_config(BASE, spec, arm, arm_dir=tmp_path, env_vars=env,
                                     trial_name="t")
        assert env[launch.PROFILE_STEPS_ENV] == "3"
        assert env[launch.PROFILE_DIR_ENV] == str(tmp_path / arm / "profile")
        assert config["actor"]["scheduling_spec"][0]["cmd"] == launch.WORKER_CMD
        assert launch.scrubbed_environment(env).get(launch.PROFILE_STEPS_ENV) == "3"
    plain = launch.arm_environment(_spec(), "native", readback_dir=tmp_path,
                                   stats_path=tmp_path / "s")
    assert launch.PROFILE_STEPS_ENV not in plain
    with pytest.raises(ValueError, match="profile_steps"):
        _spec(steps=4, profile_steps=(4,)).validate()


def test_scrubbed_environment_removes_numeric_switches():
    env = {"RL_KERNEL_LOGP_CASE": "R/R", "VIME_RL_KERNEL_STRICT": "1", "VLLM_PLUGINS": "x",
           "PATH": "/bin", "HF_HOME": "/h"}
    assert launch.scrubbed_environment(env) == {"PATH": "/bin", "HF_HOME": "/h"}


def test_read_vocabulary(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"vocab_size": 151936, "tie_word_embeddings": False}), encoding="utf-8"
    )
    assert launch.read_vocabulary(tmp_path) == (151936, False)
