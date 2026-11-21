# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the supported AReaL configuration boundary."""

from types import SimpleNamespace

import pytest

from rl_engine.integrations.orchestrators.areal import contract

pytestmark = pytest.mark.unit


def _engine(**overrides):
    config = SimpleNamespace(
        dtype="bfloat16",
        is_critic=False,
        use_lora=False,
        enable_tree_training=False,
        disable_dropout=True,
        temperature=1.0,
    )
    mcore = SimpleNamespace(
        bridge_type="mbridge",
        enable_chunked_logits=True,
        lm_head_loss_chunk_size=4096,
        entropy_requires_grad=False,
        enable_fp32_lm_head=False,
        enable_mtp=False,
        enable_mtp_training=False,
        mtp_only=False,
        use_torch_fsdp2=False,
        use_custom_fsdp=False,
        enable_router_replay=False,
        fp8_config=None,
        virtual_pipeline_parallel_size=None,
    )
    strategy = SimpleNamespace(
        tensor_parallel_size=4,
        context_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
    )
    engine = SimpleNamespace(
        config=config,
        mcore_config=mcore,
        parallel_strategy=strategy,
        bridge_cls="mbridge",
        enable_tree_training=False,
    )
    for dotted, value in overrides.items():
        owner_name, _, attr = dotted.rpartition("__")
        owner = getattr(engine, owner_name) if owner_name else engine
        setattr(owner, attr, value)
    return engine


def test_supported_engine_has_no_errors():
    assert contract.engine_config_errors(_engine()) == []


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"config__temperature": 0.7}, "temperature"),
        ({"config__dtype": "float16"}, "bfloat16"),
        ({"config__is_critic": True}, "critic"),
        ({"config__disable_dropout": False}, "dropout"),
        ({"mcore_config__enable_chunked_logits": False}, "enable_chunked_logits"),
        ({"mcore_config__lm_head_loss_chunk_size": 0}, "lm_head_loss_chunk_size"),
        ({"mcore_config__entropy_requires_grad": True}, "entropy_requires_grad"),
        ({"bridge_cls": "megatron-bridge"}, "mbridge"),
        ({"parallel_strategy__pipeline_parallel_size": 2}, "pipeline"),
        ({"parallel_strategy__context_parallel_size": 4}, "context_parallel_size"),
        ({"parallel_strategy__tensor_parallel_size": 3}, "tensor_parallel_size"),
    ],
)
def test_unsupported_engine_settings_are_rejected(override, fragment):
    errors = contract.engine_config_errors(_engine(**override))
    assert any(fragment in error for error in errors), errors


def test_model_config_requires_untied_dense_qwen3_without_sp():
    engine = SimpleNamespace(use_padded_seq=False, is_vision_model=False)
    hf = SimpleNamespace(
        architectures=["Qwen3ForCausalLM"], model_type="qwen3", tie_word_embeddings=False
    )
    tf = SimpleNamespace(sequence_parallel=False, apply_rope_fusion=False, use_mup=False)
    assert contract.model_config_errors(engine, hf, tf) == []
    tf.sequence_parallel = True
    hf.tie_word_embeddings = True
    errors = contract.model_config_errors(engine, hf, tf)
    assert any("sequence parallelism" in error for error in errors)
    assert any("tied" in error for error in errors)


@pytest.mark.parametrize(
    "rope",
    [
        {"rope_scaling": None, "rope_theta": 1_000_000},
        {"rope_scaling": {"rope_type": "default", "rope_theta": 1_000_000}},
        {"rope_parameters": {"rope_type": "default", "rope_theta": 1_000_000}},
        {"rope_scaling": {"type": "default"}, "rope_theta": 1_000_000},
        {
            "rope_scaling": {"rope_type": "default", "rope_theta": 1_000_000},
            "rope_parameters": {"rope_type": "default", "rope_theta": 1_000_000},
        },
    ],
)
def test_default_rope_schemas_are_equivalent(rope):
    hf = SimpleNamespace(architectures=["Qwen3ForCausalLM"], model_type="qwen3", **rope)
    assert contract.model_config_errors(SimpleNamespace(), hf, SimpleNamespace()) == []


@pytest.mark.parametrize(
    "rope",
    [
        {"rope_scaling": {"rope_type": "linear", "factor": 2}},
        {"rope_parameters": {"rope_type": "yarn", "factor": 2}},
        {"rope_scaling": {"rope_type": "default", "factor": 2}},
        {"rope_scaling": {"rope_type": "default", "type": "linear"}},
        {"rope_scaling": {"rope_type": "default", "rope_theta": 10000}},
        {"rope_theta": 10000, "rope_parameters": {"rope_type": "default", "rope_theta": 1e6}},
        {"rope_theta": float("nan")},
        {"rope_parameters": {"rope_type": "default", "rope_theta": None}},
        {"rope_scaling": True},
        {"rope_parameters": {"full_attention": {"rope_type": "default"}}},
    ],
)
def test_scaled_invalid_or_conflicting_rope_is_rejected(rope):
    hf = SimpleNamespace(architectures=["Qwen3ForCausalLM"], model_type="qwen3", **rope)
    errors = contract.model_config_errors(SimpleNamespace(), hf, SimpleNamespace())
    assert errors
    assert all("RoPE" in error for error in errors)


def test_vocabulary_contract_and_mode_parsing():
    env = {contract.REAL_VOCAB_ENV: "151936", contract.PADDED_VOCAB_ENV: "151936"}
    assert contract.vocabulary_contract(env) == (151936, 151936)
    with pytest.raises(ValueError, match="positive"):
        contract.vocabulary_contract({contract.REAL_VOCAB_ENV: "x"})
    with pytest.raises(ValueError, match="exceeds"):
        contract.vocabulary_contract(
            {contract.REAL_VOCAB_ENV: "10", contract.PADDED_VOCAB_ENV: "8"}
        )
    assert contract.adapter_mode({}) is None
    assert contract.adapter_mode({contract.MODE_ENV: "Strict"}) == "strict"
    with pytest.raises(ValueError):
        contract.adapter_mode({contract.MODE_ENV: "fast"})
