# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Runtime patches that attach RL-Kernel to an unmodified AReaL checkout.

Two modes, selected with ``RL_KERNEL_AREAL_MODE``:

``observe``
    Only adds the same-version consistency metrics to AReaL's actor. Numerics
    stay native. Used for the native arm of asynchronous comparisons.

``strict``
    Additionally routes Megatron training through RL-Kernel: strict
    Attention/FFN/RMSNorm/TP collectives (shared runtime), shared CUDA RoPE,
    sequence parallelism off, and the strict LM head in place of AReaL's
    ``chunked_lm_head_logprobs_entropy``. Unsupported configurations raise.

Every patch targets a module attribute that AReaL resolves at call time, so no
AReaL source file is edited.
"""

from __future__ import annotations

import contextvars
import functools
import os
from types import ModuleType
from typing import Any

from rl_engine.integrations.orchestrators.areal import contract
from rl_engine.integrations.orchestrators.areal.hooks import when_imported

ACTOR_MODULE = "areal.trainer.ppo.actor"
ENGINE_MODULE = "areal.engine.megatron_engine"
_ENGINE_MARKER = "__rl_kernel_areal_strict__"
_CURRENT_ENGINE: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "rl_kernel_areal_engine", default=None
)


def _strict_plan() -> Any:
    from rl_engine.runtime.plan import integration_plan_from_environment

    plan = integration_plan_from_environment()
    if any(case.case_id != "R/R" for case in plan.cases.values()):
        raise ValueError(
            "RL_KERNEL_AREAL_MODE=strict requires RL_KERNEL_{ATTENTION,FFN,LOGP}_CASE=R/R"
        )
    return plan


def _disable_sp_and_fused_rope(engine: Any, tf_config: Any) -> Any:
    """Force the two Megatron options the strict route cannot reproduce."""

    bridge = getattr(engine, "bridge", None)
    set_extra_args = getattr(bridge, "set_extra_args", None)
    if callable(set_extra_args):
        # mbridge applies extra args when it (re)builds TransformerConfig, which
        # also covers configs created again for model construction.
        try:
            set_extra_args(sequence_parallel=False, apply_rope_fusion=False)
        except TypeError:
            pass
        rebuilt = getattr(bridge, "config", None)
        if rebuilt is not None:
            tf_config = rebuilt
    for target in (tf_config, getattr(bridge, "config", None)):
        if target is None:
            continue
        target.sequence_parallel = False
        if hasattr(target, "apply_rope_fusion"):
            target.apply_rope_fusion = False
    return tf_config


def patch_megatron_engine(module: ModuleType) -> None:
    """Install the strict training route on ``areal.engine.megatron_engine``."""

    if getattr(module, _ENGINE_MARKER, False):
        return
    from rl_engine.integrations.common import cuda_rope
    from rl_engine.integrations.engines.train.megatron.runtime import (
        install_megatron_integration,
    )
    from rl_engine.integrations.orchestrators.areal.head import make_strict_chunked_head

    real_vocab, padded_vocab = contract.vocabulary_contract()
    if not cuda_rope.enabled():
        raise ValueError("strict AReaL training requires RL_KERNEL_CUDA_ROPE=1 on CUDA")
    install_megatron_integration(_strict_plan(), output_layer_mode="external")

    engine_cls = module.MegatronEngine
    original_initialize = engine_cls.initialize
    original_make_configs = module.make_hf_and_mcore_config
    original_make_model = module.make_mcore_model

    @functools.wraps(original_initialize)
    def initialize(self: Any, *args: Any, **kwargs: Any) -> Any:
        contract.raise_if(contract.engine_config_errors(self), "engine configuration")
        token = _CURRENT_ENGINE.set(self)
        try:
            result = original_initialize(self, *args, **kwargs)
        finally:
            _CURRENT_ENGINE.reset(token)
        if bool(getattr(self.tf_config, "sequence_parallel", False)):
            raise RuntimeError("strict AReaL engine finished initialization with SP enabled")
        return result

    @functools.wraps(original_make_configs)
    def make_hf_and_mcore_config(*args: Any, **kwargs: Any) -> Any:
        hf_config, tf_config = original_make_configs(*args, **kwargs)
        engine = _CURRENT_ENGINE.get()
        if engine is not None:
            tf_config = _disable_sp_and_fused_rope(engine, tf_config)
        return hf_config, tf_config

    @functools.wraps(original_make_model)
    def make_mcore_model(*args: Any, **kwargs: Any) -> Any:
        engine = _CURRENT_ENGINE.get()
        hf_config = kwargs.get("hf_config", args[0] if args else None)
        tf_config = kwargs.get("tf_config", args[1] if len(args) > 1 else None)
        if engine is not None:
            contract.raise_if(
                contract.model_config_errors(engine, hf_config, tf_config),
                "model configuration",
            )
        models = original_make_model(*args, **kwargs)
        for model in models:
            gpt = module.unwrap_to_gpt_model(model)
            if bool(getattr(gpt.config, "sequence_parallel", False)):
                raise RuntimeError("constructed Megatron model still uses sequence parallelism")
            if bool(getattr(gpt.config, "apply_rope_fusion", False)):
                raise RuntimeError("constructed Megatron model still uses fused RoPE")
        return models

    strict_head = make_strict_chunked_head(
        module.ChunkedLMHeadOutput,
        real_vocab_size=real_vocab,
        padded_vocab_size=padded_vocab,
    )
    engine_cls.initialize = initialize
    module.make_hf_and_mcore_config = make_hf_and_mcore_config
    module.make_mcore_model = make_mcore_model
    module.chunked_lm_head_logprobs_entropy = strict_head
    setattr(module, _ENGINE_MARKER, True)

    from rl_engine.integrations.common.state import get_active_integration

    integration = get_active_integration("megatron")
    if integration is not None:
        integration.record_installed_hook(
            "logp",
            "rl_engine.integrations.orchestrators.vime.providers.linear_logp_provider.provider,"
            f"{ENGINE_MODULE}.chunked_lm_head_logprobs_entropy",
        )
    print(
        "[RL-Kernel][areal] strict training route installed "
        f"(real_vocab={real_vocab}, padded_vocab={padded_vocab})",
        flush=True,
    )


def patch_actor(module: ModuleType) -> None:
    from rl_engine.integrations.orchestrators.areal.stats import install_actor_stats

    install_actor_stats(module)


def install(mode: str | None = None) -> str | None:
    """Register lazy patches for the current process; returns the active mode."""

    from rl_engine.integrations.orchestrators.areal import profiling

    selected = contract.adapter_mode() if mode is None else mode
    if selected is not None:
        if selected not in contract.MODES:
            raise ValueError(f"unknown AReaL adapter mode {selected!r}")
        os.environ[contract.MODE_ENV] = selected
        when_imported(ACTOR_MODULE, patch_actor)
        if selected == contract.MODE_STRICT:
            when_imported(ENGINE_MODULE, patch_megatron_engine)
    # Registered last so its ranges wrap the strict routes installed above.
    if profiling.enabled():
        when_imported(ENGINE_MODULE, profiling.install_engine_profiler)
    return selected


__all__ = ["ACTOR_MODULE", "ENGINE_MODULE", "install", "patch_actor", "patch_megatron_engine"]
