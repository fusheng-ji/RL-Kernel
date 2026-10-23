# VeRL-Omni integration outline (B4)

Target repo: community fork of `verl-project/verl-omni`. Branch **`rl-kernel/h3`** off fork **`main`**
(baseline runs use **`main`** unchanged).

RL-Kernel side: stack through [#575](https://github.com/RL-Align/RL-Kernel/pull/575) (runtime modes,
Diffusers adapter, training provider, checkpoint mapper, vLLM-Omni adapter).

## Environment routing

| Variable | Values | Effect |
| --- | --- | --- |
| `RL_KERNEL_H3_MODE` | `off` / `strict` / `audit` | Maps to `H3RuntimePolicy` |
| `RL_KERNEL_H3_MANIFEST` | path to JSON | Pins RL-Kernel + fork SHAs |
| `RL_KERNEL_H3_READBACK` | `1` | Emit per-op backend trace |

P/P: `off` on both sides. R/R: `strict` on both. R/P and P/R: mixed per arm table in `experiment-design.md`.

## Rollout (vLLM-Omni)

Module reference: vLLM-Omni `diffusion/models/minimax_h3/minimax_h3_transformer.py`.

| Native piece | RL-Kernel replacement | Notes |
| --- | --- | --- |
| `QKVParallelLinear` | Split or fused contract via `h3_qkv_gemm` + column equality test | Fused vs Diffusers split q/k/v |
| `fused_qk_norm_rope` | `h3_qk_rmsnorm_d128` + `h3_mm_rope_3axis_partial` or approved fused replay | Order: norm then partial RoPE |
| `Attention` | #553 or optional flash (#576) under strict policy | Non-causal full attention |
| `RowParallelLinear` (o_proj, fc2) | Deterministic TP GEMM + `DeterministicCollective` | TP=2 target |
| `MergedColumnParallelLinear` + `SiluAndMul` | `h3_ffn_*` + FP32 SwiGLU contract | BF16 native violates H3 FP32 SwiGLU rule |
| AdaLN Triton fusions | Registry AdaLN rows | Match six-output gather |

Integration hook: extend existing RL-Kernel vLLM-Omni adapter from #575 (`vllm_runtime_adapter_h3` row);
avoid duplicating layer names in VeRL-Omni core—prefer sitecustomize or worker plugin loaded when
`RL_KERNEL_H3_MODE=strict`.

## Actor (Diffusers + FSDP2)

File: `verl_omni/pipelines/minimax_h3_flow_grpo/diffusers_training_adapter.py`.

- Wrap **before** FSDP: replace block forward/backward with RL-Kernel `CandidateOps` or training provider.
- Ulysses: keep `h3_ulysses_forward`; ensure CP metadata row (`cp_metadata_coshard`) matches indices from rollout.
- Transition: `sample_h3_transition` must call the **same** strict transition op as rollout once B2 is merged.

## Metrics wiring

Extend trainer step to log:

```text
rollout_corr/velocity_abs_diff_max
rollout_corr/prev_sample_mean_abs_diff_max
```

(computed on overlapping tensors after batch alignment), in addition to logprob fields.

## Fusion equivalence tests (gate before B6)

1. For each head and token, fused QKV output columns equal stacked `to_q`, `to_k`, `to_v` outputs.
2. Fused qk-norm-rope equals sequential strict ops on BF16 fixtures.
3. TP=2: single-rank partial vs full tensor gather byte-equal.

## File layout (proposed in fork)

```text
verl_omni/integrations/rl_kernel/
  __init__.py
  manifest.py
  actor_patch.py
  rollout_patch.py
  metrics.py
  README.md
```

## Dependencies

- B1: native baseline timings and P/P drift numbers.
- B2: logprob comparator semantics.
- B3: rollout latency acceptable for multi-step jobs.
- RL-Kernel PRs merged or manifest-pinned SHAs built locally.
