# AReaL train/rollout consistency

Verify, on an **unmodified AReaL checkout**, that RL-Kernel makes the rollout
log-probabilities from vLLM and the recomputed log-probabilities from Megatron
bitwise identical for the same weights and tokens.

## What is compared

AReaL already logs, for every recomputed batch,
`compute_logp/r3/rollout_train_logp_abs_diff/{avg,min,max}`: the per-token
difference between the logp vLLM returned and the logp the trainer recomputes
before the update (`areal/trainer/ppo/actor.py`, `_log_r3_rollout_train_logp_stats`;
it runs whether or not R3 is enabled). The verifier judges with that metric plus
exact counts added next to it under `compute_logp/consistency/`:

| metric | meaning |
| --- | --- |
| `n_valid_tokens`, `mismatch_tokens` | compared tokens and how many differ (`!=`, not a tolerance) |
| `n_same_version_tokens`, `same_version_mismatch_tokens` | tokens generated with the weight version the actor holds at recomputation |
| `n_stale_tokens`, `stale_logp_abs_diff` | older-version tokens; differences here are real policy change |
| `n_mixed_seq_same_version_tokens`, `mixed_seq_same_version_mismatch_tokens` | same-version tokens inside sequences that were interrupted by a weight update and resumed |

Synchronous training (`max_head_offpolicyness=0`): every compared token must
match. Asynchronous training: only same-version tokens must match. AReaL pauses
generation with `clear_cache=True` on weight updates and resubmits interrupted
requests with the full prefix, and records a version per generated segment, so a
same-version token was produced from the full prefix with exactly those weights.
The mixed-sequence subset exercises that interrupt/re-prefill path explicitly.

## How RL-Kernel is attached (no AReaL source change)

| process | mechanism |
| --- | --- |
| actor worker (Megatron) | `actor.scheduling_spec[0].cmd = python3 -m rl_engine.integrations.orchestrators.areal.worker`; the worker installs lazy patches, then runs `areal.infra.rpc.rpc_server` unchanged |
| vLLM server | `VLLM_PLUGINS=rl_kernel` plus `RL_KERNEL_*` environment, inherited by AReaL's vLLM subprocess |
| controller | `run_trainer` wraps `StatsLogger.commit` to tee committed stats to JSONL |

In `RL_KERNEL_AREAL_MODE=strict` the worker:

1. installs the shared Megatron runtime (strict Attention, FFN, RMSNorm and
   fixed-tree TP collectives) with `output_layer_mode="external"`;
2. installs shared CUDA RoPE (`RL_KERNEL_CUDA_ROPE=1`, the same `RoPESM90Op` that
   vLLM uses);
3. turns off sequence parallelism and fused RoPE before the model is built, and
   checks the built model;
4. replaces `areal.engine.megatron_engine.chunked_lm_head_logprobs_entropy`
   with the strict LM head. AReaL's own chunked-head path still does label
   rolling, CP splitting, CP reassembly and unpadding;
5. adds the `compute_logp/consistency` metrics.

`RL_KERNEL_AREAL_MODE=observe` only adds the metrics (native numerics); the
native arm of an asynchronous comparison uses it so both arms report the same
breakdown. The native arm of a synchronous comparison runs stock AReaL, unless
`verify run --native-observe` is given, which switches it to observe as well.
`verify run --native-rope-fusion` turns on Megatron ``apply_rope_fusion`` for the
native arm via the RL-Kernel worker (TransformerConfig patch only; AReaL
unchanged). It cannot be combined with the RL-Kernel strict arm.

## Supported scope

Dense untied Qwen3 (validated on Qwen3-8B), BF16, mbridge, Megatron TP 1/2/4/8,
CP 1/2, PP 1, no LoRA/FP8/MTP/critic/tree training, `temperature=1.0`,
`megatron.enable_chunked_logits=true`, `entropy_requires_grad=false`, SM90 GPUs.
Anything else is rejected before the model is built.

## Running

Install RL-Kernel (with its CUDA extension) into the AReaL environment. Then:

```bash
export AREAL_ROOT=/path/to/AReaL MODEL=/path/to/Qwen3-8B
bash rl_engine/integrations/orchestrators/areal/run_all.sh
```

Stages, each in `$OUT_ROOT/<stage>`; the run stops at the first failing stage and
packs everything into `$OUT_ROOT.tar.gz`:

| stage | what | GPU |
| --- | --- | --- |
| S0 | package / GPU / source inventory | no |
| S1 | preflight: every patched AReaL, Megatron and vLLM symbol exists with the expected signature | no |
| S2 | RL-Kernel arm smoke, 2 steps, small batch, vLLM CUDA graphs | yes |
| S3 | synchronous, native vs RL-Kernel, vLLM CUDA graphs on both arms | yes |
| S4 | asynchronous (`max_head_offpolicyness=1`), native vs RL-Kernel, CUDA graphs | yes |
| S5 | synchronous, native vs RL-Kernel, vLLM `enforce_eager` (earlier setup, control) | yes |
| S6 | `PERF.md` / `perf.json`: consistency and performance over all stages | no |

Use `--from S3` or `--only S4` to resume. Individual commands:

```bash
python -m rl_engine.integrations.orchestrators.areal.verify preflight --areal-root $AREAL_ROOT --model $MODEL
python -m rl_engine.integrations.orchestrators.areal.verify run --areal-root $AREAL_ROOT \
    --model $MODEL --out out/sync --mode sync --arms native,rlkernel --steps 5
python -m rl_engine.integrations.orchestrators.areal.verify analyze --out out/sync
python -m rl_engine.integrations.orchestrators.areal.verify perf --out out
```

The default layout is training TP4 on four GPUs and one vLLM TP4 replica on the
other four (`--train-tp`, `--train-cp`, `--rollout-tp`, `--rollout-dp`). vLLM runs
with `enforce_eager` unless `--vllm-execution graph` is given. `graph` follows the
VIME CUDA experiment of RL-Kernel PR 377: `--optimization-level 0` (no
torch.compile) and `cudagraph_mode=FULL_DECODE_ONLY` captured at every batch
size `1..--max-num-seqs`. AReaL's `vLLMConfig` has no field for these, so the
verifier passes them through `RL_KERNEL_AREAL_VLLM_SERVER_ARGS` and
`run_trainer` appends them to the arguments `vLLMConfig.build_args` returns.
Both arms receive the same value. Extra environment for
the RL-Kernel arm (for example an FA4 compatibility layer) goes through
`--env KEY=VALUE` or `RLK_ENV`. The RL-Kernel arm also enables the persistent
FA4 CuTe JIT cache (`FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`;
`FLASH_ATTENTION_CUTE_DSL_CACHE_DIR` is passed through when set), so later runs
load the compiled kernels instead of rebuilding them.

On clusters without an eight-GPU node, `--nodes 2 --gpus 4 --scheduler ray`
runs the same layout on two nodes through AReaL's Ray scheduler (`--gpus` is
per node; training TP×CP and rollout TP must each fit in one node). Start the
Ray cluster first and export `RAY_ADDRESS`. If the first data-service worker on
a remote node needs more than AReaL's fixed 30 s to start, set
`RLK_AREAL_WORKERS_READY_TIMEOUT` (seconds); both arms get the same value.

Placement matters for timing: a GPU role that shares a node with the driver, the
AReaL controller and the Ray head ran up to ~4x slower in our runs. Prefer a Ray
head with `--num-gpus=0` on a separate CPU node and run the driver there. To pin a
role instead, start the chosen raylets with a custom resource
(`ray start --resources='{"rlk_actor_node": 1}'`) and set
`RLK_AREAL_RAY_ROLE_RESOURCES=actor=rlk_actor_node` (comma-separated
`role=resource` pairs); every placement-group bundle of that role then also asks
for a sliver of the resource.

`--profile-steps 3` (comma-separated actor versions) wraps every recomputed-logp
`forward_batch` and PPO `train_batch` call of those versions on actor rank 0
in `torch.profiler`, for both arms. Coarse `rlk::attention`, `rlk::mlp`,
`rlk::rope`, `rlk::lm_head` and `rlk::det_collective` ranges sit around the same
Megatron entry points in both arms, so the profiles aggregate into comparable
buckets. A JSON summary and the `key_averages` table per call land in
`<arm>/profile/`. Profiled steps include the profiler's own overhead and are
attribution data, not timing evidence.

## Outputs and verdicts

Each run directory holds `spec.json`, `REPORT.md`, `summary.json`, `perf.json`,
and per arm `config.yaml`, `run.json`, `trainer.log`, `stats.jsonl` and
`readback/`. `stats.jsonl` rows carry `wall_time`, so end-to-end step time is
measured, not estimated.

`REPORT.md` and `PERF.md` list, per arm, the steady-state (first step skipped)
`timeperf/*` means, rollout throughput (generated tokens / `timeperf/rollout` /
rollout GPUs, sync only), end-to-end step time, and graph evidence parsed from
the vLLM `rollout.log` (`Cudagraph is disabled`, `cudagraph_mode`, `Graph
capturing finished`). A run that asked for graphs but whose log shows eager
mode is flagged. Comparisons report candidate/baseline ratios with a paired
bootstrap 95% interval over steps.

| verdict | meaning |
| --- | --- |
| `PASS` | RL-Kernel arm: zero mismatching tokens (sync) or zero same-version mismatches with enough same-version tokens (async), every strict route proven to execute |
| `FAIL` | any mismatch, missing route evidence, fallback, non-finite value or trainer failure |
| `INCONCLUSIVE` | async run with fewer same-version tokens than `--min-same-version-tokens` |
| `REPORT_ONLY` | native arm; shown as the control |

Route evidence comes from `RL_KERNEL_READBACK_DIR`: Megatron and vLLM each must
report executed Attention, FFN and Logp operators, and both roles must report
the shared CUDA RoPE. The native arm must produce no readback.

## Known limits

- Equal log-probabilities do not imply that every intermediate tensor or gradient
  is bitwise equal, nor better rewards or convergence.
- The native control differs from the RL-Kernel arm in sequence parallelism
  (AReaL default) as well as in operators.
- Matching AReaL internals by name ties the adapter to the tested AReaL revision;
  S1 fails fast when a patched symbol moves.
