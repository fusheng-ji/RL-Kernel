# Experiment design (H3 VeRL-Omni, #377 pattern)

## Reference implementation

Follow layout and integrity rules from
[`examples/vime_qwen3_8b_tp4_cp2_200/`](https://github.com/RL-Align/RL-Kernel/tree/main/examples/vime_qwen3_8b_tp4_cp2_200):

- `experiment_matrix.json` — pinned versions, arms, topology
- Per-step CSV with strictly aligned step pairs
- `summary.json` — bootstrap CIs (20k resamples, fixed seed)
- No imputation; finite required fields only
- Plots: performance matrix, paired effects, consistency vs reward

## Arms (minimum + localization)

| Group | ID | Train ops | Rollout ops | Purpose |
| --- | --- | --- | --- | --- |
| Baseline | G10 | Production | Production | P/P; expect non-zero drift |
| Target | G11 | RL-Kernel strict | RL-Kernel strict | R/R; zero drift |
| Localize train | — | RL-Kernel | Production | R/P |
| Localize rollout | — | Production | RL-Kernel | P/R |

## VeRL-Omni H3 recipe (pinned in manifest)

- Pipeline: FlowGRPO / DiffusionNFT-style H3 (video+audio latents).
- Actor: Diffusers `MiniMaxH3Transformer3DModel`, FSDP2, Ulysses `ACTOR_SP` (often 2 on smoke).
- Rollout: vLLM-Omni H3 transformer; TP configurable (smoke TP=1; target **TP=2** for realism).
- Scheduler: `FlowMatchSDEDiscreteScheduler`, `sde_type=cps`, `noise_level=0.8`, dual schedulers merged 0.5/0.5.
- Smoke entry: `tests/special_e2e/run_flowgrpo_minimax_h3_tiny.py`.
- TP=2 sync: `tests/special_e2e/run_minimax_h3_lora_sync_tp2.sh`.

## Metrics

### Existing (VeRL-Omni)

From `rollout_correction.compute_rollout_corr_metrics_from_logprobs`:

- `rollout_corr/logprob_abs_diff_max`, `logprob_abs_diff_mean`, per-timestep variants
- KL / PPL diff fields when logprobs enabled

**R/R pass:** max and mean abs diff **0** every step (after SDE contract defines comparable logprobs).

### Required additions (B4)

- `velocity` and `prev_sample_mean` max abs diff (train vs rollout), per modality if split
- Readback blob: requested vs actual backend per replaced op

### Performance (#377 style)

- Rollout wall time per step
- Actor train time (and reference logp if applicable)
- End-to-end step time
- Report paired % delta with 95% bootstrap CI

## Phased horizons

| Phase | Steps | GPUs | Weights | Goal |
| --- | --- | --- | --- | --- |
| Smoke | 2–8 | 2 | Tiny random | CI-style; B6 |
| Short | 8–32 | 8 | Real | First P/P vs R/R parity |
| Scale | 3 seeds × N | 8 | Real | Bootstrap + figures for publish |

## Topology target

- **8× H100** (two nodes × 4 GPUs): FSDP actor + rollout TP=2 (matches community cluster constraint).
- Pin: torch, vLLM, vLLM-Omni, verl-omni, RL-Kernel SHAs in manifest (see `rl_kernel_h3/MANIFEST.md` pattern on probe branch).

## Artifact layout (target)

```
examples/minimax_h3_verl_omni_e2e/
  experiment_matrix.json
  analyze_performance.py
  collect_results.py
  results/<run_id>/
    rounds.csv
    summary.json
    figures/
```

Evidence bundles also land on `pr_assets` branch under `pending/h3-verl-omni-e2e/` with commit SHA cited in PR text.
