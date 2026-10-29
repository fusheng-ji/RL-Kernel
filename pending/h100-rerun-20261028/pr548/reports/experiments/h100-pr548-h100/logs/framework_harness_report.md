# #548 framework harness (H100 re-run 2026-10-28)

## Configuration this run (handoff)

- `tests.distributed.tp.check_tp_ppo_rollout`
- `--gpu-layout shared --critic-tp 1`
- 4 processes (`torch.distributed.run --nproc_per_node=4`)
- Head: `f0b58af2` on fork `feat/issue-44-tp-ppo`

## Prior published H100 run (for layout comparison only)

- Same harness with **`--gpu-layout dedicated`** (see `hopper-main/pr548/info.txt`: Megatron/DeepSpeed both PASS, two PPO rounds).
- This re-run used **`shared`** per HANDOFF-RERUN-20261028 §3.3; do not treat layout alone as root cause without a controlled re-run.

## Outcome (recorded as-is, no code patches)

| Framework | Result | Primary error (from logs) |
| --- | --- | --- |
| Megatron | **FAIL** | `ValueError: unsupported Megatron checkpoint keys: {'decoder.final_layernorm._extra_state'}` during actor install / weight sync (`frameworks_megatron.log`) |
| DeepSpeed | **FAIL** | `ModuleNotFoundError: No module named 'deepspeed'` in `check_tp_ppo_frameworks.initialize_engine` on ranks 0–2 (`frameworks_deepspeed.log`). Megatron path progressed further before checkpoint-key validation failed. |

## Regression pytest (unchanged this fix)

- `pytest_regression.log`: **108 passed** (`RUN_TP_CUDA=1`).

## Artifacts

- `logs/frameworks_megatron.log`, `logs/frameworks_deepspeed.log`
- `logs/rollout_megatron/`, `logs/rollout_deepspeed/` (partial runtime trees)
