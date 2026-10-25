# PR #548 on B200: unit tests only; candidate stage not run

Head `4585cd3183cf85864adc50f117c55fa92324f1e2`, two B200, torch 2.13.0+cu130, shared environment without Megatron Core / DeepSpeed / the vLLM sidecar environment. Runner: [run.sh](run.sh).

| Run | passed | failed | skipped |
|---|---:|---:|---:|
| TP PPO unit tests (6 files) | 45 | 0 | 1 (`test_tp_scalar.py:296`, opt-in multi-GPU validation) |

The Qwen3-8B candidate stage was not started: see [CANDIDATE-STAGE-PLAN.md](CANDIDATE-STAGE-PLAN.md) for the plan, the cost and the missing inputs (raw reference, pinned environments, checkpoint).

Draft addition for the PR body Status: *no change to the qualification claim; at head `4585cd31` the unit tests pass (45 passed, 1 skipped, B200, torch 2.13).*
