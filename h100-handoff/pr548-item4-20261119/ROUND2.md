# Round 2 (2026-11-20): continue from step 5 at `19b62661`

Round 1 ran `1bba6d19` and stopped at step 5 as the README says: 5 of 8 probe rows completed
(results branch `handoff/h100-pr548-item4-results-20261119`, commits `b6412e38` and
`3f076422`). Both causes are fixed in **`19b62661`**, together with two problems the B200
checks of the fixes found:

- **DeepSpeed BF16 actor and critic failed in `deepspeed.initialize`**
  (`unsupported mix of model_dtype=torch.bfloat16 and grad_accum_dtype=torch.float32`).
  DeepSpeed 0.19.7 builds its `BF16_Optimizer` with FP32 gradient buffers only at ZeRO
  stage 1 and raises at stage 0. DeepSpeed BF16 now runs at ZeRO-1, which partitions nothing
  with DP=1; FP32 stays at ZeRO-0.
- **The Megatron BF16 actor ran out of memory at TP2** (90.0 GiB allocated of 93.09 GiB). Both
  BF16 actors (Megatron and DeepSpeed, which holds the same state) now run at TP4 only, with
  no TP2-vs-TP4 comparison.
- **DeepSpeed BF16 critic: replicated gradients differed between TP ranks** (found on B200 once
  ZeRO-1 let it start). `BF16_Optimizer` packs the parameters into one flat buffer; the
  critic's one-element bias exists only on the TP leader and came first, so every other
  parameter sat two bytes later on the leader, and the RMSNorm backward took a different
  kernel path there. The DeepSpeed BF16 optimizer now lists the parameters every rank holds
  first, and `deepspeed_ppo_worker` checks that the flat layout agrees on all TP ranks.
- **DeepSpeed BF16 critic: `assert all_groups_norm > 0.` in `BF16_Optimizer.step`** at its
  second update. Every gradient of that update is exactly zero, as for the HF reference critic
  of an earlier B200 pre-check (grad norm 0.0 at its second update), which still applies AdamW
  there. The BF16 step now runs DeepSpeed's update without that assert when the norm is
  zero. The same probe peaked at 98.9 GiB per rank at TP2 on B200, so the **DeepSpeed BF16
  critic also runs at TP4 only**.

## What stays from round 1

**Keep steps 1-4 and do not rerun them**: the build, the smoke tests, the calibration and the
frozen reference (`acceptance.json`, bounds, rollout and target tensors). `19b62661` differs
from `1bba6d19` only in these files (`git diff --stat 1bba6d19 19b62661`):

- `rl_engine/integrations/engines/train/deepspeed/ppo.py` (ZeRO stage check; for BF16 the
  parameter order, the flat-layout check and the zero-gradient step);
- `tests/distributed/tp/check_tp_ppo_frameworks.py` (DeepSpeed engine config: ZeRO-1 for BF16),
  `check_qwen3_fixed.py` and `check_qwen3_ppo_rollout.py` (BF16 optimizer parameter order);
- `tools/validation/models/qwen3_ppo/fixed_matrix.sh`, `probe_trainer_memory.sh`,
  `qualify_candidate.sbatch` (TP sizes per row);
- two documents and four new unit-test files.

No HF reference code and no `csrc` changed, so the built `_C` and the frozen reference stay
valid. Keep the round-1 probe directory as it is; the new probe writes a new directory.

## Steps

1. **Move the worktree to `19b62661`.**
   `git -C <src> fetch origin handoff/h100-pr548-item4-20261119`, then
   `git -C <src> checkout --detach 19b62661`. `git -C <src> status --porcelain` must be empty
   and `<trainer-python> -c "import rl_engine._C as c; print(c.__file__)"` must still print a
   path inside `<src>`.
2. **Unit tests (CPU, about 3 minutes).** From `<src>` with `PYTHONPATH=<src>:<overlay>`:
   `<trainer-python> -m pytest -rs tests/integrations/engines/train/deepspeed/test_deepspeed_ppo_zero_stage.py tests/integrations/engines/train/deepspeed/test_deepspeed_bf16_flat_layout.py tests/integrations/engines/train/deepspeed/test_deepspeed_bf16_zero_norm_step.py tests/validation/models/test_qwen3_fixed_matrix.py`
   (26 tests). **Any failure: stop and hand back.**
3. **Step 5 again: trainer memory probe (8 runs).**
   `bash tools/validation/models/qwen3_ppo/probe_trainer_memory.sh <src> <trainer-python> <overlay> <checkpoint> <reference run dir> <results>/probe`
   - **No TP size argument** (round 1 passed `2`). It probes each row once at its smallest TP
     size: the four FP32 rows and the Megatron BF16 critic at TP2; the two BF16 actors and the
     DeepSpeed BF16 critic at TP4.
   - **Go on only if all 8 rows are `probe_complete` and every rank peak is at most 88 GiB.
     Otherwise stop and hand back with `summary.tsv` and the logs.**
4. **Step 6: candidate**, the command from the README, unchanged:
   `bash tools/validation/models/qwen3_ppo/qualify_candidate.sbatch <src> <trainer-python> <vllm-python> <overlay> <checkpoint> <reference run dir> <reference run dir>/acceptance.json <results>/candidate`
   - 13 runs against the frozen bounds (five rows at TP2 and TP4, three rows at TP4 only),
     5 TP2-vs-TP4 comparisons, then the tiny-GPT regression for each engine (4 GPUs).
   - The script stops at the first failing comparison (exit 1). A failure is a result: keep
     it as it is, do not rerun it, and hand back.

## What `19b62661` was checked with before (B200)

One B200 node (183 GiB per GPU, so the peaks below say nothing about OOM by themselves, only
how much each rank allocates). A fresh capture and fresh targets at `19b62661`, then
`probe_trainer_memory.sh` with no TP size argument, exactly as step 3 above:

| Probe | Status | Peak per rank | Round 1 on H100 |
|---|---|---:|---:|
| Megatron, FP32 actor, TP2 | `probe_complete` | 66.1 GiB | 66.1 GiB |
| Megatron, FP32 critic, TP2 | `probe_complete` | 61.2 GiB | 61.3 GiB |
| Megatron, BF16 actor, TP4 | `probe_complete` | 46.6 GiB | (TP2: OOM at 90.0 GiB) |
| Megatron, BF16 critic, TP2 | `probe_complete` | 86.0 GiB | 86.0 GiB |
| DeepSpeed, FP32 actor, TP2 | `probe_complete` | 66.1 GiB | 66.1 GiB |
| DeepSpeed, FP32 critic, TP2 | `probe_complete` | 61.2 GiB | 61.3 GiB |
| DeepSpeed, BF16 actor, TP4 | `probe_complete` | 53.6 GiB | (TP2: failed at init) |
| DeepSpeed, BF16 critic, TP4 | `probe_complete` | 49.6 GiB | (TP2: failed at init) |

All 8 rows complete; the highest peak is the Megatron BF16 critic at TP2 (86.0 GiB, which
already passed on H100 in round 1). On the five rows both runs share, B200 and H100 agree
within 0.1 GiB, so the three new TP4 rows are expected to stay well under 88 GiB on H100.
The 26 unit tests of step 2 also pass on CPU at `19b62661`. This is a pre-check, not
evidence: the H100 run is the result.

## Returning results

- The same results branch, new commits on top of `3f076422`: the new probe directory under
  `results/probe/`, then `results/candidate/`. Commit and push the probe before starting
  step 6.
- Add a "Round 2" section to `SUMMARY.md`: the commit check (`19b62661`), the commands, wall
  time, the probe table (peak per rank) and pass/fail per candidate run and comparison.
- Redaction, `*.log` (`git add -f`), tensors left on disk, `git commit -s` and no AI
  attribution: as in the README.
