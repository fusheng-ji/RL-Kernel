# Round 2 (2026-11-20): continue from step 5 at `246e9911`

Round 1 ran `1bba6d19` and stopped at step 5 as the README says: 5 of 8 probe rows completed
(results branch `handoff/h100-pr548-item4-results-20261119`, commits `b6412e38` and
`3f076422`). Both causes are fixed in **`246e9911`**:

- **DeepSpeed BF16 actor and critic failed in `deepspeed.initialize`**
  (`unsupported mix of model_dtype=torch.bfloat16 and grad_accum_dtype=torch.float32`).
  DeepSpeed 0.19.7 builds its `BF16_Optimizer` with FP32 gradient buffers only at ZeRO
  stage 1 and raises at stage 0. DeepSpeed BF16 now runs at ZeRO-1, which partitions nothing
  with DP=1; FP32 stays at ZeRO-0.
- **The Megatron BF16 actor ran out of memory at TP2** (90.0 GiB allocated of 93.09 GiB). Both
  BF16 actors (Megatron and DeepSpeed, which holds the same state) now run at TP4 only, with
  no TP2-vs-TP4 comparison. The other six rows run at TP2 and TP4 as before.

## What stays from round 1

**Keep steps 1-4 and do not rerun them**: the build, the smoke tests, the calibration and the
frozen reference (`acceptance.json`, bounds, rollout and target tensors). `246e9911` differs
from `1bba6d19` only in these files (`git diff --stat 1bba6d19 246e9911`):

- `rl_engine/integrations/engines/train/deepspeed/ppo.py` (ZeRO stage check);
- `tests/distributed/tp/check_tp_ppo_frameworks.py` (DeepSpeed engine config: ZeRO-1 for BF16);
- `tools/validation/models/qwen3_ppo/fixed_matrix.sh`, `probe_trainer_memory.sh`,
  `qualify_candidate.sbatch` (TP sizes per row);
- two documents and two new unit tests.

No HF reference code and no `csrc` changed, so the built `_C` and the frozen reference stay
valid. Keep the round-1 probe directory as it is; the new probe writes a new directory.

## Steps

1. **Move the worktree to `246e9911`.**
   `git -C <src> fetch origin handoff/h100-pr548-item4-20261119`, then
   `git -C <src> checkout --detach 246e9911`. `git -C <src> status --porcelain` must be empty
   and `<trainer-python> -c "import rl_engine._C as c; print(c.__file__)"` must still print a
   path inside `<src>`.
2. **Unit tests (CPU, about 3 minutes).** From `<src>` with `PYTHONPATH=<src>:<overlay>`:
   `<trainer-python> -m pytest -rs tests/integrations/engines/train/deepspeed/test_deepspeed_ppo_zero_stage.py tests/validation/models/test_qwen3_fixed_matrix.py`
   (20 tests). **Any failure: stop and hand back.**
3. **Step 5 again: trainer memory probe (8 runs).**
   `bash tools/validation/models/qwen3_ppo/probe_trainer_memory.sh <src> <trainer-python> <overlay> <checkpoint> <reference run dir> <results>/probe`
   - **No TP size argument** (round 1 passed `2`). It probes each row once at its smallest TP
     size: the six FP32 and BF16-critic rows at TP2, the two BF16 actors at TP4.
   - **Go on only if all 8 rows are `probe_complete` and every rank peak is at most 88 GiB.
     Otherwise stop and hand back with `summary.tsv` and the logs.**
4. **Step 6: candidate**, the command from the README, unchanged:
   `bash tools/validation/models/qwen3_ppo/qualify_candidate.sbatch <src> <trainer-python> <vllm-python> <overlay> <checkpoint> <reference run dir> <reference run dir>/acceptance.json <results>/candidate`
   - 14 runs against the frozen bounds (six rows at TP2 and TP4, the two BF16 actors at TP4),
     6 TP2-vs-TP4 comparisons, then the tiny-GPT regression for each engine (4 GPUs).
   - The script stops at the first failing comparison (exit 1). A failure is a result: keep
     it as it is, do not rerun it, and hand back.

## What `246e9911` was checked with before (B200)

B200_PLACEHOLDER

## Returning results

- The same results branch, new commits on top of `3f076422`: the new probe directory under
  `results/probe/`, then `results/candidate/`. Commit and push the probe before starting
  step 6.
- Add a "Round 2" section to `SUMMARY.md`: the commit check (`246e9911`), the commands, wall
  time, the probe table (peak per rank) and pass/fail per candidate run and comparison.
- Redaction, `*.log` (`git add -f`), tensors left on disk, `git commit -s` and no AI
  attribution: as in the README.
