# H100 results: vime DeepSeek V4 (PR #2) on Hopper (handoff vime-dsv4-20261126)

Hardware: 2 nodes × 4 NVIDIA H100 (95830 MiB, sm90), one Slurm allocation held by this account only (no other job on either node). Every task ran as `srun --overlap` steps inside that allocation, not as its own `sbatch`. Handoff checkout at `a517384fe` for V2, V3 and V4 (V0 built from `cd4fda86`, the same build scripts). Times are UTC. **DSA backend: `cudnn` for V3 and V4** (`cudnn_dsa.wrapper` true in V2's preflight, so `DSA_BACKEND=none` was not used).

| Task | Result | Key numbers | Different from the README's expectation |
|---|---|---|---|
| V0 build | **PASS** | no `STEP_FAIL`; capability assert `(9, 0)`; `IMPORTS_OK`, `BUILD_DONE`; `freeze-diff-vs-b200.txt` empty | `build-steps.log` is empty (0 bytes). Built with 1 GPU and 32 CPUs (README: 48). Two earlier attempts failed, see not-green-attempts |
| V1 checkpoint | **PASS** | download: 11/11 `OK` against `reference/ckpt-download.sha256`; slice `v4-4l`: 10/10 `OK` against `reference/ckpt-v4-4l.sha256`; no `*_HASH_MISMATCH` | optional BF16 version not generated (only V4's optional cases need it) |
| V2 smoke | **FAIL** (verdict, exit 1) | `pr-pytest` 96 passed rc=0; `fix-pytest` 93 passed, 3 skipped, rc=0; `smoke--layer-index_2` 546/545, `_3` 540/539 (= B200); **`smoke--num-layers_4` rc=1: loaded 2164, OOM in backward on rank 0** (tried 4.00 GiB with 90.91 GiB in use, 89.40 GiB allocated by PyTorch; 93.09 GiB card); `rollout_smoke`, `runtime_checks` rc=0; `pr-tree-status.txt` empty; `gpu-free.txt` 0.52 GiB used | the only non-zero item is the 4-layer smoke OOM on the 94 GB card. V3 and V4 were run anyway (see Deviations) |
| V2 preflight | recorded | `device.capability` [9, 0]; `deep_gemm` supported true, `e8m0_used` true; `cudnn_dsa.wrapper` true (cuDNN backend 92400); `fp8_grid` 384x640 and 300x520 `bitwise_equal` true; `l2test.log`: mixed bf16-first `nan` for TE and apex, mixed fp32-first 0.1760 for both (`ref` 0.1814; B200 0.1755) | `e8m0_used=true on H100 (README expected false): DeepGEMM uses UE8M0 scales, same rollout quant format as B200; V4 QAT comparison directly comparable to B200` |
| V3 core (`SUITE=core-r2`) | **FAIL** (4 of 8 cases) | T0-50, T1-50, T2-50: rc 0, 50 steps, 0 non-finite grad records; peaks T0 78.08 / 78.08, T1 57.57 / 57.57 / 21.44 / 21.44, T2 42.08 × 4 GiB (= B200); `logprob_diff` 0.089–0.096 at step 49; median step time from step 2: 14.02 / 14.79 / 11.42 s; T1 and T2 ranks on both nodes, NCCL `NET/IB` only, no `NET/Socket`. `skip-T1-2`: rc 0, 2 steps. **det-R0a, det-R0b, det-R1, det-R2: rc 1**, no replay numbers | the four deterministic replays stop before the first train step with `RuntimeError: deterministic mode: cuDNN's deterministic DSA sparse-attention backward needs an SM100 GPU and a query-head count in [16, 32, 64, 96, 128]; got sm90 with 64 heads` (an explicit check in vime). So `replays` in `summary.json` is empty. Not rerun with `DSA_BACKEND=none` (the README switches backends only when T0-50 fails in DSA; T0-50 passed) |
| V4 drift | **FAIL** (OOM, all 4 cases) | fp8-M2-a, qat-M2-a, fp8-M2-b, qat-M2-b: rc 1, 0 steps. Every case: rollout done, gradients of step 0 finite (per-rank finite grad norm 2.06–3.28, no non-finite tensors), then **OOM in the optimizer step of step 0 on ranks 0 and 3** (PP1 × EP4 on node-a): tried 1.51 GiB with 92.2–92.3 GiB in use, 84.46 GiB allocated by PyTorch. Probe peaks before the step: 59.64 / 57.56 / 57.56 / 57.56 GiB | M2 (FP32 main params) does not fit 4 × 94 GB at PP1 × EP4 (B200 smoke peaked at 85.4–87.5 GiB). Reported as is: no offload, no other split, no `--max-tokens-per-gpu` change. No drift numbers, so the fp8 vs QAT comparison could not be made on H100 |

## Deviations from the README

- **V3 and V4 ran although V2's verdict is FAIL.** The schedule's V2 gate was set by hand to "OK known: num-layers_4 OOM structural (94 GB)" on 2026-11-27 22:39Z, because the only failing V2 item is the 4-layer smoke OOM that the 1-GPU smoke cannot avoid on a 94 GB card. `V2-verdict.txt` itself says `FAIL` and is shipped unchanged.
- **V4 ran before a valid V3.** The first V3 (`SUITE=core`, 2026-11-28 01:50–01:54Z) failed for a scheduling reason (below), and V4 (01:54–02:20Z) followed it in the same chain. V3 was then rerun from scratch as `SUITE=core-r2` (02:21–03:47Z), as the README says for a rerun. V4 used the same `cudnn` DSA backend that V3 then passed with in T0–T2.
- **First V3 attempt used the wrong checkpoint** (`not-green-attempts/V3_attempt1_core_wrong_ckpt/`): the schedule exported a `CKPT` for another task and the handoff's `common.sh` keeps an inherited `CKPT`, so all four cases failed at HF config load within 40 s. Not a vime result. V4 was not affected (`drift.sbatch` sets `CKPT` per case).
- Resources: tasks ran as overlap steps in one 2-node allocation (not `sbatch --exclusive`); no other job shared the nodes. V0 used 32 CPUs instead of 48.

## Files

```
V0/  build.log (redacted), env-freeze-sm90.txt, base-image-freeze.txt, freeze-diff-vs-b200.txt, build-steps.log
V1/  sha256-check.txt (the two `sha256sum -c` outputs of prep_ckpt.sh)
V2/  every file of runs/smoke/
V3/  summary.json, progress.txt, pins-*.txt, gpus-*.txt, preflight-snapshot.txt; per case train.log, node*.log, probe-rank*.jsonl, rc, nccl-NET.txt (grep "NET/" of the nccl logs)
V4/  same as V3
not-green-attempts/
  V0_attempt1_unconstrained_pip/, V0_attempt1_gate_missing_outdir.txt   flash_attn.cute / cutlass import error, NGC index unreachable from the node
  V0_attempt2_home_mounted_as_root/                                    host home mounted as /root by --root (fixed with ENROOT_MOUNT_HOME=n)
  V2_attempt1_shared_gpu/                                              V2 on a card another step already held 13.9 GiB of (invalid run)
  V3_attempt1_core_wrong_ckpt/                                         see Deviations
```

`*.pt` are not included. Job ids, node names, IPs, user and paths are redacted (`<JOB_1>`, `JOB_1` in file names, `node-a`/`node-b`, `<IP>`, `<WORK>`).
