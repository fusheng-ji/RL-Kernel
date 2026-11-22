# PR #548 item 4 on 4×H100 (2026-11-19 to 2026-11-22)

This is the fixed-rollout Qwen3-8B qualification, as run and redacted by the H100 tester. The
large `*.pt` tensors stay on the tester's scratch storage, and the committed manifests anchor their
hashes. Scheduler dumps (`slurm.txt`) were removed before upload.

- `SUMMARY.md`: every step, with commands, wall times and deviations. It covers round 1 at
  `1bba6d19` and round 2 at `19b62661`.
- `calibration/`, `reference/`: the calibration, and the frozen reference with its 12 HF runs,
  bounds files and `acceptance.json`.
- `probe/probe-20261122T000414Z/`: round 2 trainer memory probe, 8 of 8 rows complete, peak
  86.0 GiB per rank.
- `candidate/`: round 2 step 6, stopped at run 2. The Megatron FP32 actor passed at TP2 and failed
  at TP4 on one tensor.
- `round2/diagnosis/`: CPU diagnosis of that tensor (`diagnose_update_sensitivity.py`). It holds
  the step-0 gradient and update tensors of the three tensors it examined, for five runs.
- `not-green-attempts/`: failed environment attempts, unedited.
