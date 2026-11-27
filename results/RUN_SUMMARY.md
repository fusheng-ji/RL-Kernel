# Partial run summary (JOB_X still RUNNING as of push time)

**Allocation:** two 2-node x 4 H100 jobs, non-exclusive, 64 CPU per node. **JOB_X** RUNNING; **JOB_Y** PENDING. A launched 2026-11-27 after step0 retry; **nodea_done** and **nodeb_done** not set at push time.

## Deviations (section 0)

Same as PLAN section 0: not `--exclusive`; 64 CPU; four fixed CUDA lanes on node-b; vime V0 lane uses 32 CPUs in cpuset slice; early step0 failures from `~/.local/bin/env` shadowing (fixed in scripts, attempts in `not-green-attempts/`).

## Gates at push

See `checks/gates-snapshot.txt`. Notable: `step0_A` OK, `nested_srun` OK, `miles_H1` done, `verlomni_F0` OK, `verlomni_F2` done, `vime_V0` FAIL, `vime_V2` FAIL skipped (B1 runs P3/P4 only).

## Failures (retained as-is)

1. **vime V0:** `flash_attn.cute` / `cutlass.cute.core` `ThrMma`; `pypi.ngc.nvidia.com` DNS failed on node. Log: `partial-results/vime/V0.log`.
2. **Early A step0:** env permission denied; nested verdict parser bug (fixed before successful A launch). Logs in `not-green-attempts/`.
3. **node-b restarts:** lane cpuset range bug, missing `out/vime` dir (fixed). Attempt logs in `not-green-attempts/`.

## Partial results bundled (for B200 handoff)

| Slice | Location under `partial-results/` | Handback branch (when complete) |
|---|---|---|
| step0 + checks | `checks/run/`, `checks/nested-*` | schedule-results |
| miles H0+H1 | `miles/` | miles (partial) |
| pipeline P1+P2 | `pipeline/P1,P2` | pipeline (partial) |
| QK2 | `QK2/` | qk-rmsnorm (partial) |
| T10 | `T10/` (logs + metadata; no `tmp/` blobs) | h3-image (partial) |
| vime V0 fail | `vime/` | vime-dsv4 (partial) |
| verl-omni | `verlomni/` (F0, summaries, small runs metadata) | verl-omni (partial) |

## Still running / not started

- node-a: verl-omni F1/F3, optional F3 after gates.
- node-b: T10-a remainder, lanes active.
- B2, C, B1: not started (JOB_Y pending).

## Script fixes in this push

`/usr/bin/env` for srun steps; PATH strips non-executable `~/.local/bin/env`; `out/checks` mkdir; nested `===` block parser; `cpu_lane_specs` comma lists; `stage_push.sh` full redaction + leak grep.