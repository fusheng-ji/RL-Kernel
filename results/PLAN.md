# H100 integrated schedule: PLAN (2026-11-27)

Status: preparation done; scripts revised per review (`9e1874f6` amend). **Auto-run enabled**: when `<JOB_1>` or `<JOB_2>` turns `RUNNING`, `scripts/on_job_start.sh` runs `prep_verify.sh`, step 0, and allocation A (first job) or C+B1 (second job) without waiting for human OK if must-pass gates pass. Paths: `<WORK>` = scratch work dir, `<OUT>` = `<WORK>/out`.

**Every `SUMMARY.md` must state these deviations from the schedule README:** not `--exclusive`; 64 CPU per node (not 96); two 48 h jobs mapped to A/B1/B2/C time slices; vime V0 uses 32 CPU on node-b lane 3 (not 48); node-b uses four fixed `CUDA_VISIBLE_DEVICES` lanes (not seven overlapping `srun` steps); clean-timing rows on a node with `other_jobs_on_node` > 1 in step 0 are labelled **shared node** in the summary.

## 0. Stop-and-ask items (read first)

1. **The allocation is not the one in the README.** The README asks for five `salloc -N2 --gres=gpu:4 --exclusive` allocations. What exists are two running-to-be jobs, `<JOB_1>` and `<JOB_2>` (2 nodes x 4 H100, 725 GB and **64 CPU per node**, not exclusive, 48 h each, body is `sleep 2d`). A fresh `--exclusive` two-node request of 18 h was tested with `sbatch --test-only` and would start on 2026-12-10 (`checks/test_only_N2_exclusive_18h.txt`), so the existing jobs are the only way to run this week. Consequences, to be written into every `SUMMARY.md`:
   - not `--exclusive`; the nodes are full by memory (725 of about 750 GB), but they are not exclusive in the Slurm sense;
   - 64 CPU per node: vime V0 gets 32 CPU instead of 48 and may take longer than 3 to 5 h;
   - two 48 h jobs replace five allocations; the README's A, B1, B2a, B2b, C become time slices inside them (section 3);
   - `pkill`: the miles launcher runs `pkill -9 sglang / miles / redis`. Inside my own job that can only hit my own processes, so nothing else may run on these nodes while a miles stage is running.
2. **AReaL (allocation C)** follows release `areal-perf-round1-logs` (`areal-ropefusion-placement-run.tar.gz`, `NOTES.md`, bundle `start_ray.sh`, `areal_run_verify.sh` with cwd `<OUT>`). L1/L2 pin actor via `RLK_AREAL_RAY_ROLE_RESOURCES` and `RAY_NODE_RESOURCES`. C4 (CPU-node driver) is **skipped** unless a CPU job can reach both GPU nodes. Ray teardown uses `scancel` on known `start_ray` step ids only (no `ray stop --force`).
3. **Node-local disk, driver, `memory.total`, InfiniBand and Hub access from the H100 nodes cannot be measured without a GPU allocation.** They are step 0 of allocation A (`scripts/step0_checks.sh`). The README says: Hub not reachable from compute nodes -> run F0 once on a node that has it (the FA3 `v1` kernel is already cached, see 2); node-local disk too small for vime -> stop and ask. `VIME_LOCAL` (default `/tmp`) is where vime's enroot data goes.
4. **Shared file system.** The project disk behind `datastorage` is at 100% quota (39 GB free), so the 700 GB of downloads went to the scratch file system (4.3 TB free, 97% full). Scratch can be purged by policy; if the files disappear the downloads have to be redone (`scripts/miles_h0_noimport.sh`, `scripts/vime_prep.sh`, `scripts/verlomni_prep.sh`). Both-node read/write of `<WORK>` is tested in step 0.

## 1. Cluster checks (README 2.1)

| Check | Required | Result | Evidence |
|---|---|---|---|
| Card | H100, `memory.total` >= 95,000 MiB | **open**: node feature says `H100-92GB`; the number comes from step 0 | `checks/step0-*/gpus.csv` |
| Driver | >= 580 | 580.173.02 on this partition (last run, 2026-11-23); re-measured in step 0 | `checks/step0-*/nvidia-smi.txt` |
| Inter-node network | InfiniBand | **open** (step 0) | `checks/step0-*/ib.txt` |
| Shared FS, about 700 GB | both nodes read/write | scratch: 4.3 TB free; project disk: 39 GB free (not usable). Write test from both nodes in step 0 | `checks/prep-verify.txt` (disk line) |
| Node-local disk, short `TMPDIR` | enough for enroot data and vLLM sockets | CPU node: 22 GB root, `/dev/shm` 182 GB. H100 nodes: **open** (step 0) | `checks/cpu_job_*.log`, `checks/step0-*/df-local.txt` |
| Job time limit >= 18 h | yes | partition `MaxTime` 14 days; `--test-only` accepted `-t 18:00:00` | `checks/test_only_N2_exclusive_18h.txt` |
| Hub from compute nodes | F0 needs the FA3 kernel | login node and CPU node reach the Hub (HTTP 200). H100 nodes: **open** (step 0) | `checks/cpu_job_*.log`, `checks/step0-*/hub.txt` |
| enroot / pyxis | miles, vime | `enroot 3.5.0` on the CPU node (root disk only 22 GB, so imports need `/dev/shm`), not on the login node; `srun --container-image` (pyxis) exists | `checks/cpu_job_*.log` |
| Account / partition | `-N2 --gres=gpu:4 --exclusive` schedulable | schedulable but only from 2026-12-10 (see 0.1); the two existing jobs are used instead | `checks/test_only_N2_exclusive_18h.txt` |

## 2. Preparation status (README 2.2 to 2.4)

All verification output is in `checks/prep-verify.txt`.

| Row | Status | Verification |
|---|---|---|
| Fork heads vs README table | done, all equal | `ls-remote` lines in `prep-verify.txt` |
| `<W10>` `4cfa3cbfc` | done (worktree, not built) | commit and tree `d407d539...` equal `reference/tree.txt`; T10 bundle verified after fetching the prerequisite commit `a5989aede` |
| `<WQ>` `c56b81c4` | done (worktree, **must be rebuilt**) | the previous `build.log` does not show whether `KERNEL_ALIGN_DET_GEMM_SM90=1` was set, so QK2 recompiles `<WQ>` with it and keeps `build.log` |
| `<WB>` `13d0f906` | done (worktree, not built) | tree `c563a92f...` |
| `<WP>` `307cf4747` | done (worktree, not built) | parent is `008ca510d` as the README says |
| `<E1>` / `<E>` | present, torch 2.13.0+cu130, not touched | |
| `<V>` | present: diffusers 0.42.0.dev0, accelerate 1.16.0 | |
| `<E2>` | present: torch 2.14.0+cu130, vLLM 0.33.0, vLLM-Omni `6039984e3`, verl 0.10.0.dev0 | |
| verl-omni `708f4d63` | done, clean | `<WORK>/verlomni/verl-omni` |
| `<OVERLAY>` | rebuilt with `uv pip install --no-deps --target` (`<E2>` has no pip; `<E2>` itself unchanged); the six reward packages import | |
| `<REWARDS>` | done: CLAP at `ada0c23a`; `imagebind_huge.pth` sha256 `d6f6c22b...` equal | |
| FA3 kernel `v1` | cached in `<WORK>/verlomni/hf` at `7cb368cf...`, build `torch-stable-abi29-cu130-x86_64-linux`; use `HF_HOME=<WORK>/verlomni/hf` with `HF_HUB_OFFLINE=1` if the compute node has no Hub | |
| `<IMG>` | VAE sha256 `5bb28096...` equal | |
| `<CKPT>` / `<WEIGHTS>` | `h3_conditioning`, `h3_block0`, `hf_full/transformer/*.index.json` all present; `h3_shard_root()` resolves on the CPU with `RL_KERNEL_H3_WEIGHTS=<WEIGHTS>`. The cause of last round's skip is therefore the unset variable (QK) and was not the layout; `prepare_h3_weights.py` was not needed | |
| T10-a step c input `<T9 run-a>` | present | |
| miles H0 | image digest `amd64 sha256:0a106222...` equal; 16 files, `download_sha256` equal to `reference/b200-summary.json`. The model directory is 19 GB, not the 70 GB the schedule README states; the hashes are what is gated. Image imported on a CPU node (`checks/cpu_job_*.log`): sha256 of the squashfs file is in `prep-verify.txt`, size 39.2 GB, `unsquashfs -s` reads it (`checks/sqsh-superblock.txt`). **First attempt failed** (`not-green-attempts/`): enroot's temporary tree on the shared file system made `mksquashfs` print `Cannot stat` for hundreds of files and the job was cancelled by me (exact job id) before it finished; the second attempt used `/dev/shm` for the temporary tree and ended without such lines. Gate `miles_H0` is set | |
| vime `src/` | done: vime `aebf8fea`, fix `a2d04354` + 0002 diff `ea9498d46771cd6e`, Megatron-LM `ae76c60e6`, Megatron-run + 0004 diff `95f19b25da8b1ff7` | all four equal the README |
| vime V1 | done: 11 downloaded files all OK; 4-layer slice matches `reference/ckpt-v4-4l.sha256` (no `SLICE_HASH_MISMATCH`). The optional BF16 copy (52 GB) was not made; it is only needed for the optional `bf16-M2-a` / `rnd-M2-a` cases of V4 | |
| vime image, miles H1 environment, RL-Kernel builds | by design not prepared (V0 in A, H1 in A, builds in A) | |

## 3. Allocations mapped onto the two jobs

Notation: `JOB_X` first, `JOB_Y` second. **node-b** uses one `node_step` (4 GPUs, 64 CPU) and `u_nodeb.sh` with four fixed lanes (`CUDA_VISIBLE_DEVICES` 0–3 + `taskset`). **node-a** builds `<WP>` inside `u_a_nodea.sh` then runs clean-timing work on all four GPUs. **B1/B2** need gate `nested_srun` (`step0_nested.sh`): two host hashes and 4 GPUs each from `srun -N2` inside a driver step; if that fails, `driver_2node` falls back to <CLUSTER> `bash` with only `SLURM_JOB_ID` set (`checks/nested-srun-2node.txt.verdict`). Miles/vime drivers use in-allocation `srun` when the gate passes.

| Slice | Where | Script | Order and time | Output |
|---|---|---|---|---|
| step 0 | both nodes JOB_X | `step0_checks.sh` (`step0-node-a/b`, `rw-node-a/b`, `other_jobs_on_node` count), `step0_twocards.sh` (record), `step0_nested.sh`, `step0_mustpass.sh` | ~10 min | `<OUT>/checks/`, `gates/step0_A`, `gates/nested_srun` |
| A node-b | 4 lanes on one 8 h `node_step` | `u_nodeb.sh`: L0 W10+T10-a; L1 T10-edit after `ext.txt`; L2 WQ+WB+QK2; L3 V0+V2 (skip if `vime_local` FAIL) | 8 h | `<OUT>/build`, `<OUT>/T10`, `<OUT>/QK2`, `<OUT>/vime` |
| A node-a | build WP + F0–F2, P1–P2, H1, opt F3 | `u_a_nodea.sh` | 8 h | `<OUT>/verlomni`, `<OUT>/pipeline/P1,P2`, `<OUT>/miles` |
| B1 (12 h) | node-a driver spanning both nodes | `alloc_B1.sh` (P3 -> P4 -> V3 -> V4) | 12 h; 4 h if `vime_V0` is FAIL | `<OUT>/pipeline/P3,P4`, `<OUT>/vime/V3,V4` |
| B2a (8 h) | node-a driver | `alloc_B2.sh JOB a` (H2 -> H3 -> H4 -> H5) | 8 h; H4's SGLang self-check runs on one node after the trainer | `<OUT>/miles` |
| B2b (18 h) | node-a driver | `alloc_B2.sh JOB b` (H6 -> H7; H8 is CPU work done outside) | 18 h; on requeue move the partial run to `not-green-attempts/` and set `TAG6=r1` | `<OUT>/miles` |
| C (8 h) | node-a driver, 64 CPU, Ray on both nodes via bundle `start_ray.sh` | `alloc_C.sh`, `u_areal.sh`, `areal_run_verify.sh` | 8 h | `<OUT>/areal` |

Assignment to the two jobs:
- **JOB_X** (starts first, estimated 07:38 today, the Slurm estimate moved from 10:50 earlier): step 0, A (8 h), then B2a (8 h), then B2b (18 h). About 34 h of the 48 h; the rest is slack for H6 overruns.
- **JOB_Y** (starts second, estimated 11:13): C first (it is independent), then B1 once the gates `vime_V0`, `vime_V2` and `build_WP` exist (A has run 5 to 6 h by then). About 20 h of the 48 h.
- If only one job starts: A, then B1, then B2a; B2b only if more than 18 h are left, otherwise it waits for the other job; C goes wherever there is time.
- Order is as in the README: preparation, A, then B1, B2a-B2b and C in any order.
- Clean timing (verl-omni F1 to F3, pipeline `seconds`, miles H5, vime V3/V4) only runs where nothing else runs on the node: node-a of A (for F1 to F3 and the pipeline single-node rows) and the two-node slices of B1/B2. Because the two jobs are not exclusive, B1/B2/C must not overlap on the same job.
- Each time slice begins with `prep_verify.sh` on the login node (`on_job_start.sh`); restore anything missing before runs.

## 4. Stop conditions mapped to script steps

| Condition | Where it is checked | Effect |
|---|---|---|
| Card is not 94 GB | `step0_checks.sh` -> `gpus.csv`; recorded, not gated | schedule as is; T10-edit and V4 OOMs are returned as they are |
| F0 `same_kernel_file` false, a repeat not bitwise, or a kernel does not load | `u_verlomni.sh`, python gate after `preflight.json`, sets `gates/verlomni_F0` to `FAIL` and exits | verl-omni stops at F0; `u_a_nodea.sh` continues with P1/P2; F3 is skipped |
| vime V0 `STEP_FAIL` or no `BUILD_DONE` | `u_vime.sh V0` greps `V0.log`, sets `gates/vime_V0` | V2 refuses to start; `alloc_B1.sh` sees the FAIL gate and runs P3/P4 only with `--time 04:00:00` |
| vime V2 cuDNN DSA fails | `u_vime.sh V2` reads `smoke/preflight.json` `cudnn_dsa.wrapper`, writes `gates/vime_dsa` | not a stop: V3/V4 run with `DSA_BACKEND=none`; a failure inside V3's T0-50 must also be moved to `not-green-attempts/` and V3/V4 re-launched with `DSA_BACKEND=none` by hand |
| miles H0 digest wrong | prep: `gates/miles_H0` set only if `image_digest.txt` equals `sha256:0a106222...`; `u_miles.sh` and `alloc_B2.sh` refuse without it | no miles at all (A's H1, B2a, B2b) |
| miles H2: b, c, d all fail | `u_miles.sh H2` auto-judges README pass rule → `gates/miles_H2` `OK setting=<x>` or `FAIL b,c,d` | H3 onward and B2b do not start |
| miles H5 estimates H6 > 16 h | `u_miles.sh H5` parses step 1–9 mean → `gates/miles_H5` `OK est=<h>h` or `FAIL H6>…h` / `FAIL unparsed` | `alloc_B2.sh b` refuses to start B2b |
| Driver < 580 | step 0 | no miles |
| pipeline rows 925/928 do not fit | `run` in `u_pipeline.sh` records the exit code per row in `rows.txt` | rerun the row with `--size 40` under a new output name and record the OOM; no code change |
| Dirty tree before/after any step | `tree_status` in every `u_*.sh` | step stops (before) or is flagged (after) |

## 5. Hand-back order (one results branch at a time: fetch, new branch on `FETCH_HEAD`, no force push)

1. `handoff/h100-schedule-results-20261127` (this branch, now).
2. After A's single-card items: `handoff/h100-h3-image-results-20261123` (`results/T10/`, on top of `2f7c3bed6`, the old local commits are void), then `handoff/h100-h3-qk-rmsnorm-results-20261125` (`results/QK2/`).
3. After node-a of A: `handoff/h100-verl-omni-fa3-results-20261124`, then `handoff/h100-h3-pipeline-results-20261126` (P1/P2, extended after B1).
4. vime V0 to V2 results go to `handoff/h100-vime-dsv4-results-20261126` after A, V3/V4 after B1.
5. miles to `handoff/h100-miles-inkling2-results-20261126`: H0 to H1 after A, H2 to H5 after B2a, H6 to H7 after B2b.
6. AReaL to the new `handoff/h100-areal-placement-results-20261126` after C.

Each `SUMMARY.md` states the allocation, nodes and cards, and the deviations of section 0.

## 6. Monitoring and launch (review item 1)

`scripts/monitor.sh` polls every 5 minutes. On `RUNNING`: `on_job_start.sh` runs `prep_verify.sh`. **Before `gates/patch_autorun_A`:** first job runs `alloc_A_step0.sh` only (no node lanes). **After patch:** first job runs full `alloc_A.sh` or `resume_A_nodes.sh` if `step0_A` already OK. **First job B2:** waits `gates/user_release_B2`, then `nodea_done` **and** `nodeb_done`, then `miles_H0`; B2b after `miles_H5` OK. **Second job C:** waits `gates/user_release_C`. **B1:** waits `user_release_B1`, `build_WP`, `vime_V2`, **`areal_done`**, `nested_srun` OK. `gate_wait_stop` on these gates (no `|| true`). Stop and ask on must-pass fail, automated miles stop gates, or surprises outside section 4.

**Must-pass A:** `step0_mustpass.sh` -> `gates/step0_A`. **Must-pass B1/B2:** `gates/nested_srun` OK (both miles-style and vime-style probes; or documented `DRIVER_MODE=login` fallback in `nested-srun-2node.txt`).

**Nested driver mode** (item 4): `step0_nested.sh` exports job-level `SLURM_NNODES=2` and tests inner `srun --ntasks-per-node=1` and `srun -N$SLURM_NNODES` separately; `driver_2node` login path uses the same job-level exports.
