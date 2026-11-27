# H100 handoff: native miles x Inkling-2-Mini end-to-end baseline on 2 nodes x 4 H100 (2026-11-26)

## Round 2 (2026-11-27): what changed and what to run

Round 1 stopped at H2: no probe passed. H0 and H1 passed and stay valid; do not rerun them. Round 1 ran on nodes shared with other steps, so its memory numbers are not usable. Read from the round-1 logs:
- `a` (TP1, 0.55): SGLang does not start (hybrid state cache memory negative). As expected; **not rerun**.
- `b` (TP1, 0.80): rollout and one training step finished (`perf 0`), then an NCCL ALLTOALL watchdog timeout, peak about 94 GB per GPU. Other steps held memory on the same GPUs, so this is **rerun** on idle nodes.
- `c`, `d` (TP2): `Inkling2 tensor parallelism requires NVLink multicast`. The image's SGLang notes that H100 TP2 reports multicast support but the driver refuses the mapping (`symm_workspace.py`, `is_multicast_supported`), and Inkling2's custom all-reduce only takes groups of 4, 6 or 8. **Not rerun.**

Changes in this round:
1. **New probes `e` and `f`: TP4**, one engine over the 4 GPUs of a node (2 engines in all), at 0.55 and 0.70. Each GPU then holds 12.5 GB of weights instead of 50 GB. Whether it starts depends on NVLink multicast inside one node; the fabric state is recorded (point 3).
2. **Idle-GPU check at every stage start** (`scripts/gpu_idle_check.sh`, called by `node_entry.sh` on both nodes and by `run_stage_1node.sbatch`). The stage does not start, rc 96, if a node does not see exactly 4 GPUs or a GPU already uses more than 1 GiB. **Nothing else may run on these GPUs for the whole job**: no other `srun` step, before or during. The cleanup in `lib_train.sh` kills every SGLang server on both nodes, other jobs' included. A rc 96 is not a result: wait for idle nodes and rerun with a new `TAG`.
3. Each stage writes `gpu_check_node{0,1}.txt`: memory and processes per GPU, `nvidia-smi topo -m`, `nvlink -s` and the fabric section of `nvidia-smi -q`.
4. H2 sets `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` (unless already set), so `train.log` shows which NIC and HCA each rank uses. A debug setting only.

Run in round 2: H2 probes `b2`, `e`, `f` (each on its own idle 2-node job, one after the other), then continue with H3 as written if one passes. Round-1 outputs go to `not-green-attempts/round1/` unchanged; results go on the same results branch.


## Why

The native miles path (torchtitan FSDP2 trainer + colocated SGLang rollout, no RL-Kernel code) for `thinkingmachines/Inkling-2-Mini` is the Path A baseline for the later miles + RL-Kernel comparison (RFC #736, INK-55 / INK-66 / INK-72). It ran on 8 x B200, in one node (`reference/b200-summary.json`). This round measures the same baseline on H100:
- **Support**: does the recipe run on H100, and on two nodes;
- **Consistency**: train vs rollout log-prob difference, and how it is split between trainer, SGLang decode and SGLang prefill;
- **Speed**: the per-step time split, its run-to-run noise, and peak memory;
- **Drift**: on B200 the train-rollout difference rose slowly over 100 steps (+22% to +30% inside every confidence bucket). H6 and H7 save weights along the way so the cause can be located.

No RL-Kernel code runs here. miles and SGLang are the image's, unmodified.

**The recipe does not fit H100 as it is.** On B200 each GPU peaked at about 136 GiB. The recipe runs one TP1 SGLang engine per GPU, each holding the whole 50 GB BF16 model, with `--sglang-mem-fraction-static 0.55`. On an 80 or 94 GB card, 0.55 does not even hold the weights. So H2 probes a few SGLang settings first, and every later task uses the first one that works. This is a change of recipe arguments only, not of code, and it must be reported as a deviation.

| Task | What | Hardware | Time (est.) |
|---|---|---|---|
| **H0** | import the image, pinned downloads | login / CPU node with internet | 1 h |
| **H1** | versions, BF16 conversion twice, the PR's own tests | 1 node x 4 H100 | 30 min |
| **H2** | memory probe: one step per SGLang setting, pick one | 2 nodes x 4 H100 | 1 to 2 h |
| **H3** | 3-step smoke with the weight-sync check and dumps | 2 nodes | 40 min |
| **H4** | consistency split: trainer replays, option probes, SGLang self-check | 2 nodes + 1 node | 1.5 h |
| **H5** | 10 steps, twice, clean (noise line) | 2 nodes | 2 x 1 h |
| **H6** | 100 steps with dumps and 4 HF exports | 2 nodes | 8 to 14 h |
| **H7** | drift: fresh SGLang on the exports, trainer replay with the original weights | 1 node + 2 nodes | 1.5 h |
| **H8** | summaries | 1 node (CPU work) | 30 min |

H0 to H3 are the minimum. If H2 finds no setting that fits, stop there and report H2.

Placeholders:
- `<WORK>`: a directory on a filesystem **shared by both nodes**. The trainer ranks of both nodes write the dumps there. It needs about 450 GB: image 43 GB, models 70 GB, H6 exports 4 x 50 GB, checkpoints, dumps;
- `<THIS_DIR>`: this folder; `<REPO>`: a clone of the fork;
- `<PART>`, `<ACCOUNT>`: your Slurm partition and account. Job names stay `exp`.

All sbatch lines below take the same prefix; it is written `SB` here. `--export=ALL,...` keeps `PATH` for `srun` / `scontrol`; the sbatch files unset `WANDB_API_KEY` and `HF_TOKEN` inside the container:

```bash
SB="sbatch -p <PART> -A <ACCOUNT> --output=<WORK>/jobs/exp-%j.out"
mkdir -p <WORK>/jobs
```

## Rules (as in the earlier rounds)

- **Unmodified software.** `git status --porcelain -uno` of `/root/miles` and `/sgl-workspace/sglang` inside the container is recorded in H1 and must be empty. Change no file in the image. Pass only the arguments written here.
- **Idle, exclusive hardware.** Every 2-node job is `--exclusive` on both nodes. The miles launcher runs `pkill -9 sglang / miles / redis`, and enroot does not isolate PIDs, so a shared node would lose other jobs' processes.
- **Upload nothing.** `WANDB_API_KEY` and `HF_TOKEN` are unset inside the container; downloads are anonymous.
- **Report failures as they are.** Never change a script, a bound or a config to make a result pass. A failed run is kept as it is under `not-green-attempts/`; rerun it only with a new `TAG`. An out-of-memory error in H2 is a result, not a failure.
- **Never relax a security setting** of the node or of a library. Stop and report instead.
- **Push only the results branch.** No PR, no comment.
- **Redact everything sent back.** Remove cluster, job, node, partition, account, user, `/home/...` and `/data/...` paths, `<WORK>` itself, IP addresses and email. Hosts become `node-a` / `node-b`. `git add -f` the `*.log` files. Commit with `git commit -s`, no AI attribution lines.

## What has not been run anywhere

The two-node launch (`node_entry.sh`, `lib_train.sh`) has not run on any GPU. B200 was a single node. What was checked on B200:
- every script parses (`bash -n`, `py_compile`);
- a dry run of the final miles command line: `--expert-model-parallel-size 8`, the H2 engine size and memory fraction come last and win (miles resolves repeated flags last-wins), `--actor-num-nodes 2`, `--actor-num-gpus-per-node 4`, `--colocate`;
- `collect.py`, `summ.py`, `drift.py` and `sync_compare.py`, run on the B200 data in the H100 file layout (memory and metrics split into two node files), reproduce the B200 report files exactly. `sync_compare.py` also flags a wrong export (440 of 574 tensors differ).

Also not run: the between-run cleanup in `lib_train.sh` (stop leftover Ray jobs, kill SGLang on both nodes) and the per-run timeout. If a later run in the same stage fails because GPUs are still busy, report it.

How the two nodes work: node 0 starts a Ray head, node 1 joins as a worker, node 0 waits for 8 GPUs and then runs the stage with `MILES_SCRIPT_EXTERNAL_RAY=1`. The recipe takes `--expert-model-parallel-size` from `num_gpus_per_node` (4), so `lib_train.sh` appends EP8 to keep B200's layout. If Ray, NCCL or the placement fails, report the error; setting `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` for your fabric is allowed (miles passes them through) and must be reported.

## H0. Prepare (login or CPU node with internet)

Needs `enroot`, `curl`, and `hf` (huggingface_hub). Do not run it inside the image.

```bash
WORK=<WORK> bash <THIS_DIR>/scripts/h0_prepare.sh
```

**Expected:**
- `runs/h0/image_digest.txt` reads `amd64 sha256:0a106222fae2ac646cbd8ca69b2d883dcd1da866b29d0b322e1bcbbbd8f789fa`. **If not, stop and report**: the tag has moved;
- `runs/h0/download.sha256` equals `download_sha256` in `reference/b200-summary.json` (16 files).

On the login node, `ENROOT_*` paths point into `<WORK>` (the script sets them).

## H1. Environment, BF16 conversion, the PR's tests (1 node)

```bash
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=r0 --time=01:00:00 <THIS_DIR>/scripts/run_stage_1node.sbatch <THIS_DIR>/scripts/h1_env.sh
```

**Expected** (`runs/h1_env-r0/`):
- `env.txt`: miles `1e1d3fe8...`, SGLang `99f0be56...`, torch `2.14.1+cu130`, 4 x H100, and both `git status` lines empty. Report the GPU memory (80 or 94 GB) and driver; the image needs a CUDA 13.0 driver (>= 580);
- the job log says `BF16_BYTE_IDENTICAL`, and `bf16 tensors: 1116`;
- `sha256(json.dumps(bf16_tensors.json, sort_keys=True))` equals `bf16.checkpoint_sha256` in the reference (`collect.py` prints it as `checkpoint_sha256`);
- the four pytest files pass (B200: 9 + 1 + 1 + 1).

## H2. Memory probe (2 nodes)

Each probe is one real recipe step (rollout, train, weight sync), in its own job so a hang cannot block the others (each run is also cut at 1 h):

| probe | SGLang GPUs per engine | `--sglang-mem-fraction-static` | note |
|---|---|---|---|
| `a` | 1 | 0.55 | the recipe as it is; expected to fail at engine start |
| `b` | 1 | 0.80 | the recipe's TP1 layout |
| `c` | 2 | 0.55 | round 1 only (multicast refused, see Round 2) |
| `d` | 2 | 0.70 | round 1 only (multicast refused, see Round 2) |
| `e` | 4 | 0.55 | round 2: one engine per node |
| `f` | 4 | 0.70 | round 2 |

```bash
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=a --time=01:30:00 <THIS_DIR>/scripts/run_stage.sbatch <THIS_DIR>/scripts/h2_memprobe.sh a 1 0.55
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=b --time=01:30:00 <THIS_DIR>/scripts/run_stage.sbatch <THIS_DIR>/scripts/h2_memprobe.sh b 1 0.80
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=c --time=01:30:00 <THIS_DIR>/scripts/run_stage.sbatch <THIS_DIR>/scripts/h2_memprobe.sh c 2 0.55
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=d --time=01:30:00 <THIS_DIR>/scripts/run_stage.sbatch <THIS_DIR>/scripts/h2_memprobe.sh d 2 0.70
```

Round 2 (new tags, idle nodes, one job at a time):

```bash
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=b2 --time=01:30:00 <THIS_DIR>/scripts/run_stage.sbatch <THIS_DIR>/scripts/h2_memprobe.sh b2 1 0.80
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=e --time=01:30:00 <THIS_DIR>/scripts/run_stage.sbatch <THIS_DIR>/scripts/h2_memprobe.sh e 4 0.55
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=f --time=01:30:00 <THIS_DIR>/scripts/run_stage.sbatch <THIS_DIR>/scripts/h2_memprobe.sh f 4 0.70
```

A probe **passes** when its `train.log` ends with `train rc=0`, has one `perf 0:` line, and shows no out-of-memory error. Pick the first passing probe in the order `b2`, `e`, `f` (TP1 first, it is the recipe's layout; in round 1 the order was `b`, `c`, `d`), and write it down for every later task:

```bash
echo "export SG_TP=1 SG_MEM=0.80" > <WORK>/h100.env    # the chosen probe's values
```

If none of `b2`, `e`, `f` passes, **stop and report H2** with the `gpu_check_node*.txt` files. Do not try other settings and do not turn on offload. The B200 reference is one TP1 engine per GPU at 0.55, peak 135 to 136 GiB per GPU.

## H3. Smoke, 3 steps (2 nodes)

```bash
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=r0 --time=01:30:00 <THIS_DIR>/scripts/run_stage.sbatch <THIS_DIR>/scripts/h3_smoke.sh
```

**Gates:**
- `train rc=0`, three `perf N:` lines, no NaN;
- the first weight sync passes `--check-weight-update-equal` (skip list `mlp.gate._one`: the checker reports that tensor falsely; B200 needed the same skip). The check covers the first sync only;
- `dump/rollout_data/` and `dump/policy_loss_debug/` exist.

**Reported, not gated** (the B200 values are in `reference/b200-summary.json` -> `smoke_3_steps`): abs diff about 0.01 to 0.02, KL about 1e-3, mismatch fraction about 0.9.

## H4. Consistency split

Trainer replays and option probes (2 nodes):

```bash
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=r0 --time=03:00:00 <THIS_DIR>/scripts/run_stage.sbatch <THIS_DIR>/scripts/h4_trainer.sh <WORK>/runs/h3_smoke-r0/dump
```

SGLang self-check on H3's rollout 0 (1 node; it uses `SG_TP` GPUs):

```bash
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=r0 --time=01:30:00 <THIS_DIR>/scripts/run_stage_1node.sbatch <THIS_DIR>/scripts/h4_sglang.sh <WORK>/runs/h3_smoke-r0/dump
```

**Gates:** `replay_bitwise.txt` shows `0/... differ` for `train_log_probs` and `old_log_probs` (B200: 0 of 2,284,383); the two `sglang_*.json` files exist.

**Reported, not gated:**
- `probe_r3` (`--use-rollout-routing-replay`): on B200 the rollout said `routed_experts payload is all zeros` and retried forever. Each probe here is cut at 40 min; a timeout is the same result;
- `probe_top` (`--true-on-policy-mode`): accepted on B200, 1 step ran;
- the SGLang numbers (prefill repeat, solo vs 16 concurrent, greedy repeat, decode vs prefill), against `consistency` in the reference.

## H5. Speed and noise line, 10 steps twice (2 nodes)

```bash
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=a --time=03:00:00 <THIS_DIR>/scripts/run_stage.sbatch <THIS_DIR>/scripts/h5_run.sh 10
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=b --time=03:00:00 <THIS_DIR>/scripts/run_stage.sbatch <THIS_DIR>/scripts/h5_run.sh 10
```

Run them one after the other, not at the same time. **Gate:** both reach `perf 9:` with `train rc=0`. The noise line is the difference of the step-1-to-9 mean step times (step 0 is warm-up). B200: 219.8 / 223.2 s.

Before H6, estimate its length as 100 x the H5 mean step time + 4 exports (about 2 min each on B200). **If it is above 16 h, stop and report** with the estimate.

## H6. 100 steps with exports (2 nodes)

```bash
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=r0 --time=18:00:00 <THIS_DIR>/scripts/run_stage.sbatch <THIS_DIR>/scripts/h6_run100.sh
```

HF exports land in `runs/h6_run100-r0/run/hf/{32,65,98,99}`. The weights in export 32 are the ones rollout 33 used, and so on. After each export, `sidecar_checksum.py` on each node checksums the live engines on that node (`live_node0.json`, `live_node1.json`).

**If the job is requeued**, the run restarts from step 0. Move the partial directory to `not-green-attempts/` and rerun with `TAG=r1`.

**Gates:** `train rc=0`, 100 `perf` lines, no NaN; four exports; each `live_node*.json` has entries for 4 exports.

## H7. Drift localization

Fresh SGLang on the exports and on the original model (1 node). Each name scores one rollout with one set of weights:

```bash
D=<WORK>/runs/h6_run100-r0/run
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=r0 --time=03:00:00 <THIS_DIR>/scripts/run_stage_1node.sbatch <THIS_DIR>/scripts/h7_fresh.sh \
  $D/hf/32:$D/dump/rollout_data/33.pt:w32_r33 $D/hf/65:$D/dump/rollout_data/66.pt:w65_r66 $D/hf/98:$D/dump/rollout_data/99.pt:w98_r99 \
  <WORK>/models/Inkling-2-Mini-bf16:$D/dump/rollout_data/0.pt:orig_r0 <WORK>/models/Inkling-2-Mini-bf16:$D/dump/rollout_data/99.pt:orig_r99
```

The trainer with the original weights on rollout 99 (2 nodes):

```bash
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=r0 --time=01:30:00 <THIS_DIR>/scripts/run_stage.sbatch <THIS_DIR>/scripts/h7_replay.sh <WORK>/runs/h6_run100-r0/run/dump 99
```

**Gates:**
- every `<name>.pt` exists, and `orig_r0_checksum.json` equals `orig_r99_checksum.json` (`per_engine_checksum`);
- `sync_compare.py` (run in H8): `all_equal` is true for exports 0, 1, 2. **A false value is the main finding of this round**: the live engines then hold other weights than the trainer exported. Report it with the first differing tensors.

**Reported:** the 2 x 2 table, weights (original / late) x data (rollout 0 / rollout 99), each row as overall and per-bucket mean |d|:

| cell | file | pair |
|---|---|---|
| original x data 0 | `h7_orig_r0.json` | `trainer_vs_fresh_prefill` (also `trainer_vs_live_decode`) |
| original x data 99 | `h7_orig_r99.json` | `trainer_vs_fresh_prefill` |
| late x data 99 | `h7_w98_r99.json` | `trainer_vs_live_decode`, `trainer_vs_fresh_prefill`, `live_decode_vs_fresh_prefill` |
| late x data 33 / 66 | `h7_w32_r33.json`, `h7_w65_r66.json` | the same three |

The B200 version (`localization` in the reference): original x data 99 rose +6% to +17% per bucket against original x data 0, late x data 99 rose +19% to +29%. B200 later separated the rest (rerun with exports, 2026-11-26, `localization_rerun` in the reference): weight sync is bitwise (574 tensors on all 8 engines after exports 32 and 65), the live engine equals a fresh prefill on the same weights, and late x data 99 is +25% to +31% per bucket on the fresh-prefill pair. About half of the rise comes from the data, half from the trained weights. H100 repeats this with its own exports.

## H8. Summaries (1 node)

Write the stage directories you used into `<WORK>/h8.env`, for example:

```bash
cat > <WORK>/h8.env <<'EOF'
H1=<WORK>/runs/h1_env-r0
H2_DIRS="<WORK>/runs/h2_memprobe-b2 <WORK>/runs/h2_memprobe-e <WORK>/runs/h2_memprobe-f"
H3=<WORK>/runs/h3_smoke-r0
H4T=<WORK>/runs/h4_trainer-r0
H4S=<WORK>/runs/h4_sglang-r0
H5A=<WORK>/runs/h5_run-a
H5B=<WORK>/runs/h5_run-b
H6=<WORK>/runs/h6_run100-r0
H7F=<WORK>/runs/h7_fresh-r0
H7R=<WORK>/runs/h7_replay-r0
EOF
$SB --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=r0 --time=01:30:00 <THIS_DIR>/scripts/run_stage_1node.sbatch <THIS_DIR>/scripts/h8_analyze.sh
```

Everything lands in `<WORK>/runs/report/`. `run100_summary.txt` and `run100_drift.txt` have the same layout as the B200 ones the reference was built from.

## B200 reference

`reference/b200-summary.json` (and `reference/b200-run100-curve.json`, one row per step). One node x 8 B200, TP1 engines at 0.55, EP8:

| | B200 |
|---|---|
| step time, steps 1-9 (3 runs) | 219.8 / 223.2 / 221.7 s, noise 1.5% |
| split, steps 1-9 of 100-step run | rollout 100.7, actor train 78.9, no-grad scoring 19.9, weight sync 9.4 s |
| generated tokens/s, 8 GPUs | 3412 (steps 1-9), 2042 (steps 90-99, shorter answers) |
| peak memory per GPU | about 136 GiB |
| 100 steps | 5 h 29 min, no NaN |
| reward, first / last 10 steps | 0.542 / 0.857 |
| KL | 94 of 100 steps in 0.7 to 1.2e-3 |
| abs diff, first / last 10 steps | 0.01441 / 0.01488; +22% to +30% inside every confidence bucket (steps 0-19 vs 80-99) |
| tokens that differ, train vs rollout | 78% to 92% (signed zero counted equal); no bitwise path |
| trainer replay, SGLang prefill repeat | bitwise equal |

**Do not compare H100 step times with B200 directly.** The SGLang setting (H2) and the topology (2 nodes, the EP all-to-all crosses nodes) differ. Compare the shape of the curves, the size of the consistency numbers, and H100's own noise line.

## What to send back

Push the fork branch `handoff/h100-miles-inkling2-results-20261126` (from this branch), with `results/`:
- `H0/`: `image_digest.txt`, `download.sha256`, `revisions.txt`;
- `H1/`: `env.txt`, `bf16_a.sha256`, `bf16_b.sha256`, `pytest_*.log`, the job log;
- `H2/`: `h2_probe_summary.json`, and per probe `train.log`, `h100_config.txt`, `gpu_mem_node*.csv`, `gpu_check_node*.txt`, the job log; for a probe that fails, the first traceback with file and line;
- per H3 to H7 run: `train.log` (and the sub-run ones), `h100_config.txt`, `gpu_mem_node*.csv`, `gpu_check_node*.txt`, `ray_gpus.txt`, the job log;
- `report/`: everything H8 wrote **except** `run100_drift_rows.pt` if it is above 20 MB;
- `SUMMARY.md`: the H2 choice, each gate's verdict, every deviation (including the SGLang setting and any NCCL variable), and the tables: H5 speed, H6 curve (first / last 10 steps), H7 2 x 2.

Do not send the dumps (about 4 GB), the exports (50 GB each), the checkpoints or `ray_logs/`. Keep them until the results are read.
