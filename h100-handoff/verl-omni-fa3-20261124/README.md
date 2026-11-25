# H100 handoff: stock verl-omni H3 with its native FA3 pair (2026-11-24)

## Why

verl-omni's own MiniMax-H3 FlowGRPO recipe is the native P/P baseline for RL-Kernel's R/R run. Its default attention pair is FlashAttention 3 on both sides:
- actor: `attn_backend=_flash_3_varlen_hub`;
- rollout: `rollout_attn_backend=FLASH_ATTN_3_HUB`.

vLLM-Omni accepts `FLASH_ATTN_3_HUB` only on compute capability 9.x, so on B200 the recipe ran on its SDPA pair (`native` / `TORCH_SDPA`) instead. This round runs the FA3 pair, with the SDPA pair on the same H100 node as the control. It measures:
- **Consistency**: actor velocity against rollout velocity on the same transition, for each pair.
- **Speed**: the per-step time split and peak memory, for each pair.

No RL-Kernel code runs here. verl-omni is stock and unmodified.

| Task | What | Hardware | Time (est.) |
|---|---|---|---|
| **F0** | preflight: packages, which FA3 kernel each side loads, one FA3 call at H3's shape | 1 H100 | 10 min |
| **F1** | SDPA pair, 256x384, 121 frames: clean timing, then capture | 1 node x 4 H100, exclusive | about 45 min |
| **F2** | FA3 pair, same workload: clean timing twice, then capture twice | the same node | about 1.5 h |
| **F3** (optional) | 1344x768, 107 frames: FA3 and SDPA, clean timing | the same node | about 1.5 to 2 h |

F0 to F2 are the round. F3 may run out of memory; that is a result, so report it and stop there.

Paths are placeholders, as in the earlier rounds:
- `<REPO>`: a clone of the fork; `<THIS_DIR>`: this folder; `<OUT>`: where results are written;
- `<CKPT>`: the H3 checkpoint directory, revision `42ed227ee7df40d41602854ae760620d6eb651fe`, with `FL2VA/` (complete) and `transformer/`. The symlink directory from the 2026-11-22 S2 round is right;
- `<E2>`: the 2026-11-22 S2 environment (torch 2.14.0+cu130, vLLM 0.33.0, vLLM-Omni `6039984e`, verl `2b4ed682`, diffusers 0.43.0, transformers 5.19.0). **Do not change it**;
- `<VERL_OMNI>`, `<OVERLAY>`, `<REWARDS>`: made below.

## Rules (unchanged from the earlier H3 rounds)

- **Clean tree.** `git status --porcelain --untracked-files=no` in `<VERL_OMNI>` must be empty before and after each run (`run_arm.sh` checks and records it). Change no file.
- **Idle hardware.** Use one exclusive node for F1 to F3. `nvidia-smi` is recorded at the start of each run.
- **Report failures as they are.** Never change a script, a bound or a config to make a result pass. Failed attempts go, as they are, into `not-green-attempts/`. Rerun a failed run only with a new `TAG`.
- **Never relax a security setting** of the node or of a library to get a run through (ptrace scope, the kernel library's publisher check, `trust_remote_code` beyond what `run_arm.sh` passes). Stop and report instead.
- **Push only the results branch.** No PR, no comment.
- **Redact everything sent back.** Remove cluster, job, node, partition, user, conda environment name, `/home/...` and `/data/...` paths, IP addresses and email. Hosts become `node-a`. `git add -f` the `*.log` files. Commit with `git commit -s`, no AI attribution lines.

## Setup

```bash
git clone https://github.com/verl-project/verl-omni.git <VERL_OMNI>
git -C <VERL_OMNI> checkout --detach 708f4d631b5c77f005fdc112e5de75b52237c5ae
```

`run_arm.sh` puts `<VERL_OMNI>` first on `PYTHONPATH`, so this checkout is used instead of the editable one in `<E2>`. Do not install it.

Reward packages, in an overlay directory (`<E2>` has everything else):

```bash
<E2>/bin/python -m pip install --no-deps --target <OVERLAY> ftfy==6.3.1 iopath==0.1.10 portalocker==4.4.2 timm==1.0.31 \
  "pytorchvideo @ git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f" \
  "imagebind @ git+https://github.com/facebookresearch/ImageBind.git@53680b02d7e37b19b124fa37bae4b6c98c38f5be"
```

`--no-deps` matters: it must not bring in another torch. If an import then misses a package, install exactly that one the same way and say so.

Reward weights, in `<REWARDS>`:
- `larger_clap_general/`: `laion/larger_clap_general` at revision `ada0c23a36c4e8582805bb38fec3905903f18b41`;
- `imagebind_huge.pth`: `seung275/imagebind_huge.pth` at revision `b17ac77b71715c0dcefe5600b6a4b25f7cbc8012` (Meta's own URL returns 403). sha256 must be `d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157`.

The reward values are not compared. The rewards run only because the recipe needs them.

## F0. Preflight (1 H100)

```bash
cd <OUT> && CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=<OVERLAY> \
  <E2>/bin/python <THIS_DIR>/preflight.py > <OUT>/F0/preflight.json 2> <OUT>/F0/preflight.err
<E2>/bin/python -m pip list --format=freeze > <OUT>/F0/pip-list.txt
```

This is the first place the FA3 kernel (`kernels-community/flash-attn3`, kernel repo type) is downloaded and loaded, through the two library calls the recipe itself makes. **It has not been run on our side** (B200 cannot run the kernel, and we did not load it there). On 2026-11-24 the kernel repo's `v1` was `7cb368cf8278b583132eb72cbf312d54586df2e2` and `v2` was `60400de0ac5203ddd3cf4658a19944cbfea1d684`. Diffusers asks for `v1` only; vLLM-Omni tries `v1`, then `v2`. The expected build is `torch-stable-abi29-cu130-x86_64-linux`.

**Expected:**
- `capability` is `[9, 0]`; every package has a version (`pip-list-b200.txt` here is the B200 environment);
- `same_kernel_file` is `true`. If it is `false`, or either side cannot load a kernel, **stop and report**: the recipe's default pair would then not be one implementation, which is a finding;
- `fa3.actor_repeat_bitwise` and `fa3.rollout_repeat_bitwise` are `true`. `actor_equals_rollout_bitwise` and the two relative L2 values are reported, not gated.

**No Hub access on the compute node?** Run F0 once on a GPU node that has it, then cache the `v1` reference the normal way, and add `export HF_HUB_OFFLINE=1` for F1 to F3:

```bash
<E2>/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('kernels-community/flash-attn3', repo_type='kernel', \
  revision='v1', allow_patterns=['build/torch-stable-abi*-cu130-x86_64-linux/*']))"
```

Use no `LOCAL_KERNELS` and no `DIFFUSERS_TRUST_REMOTE_KERNELS`. If no GPU node has Hub access, stop and report.

## F1. SDPA pair, the control (4 H100)

```bash
export E2=<E2> VERL_OMNI=<VERL_OMNI> CKPT=<CKPT> REWARDS=<REWARDS> OVERLAY=<OVERLAY> OUT=<OUT> CUDA_VISIBLE_DEVICES=0,1,2,3
ARM=sdpa WL=w0 CAP=0 STEPS=4 TAG=r0 bash <THIS_DIR>/run_arm.sh
ARM=sdpa WL=w0 CAP=1 STEPS=2 TAG=r0 bash <THIS_DIR>/run_arm.sh
```

`run_arm.sh` writes `<OUT>/runs/<WL>-<ARM>-cap<CAP>-<TAG>/`. Layout: rollout TP4 (one engine), actor FSDP2 DP4, 2 prompts x 4 samples per step, 10 denoise steps, LoRA r64 (the recipe's). `SHORT_TMP` sets a short node-local temp directory (default `/tmp`); vLLM's socket paths fail when it is long.

## F2. FA3 pair (the same node)

```bash
ARM=fa3 WL=w0 CAP=0 STEPS=4 TAG=r0 bash <THIS_DIR>/run_arm.sh
ARM=fa3 WL=w0 CAP=0 STEPS=4 TAG=r1 bash <THIS_DIR>/run_arm.sh
ARM=fa3 WL=w0 CAP=1 STEPS=2 TAG=r0 bash <THIS_DIR>/run_arm.sh
ARM=fa3 WL=w0 CAP=1 STEPS=2 TAG=r1 bash <THIS_DIR>/run_arm.sh
```

Then, for both tasks:

```bash
cd <OUT>/runs && <E2>/bin/python <THIS_DIR>/analyze.py w0-*-cap0-* w0-*-cap1-* --repeat w0-fa3-cap1-r0 w0-fa3-cap1-r1 --out <OUT>/summary-w0.json
```

**Gates:**
- every run: `train exit=0`, `verl-omni dirty files after: 0`, `commit.txt` = `708f4d63...`;
- every capture: `input_equal` = `pairs` in all four groups, `matched` = `rollout_transitions`, `rollout_tp_copies_that_differ` = 0;
- FA3 captures: `backends` lists `actor_attention_backend: _flash_3_varlen_hub` and one `actor_kernel_file` and one `rollout_kernel_file` naming the same snapshot and build, and `rollout_backend_log_lines` > 0. SDPA captures list `actor_attention_backend: native` and no kernel file.

**Reported, not gated:** bitwise counts, relative L2 (overall and by denoise step), the recipe's log-prob difference, time split, peak memory, and how many rollout transitions the two FA3 captures share.

**B200 reference** (SDPA pair, the same script and layout; `reference/summary-b200.json`):

| | 256x384, 121 frames | 1344x768, 107 frames |
|---|---|---|
| steady step (median) | 35.2 s | 430.8 s |
| generate / old_log_prob / update / weight sync | 20.2 / 2.5 / 7.2 / 4.8 s | 298.6 / 24.4 / 100.4 / 4.8 s |
| peak memory per GPU (MiB) | 71376 / 76650 / 69534 / 69174 | 96872 / 101876 / 95904 / 95264 |
| video velocity, actor vs rollout, no grad (capture, 2 steps) | input equal 48/48, bitwise 0, rel-L2 median 3.34% (max 9.36%) | not captured |
| the same, audio | input equal 48/48, bitwise 0, rel-L2 median 1.17% (max 3.16%) | not captured |

4 x B200, the same layout and commit, `verl-omni dirty files after: 0` and exit 0 in all three runs. The capture run steps are slower than the clean ones (the capture writes files); use the clean runs for timing.

The two machines need not agree. The comparison that matters is FA3 against SDPA on the same H100 node.

## F3. 1344x768, 107 frames (optional, the same node)

```bash
ARM=fa3  WL=w1 CAP=0 STEPS=3 TAG=r0 bash <THIS_DIR>/run_arm.sh
ARM=sdpa WL=w1 CAP=0 STEPS=3 TAG=r0 bash <THIS_DIR>/run_arm.sh
cd <OUT>/runs && <E2>/bin/python <THIS_DIR>/analyze.py w1-* --out <OUT>/summary-w1.json
```

At 1344x768 the SDPA pair peaked at 96872 / 101876 / 95904 / 95264 MiB per GPU on B200, above the H100's 95,830 MiB on at least one GPU, so the SDPA run probably runs out of memory on H100; FA3's peak is not known. An out-of-memory error is a result. Do not enable offload to make it fit. Report the out-of-memory error and where it happened.

## What to send back

Push the fork branch `handoff/h100-verl-omni-fa3-results-20261124`, with `results/`:
- `F0/` (`preflight.json`, `preflight.err`, `pip-list.txt`);
- `summary-w0.json`, and `summary-w1.json` if F3 ran;
- per run: `commit.txt`, `env.txt`, `summary.txt`, `steps.txt`, `smi.csv`, `nvidia-smi-start.txt`, `train.log`;
- `SUMMARY.md`: each gate's verdict, every deviation, and the time split and memory tables for the two pairs.

Do **not** send `cap/` or `out/` (large tensors and media). Keep `cap/` of the four captured runs until the results are verified.
