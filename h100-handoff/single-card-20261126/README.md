# H100 单卡任务包：T7、T8、QK（2026-11-26）

本文位于 fork 分支 `handoff/h100-single-card-20261126` 的 `h100-handoff/single-card-20261126/README.md`。这个分支从 `handoff/h100-h3-image-20261123`（`c2385081b`）切出，所以 T7 和 T8 的文件夹也在里面。QK 的文件夹在 `handoff/h100-h3-qk-rmsnorm-20261125` 上。

三项 handoff 都已推到 fork，各自只需要**一张 H100**，可以放在 mix 节点上，作为三个独立的单卡作业同时跑。本文只把它们合在一起排，代码、命令和期望值与原 README 一致。唯一的改动是 GPU 编号（见“通用要求”第 2 条）。遇到不清楚的地方，以原 README 为准：

- T7：`handoff/h100-h3-image-20261123` 的 `h100-handoff/h3-image-t7-20261124/README.md`
- T8：同一分支的 `h100-handoff/h3-image-t8-20261125/README.md`
- QK：`handoff/h100-h3-qk-rmsnorm-20261125` 的 `h100-handoff/h3-qk-rmsnorm-20261125/README.md`
- 占位符、环境和规则：`h100-handoff/h3-image-20261123/README.md`（T1–T6 那一轮）

## 总览

| 作业 | 内容 | 代码 | 卡 | 显存峰值（B200） | 主机内存 | 时长（估） | 结果分支 |
|---|---|---|---|---|---|---|---|
| **T8** | #738 `h3_image_pdd_head_contract` + `ws1_one_h3_image_step`。SM90 上第一次走真正的 strict 行 | bundle `h3-image-t8-20261125.bundle` → `b685c4991` | 1 张 **94 GB** H100 | **79.4 GiB** | ≥ 128 GB（B200 RSS 约 46 GiB） | 2–3 h（B200 2 h 11 min） | `handoff/h100-h3-image-results-20261123` |
| **T7** | #777 PDD schedule、#778 velocity step（含 Triton）、#779 的最终 chain | 公开分支 `feat/h3-image-phase-b` @ `a5989aede` | 1 张 80/94 GB H100 | 67.2 GiB | ≥ 96 GB | 约 1 h | 同上 |
| **QK** | #420 `h3_qk_rmsnorm_d128`：Q1 测试 + Q2 survey | bundle `h3-qk-rmsnorm-20261125.bundle` → `c56b81c4` | 1 张 H100 | 小 | ≥ 64 GB | 约 50 min | `handoff/h100-h3-qk-rmsnorm-results-20261125`（新建） |

三项互不依赖，可以同时提交。T8 最长，先提交。每项建议申请 1 GPU、16–32 CPU、上表的主机内存，时限 T8 4 h、T7 2 h、QK 1.5 h。

## 通用要求（mix 节点）

1. **分到的那张卡必须空着。** 每个任务开始时，那张卡的 `memory.used` 应为 0 MiB，卡上没有别的进程。同节点其他卡上有别人的作业没关系，在 `env.txt` 里写一句“shared node, N other GPUs busy”即可（只写数量，不写别人的进程、用户或作业）。
2. **GPU 编号。** 原 README 写死了 `CUDA_VISIBLE_DEVICES=0`。在 mix 节点上，如果作业没有用 cgroup 限制设备，0 号卡可能是别人的。所以本文的命令都用 `$GPU`，由下面的预检设好。这是相对原 README 唯一的改动，在 SUMMARY 里注明。
3. **T8 只能用 94 GB 卡**（`memory.total` ≥ 95,000 MiB），不要加 `--plain-autograd`（那样需要 111 GiB）。显存不够或 OOM 都是结果，照实交回，不要改代码。
4. **规则不变**：detached 干净 worktree，`git status --porcelain --untracked-files=no` 前后都为空；不改任何文件；`_C` 必须来自本 checkout；失败照实报告，失败的尝试放进 `not-green-attempts/`；不往共享环境装东西。
5. **编译并行度**：`export MAX_JOBS=<分到的 CPU 数>`，避免占满 mix 节点的 CPU。只影响编译速度，不影响产物。
6. **临时目录放在作业自己的目录下**：`export TMPDIR=<OUT>/<任务>/tmp`，先 `mkdir -p`。这个目录不提交。
7. **计时只记录，不比较。** T7 d、T8 和 QK Q2 的时间在共享节点上本来就有噪声，期望里也只要求记录。
8. **交回前脱敏**：集群、作业、节点、分区、用户、conda 环境名、`/home/…` 和 `/data/…` 路径、IP、邮箱都去掉，主机名写成 `node-a`。`git add -f` 日志，`git commit -s`，不加 AI 署名行。只推结果分支，不开 PR，不发评论。

### 每个作业开头的预检

```bash
# Slurm 分配了一张卡时，CUDA_VISIBLE_DEVICES 是它的编号；有 cgroup 限制时通常就是 0
GPU=${CUDA_VISIBLE_DEVICES:?no GPU allocated}
case "$GPU" in *,*) echo "more than one GPU allocated: $GPU"; exit 1;; esac
mkdir -p <OUT>/<任务>
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv -i "$GPU" > <OUT>/<任务>/nvidia-smi.txt
# 若 memory.used 不为 0，停下，换一张卡
nvidia-smi --query-gpu=index --format=csv,noheader | wc -l   # 节点上的 GPU 总数，写进 env.txt
```

## 一次性准备（三项共用）

用 T1–T6 那一轮在 H100 上留下的环境和权重。都还在的话，跳过这一节。

- `<E1>`：上一轮的环境（torch 2.13.0+cu130）。T7 和 T8 用它编译。
- `<V>`：在 `<E1>` 上建的 venv，装了 Diffusers `875373ac`（`0.42.0.dev0`）和 `accelerate==1.16.0`。T7 和 T8 用它运行。
- `<IMG>`：`MiniMaxAI/MiniMax-H3-Image` @ `8ced6006`，包含 T6 的下载（`transformer_diffusers/` 约 66 GB、`examples/`、`assets/showcase/photo.jpg`）。
- `<CKPT>`：MiniMax-H3 checkpoint @ `42ed227e`（T7/T8 从这里读 text encoder）。
- `<E>`：QK 用的环境（之前 H3 轮次的 torch 2.13.0+cu130 或 2.14.0+cu130，写明是哪个）。
- `<WEIGHTS>`：含 `hf_full/` 的目录（QK）。
- `<PROVIDER>`：Diffusers checkout @ `f53d552036a0d1bd5570782a39cd40cfabf112bc`，干净（QK）。

缺 `<V>`、`accelerate` 或 `<IMG>` 的话，按 `h3-image-20261123/README.md` 的 “Environment”、“Weights” 和 T6 的 “Extra setup” 补齐，并在 SUMMARY 里写明是重建的。

拿 handoff 文件：

```bash
cd <REPO>
F=https://github.com/fusheng-ji/RL-Kernel.git
git fetch $F handoff/h100-h3-image-20261123:refs/h100/handoff-image      # c2385081b
git fetch $F handoff/h100-h3-qk-rmsnorm-20261125:refs/h100/handoff-qk    # aedb56de6
git worktree add --detach <H_IMG> refs/h100/handoff-image
git worktree add --detach <H_QK> refs/h100/handoff-qk
```

- `<T7_DIR>` = `<H_IMG>/h100-handoff/h3-image-t7-20261124`
- `<T8_DIR>` = `<H_IMG>/h100-handoff/h3-image-t8-20261125`
- `<QK_DIR>` = `<H_QK>/h100-handoff/h3-qk-rmsnorm-20261125`

这两个 worktree 只用来读 handoff 文件，不在里面跑代码。

---

## T8：PDD head contract 和一个真实训练步（94 GB 卡）

```bash
cd <REPO>
git bundle verify <T8_DIR>/h3-image-t8-20261125.bundle            # 需要 a5989aede
git fetch <T8_DIR>/h3-image-t8-20261125.bundle feat/h3-image-one-step
git rev-parse FETCH_HEAD        # 必须是 b685c4991b09f83a87d21cf2ffa29bf556406df5
git worktree add --detach <W8> FETCH_HEAD
cd <W8>
git rev-parse HEAD^{tree}       # 必须是 81e62fa8ab01aa3dda61158bdf93f3200119da26（reference/tree.txt）
RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 <E1>/bin/python setup.py build_ext --inplace > <OUT>/T8/build.log 2>&1
PYTHONPATH=<W8> <V>/bin/python -c "import os, rl_engine._C as c, rl_engine._H3_C as h, triton; p=os.path.realpath; print(p(c.__file__).startswith(p('<W8>')), h.h3_attention_build_info()['profile'], triton.__version__)" > <OUT>/T8/ext.txt

export RL_KERNEL_H3_IMAGE_ROOT=<IMG> RL_KERNEL_H3_IMAGE_DIR=<IMG> RL_KERNEL_H3_IMAGE_VAE_DIR=<IMG>/vae_diffusers RL_KERNEL_H3_IMAGE_TEXT_ENCODER_ROOT=<CKPT>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYT="env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=<W8> <V>/bin/python"
TOOL=tools/validation/models/h3_image_step.py
# a. 新测试
$PYT -m pytest -q -p no:cacheprovider -rs tests/models/minimax_h3_image/test_h3_image_pdd_head.py \
  tests/models/minimax_h3_image/test_h3_image_step.py > <OUT>/T8/pytest-new.txt 2>&1
# b. 整个 image suite
$PYT -m pytest -q -p no:cacheprovider -rs tests/models/minimax_h3_image > <OUT>/T8/pytest-image.txt 2>&1
# c. 抓真实输入，再 replay
$PYT $TOOL capture --root <IMG> --text-encoder-root <CKPT> --out <OUT>/T8/inputs > <OUT>/T8/capture.log 2>&1; echo "exit=$?" >> <OUT>/T8/capture.log
$PYT $TOOL replay --root <IMG> --inputs <OUT>/T8/inputs --out <OUT>/T8/run-t2i --fixtures t2i_upscaled > <OUT>/T8/run-t2i.log 2>&1; echo "exit=$?" >> <OUT>/T8/run-t2i.log
$PYT $TOOL replay --root <IMG> --inputs <OUT>/T8/inputs --out <OUT>/T8/run-edit --fixtures edit_two_refs > <OUT>/T8/run-edit.log 2>&1; echo "exit=$?" >> <OUT>/T8/run-edit.log
# d. 第二个进程复现第一个
$PYT $TOOL replay --root <IMG> --inputs <OUT>/T8/inputs --out <OUT>/T8/run-b --fixtures t2i_upscaled --steps 0,5 --against <OUT>/T8/run-t2i > <OUT>/T8/run-b.log 2>&1; echo "exit=$?" >> <OUT>/T8/run-b.log
```

**期望**（B200 值在 `<T8_DIR>/reference/`）：
- `ext.txt`：`True h3-cuda-strict-v1` 和 Triton 版本。
- a：B200 `83 passed`，0 skipped；`test_cuda_binding_is_the_row_on_sm90_and_interim_elsewhere` 这次走 SM90 分支。
- b：B200 `514 passed`，0 skipped。其中包含已封存的 H100 layer-50 text digest，不一致就是发现。
- c：两个 fixture 都 `exit=0`、`ok True`。`summary.json` 里 `proj_in`、`pdd_proj_out`、`block.o_proj`、`block.ffn_gate_up`、`block.ffn_down` 的 status 是 `row`；`controls` 除 `next_head_first_drift`（节点为 `pdd_proj_out`）外全部为 null；`training.ok` 为 true。capture 的 layout 和 timestep 与 B200 相同；第 1–5 步的 image 行可能与 B200 不同（来自本 GPU 上的 Diffusers loop）。
- d：`exit=0`，两步的 `equal_to_other_process` 都为 true。
- 到 Diffusers 的距离和时间只记录，不比较。checkpointing 下 GEMM 行失败、或 `canonical_backward_session` 失败，都是发现，照实交回。

**交回 `results/T8/`**：`ext.txt`、`build.log`、两个 pytest 文件、所有日志、每个 `summary.json` 和 `*.digests.json`、`inputs/*.json`（**不要** `*.safetensors`）、`nvidia-smi.txt`、`env.txt`，以及 SUMMARY.md 里的一行 T8。

---

## T7：schedule、velocity step 和最终 Phase B chain

```bash
cd <REPO>
git fetch https://github.com/fusheng-ji/RL-Kernel.git feat/h3-image-phase-b
git rev-parse FETCH_HEAD        # 必须是 a5989aede979e793eea1d0e7ee3c98aca3f84fad
git worktree add --detach <W7> a5989aede
cd <W7>
git rev-parse HEAD^{tree}       # 必须是 6351519958226c71775002d88fa116274d089d1e（reference/tree.txt）
RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 <E1>/bin/python setup.py build_ext --inplace > <OUT>/T7/build.log 2>&1
PYTHONPATH=<W7> <V>/bin/python -c "import os, rl_engine._C as c, triton; print(os.path.realpath(c.__file__).startswith(os.path.realpath('<W7>')), hasattr(c, 'conditioning_noise_mix_forward'), triton.__version__)" > <OUT>/T7/ext.txt

export RL_KERNEL_H3_IMAGE_DIR=<IMG> RL_KERNEL_H3_IMAGE_VAE_DIR=<IMG>/vae_diffusers RL_KERNEL_H3_IMAGE_TEXT_ENCODER_ROOT=<CKPT>
# a. 两个新行
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=<W7> <V>/bin/python -m pytest -q -p no:cacheprovider -rs \
  tests/models/minimax_h3_image/test_h3_image_pdd_schedule.py \
  tests/models/minimax_h3_image/test_h3_image_velocity_step.py \
  tests/ops/flow/test_h3_image_velocity_step.py > <OUT>/T7/pytest-new-rows.txt 2>&1
# b. stack head 上的整个 image suite
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=<W7> <V>/bin/python -m pytest -q -p no:cacheprovider -rs \
  tests/models/minimax_h3_image tests/ops/flow/test_h3_image_velocity_step.py > <OUT>/T7/pytest-image.txt 2>&1
# c. 发布权重上的最终 chain
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=<W7> <V>/bin/python tools/validation/models/h3_image_full_chain.py \
  --root <IMG> --text-encoder-root <CKPT> --out <OUT>/T7/run --repeat > <OUT>/T7/evidence.log 2>&1; echo "exit=$?" >> <OUT>/T7/evidence.log
# d. schedule 和 velocity step 的数据
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=<W7> <V>/bin/python <T7_DIR>/chart_data.py --out <OUT>/T7/data-h100.json > <OUT>/T7/data.log 2>&1; echo "exit=$?" >> <OUT>/T7/data.log
```

**期望**（B200 值在 `<T7_DIR>/reference/`）：
- `ext.txt`：`True True` 和 Triton 版本。
- a：B200 `66 passed`，0 skipped。不应有失败或跳过；因缺 Triton/CUDA 跳过，说明这一行没测到，照实交回。
- b：B200 `460 passed`，0 skipped。`test_release_fixtures_bitwise_against_diffusers` 不一致就是发现。
- c：`exit=0`、`all_gates_pass True`；每个 fixture 的 `gate`、`secondary_equal`、`run_to_run_equal` 为 true，`first_drift` 为 null；每个 `*.rl-kernel.json` 的 `denoise` 阶段里，sampler `backend_id` 是 `H3ImageVelocityStepTritonOp`。最终 latent 和输出应与上一轮 T6 相同，比较在这边做。
- d：`exit=0`；`velocity.steps` 里每一步 `triton_vs_reference_differing` 为 0；`pdd_schedule` 的 sigma 和 timestep 与 `reference/data-b200.json` 完全一致。时间只记录。

**交回 `results/T7/`**：`ext.txt`、`build.log`、`pytest-new-rows.txt`、`pytest-image.txt`、`evidence.log`、`data-h100.json`、`data.log`、`run/summary.json` 和 8 个 `run/*.json`（**不要** `*.png`，约 110 MB）、`nvidia-smi.txt`、`env.txt`，以及 SUMMARY.md 里的一行 T7。

---

## QK：`h3_qk_rmsnorm_d128`

```bash
cd <REPO>
git fetch https://github.com/RL-Align/RL-Kernel.git test-h3
git bundle verify <QK_DIR>/h3-qk-rmsnorm-20261125.bundle          # 需要 test-h3 13d0f906
git fetch <QK_DIR>/h3-qk-rmsnorm-20261125.bundle 'refs/heads/feat/h3-qk-rmsnorm-d128:refs/h100/qk'
git worktree add --detach <WQ> refs/h100/qk      # c56b81c4f61ad725c361c964e466d6c256e878a5
cd <WQ> && RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 KERNEL_ALIGN_DET_GEMM_SM90=1 <E>/bin/python setup.py build_ext --inplace > <OUT>/QK/build.log 2>&1
<E>/bin/python -c "from rl_engine.backends.extension import _C; print(_C.__file__)" > <OUT>/QK/c-file.txt

# Q1. 测试
export PYTHONPATH=<WQ>:<PROVIDER>/src RL_KERNEL_H3_WEIGHTS=<WEIGHTS>
CUDA_VISIBLE_DEVICES=$GPU <E>/bin/python -m pytest -v -p no:cacheprovider -rfs \
  tests/models/minimax_h3/test_h3_qk_rmsnorm.py tests/models/minimax_h3/test_h3_rmsnorm.py \
  tests/models/minimax_h3/test_h3_one_block.py tests/models/minimax_h3/test_h3_token_refiner.py \
  tests/models/minimax_h3/test_h3_operator_trace.py tests/models/minimax_h3/test_h3_parallel.py \
  tests/models/minimax_h3/test_h3_parallel_model.py tests/models/minimax_h3/test_h3_full_model.py \
  tests/runtime/test_dispatch.py tests/validation/operators > <OUT>/QK/Q1/tests.txt 2>&1

# Q2. Survey
CUDA_VISIBLE_DEVICES=$GPU <E>/bin/python tools/validation/models/h3_prior_art.py --op qk_norm --out <OUT>/QK/Q2/qk_norm.json > <OUT>/QK/Q2/prior-art.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU <E>/bin/python tools/validation/models/h3_bitwise_reuse.py --op qk_norm --out <OUT>/QK/Q2/qk_norm_bitwise.json > <OUT>/QK/Q2/bitwise.log 2>&1
```

先 `mkdir -p <OUT>/QK/Q1 <OUT>/QK/Q2`。如果编译参数和你之前 H3 轮次的不同，用你自己的，并写明。

**期望**：
- Q1：`test_h3_qk_rmsnorm.py` 全部通过，包括与 `F.rms_norm` 在 BF16、FP16、FP32 下的逐位比较。除了 refiner SwiGLU 容差用例 `test_nodes_accuracy_repeat_batch_and_isolated[1-257]`（B200 上在 base 也失败），以及没装 matplotlib 时的 plot 测试，不应有其他失败。有其他失败的话，在第二个 worktree 里用 `13d0f906` 跑同一个测试，两边结果都交回。B200 参考：554 passed、3 failed、32 skipped @ `944a585d`。
- Q2：`rl_kernel` 和 `diffusers` 都 batch-invariant；`rl_kernel` 的 forward 在每个尺寸上都等于 provider 的字节。`<E>` 里没装的库（TransformerEngine、Liger、SGLang）报 unavailable 即可，不要安装。时间照实记录。

**交回**：新建分支 `handoff/h100-h3-qk-rmsnorm-results-20261125`，`results/` 下放 `Q1/tests.txt`、`Q2/`、`c-file.txt`、`build.log`、`nvidia-smi.txt`、`env.txt`、`pip-list.txt`（torch、triton、diffusers 和 survey 用到的库），以及 `SUMMARY.md`（写清每一处偏差，包括第 2 条的 `$GPU`）。

---

## T7 和 T8 写同一个结果分支

两项都加到 `handoff/h100-h3-image-results-20261123`（当前 `3dbfe6562`），**一个接一个推，不要 force push**：

```bash
git fetch https://github.com/fusheng-ji/RL-Kernel.git handoff/h100-h3-image-results-20261123
git switch -c results-t7-t8 FETCH_HEAD        # 后完成的那项，先重新 fetch，在最新的 tip 上加
# 加 results/T7/ 或 results/T8/，SUMMARY.md 追加一行，脱敏，git add -f 日志
git commit -s -m "results(h3-image): T7 on H100"   # 或 T8
git push https://github.com/fusheng-ji/RL-Kernel.git HEAD:handoff/h100-h3-image-results-20261123
```

SUMMARY 那一行写：命令、退出码、主要数字，再加一句“mix node, card idle at start, other GPUs busy: N”，以及用了 `$GPU` 而不是 `CUDA_VISIBLE_DEVICES=0`。
