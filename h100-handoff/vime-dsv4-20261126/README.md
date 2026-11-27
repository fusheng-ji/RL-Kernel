# H100 任务包：vime DeepSeek V4（PR #2）在 Hopper 上的验证（2026-11-26）

## 为什么

[inaniloquentee/vime#2](https://github.com/inaniloquentee/vime/pull/2)（head `aebf8fea`）给 vime 加了 DeepSeek V4（`dsv4_hybrid`、mHC、hash-MoE、CSA/DSA、FP8 E4M3 + UE8M0 128×128 块缩放）。作者只在单机 MI300X 上跑过 4 层 50 步。我们在 B200（sm100）上做了 4 层的单机、跨机、一致性归因和 QAT，Hopper（sm90）上还没有任何数据。本轮回答两件事：

1. **能不能跑**：PR 加修复分支在 sm90 上能否构建、通过 PR 自带的检查、在单机和跨机上训练 50 步。
2. **B200 上的结论在 H100 上是否成立**：
   - 训练/rollout 的 logprob 差从第 0 步的约 0.09 开始，FP32 主参数（M2）下随训练增长到 3–4；
   - 训练端 QAT 能让它保持在 0.08–0.10。

sm90 和 sm100 的区别正好落在这条路径上：DeepGEMM 在 Hopper 上用 FP32 块缩放（Blackwell 用 UE8M0），cuDNN DSA 内核和 FlashMLA 也走另一套实现。所以 H100 的结果不能从 B200 推出来。

本轮只跑 4 层的 checkpoint，43 层不在范围内。

| 任务 | 内容 | 硬件 | 时长（估） |
|---|---|---|---|
| **V0** | 导入 vLLM 镜像，在里面编译 sm90 的依赖，导出成 `vime-v4-env-sm90.sqsh` | 1 节点，1 GPU，48 CPU | 3–5 h（B200 上的编译约 4 h） |
| **V1** | 下载 7 个分片，切出 4 层 checkpoint（BF16 版可选） | CPU，能访问 Hub | 约 30 min |
| **V2** | preflight、PR 自带的 smoke 和单测、修复分支单测 | 1 GPU | 约 40 min |
| **V3** | T0、T1、T2 各 50 步；4 次确定性回放；`VIME_SKIP_GRAD_NORM` 在 EP2 下的检查 | 2 节点 × 4 H100，独占 | 3–5 h |
| **V4** | 漂移和 QAT：fp8-M2 两次、qat-M2 两次，各 24 步 | 2 节点 × 4 H100，独占 | 约 2 h |

V0 → V1 → V2 → V3 → V4 依次跑。V3 和 V4 可以排在同一对节点上，先后执行。V2 的 cuDNN DSA 项失败**不是**停止条件，见 V2。

## 占位符

- `<REPO>`：这个 fork 的 clone；`<THIS_DIR>`：本文件夹（`<REPO>/h100-handoff/vime-dsv4-20261126`）。
- `<WORK>`：共享文件系统上的工作目录，约 200 GB（镜像 19 + 22 GB，checkpoint 35 + 27 GB，可选 BF16 52 GB，jit 缓存和日志）。
- `<LOCAL_SCRATCH>`：节点本地盘。enroot 的数据和临时文件放这里，不要放网络文件系统。
- `<ACCOUNT>`、`<PARTITION>`：H100 集群上的账号和分区。

## 规则（与以前的 H100 轮次相同）

- **不改文件。** `src/` 下的 checkout 由 `setup_sources.sh` 建好以后，除了它打的补丁之外不再改动。每个作业把 commit 和 `git diff | sha256sum` 写进 `runs/<suite>/pins-<job>.txt`。本文件夹里的脚本也不改；只填 `env.sh`。
- **失败照实报告。** 不为了让结果通过而改脚本、配置、阈值或布局。失败的尝试原样放进 `not-green-attempts/`。重跑失败的项时换一个 `SUITE` 名。
- **OOM 是结果。** 不开 offload、不改切分、不调 `--max-tokens-per-gpu` 来硬塞。报告是哪个 rank、哪一段、峰值多少。
- **共享节点上不用 `pkill` 或 `ray stop --force`。** 它们会杀掉同账号其他作业的进程。每个 case 是独立的 `srun` step，由 Slurm 收尾。
- **不放宽安全设置**（ptrace、内核库发布者检查、超出脚本已有的 `trust_remote_code`）。遇到了就停下来报告。
- **只推结果分支**，不开 PR，不发评论。
- **交回前脱敏**：去掉集群、作业号、节点名、分区、用户名、`/home/...` 和 `/data/...` 等路径、IP、邮箱。`analyze.py` 的输出已经把主机名换成 `node-a`、`node-b`；日志要自己处理。`git add -f` 日志，`git commit -s`，不加 AI 署名行。

## 准备

```bash
export H=<THIS_DIR>
# 填好 env.sh 里的 W 和 SCRATCH（或在每次 sbatch 前 export W=<WORK> SCRATCH=<LOCAL_SCRATCH>）
bash $H/setup_sources.sh                 # 登录节点，无需 GPU
```

`setup_sources.sh` 在 `<WORK>/src` 下建好四个目录：

| 目录 | 内容 |
|---|---|
| `vime` | PR head `aebf8fea`，干净。PR 自带的检查在这里跑 |
| `vime-fix-run` | 修复分支 `a2d04354`（`vime-fix.bundle` 里 3 个 commit，基于 `aebf8fea`）+ 补丁 0002，补丁不提交 |
| `Megatron-LM` | `ae76c60e6`，干净 |
| `Megatron-run` | `ae76c60e6` + 补丁 0004，补丁不提交 |

脚本最后打印四个目录的 commit 和 diff 哈希。把这几行原样写进 `SUMMARY.md`。期望：`vime` 和 `Megatron-LM` 的 diff 为空（`e3b0c442…`），`vime-fix-run` 为 `ea9498d46771cd6e`，`Megatron-run` 为 `95f19b25da8b1ff7`（与我们的 checkout 相同）。

修复分支的 3 个 commit：

- `973bca33`：依次在 `entrypoints/common`、`launchers`、`openai` 里找 vLLM 的 `FrontendArgs`。在 pin 的 vLLM 上，原来的路径不存在。
- `05803d49`：
  - `VIME_SKIP_GRAD_NORM` 在 EP>1 时不再在初始化阶段崩溃；
  - 梯度范数非有限时报错，不再静默继续。
- `a2d04354`：模型脚本加上 `--dsa-indexer-loss-coeff 0`，并补 README。

补丁：

- 0002：`V4_ALLOW_NUMPY2=1` 时允许 numpy 2。镜像自带 numpy 2.2.6；装 numpy 1.26 会让 pandas 3 坏掉。
- 0003：上游 vLLM `14a5bb4ad` 的回移植，修复权重同步时 V4 mHC 的 post-load hook。由 `node_run.sh` 在容器里每次运行时打上，不进 `src/`。
- 0004：Megatron 按 dtype 分组求梯度范数。mHC 参数保持 FP32，而 TE 的 `multi_tensor_l2norm` 对 BF16/FP32 混合的列表返回 NaN；V2 的 `l2test.log` 会在 H100 上复现这一点。

## V0. 构建环境（1 GPU）

```bash
sbatch -A <ACCOUNT> -p <PARTITION> --gres=gpu:1 $H/build.sbatch
```

- 镜像固定为 `vllm/vllm-openai@sha256:cb12c301…`（vLLM `0.32.1rc1.dev361+g83c078d95`，torch 2.14.0+cu130）。enroot 里要写成 `docker://vllm/vllm-openai:sha256:<digest>`：`@sha256` 会被当成 user@host。脚本已经这样写了。
- 镜像的 entrypoint 会吞掉命令，所以每次 `enroot start` 都带 `--rc rc.sh`。
- **所有 pip 安装都以 `reference/constraints-b200.txt` 为约束**。它是 B200 的 freeze，去掉了下面这些本地编译的包。只从 pypi.org 装：宿主机传进容器的 `PIP_*` 变量会被清掉（包括 NGC 之类的 extra index），`PIP_CONFIG_FILE=/dev/null`。装不上就记一条 `STEP_FAIL`，不会再不带约束重试。
- 第一轮（`b88c1f51a`）在 H100 上失败过：受约束安装失败后回退成不带约束重装，pip 按当天的 PyPI 重新解析，装进了 `flash-attn-4`，`nvidia-cutlass-dsl` 也升到 4.8.1（`libs-cu13` 还是 4.8.0）。于是导入 `vllm` 时 `cutlass.cute` 缺 `ThrMma`。这个回退已经去掉。
- 脚本额外做三项检查，任何一项不过都记 `STEP_FAIL`，也不导出 sqsh：
  - 镜像自带的包要和 B200 freeze 一致（写到 `wheels-sm90/base-image-freeze.txt`）。不一致说明镜像不同。注意：`image.sha256` 是本地 sqsh 文件的哈希，每次 `enroot import` 都会变，不能用来比较镜像；镜像身份看 digest 和这项检查。
  - 最终环境除本地编译的包外要和 B200 freeze 逐行相同（差异写到 `wheels-sm90/freeze-diff-vs-b200.txt`）。
  - 不能装有 `flash-attn-4`。
- 编译并发 `MAX_JOBS` 取 `SLURM_CPUS_PER_TASK`（没设时为 32）。
- 另外编译这些，都针对 sm90：
  - FlashAttention 2.8.3，打 C++20 补丁；
  - fast-hadamard-transform `f134af63`；
  - apex `10417ace`；
  - TransformerEngine `27486e03`（v2.18）；
  - FlashMLA `b7643bd5`（`nv_dev`）；
  - torch_memory_saver `8d30c59c`。
- wheel 缓存在 `<WORK>/wheels-sm90`。被抢占或失败后重新提交，已经编好的会跳过。

**期望：** 日志里没有 `STEP_FAIL`，结尾是 `IMPORTS_OK` 和 `BUILD_DONE`。`build_inner.sh` 在最后的导入检查里断言 capability 为 `(9, 0)`。把 `<WORK>/wheels-sm90/` 下的 `env-freeze-sm90.txt`、`base-image-freeze.txt`、`freeze-diff-vs-b200.txt`、`build-steps.log` 交回。B200 的对照是 `reference/env-freeze-b200.txt`，除了几个编出来的包的本地版本号以外应该相同。**任何一步 `STEP_FAIL` 都停下来报告**，不要换版本。

## V1. checkpoint（CPU）

```bash
bash $H/prep_ckpt.sh            # 加 --bf16 再生成 52 GB 的 BF16 版（只有 V4 的可选项要用）
```

- 来源是 `deepseek-ai/DeepSeek-V4-Flash-Base`，revision `8855555deef230a27a21a8d6f294b7b7497759b6`。只下载第 1–5、45、46 号分片和 4 个 json。
- `slice_ckpt.py` 保留嵌入、输出头和第 0–3 层，去掉 MTP，维度不变。
- 下载文件对 `reference/ckpt-download.sha256` 校验，必须全部 OK。
- 切片和 BF16 版对 `reference/ckpt-v4-4l*.sha256` 校验。若不一致，打印 `*_HASH_MISMATCH`，照实报告，可以继续跑。

## V2. preflight、smoke、单测（1 GPU）

```bash
sbatch -A <ACCOUNT> -p <PARTITION> --gres=gpu:1 $H/smoke.sbatch
```

结果在 `<WORK>/runs/smoke/`：

| 文件 | 期望（B200 上的结果） |
|---|---|
| `preflight.json` → `device.capability` | `[9, 0]` |
| `preflight.json` → `deep_gemm` | 照实记录。B200 上是 `supported: true`、`e8m0_used: true`。**H100 上预计 `e8m0_used: false`**，这正是要看的差别 |
| `preflight.json` → `cudnn_dsa.wrapper` | 照实记录。B200 上是 `true` |
| `preflight.json` → `fp8_grid.*.bitwise_equal` | `true`（两种形状都是）。探针的舍入和 vime 同步导出用的是同一张 UE8M0 网格。若为 `false`，V4 的 QAT 结果不能直接解释，照常跑 V4，但在 SUMMARY 里写明 |
| `l2test.log` | 纯 BF16、纯 FP32 的列表，TE 和 apex 都与 `ref` 一致。BF16/FP32 混合的列表都不对：B200 上 `mixed bf16-first` 两者都是 `nan`，`mixed fp32-first` 两者都是 0.1755（`ref` 0.1814）。这就是补丁 0004 的原因 |
| `pr-pytest.log` | B200：除 ROCm 专用项跳过外全部通过，`rc=0` |
| `smoke--layer-index_2.log`、`smoke--layer-index_3.log`、`smoke--num-layers_4.log` | B200 上分别匹配 546/545、540/539、2164/2160（PR 描述里写的数字是旧的，`aebf8fea` 加了 shared experts）。照实记录 H100 的数字 |
| `rollout_smoke.log`、`runtime_checks.log` | `rc=0` |
| `fix-pytest.log` | B200：`93 passed`，`rc=0` |
| `pr-tree-status.txt` | 空 |

**cuDNN DSA：** 若 `cudnn_dsa` 报错，或 V3 的 T0-50 在 DSA 内核里失败，先把那次尝试放进 `not-green-attempts/`，然后 V3、V4 全部改用 `export DSA_BACKEND=none`（`--dsa-kernel-backend none`，不用融合内核），在 SUMMARY 里写明。两种后端不要混用。

## V3. 核心验证（2 节点 × 4 H100，独占）

```bash
sbatch -A <ACCOUNT> -p <PARTITION> -N2 --gres=gpu:4 --exclusive $H/core.sbatch
```

设置：修复分支加 0002；M1，即 PR 默认的 StatelessAdam + precision-aware（直接在 BF16 上更新）；lr 1e-6。数据是 gsm8k 的提示（`data/gsm8k.jsonl`）。每步 8 个提示 × 4 个样本，回复 32 token，GRPO。

| case | 布局 | 每节点 GPU | 说明 |
|---|---|---|---|
| `T0-50` | 单机，训练 PP1 × EP2 + rollout 1 卡 | 3 | 同时保存 rollout，供回放使用 |
| `T1-50` | PP2 × EP2，PP 边界跨机 | 2 + 3 | 第 2 节点多一张卡给 rollout |
| `T2-50` | EP4，专家 all-to-all 跨机 | 2 + 3 | |
| `det-R0a`、`det-R0b` | 单机 PP1 × EP2，`--deterministic-mode` | 2 | 用 T0-50 第 0 步的 rollout 训一步，权重等于 checkpoint |
| `det-R1`、`det-R2` | T1、T2 的布局，确定性 | 2 + 2 | 同上 |
| `skip-T1-2` | T1 布局，`VIME_SKIP_GRAD_NORM=1`，2 步 | 2 + 3 | 修复前在初始化阶段崩溃 |

**T0 的布局与 B200 不同。**
- B200 的 T0 是 PP2 × EP1，切分 `Et*3|t*1L`，第 0 段峰值 113 GiB，94 GB 的卡放不下。
- 也不能把第 0 段切小：vime 要求 3 个 hash-MoE 层和嵌入在同一段（`moe_num_hash_layers <= n_decoders_with_embedding` 的断言）。
- 所以 H100 上的 T0 改为 PP1 × EP2，仍是单机。T1、T2 与 B200 相同。
- 我们在 B200 上用新布局冒烟跑过，峰值见 `reference/pkgtest-b200.json`。

每个 case 的输出在 `<WORK>/runs/core/<case>/`：`train.log`、`node*.log`、`probe-rank*.jsonl`、`nccl-*.log`、`rc`。

被抢占后直接重新提交：已有 `rc` 的 case 会跳过。只想重跑某几个 case 时：
- 删掉那几个 case 的目录；
- 再设 `ONLY="T1-50 det-R1"` 重新提交。

之后：

```bash
python3 $H/analyze.py <WORK>/runs/core > <OUT>/V3/summary.json
```

**期望**（对照 `reference/summary-b200.json`，其中 `core-*` 和 `det_cross_topology` 两节）：

- 每个 case 都是 `rc` 0，`grad_records_nonfinite` 为 0，`n_steps` 等于设定的步数。
- 训练端各 rank 的峰值显存：
  - T0（PP1 × EP2）：B200 冒烟 78 GiB（`reference/pkgtest-b200.json`）。若 H100 上 OOM，照实报告，T1、T2 照常跑；
  - T1：58 / 58 / 21 / 21 GiB；
  - T2：每个 rank 42 GiB。
- `logprob_diff`：第 0 步约 0.09，50 步内保持在 0.08–0.10（lr 1e-6 时权重几乎不变）。
- `T1-50`、`T2-50` 的 `ranks` 分布在 `node-a` 和 `node-b` 上，否则不算跨机。`nccl-*.log` 里应只有 `NET/IB`，没有 `NET/Socket`。记录看到的是哪一种。
- 回放（`replays`）：
  - `det-R0b_vs_det-R0a` 的 `bitwise_equal_frac` 为 1.0（B200：1.0）；
  - `det-R1_vs_det-R0a`：B200 上用本包的布局（两者都是 EP2）**逐位相同**（`reference/pkgtest-b200.json`）。PP 切分不改变数值；
  - `det-R2_vs_det-R0a`（EP4 对 EP2）：`mean_abs` 不超过约 0.004（B200 上 R2 对旧的 PP2 × EP1 单机布局为 0.0035，逐位相同 72%）。若 T0 因 OOM 没有生成 rollout，回放会被跳过，`progress.txt` 里会写明。
- `skip-T1-2`：`rc` 0。

## V4. 漂移和 QAT（2 节点 × 4 H100，独占）

```bash
sbatch -A <ACCOUNT> -p <PARTITION> -N2 --gres=gpu:4 --exclusive $H/drift.sbatch
```

设置：
- 修复分支；M2，即 StatelessAdam 但不用 precision-aware（保留 FP32 主参数）；lr 1e-5，24 步。
- reward 是回复里小写 ASCII 字母的比例（`V4REWARD=lower`）。它只用来让策略动起来，不代表模型质量。
- 训练端在第一个节点的 4 张卡上：PP1 × EP4。
- rollout 用第二个节点的 1 张卡。

| case | 内容 |
|---|---|
| `fp8-M2-a`、`fp8-M2-b` | 现状：FP8 checkpoint，同步时把 BF16 权重量化成 FP8 |
| `qat-M2-a`、`qat-M2-b` | 训练端 QAT（`V4_QAT=1`，在探针里实现）：FP32 主参数照常累积，每次优化步之后把模型参数舍入到同一张 FP8 网格，前向、同步、rollout 用的是同一份权重 |
| `bf16-M2-a`、`rnd-M2-a`（可选，`CASES="bf16-M2-a rnd-M2-a"`，需要 V1 的 `--bf16`） | BF16 rollout；以及 BF16 rollout 加上同步时舍入到 FP8 网格的权重 |

和 B200 的两点不同：
- B200 上漂移是在单机 5 卡上跑的：PP2 × EP2，切分 `Et*3|t*1L`，第 0 段峰值 116–119 GiB，94 GB 的卡放不下。和 T0 一样不能把第 0 段切小。
- 所以这里改成 PP1 × EP4，并把 rollout 放到另一个节点上。我们在 B200 上用这个布局冒烟跑过 2 步，峰值见 `reference/pkgtest-b200.json`。

H100 的数字只和 H100 自己比：fp8 对 qat。不要和 B200 的步数逐一对比。

```bash
python3 $H/analyze.py <WORK>/runs/drift > <OUT>/V4/summary.json
```

**期望**（B200：`reference/summary-b200.json` 的 `drift` 节）：

- 每个 case 都是 `rc` 0，24 步，无非有限梯度。
- `fp8-M2-*` 的 `logprob_diff` 从第 0 步约 0.09 开始增长，第 4 步约 0.4–0.6，第 23 步 > 1（B200：3.2 和 4.3）。
- `qat-M2-*` 的 `logprob_diff` 全程保持在 0.08–0.10（B200：第 23 步 0.079 和 0.083）。
- 熵（`entropy`）：qat 24 步内不塌到 0（B200：2.57、2.66）。
- `qat_moved_frac_rank0` 每步都有记录。
- 记录训练端各 rank 的峰值和 `median_step_time_s_from_step2`。B200 冒烟时 PP1 × EP4 的峰值是 85.4–87.5 GiB，离 H100 的约 93.6 GiB 只有约 6 GiB 余量。OOM 照实报告。
- 若 H100 上 fp8 不漂移，或者 qat 也漂移，这就是结果，照实报告。不要改学习率或步数去追 B200 的数字。

## 交回

结果推到新分支 `handoff/h100-vime-dsv4-results-20261126`，放在 `h100-handoff/vime-dsv4-20261126-results/` 下：

```
SUMMARY.md                 每个任务一行：通过/失败、关键数字、与期望不同的地方、DSA 后端、所有偏离本文的地方
V0/  build 日志（脱敏）、env-freeze-sm90.txt、base-image-freeze.txt、freeze-diff-vs-b200.txt、build-steps.log
V1/  sha256sum -c 的输出
V2/  runs/smoke/ 下所有文件
V3/  summary.json、progress.txt、pins-*.txt、gpus-*.txt、每个 case 的 train.log、node*.log、probe-rank*.jsonl、rc；nccl 日志只要 grep "NET/" 的结果
V4/  同 V3
not-green-attempts/   失败的尝试，原样
```

- `*.pt` 不交回（回放的数字已经在 `summary.json` 里）。
- 日志超过 50 MB 的，用 `gzip` 压缩后再交。
- 推之前用 `grep` 检查节点名、IP、用户名、`/home`、`/data`、作业号。
