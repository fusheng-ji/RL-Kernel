# H100 整合任务包：所有未完成的 handoff，双节点 × 4 H100（2026-11-27）

本文把当前所有未完成的 RL-Kernel H100 handoff 合在一起，供 H100 端**在拿到资源之前**写好完整的调度计划，并把一切不需要 GPU 的准备做完。拿到分配后只剩编译和运行。

各项的代码、命令、期望值和交回清单仍以各自 README 为准；本文只规定：
- 资源的形状；
- 每项的准备工作；
- 先后顺序和停止条件；
- 调度计划里要写什么。

仓库：https://github.com/fusheng-ji/RL-Kernel（下面所有分支都在这个 fork 上）。

## 0. 资源形状

所有任务都放在**双节点 × 4 H100（共 8 卡）的独占分配**里，不再用 mix 节点：

```bash
salloc -N2 --gres=gpu:4 --exclusive --time=<见第 3 节> ...     # 或等价的 sbatch
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); NA=${NODES[0]}; NB=${NODES[1]}
```

一共五个分配：A、B1、B2a、B2b、C（见第 3 节）。CPU 工作（下载、导入镜像、建 checkout、校验 bundle）都在分配之外做完。

## 1. 任务清单

| 项 | handoff 分支 @ head | README | 结果分支 | 负责 |
|---|---|---|---|---|
| T10-a、T10-edit | `handoff/h100-h3-image-20261123` @ `767fc1b99` | `h100-handoff/h3-image-t10-20261126/README.md` | `handoff/h100-h3-image-results-20261123` → `results/T10/` | RFC #738 PR #868 / #869 / #870 |
| QK2 | `handoff/h100-single-card-20261126` @ `ea59c4c7d` | `h100-handoff/single-card-20261126/README.md` 的 “QK2” 一节 | `handoff/h100-h3-qk-rmsnorm-results-20261125` → `results/QK2/` | RFC #420 QK RMSNorm |
| verl-omni FA3 F0–F3 | `handoff/h100-verl-omni-fa3-20261124` @ `2ff9c5253` | `h100-handoff/verl-omni-fa3-20261124/README.md` | `handoff/h100-verl-omni-fa3-results-20261124` | H3 native baseline |
| pipeline P1–P4 | `handoff/h100-h3-pipeline-20261126` @ `307cf4747` | `h100-handoff/h3-pipeline-20261126/README.md` | `handoff/h100-h3-pipeline-results-20261126` | RFC #420 ws2_pipeline_parallel |
| miles H0–H8 | `handoff/h100-miles-inkling2-20261126` @ `25cb8d21c` | `h100-handoff/miles-inkling2-20261126/README.md` | `handoff/h100-miles-inkling2-results-20261126` | RFC #736 Path A baseline |
| vime V0–V4 | `handoff/h100-vime-dsv4-20261126` @ `b88c1f51a` | `h100-handoff/vime-dsv4-20261126/README.md` | `handoff/h100-vime-dsv4-results-20261126` | vime PR #2 |
| AReaL | `handoff/h100-areal-placement-20261124` @ `85833e982` | `h100-handoff/areal-placement-20261124/HANDOFF.md`（续做笔记） | `handoff/h100-areal-placement-results-20261126`（新建） | AReaL placement |

已经交回、**不要再跑**的：T1–T9、QK（Q1/Q2）。T8 run-edit、T8 run-b 和 T9-edit 由 T10 取代，不要在旧的 `b685c4991` / `69e0e42c9` 上重跑。

## 2. 拿到资源之前要做完的事

### 2.1 集群核对（写进计划的“资源核对”一节）

逐项记下结果。“不满足时”那一列是规定好的处理方式，照做，不要自己想办法绕过去。

| 检查 | 要求 | 不满足时 |
|---|---|---|
| 卡型 | H100，`memory.total` ≥ 95,000 MiB（94 GB） | 80 GB 卡上：T10-edit（峰值约 82 GiB）放不下，vime V4（B200 峰值 85–88 GiB）大概率放不下。照样排，OOM 照实交回 |
| 驱动 | ≥ 580（CUDA 13.0；miles 镜像要求） | miles 全部不跑，报告 |
| 节点间网络 | InfiniBand（vime 期望 `NET/IB`） | 照跑，记录看到的是哪种 |
| 共享文件系统 | 两个节点都能读写，空间约 **700 GB**：miles 450 GB + vime 200 GB + verl-omni 和 H3 输出 | 先停下问 |
| 节点本地盘 | enroot 数据（vime）、短 `TMPDIR`（vLLM 的 socket 路径太长会出错） | 先停下问 |
| 单个作业时限 | 至少 18 h（B2b 的 miles H6） | 先停下问；不要自己减 H6 的步数 |
| 计算节点能否访问 Hub | verl-omni F0 要下载 FA3 kernel | 按 verl-omni README：在能联网的 GPU 节点上跑一次 F0 缓存 `v1`，F1–F3 用 `HF_HUB_OFFLINE=1`。哪儿都不行就停下报告 |
| enroot / pyxis | miles、vime 都用 enroot | 先停下问 |
| 账号、分区 | `-N2 --gres=gpu:4 --exclusive` 能排上 | 先停下问 |

### 2.2 代码和 handoff 文件（CPU，登录节点）

```bash
cd <REPO>
F=https://github.com/fusheng-ji/RL-Kernel.git
for b in h3-image-20261123 single-card-20261126 verl-omni-fa3-20261124 h3-pipeline-20261126 \
         miles-inkling2-20261126 vime-dsv4-20261126 areal-placement-20261124; do
  git fetch $F handoff/h100-$b:refs/h100/$b
done
git ls-remote $F 'refs/heads/handoff/h100-*'      # 对照第 1 节的 head，不一致就停下问
```

然后按各自 README 的 “Code” 一节做到 **build 之前** 为止：
- 建 detached worktree；
- `git bundle verify`；
- 核对 commit 和 tree hash。

| worktree | commit | 来源 |
|---|---|---|
| `<W10>` | `4cfa3cbfc`（tree 见 T10 `reference/tree.txt`） | T10 bundle，或 fork 分支 `feat/h3-image-ws1-full` |
| `<WQ>` | `c56b81c4` | QK bundle；上一轮在 H100 上编过的可以直接用，但要保留或补上 `build.log` |
| `<WB>` | `13d0f906` | `RL-Align/RL-Kernel` 的 `test-h3` |
| `<WP>` | `307cf4747` | pipeline handoff 分支本身（代码就是它的父 commit `008ca510d`） |
| verl-omni | `708f4d63…` | `verl-project/verl-omni`，不安装 |
| vime `src/` | `aebf8fea` + 修复 `a2d04354` + 补丁 | `bash setup_sources.sh`（登录节点），四个 diff 哈希写进计划 |

**RL-Kernel 的编译必须在 H100 上做。** `setup.py` 按可见 GPU 的 compute capability 生成代码，不看 `TORCH_CUDA_ARCH_LIST`，登录节点上编不出 sm_90。所以 `<W10>`、`<WQ>`/`<WB>`、`<WP>` 的编译排在分配 A 的开头（见第 3 节），不要提前在登录节点上编。

### 2.3 环境（CPU）

| 名称 | 内容 | 谁用 | 准备 |
|---|---|---|---|
| `<E1>` | torch 2.13.0+cu130 | T10（编译）、pipeline | 前几轮留下的，不要动 |
| `<V>` | `<E1>` 上的 venv，Diffusers `875373ac`（0.42.0.dev0）+ `accelerate==1.16.0` | T10（运行） | 前几轮留下的；缺了就按 `h3-image-20261123/README.md` 的 “Environment” 一节重建，并写明 |
| `<E>` | QK 那次的环境 | QK2 | **只用一个环境**：上一轮 `env.txt` 和 `pip-list.txt` 的版本不一致 |
| `<E2>` | 2026-11-22 S2 环境：torch 2.14.0、vLLM 0.33.0、vLLM-Omni `6039984e`、verl `2b4ed682` | verl-omni | 不要改；reward 包装进 `<OVERLAY>`（`pip --no-deps --target`） |
| miles 镜像 | 摘要 `sha256:0a106222…` | miles | H0（`h0_prepare.sh`），约 1 h，需要联网 |
| vime 镜像 | `vllm/vllm-openai@sha256:cb12c301…` | vime | V0 在分配 A 里导入并编译（见第 3 节） |
| AReaL | 按 HANDOFF.md 原来那份 handoff 的环境 | AReaL | venv 放在节点本地盘或预热，见 HANDOFF.md 第 3 条（worker import 时间） |

### 2.4 权重和数据（CPU，需要联网）

| 项 | 内容 | 大小（约） | 校验 |
|---|---|---|---|
| `<IMG>` | `MiniMaxAI/MiniMax-H3-Image` @ `8ced6006`，含 T6 下载的 `transformer_diffusers/`、`examples/`、`assets/showcase/photo.jpg` | 70 GB | VAE sha256 `5bb28096…` |
| `<CKPT>` | MiniMax-H3 @ `42ed227e`，含完整的 `FL2VA/` 和 `transformer/` | 前几轮已有 | T10 读 text encoder；verl-omni 和 pipeline 读模型 |
| `<WEIGHTS>`（QK2）/ `RL_KERNEL_H3_WEIGHTS`（pipeline） | 要有 `h3_conditioning.safetensors`、`h3_block0.safetensors` **和** `hf_full/`，与前几轮相同 | 一个约 4.8 GB 的分片 + `hf_full/` | 上一轮 QK 没设这个变量，两个真实权重测试被跳过了。缺前两个文件时（CPU，需要联网）：`cd <WQ> && PYTHONPATH=<WQ> <E>/bin/python tools/weights/prepare_h3_weights.py --out <WEIGHTS> --block`，它会对 manifest 校验 sha256 |
| verl-omni `<REWARDS>` | `laion/larger_clap_general` @ `ada0c23a`；`seung275/imagebind_huge.pth` @ `b17ac77b` | — | `imagebind_huge.pth` 的 sha256 `d6f6c22b…` |
| verl-omni FA3 kernel | `kernels-community/flash-attn3` `v1` | 小 | 见 2.1 |
| miles 模型 | H0 下载的 16 个文件 | 70 GB | 对照 `reference/b200-summary.json` 的 `download_sha256` |
| vime checkpoint | V1：`DeepSeek-V4-Flash-Base` @ `8855555d` 的第 1–5、45、46 号分片 → 4 层切片（可选 BF16 版） | 35 + 27 GB（+52 GB） | `reference/ckpt-*.sha256` |

## 3. 分配和顺序

### 分配 A：单卡和单节点（`--time=08:00:00`）

开头 20–30 分钟（编译 sm_90，两个节点同时做）：
- **node-a**：编 `<WP>`（pipeline）。
- **node-b**：
  - GPU 0：编 `<W10>`（约 15 min）；
  - GPU 2：编 `<WQ>`（如需）和 `<WB>`；
  - GPU 3：启动 vime V0（它自己编，3–5 h，48 CPU）。

| 节点 / 卡 | 顺序 | 时长（估） |
|---|---|---|
| **node-a**（整节点，只跑需要干净计时的） | verl-omni F0 → F1 → F2 → pipeline P1 → P2 → miles H1；剩下的时间跑 verl-omni F3（可选） | 0.2 + 0.75 + 1.5 + 0.3 + 1 + 0.5 ≈ 4.3 h；F3 再 1.5–2 h |
| **node-b GPU 0** | T10-a：suite → T8 capture / run-edit / run-t2i / run-b → `t2i_upscaled` 对比 T9 的 `run-a` | 2–3 h |
| **node-b GPU 1** | T10-edit（`<OUT>/T10/ext.txt` 第一项为 `True` 后开始） | ≥ 6 h，给满 |
| **node-b GPU 2** | QK2 | 约 1 h |
| **node-b GPU 3** | vime V0 → V2 | 3–5 h + 40 min |

**依赖：**
- T10-edit 等 `<W10>` 编完。
- T10-a 的 c 步需要上一轮的 `<OUT>/T9/t2i/run-a`；不在了就跳过并写明。
- miles H1 需要 H0 已完成。
- vime V2 需要 V0、V1 已完成。

### 分配 B1：pipeline 双节点 + vime（`--time=12:00:00`）

需要 A 里的 vime V0、V2 已完成，`<WP>` 已编译。

| 顺序 | 项 | 时长（估） |
|---|---|---|
| 1 | pipeline P3（5 条，`--local-size 4`）→ P4（G5） | 1.5 h + 1 h |
| 2 | vime V3 → V4 | 3–5 h + 2 h |

### 分配 B2a：miles H2–H5（`--time=08:00:00`）

需要 H0 和 H1 已完成。H2 → H3 → H4 → H5，约 5–6 h。H4 里单节点的 SGLang self-check 在这两个节点之一上**先后**跑，不要再申请第三个节点。

### 分配 B2b：miles H6–H7（`--time=18:00:00`）

在 B2a 之后。H6 → H7，约 10–16 h。H8 是 CPU 工作，放到分配外做。H6 被 requeue 时按 README：移走半成品，用 `TAG=r1` 重来。

### 分配 C：AReaL（`--time=08:00:00`）

按 `HANDOFF.md` “Next steps on H100” 的第 1–3 步。
1. actor 在 driver 节点上和不在 driver 节点上，各跑一次 3–5 步的 native sync run。
2. 两次都记录 actor 和 vLLM 进程的这些信息：
   - `taskset -cp`；
   - `Cpus_allowed_list`；
   - 线程数和 NUMA。
3. 然后每次只改一处，下面四种依次试：
   - `srun --cpu-bind=none`；
   - 把 driver 绑到别的核；
   - 限制线程数；
   - 把 driver 和 Ray head 放到 CPU 节点。

注意事项（见 HANDOFF.md 前两节）：
- `verify run` 的退出码不可信，看 `run.json` 的 `returncode` 和 `stats.jsonl` 的行数；
- 不要从源码树启动 `verify`；
- 记录 worker import 时间。

验收：两种布局下 actor train 是否相差 5% 以内。

### 顺序

```
CPU 准备（第 2 节） ─→ A ─┬─→ B1
                          ├─→ B2a ─→ B2b
                          └─→ C
```

A 之后，B1、B2a、C 可以按排队情况任意先后。

### 停止条件（会改变后面的排程，计划里要写清楚）

| 条件 | 处理 |
|---|---|
| 卡不是 94 GB | 照排；T10-edit、vime V4 的 OOM 照实交回 |
| verl-omni F0 `same_kernel_file` 为 false，或任一侧加载不了 kernel | verl-omni 停在 F0 报告；node-a 接着跑 pipeline P1/P2 |
| vime V0 出现 `STEP_FAIL` | vime 停下报告；B1 只跑 pipeline P3/P4（约 2.5 h，`--time=04:00:00`） |
| vime V2 的 cuDNN DSA 失败 | 不停；V3、V4 全部用 `DSA_BACKEND=none`，写明 |
| miles H0 镜像摘要不对 | miles 全部停下报告；不排 B2a、B2b |
| miles H2 的 b、c、d 都不通过 | 停在 H2 报告；不排 H3 之后和 B2b |
| miles H5 估出 H6 超过 16 h | 停在 H5 报告；不排 B2b |
| pipeline 925 / 928 行放不下 | 按 README 换 size 40，记录 OOM |

## 4. 调度计划要写什么

拿到资源前写好 `PLAN.md`，内容包括：
1. **资源核对**：2.1 的每一项和结果（卡型、驱动、网络、共享盘可用空间、本地盘、时限、Hub、enroot）。
2. **准备状态**：2.2–2.4 的每一行写“完成 / 缺 / 重建”，附校验输出（tree hash、bundle verify、sha256）。
3. **每个分配的作业脚本**：
   - 哪个节点、哪张卡、多少 CPU；
   - 起步顺序；
   - 每一步的 `--time`；
   - 输出目录。
   单卡步骤用 `srun -N1 -n1 -w <node> --gres=gpu:1 --exact`，并写明 README 里的 `$GPU` 设成哪张卡。miles 和 vime 的 sbatch 脚本有两种起法：
   - 在分配里按脚本内容用 `srun` 起；
   - 单独 `sbatch -N2 --gres=gpu:4 --exclusive`，单节点的 stage 只用第一个节点。
   选哪种都写明。
4. **停止条件**：第 3 节那张表，每条对应到脚本里的哪一步。
5. **交回顺序**：哪些结果分支先推。同一个结果分支一个接一个推，每次先 fetch。

**拿到资源前，先把 `PLAN.md` 和 2.1–2.4 的核对输出推到 `handoff/h100-schedule-results-20261127`。** 这边看过再开跑；如果 2.1 有需要“先停下问”的项，就在 `PLAN.md` 里写明。

## 5. 通用规则

- **README 为准**：
  - detached 干净 worktree，核对 commit 和 tree hash，不改任何文件；
  - 每步前确认 `_C` 来自本 checkout；
  - 容器里的软件不改。
- **失败就是结果**：
  - 原样保留，放进 `not-green-attempts/`，不要重跑到绿，不要改代码、阈值或布局；
  - OOM 是结果，不要开 offload 硬塞；
  - 重跑只用新的 `TAG` / `SUITE`。
- **计时**：需要干净计时的步骤只放在 A 的 node-a 和整个双节点分配里，分别是：
  - verl-omni F1–F3；
  - pipeline 的 `seconds`；
  - miles H5；
  - vime V3/V4。
- **不杀别人的进程**：不要 `pkill`，不要 `ray stop --force`（vime、miles README 写了原因）。
- **不放宽安全设置**（ptrace、kernel 发布者检查、`trust_remote_code`）；遇到了就停下报告。
- **不上传**：`WANDB_API_KEY`、`HF_TOKEN` 不设。

## 6. 交回

- 每项只推自己的结果分支（第 1 节）：先 fetch 最新 tip，在 `FETCH_HEAD` 上新建分支再加，**不要 force push**。
- `handoff/h100-h3-image-results-20261123` 的历史 2026-11-26 被重写过，现在是 `2f7c3bed6`。旧的本地 `da4e9bc91` / `630266d29` 作废，**不要 merge 或 rebase 旧的本地分支**。
- **推之前**对要提交的所有文件**和文件名**跑一次 `grep -rE '<你的用户名>|<项目号>|<集群文件系统前缀>'`，结果必须为空。上一轮就漏在 traceback 和 `tmp/pytest-of-<user>/` 里。rank / Ray 报告里的 `host` 字段改成 `node-a` / `node-b`。
- 不要提交：`*.png`、`*.safetensors`、`*.pt`、checkpoint、镜像、dump、`tmp/`。`*.log` 需要 `git add -f`；超过 50 MB 的 gzip。
- 脱敏：
  - 集群、分区、账号、作业号、节点名，主机名；
  - conda 环境名，绝对路径（用 `<HOME>` / `<WORK>` / `<OUT>`）；
  - 用户名，项目号，邮箱，IP，GPU UUID，PCI bus id。
- 每个结果分支的 `SUMMARY.md` 写明：
  - 分配（A / B1 / B2a / B2b / C）、节点和卡号；
  - 与 README 的每处偏差，包括本文的排法。
- `git commit -s`，不加 AI 署名。不动 handoff 分支、代码分支和 `pr_assets`，不开 PR，不评论。
