# H100 任务：`ws2_pipeline_parallel`（h3-pipeline-v1，2026-11-26）

H3 的 50 个 block 分成 PP2 / PP4 个 stage，每个 stage 跑自己的 TP × CP (× SP) 网格。这一轮在 H100 上补 B200 做不了的两件事：
**strict 边界行**（video 投影只有 Hopper 能跑）和 **stage 边界跨节点**。

- 代码：本分支的父 commit `008ca510d` = PR 2（训练步，叠在 PR 1 `073b9cd77` 模型层上，基于 `test-h3` 的 `60c6aaf94`）。P1–P4 都在本分支 checkout 上跑（P4 的阴性对照需要 `008ca510d` 的修复）；本目录只有这份说明，不影响代码。
- 环境：与 GEMM 行那一轮相同的 `<E1>`，torch 2.13 及以上（`torch.distributed.pipelining` 的静态元数据接口）。先 `RL_KERNEL_H3_STRICT=1` 构建，确认 `_C` 来自这个 checkout。
- 权重：checkpoint 与前几轮相同；`RL_KERNEL_H3_WEIGHTS` 指向含 `h3_block0.safetensors` 的目录。
- 显存：每卡先跑单卡 canonical（33B 全权重加一个 block 的梯度），925 / 928 行在 94 GB 上与 #701 那轮相同。放不下就换 size 40 并记录 OOM，不要改代码。

## 任务

| 任务 | 内容 | 硬件 | 时长 |
|---|---|---|---|
| P1 | 测试：`tests/models/minimax_h3/test_h3_pipeline.py`、`test_h3_parallel_model.py`、`test_h3_packed_forward.py`、`test_h3_sharded_load.py`、`test_h3_grid_actor.py` | 1 × 4 | 20 min |
| P2 | 50 block，strict 边界，单节点 | 1 × 4 | 1 h |
| P3 | 50 block，strict 边界，stage 边界跨节点 | 2 × 4 | 1.5 h |
| P4 | G5 训练步（provider 边界：refiner LoRA 要求） | 2 × 4 | 1 h |

### P2（每条一个 `--out`）

```bash
run="srun --ntasks=N python tools/validation/models/h3_parallel_full_model.py --checkpoint $CKPT --sharded"
$run --pp 2 --tp 1 --size 40  --train-time-embedder            # N=2
$run --pp 4 --tp 1 --size 925 --train-time-embedder            # N=4
$run --pp 2 --tp 2 --size 925 --train-time-embedder            # N=4
$run --pp 2 --tp 2 --sp --size 928                             # N=4
$run --pp 2 --tp 1 --cp 2 --cp-mode ring --size 925            # N=4
$run --pp 4 --tp 1 --size 40 --inject 30:5:2                   # N=4，阴性对照
$run --pp 4 --tp 1 --size 40 --inject-handoff 26:5             # N=4，应报 handoff.26 第 5 行
$run --pp 4 --tp 1 --size 40 --inject-handoff 26:5:grad        # N=4，应报 block.25.grad 第 5 行
$run --pp 4 --tp 1 --size 40 --inject-handoff 26:5:wire        # N=4，应在每个 rank 上拒绝
```

不带 `--provider-boundaries`，所以边界行是 strict 的。

### P3（两节点各 4 卡，rank 按节点连续）

```bash
$run --pp 2 --tp 4 --size 925 --lora-rank 64 --train-time-embedder --local-size 4   # N=8，一个 stage 一个节点
$run --pp 4 --tp 2 --size 925 --lora-rank 64 --train-time-embedder --local-size 4   # N=8
$run --pp 2 --tp 2 --cp 2 --sp --size 928 --lora-rank 64 --local-size 4             # N=8
$run --pp 8 --tp 1 --size 925 --local-size 4                                        # N=8
$run --pp 1 --tp 8 --size 925 --lora-rank 64 --local-size 4                         # N=8，显存与时间的对照
```

### P4

```bash
g5="srun --ntasks=8 python tools/validation/models/h3_grid_train_step.py --checkpoint $CKPT --provider-boundaries --refiner-lora --size 928"
$g5 --pp 2 --tp 4 --repeat
$g5 --pp 4 --tp 2 --repeat --passes 5
$g5 --pp 2 --tp 2 --cp 2 --sp --repeat
$g5 --pp 2 --tp 4 --max-norm 1e9 --inject 30.attn.to_v.A:5   # 阴性对照：看 inject_detected，pass 为 false
$g5 --pp 2 --tp 4 --repeat --size recipe                      # 后一个 --size 生效；recipe 大小，记录每步秒数和峰值显存
```

## 期望

- P2 / P3：`all_bitwise` 为 true（对照项除外），`pipeline_first_drift_*` 为 null，`handoffs_verified` 全部为 true，`packed_bindings` 和 `boundary_bindings` 里本 stage 拥有的行是 `row`，其余是 `other stage`。
- 对照项在指定位置被报告；`:wire` 的 `refused` 非空且 8 / 4 个 rank 都有。
- P4：`pass`、`every_canonical_tensor_is_held`、`export_is_canonical` 为 true。
- 单节点和跨节点都要记录 `seconds`、`sharded.parallel_peak_bytes`、`resident_param_bytes`。
- 出现不逐位相等时：保留报告，不要改代码，记下第一个漂移节点。

## B200 参考结果（2026-11-26，provider 边界，1 节点 8 卡）

- 模型层（`073b9cd77`）：PP2、PP2 不均匀 10/40、PP4、PP8、TP4×PP2、TP2×PP4（LoRA）、TP2×CP2(ring)+SP×PP2、TP2×CP2(Ulysses)×PP2 全部逐位相等；四个对照分别报告为 `block.30` 第 5 行、`handoff.26` 第 5 行、`block.25.grad` 第 5 行、8 个 rank 全部拒绝。
- 每卡常驻权重 / 峰值（GiB，925 行）：TP8 30.3 / 32.2，TP4×PP2 18.2 / 20.0，TP2×PP4 12.5 / 14.3，PP8 10.0 / 11.1。
- G5（`5e8153fbc`，对照在 `008ca510d`）：928 行和 recipe 大小全部通过，1F1B 与 serial / GPipe 逐位相同，导出文件等于 canonical；对照只在 rank 5 报告一个张量。recipe 大小每 pass 秒数：TP8 5.0、TP4×PP2 5.1、TP2×PP4 5.7；峰值 35.3 / 23.0 / 18.0 GiB。

## 交回

分支 `handoff/h100-h3-pipeline-results-20261126`：每个 `report.json`、日志、`env.txt`（torch、驱动、GPU 型号、节点数），脱敏（集群、节点、路径、用户名）。
