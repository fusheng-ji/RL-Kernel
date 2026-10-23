# B200 Handoff：Euler step-3 dual-trajectory 验证（seed 33）

**分支：** `handoff/b200-euler-step3-probe`（基于 **`eb7840bb`**，与 #571 头一致；**不含**对 feat/h3-full-transformer 的新增 PR 改动）

**目的：** 在 **sm100 / B200** 上跑 step-3 反事实探针，区分 Euler 累积 vs step-3 layout 敏感。

---

## 0. B200 拉代码（推荐）

```bash
git clone https://github.com/fusheng-ji/RL-Kernel.git
cd RL-Kernel
git fetch origin handoff/b200-euler-step3-probe
git checkout handoff/b200-euler-step3-probe
git rev-parse HEAD   # 应为 eb7840bb… + handoff 目录
```

脚本位置：`handoff/b200-euler-step3-probe/h3_dual_trajectory_step3_probe.py`

---

## 1. 环境与权重

| 项 | 要求 |
|----|------|
| GPU | **NVIDIA B200**（sm100） |
| 权重 | `MiniMax-H3@42ed227`，HF 布局 `hf_full/transformer/` **14 个 shard** |
| torch | **2.13.0+cu130**（与现有 B200 证据一致） |
| TF32 | 关闭（脚本内已设） |

```bash
pip install -e .
python -c "import rl_engine._C as C; print('ok', hasattr(C,'deterministic_attention_forward'))"
```

**不要**从 Hopper 拷贝 `_C*.so` 到 B200。

---

## 2. 一键跑

```bash
export CKPT=/path/to/MiniMax-H3/hf_full
export OUT=/path/to/output/euler-probe-b200-seed33
chmod +x handoff/b200-euler-step3-probe/run_b200.sh
./handoff/b200-euler-step3-probe/run_b200.sh
```

完整 4-step strict dual trajectory 会自动跑（若无 safetensors），**预留 ≥2h、单卡 ~68GiB**。

已有 safetensors 时：

```bash
./handoff/b200-euler-step3-probe/run_b200.sh \
  --latent-dir /path/to/dual_trajectory/output \
  --skip-full-run
```

---

## 3. 产出与判读

交付：**`$OUT/seed33_report.json`**

| 条件 | 结论 |
|------|------|
| **A** video velocity max ≈ **0.55–0.70** | 基线 OK |
| **B** ≪ **A**（B < **1e-3**） | `supports_latent_accumulation` |
| **B** ~ **0.5–0.7**（与 A 同量级） | `supports_step3_layout_sensitivity` |

H100 对照（勿在 B200 当最终结论）：  
https://raw.githubusercontent.com/fusheng-ji/RL-Kernel/1757ebd0fd7f458f67cedd968b8b3545c6a49169/pending/h3-full-transformer/reports/experiments/h3-euler-step3-probe-h100/seed33_report.json

---

## 4. 跑完后回传

1. `seed33_report.json`  
2. （可选）两个 safetensors 的 sha256  
3. `nvidia-smi` 一行 + `python -c "import torch; print(torch.__version__)"`

上下文：PR [#571](https://github.com/RL-Align/RL-Kernel/pull/571)，B200 历史 A≈0.62 见 `pr_assets` …/h3-full-transformer-b200/dual_trajectory_seed33.json
