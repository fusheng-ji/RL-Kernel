# HANDOFF-RERUN-20261028 — §6 总报告（Hopper agent → Wenbo）

**生成时间（UTC）**: 2026-10-29T09:57:34Z  
**Git（已 push）**: `https://github.com/fusheng-ji/RL-Kernel` branch **`h100-rerun-20261028`**, commit **`70160557`**  
**路径**: `pending/h100-rerun-20261028/`（**未**覆盖仓库根下 legacy `pr537/` … `pr548/`）  
**PR 评论 / 正文**: 未改（由 Wenbo 决定是否 merge → `pr_assets` 并更新 PR 链接）  
**编排**: `run_all.sh` 首轮结束 `2026-10-29T00:38:43+01:00`；#537/#538 验收补跑 `2026-10-29T01:45:57+01:00`；全树脱敏+校验 `finalize_all_evidence.sh` 见 `evidence_finalize_summary.txt`。

## 6.1 运行环境（可发布占位符版）

| 项 | 值 |
| --- | --- |
| Slurm job（overlap 母 job） | `<job>`（本地记录：5875954） |
| 节点 | `node-h100-a`（本地：mcml-hgx-h100-021，单节点 4×H100） |
| GPU 驱动 / 栈 | 见各 PR `env.txt`（已脱敏：`cuda_home`、`_C`、路径均为 `<SCRATCH>` / `<WORKTREE_ROOT>`） |
| Python / Torch | 3.12.15 / 2.13.0+cu130（与 q3n `env.sh` 一致） |
| Qwen-Image 权重 | `<weights dir>`（本地：`~/datastorage/models/Qwen-Image-75e0b4be`） |

## 6.2 证据根目录与完整性

每个 PR 的公开证据树：

`<HOME>/pending/h100-rerun-20261028/pr{PR}/reports/experiments/h100-pr{PR}-h100/`

| PR | `sha256sum -c SHA256SUMS` | `evidence_check.sh` + denylist |
| --- | --- | --- |
| #537 | OK | **CLEAN** |
| #538 | OK | **CLEAN** |
| #495 | OK | **CLEAN** |
| #547 | OK | **CLEAN** |
| #536 | OK | **CLEAN** |
| #548 | OK | **CLEAN** |

校验命令（Wenbo 本地复现）：

```bash
EV=~/pending/h100-rerun-20261028/pr537/reports/experiments/h100-pr537-h100
(cd "$EV" && sha256sum -c SHA256SUMS)
~/datastorage/rl-kernel-h100-split/handoff_v2/evidence_check.sh "$EV" \
  ~/datastorage/rl-kernel-h100-split/handoff_v2/private_denylist.txt
```

（将其余 PR 的 `537` 替换即可。）

## 6.3 分 PR 结论（固定 head，与 handoff §2 一致）

| PR | Head（短 SHA） | 分支 | H100 结论摘要 |
| --- | --- | --- | --- |
| **#537** | `274b6c52` | `feat/qwenimage-adaln-mod-projection` | pytest **105 passed**；`qwenimage_strict_math_compiled()` 探针 **True**；验收链 + `validate_adaln_repair` → **`qualified: true`**（补跑时 `PYTHONPATH=<worktree>`；`cancellation-screen.json` 在 `acceptance/fixed/`） |
| **#538** | `e23c4a97` | `feat/qwenimage-final-adaln-out` | pytest **140 passed**；token_sum 探针 **`[6030, 6030, 6030]`**；验收 + validate → **`qualified: true`** |
| **#495** | `a320e65a` | `fix/174-chunked-logp-backward` | 默认构建 fused_logp pytest **394 passed, 16 skipped**；`KERNEL_ALIGN_FORCE_SM90=1` 重建后 SM90 **18 passed**；benchmark 日志/`report.json` 在树内 |
| **#547** | `6b8b05e5` | `feat/b200-deterministic-logp-autograd` | recursive 套件 **89 passed**（另有 broader logprob/layout/higher_order 日志）；`probe_deterministic_logp_autograd.py` JSON traces **passed** |
| **#536** | `f203661c` / base `16b81175` | `feat/issue-50-hal-autotuning` | head/base × autotune 0/1：**det_gemm / platform / tests/ops** 分进程；head ops **8 failed, 2095 passed**（base 同型失败计数见 `logs/*_ops.log`）；**未**在日志中 grep 到 illegal memory |
| **#548** | `f0b58af2` | `feat/issue-44-tp-ppo` | 回归 pytest **108 passed**（`RUN_TP_CUDA=1`）；4-GPU 框架 harness **Megatron + DeepSpeed 均 FAIL**（**未打补丁**）；见 `logs/framework_harness_report.md` |

### #548 框架失败（原样记录，handoff §3.3）

- 本轮：`check_tp_ppo_rollout --gpu-layout shared --critic-tp 1`（4 进程）。
- 旧 hopper-main 公开结论：`--gpu-layout dedicated` 曾双框架 PASS（layout 差异已在 report 中注明，不单归因）。
- Megatron：`unsupported Megatron checkpoint keys: {'decoder.final_layernorm._extra_state'}`。
- DeepSpeed：`ModuleNotFoundError: No module named 'deepspeed'`（本轮 srun 未像专用脚本那样配 `PYTHONPATH`+`q3n/fw`）。

## 6.4 首轮 orchestrator 后的修复（仍属同一 pending 目录）

1. **#537/#538**：首轮验收因缺 `PYTHONPATH` 失败；已用 worktree `PYTHONPATH` 重跑验收链 + validate（pytest/探针/token_sum 保留首轮）。
2. **`cancellation-screen.json`**：validator 要求位于 `acceptance/fixed/`（`adaln_parameter_screen.py` 输出）；已修正 `fix_qwenimage_acceptance_and_reports.sh` 与 `run_all.sh`。
3. **脱敏**：`redact_tree` + 重算 `SHA256SUMS`；`evidence_check_last.txt` 写入后再脱敏一轮（避免 CLEAN 行含站点路径）。
4. **并发**：曾有多实例 `run_acceptance` 重叠；已加 `acceptance.lock` 并清理重复 srun。

## 6.5 辅助脚本与日志索引

| 文件 | 用途 |
| --- | --- |
| `run_all.sh` | 六 PR 首轮编排 |
| `orchestrator.log` | 首轮全量日志 |
| `fix_qwenimage_acceptance_and_reports.sh` | #537/#538 验收 + #548 framework report |
| `finalize_all_evidence.sh` | 六 PR 脱敏 + SHA256 + evidence_check |
| `evidence_finalize_summary.txt` | §6.2 表的一行摘要 |
| `rerun_acceptance.log` | #538 完整验收补跑 |

## 6.6 未做 / 需 Wenbo 决定

- **未**更新 GitHub PR 正文（H100 Results 小节若需与本次 head 对齐，由 Wenbo 指示后再 edit）。
- **未** push 至 `datastorage/RL-Kernel-worktrees/pr_assets/` 或任何 `hopper-*` 路径。
- **#548** 框架 FAIL：若需与旧 dedicated run 对比，需另开**诊断 run**（handoff：不得为“通过”而偷偷改代码或混写结论）。
- **#536** ops **8 failed**：失败用例详情在 `pr536/.../logs/head_autotune*_ops.log` 等；是否阻塞 merge 由 Wenbo 定。

## 6.7 Agent 签收

- 六 PR 证据树：**CLEAN + SHA256 OK**（2026-10-29 UTC 脱敏批次）。
- 边界遵守：**无** `pr_assets` 写入、**无** PR 评论、**无** 框架失败补丁。
- 下一步（仅当 Wenbo 说 “push evidence”）：按 v2 规则打包/推送**新标签目录**，勿覆盖已有 `pr537/538/495/536/547/548`。
