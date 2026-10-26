# Handoff → B200 统筹（H100 Qwen3-Next norm #467 / #468 / #469）

**From:** H100 agent (Hopper, sm90). **To:** B200 统筹对话 / Wenbo.  
**Date (UTC):** 2026-10-25. **HANDOFF v2** 脱敏与 RFC #428 §2.2（H100/B200 分目录、不混 claim）已遵守。

---

## 1. 已上交的脱敏证据（`pr_assets`）

**Remote:** `https://github.com/fusheng-ji/RL-Kernel` branch **`pr_assets`**.

| Commit | 内容 |
|---|---|
| `1a28d1c0` | 新增 `pending/h100-qwen3-next-norm/pr467|468|469/.../h100-pr*-h100/`（A–D、467/468 B/C 等） |
| `e4ad944a` | #469：`gdn_provider_agreement.json` 有效 JSON；F @ `d1129d9f` |
| （待 push） | `tests.json` 时间戳修正 + 本 `HANDOFF.md` |

**根路径：**

```text
pending/h100-qwen3-next-norm/
  pr467/reports/experiments/h100-pr467-h100/
  pr468/reports/experiments/h100-pr468-h100/
  pr469/reports/experiments/h100-pr469-h100/
```

每包含：`README.md`（第 3 行 PR/分支/pin）、`tests.json`、`env.txt`、`redaction.json`、`SHA256SUMS`、`logs/`、`report.json`（467/468）、figures、reuse JSON/PNG；469 含 `gdn_provider_agreement.json`。

**校验：** 脱敏后 `evidence_check.sh`（H100 本地脚本）三目录均为 **CLEAN**；`sha256sum -c SHA256SUMS` **0 mismatch**。占位符 `<HOME>`、`<REPO>`，见各 `redaction.json`。

**勿与 B200 混用：** 现有 B200 仍在 `pr467/`、`pr468/`（`docs/usage/evidence/*-b200/`），不要写入 `h100-pr*-h100/`。

---

## 2. H100 测量摘要（给 PR 正文 **Results (H100 …)** 用）

| PR | Pin（测量） | A pytest | B | C | D | F |
|---|---|---|---|---|---|---|
| **#467** | `d42c8357` | **203 / 0 / 73** | ok | `qwen3_next_rms_norm` reuse ok | — | — |
| **#468** | **`5aa9aa1c`** (B/C; pytest/D @ `1e2eaec0`) | **300 / 0 / 73** | ok @ 5aa9aa1c | qwen3 + gated reuse ok @ 5aa9aa1c | **3× check_operator pass_rate=1.0** @ 1e2eaec0 | — |
| **#469** | A: `836c483d`；F: **`d1129d9f`** | **304 / 0 / 73** | — | — | — | **ok**（`git_dirty: false`） |

- **Skips（三 PR 相同类）：** 1× 需 2 CUDA；67× Ascend `rmsnorm_ascend` 未编（与 B200 正文 Ascend skip 同类）。
- **Idle GPU：** 467/468 B/C 前 `nvidia-smi` 无占用（见 `logs/idle_before_bc.txt`）。
- **环境（代表）：** NVIDIA H100 sm90，driver 580.173.02，torch 2.13.0+cu130，vLLM **0.30.0**，见各 `env.txt`。
- **Worktree（H100 机，未上交）：** `<HOME>/h100_norm_handoff/worktrees/pr{467,468,469}-h100` @ 上表 pin；每 PR 独立 `pip install -e` + `_C` 校验。

**未跑：** E 负对照（`pr467/negative-control` 仅 B200 已有 logs；H100 未 `RUN_E_NEGATIVE=1`）。

---

## 3. Fork 侧代码变更（统筹需知）

| 项 | SHA | 说明 |
|---|---|---|
| #469 F 路径修复 | **`d1129d9f`** | `ws1_gdn_provider_agreement.py` → `tests/models/qwen3_next/check_gdn_recurrent_golden.py` |
| 已 push | `origin/feat/ws1-c6-gdn-recurrent-golden` | `836c483d..d1129d9f` |

**统筹建议：** upstream #469 / stack tip 应对齐 **`d1129d9f`**（至少 F 可复现）；H100 pytest 仍在 **`836c483d`**，与 F 差 1 commit，已在 #469 `README.md` / `tests.json` 标明。

#467/#468 fork pin 未改：`d42c8357`、`1e2eaec0`。

---

## 4. B200 统筹待办（H100 侧不再做，除非 Wenbo 重开）

1. **PR 正文：** 在 #467 / #468 / #469 增加 **Results (H100 sm90, …)**，链到 `pr_assets` 上 `pending/h100-qwen3-next-norm/...`（raw 或相对路径按仓库惯例）；维持 §2.2 不跨卡声称。
2. **#469：** 可选在 B200 补 `pr469/` 下 B200 版 `gdn_provider_agreement.json`（正文已有历史 B200 数字）；与 H100 文件分目录。
3. **误跑丢弃：** `~/datastorage/q3n/out/hopper_base_gates/pr467|468|469`（非 pin HEAD）**不是**本 handoff 证据。
4. **Push 规则：** H100 证据已按 Wenbo 指示 push；后续改 bundle 仍须脱敏 + `SHA256SUMS` + CLEAN 后再 push。

---

## 5. H100 机本地工具（未进 git）

`<HOME>/h100_norm_handoff/`：`run_h100_qwen3_norm.sh`、`finalize_evidence.sh`、`evidence_check.sh`、`denylist.txt`、`HANDOFF_B200_NORM.md`（给 B200 机参考的对称说明）。

---

## 6. 短报告（归档）

```text
H100 #467/#468/#469, 2026-10-25T13:24Z (F 2026-10-25T14:10Z)
heads: 467=d42c8357 468=1e2eaec0 469-pytest=836c483d F=d1129d9f
A: 203/0/73  300/0/73  304/0/73   failures: none
B: 467 ok 468 ok   C: 467 ok 468 ok   D: 3/3 pass   F: ok @ d1129d9f
pr_assets: pending/h100-qwen3-next-norm/  commits 1a28d1c0 + e4ad944a (+ handoff commit)
committed/pushed: yes (pr_assets)   local patches on fork: d1129d9f (#469 only)
```

---

**Owner:** Wenbo。H100 agent 停在此处；后续决策在 B200 统筹对话。
