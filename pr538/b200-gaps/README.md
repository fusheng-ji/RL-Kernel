# PR #538 on B200: tests and fixed-mode acceptance at the exact PR head

Head `2d9680d7b8b9c0f7d150bd7cb296cf496652963a`, clean tree, extension built `KERNEL_ALIGN_USE_FAST_MATH=0 TORCH_CUDA_ARCH_LIST=10.0`, B200, driver 580.126.20, CUDA 13.0, torch 2.13.0+cu130. Runner: [run.sh](run.sh). Same setup and caveats as [#537](../../pr537/b200-gaps/README.md) (stacked on it).

| Run | passed | failed | skipped |
|---|---:|---:|---:|
| tests (six-path command), 2 GPUs visible (`results-round1-wrong-model-path/pytest.log`) | 135 | 0 | 0 |
| fixed-mode acceptance (`results/fixed/acceptance.json`): `qualified: true`, `commit` = head, `clean_commit: true` | 2738 checks | 0 | - |

`results/report.json` (with `--fp64-check`) has the same weight hashes and source hashes as the earlier evidence; only the extension hash differs. Reuse-survey modes that need vLLM / SGLang / Megatron / Liger / flash-attn sources were not rerun.

Draft addition for the PR body Results:

> **Re-run at head `2d9680d7` (B200, torch 2.13).** Tests: 135 passed, 0 failed, 0 skipped on two GPUs. Fixed-mode acceptance from a clean checkout of this commit: qualified, 2738 checks, 0 failures (all three streams). Raw data: `pr_assets` `pr538/b200-gaps/` @ `<EVIDENCE_SHA>`.
