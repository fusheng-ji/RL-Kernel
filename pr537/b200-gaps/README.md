# PR #537 on B200: tests and fixed-mode acceptance at the exact PR head

Head `a73c66c5e2db7ec844e30ac763ce85d410179259`, clean tree, extension built `KERNEL_ALIGN_USE_FAST_MATH=0 TORCH_CUDA_ARCH_LIST=10.0`, B200, driver 580.126.20, CUDA 13.0, torch 2.13.0+cu130. Runner: [run.sh](run.sh). Qwen-Image weights from a full local checkout; the tensor SHA-256 values in `results/report.json` equal those of the earlier evidence (only the extension hash differs, being a fresh build).

Why: the existing evidence names kernel commits that no longer exist after the history rewrite; the source hashes match this head, but nothing was run at this SHA.

| Run | passed | failed | skipped |
|---|---:|---:|---:|
| tests (the PR's six-path command), 2 GPUs visible (`results-round1-wrong-model-path/pytest.log`) | 101 | 0 | 0 |
| fixed-mode acceptance (`results/fixed/acceptance.json`): `qualified: true`, `commit` = head, `clean_commit: true` | 1808 checks | 0 | - |

Notes:
- The first round's evidence / audit commands failed because my script pointed at an HF cache snapshot without the transformer weights (`results-round1-wrong-model-path/`); the second round fixed only the model path (runner commit `2fc84fef`-era). Pytest was not repeated.
- `adaln_evidence.py` at this head has no `--fp64-check` (it exists on #538); the `report.json` here is without the FP64 diagnostic. The reproduction block in the older evidence README uses the #538 flag.
- Reuse survey modes needing vLLM/SGLang/Megatron sources were not rerun.

Draft addition for the PR body Results:

> **Re-run at head `a73c66c5` (B200, torch 2.13).** Tests: 101 passed, 0 failed, 0 skipped on two GPUs. Fixed-mode acceptance from a clean checkout of this commit: qualified, 1808 checks, 0 failures; tensor hashes of the real weights are identical to the earlier evidence. Raw data: `pr_assets` `pr537/b200-gaps/` @ `<EVIDENCE_SHA>`.
