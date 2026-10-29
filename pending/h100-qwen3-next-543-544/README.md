# H100 evidence: #543 / #544 chain (2026-10-29, HANDOFF §7)

Desensitized Run A (public heads) and Run B (#543 head + **three** local candidate commits, tree `46c64a9b…`, **not** a PR head).

Layout:

```text
pending/h100-qwen3-next-543-544/
  report-543-544.md
  operators/          # Run B §7 preflight operators (passed after patch 0003)
  logs/               # incl. timeout reruns, runB3 chain (layers pass, l2 fail)
  env-runA-543.txt, env-runA-544.txt, env-runB-three-patches.txt
  patch_SHA256SUMS.txt  # three patches
  SHA256SUMS
  evidence_check.txt
  HANDOFF-543-544-CHAIN.md
```

Run B: preflight + layers **passed**; L2 failed at Megatron `load_checkpoint(..., skip_load_to_model_and_opt)` (see report + `operators/l2_failure_excerpt.txt`).
