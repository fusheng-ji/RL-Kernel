# H100 evidence: Qwen3-Next norm PR #468

PR #468, fork branch `feat/cuda-qwen3-next-gated-rmsnorm`, pinned head `5aa9aa1c` (sm90, labelled H100 only).

## Commands

Unit (unchanged from prior bundle): `pytest tests/models/qwen3_next … -q -p no:randomly -rfs` @ `1e2eaec0` counts in `tests.json`.

Norm + reuse (re-run 2026-10-26 @ `5aa9aa1c`, idle GPU):

- `tools/validation/models/qwen3_next_norm_evidence.py --out report.json`
- `tools/validation/models/qwen3_next_norm_reuse_check.py --op qwen3_next_rms_norm`
- `tools/validation/models/qwen3_next_norm_reuse_check.py --op rms_norm_gated`

See `tests.json`, `logs/`, `report.json`, figures, reuse JSON/PNG.
