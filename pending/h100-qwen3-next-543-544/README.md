# H100 evidence: #543 / #544 chain (2026-10-29)

Desensitized Run A (public heads) and Run B (#543 head + two local candidate commits, **not** a PR head).

Layout:

```text
pending/h100-qwen3-next-543-544/
  report-543-544.md
  operators/          # Run B operators step (summary.json, collection errors, full step log)
  logs/
  env-runA-543.txt, env-runA-544.txt, env-runB-patched.txt
  SHA256SUMS
  evidence_check.txt  # evidence_check.sh CLEAN
  HANDOFF-543-544-CHAIN.md
```

Run B operators failure: pytest collection requested unparametrized `match_the_fold` nodeid; see `operators/operators_collection_pytest_errors.txt`.
