# Evidence of PR #575 (`feat/h3-checkpoint-mapper-vllm`)

Measurement outputs, kept here instead of in the pull request.

- Hardware / software: NVIDIA H100 (93 GiB, sm90), CUDA 13.0, torch 2.13.0+cu130; pinned `MiniMax-H3@42ed227` checkpoint (real weights, 14 shards).
- Command: `tools/validation/models/h3_mapper_vllm_report.py --checkpoint <dir> --out reports/experiments/h3-mapper-vllm-h100/report.json --png .../figure.png`.
