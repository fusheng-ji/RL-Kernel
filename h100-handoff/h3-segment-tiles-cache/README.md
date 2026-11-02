# H100 handoff index (2026-11-02, narrowed to 2 nodes x 4 H100)

Hardware available: two nodes with 4 H100 each, no single 8-GPU node. Only experiments that fit one
node (1 to 4 GPUs) are kept. Each file is self-contained (commands, expected result, what to send back).
Nothing here is submitted anywhere; paths are placeholders (`<REPO>`, `<CKPT>`, `<WEIGHTS>`, `<OUT>`).
Do not put cluster, job, node or personal paths into anything that goes into a PR.

| # | File | Needs | Size | Status |
|---|---|---|---|---|
| 1 | `01-segment-tiles-cache.md` | 1 GPU (tests, timings); 1 to 2 H100 + checkpoint for the real-model count | hours | **run** |
| 2 | `02-generation-gate-h100.md` | 1 H100, steps A and B only (resume check, preflight with memory/time readout) | ~5 h | **run A and B**; C (gate + 3 x ~20 h generation) is out of scope |
| 3 | `03-single-node-8gpu-hopper.md` | one node with 8 H100 | - | **not possible**; kept for whoever has such a node |

Why 3 is dropped rather than approximated: the in-node 8-rank TP/CP/SP layouts of #569 were already
measured across two nodes x 4 H100 (published in that PR), and the deterministic all-gather of #491 and
the CUDA-IPC fixed tree are single-node mechanisms, so a two-node run is a different code path, not a
substitute. #575 at tp = 8 has no runner at all.

Checked and **not** H100 work: B200 sections of #567-#575 now exist in the PR bodies; ROCm rows are
excluded; the VeRL-Omni comparison is parked; `KERNEL_ALIGN_USE_FAST_MATH=1` builds are not device specific.

Files in this directory: `0001-segment-tiles-cache.patch` (git am-able, applies on `upstream/test-h3`
at `a88b5a09` and on the #588 head `67365734`; commit `6bd3fe74`), `scripts/`.
