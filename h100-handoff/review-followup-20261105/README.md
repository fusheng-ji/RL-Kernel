# H100 handoff: reruns after the 2026-11-05 review fixes (RFC #420, MiniMax-H3)

Hardware: the 2 nodes x 4 H100 (sm90). Every item fits one node. Items 2 and 3 need one GPU each and can run in parallel.
Paths are placeholders (`<REPO>`, `<CKPT>`, `<WEIGHTS>`, `<PROVIDER>`, `<OVERLAY>`, `<VLLM_OMNI>`, `<VERL_OMNI>`, `<OUT>`).

**Why H100.** Since #571 (merged), strict mode binds `proj_in` / `proj_out` to the video projection rows of #646
(`video_input_projection_fp32` / `video_output_projection_fp32`). Those rows accept only cc == (9, 0). On B200 every strict
full-model run stops with `ValueError: video input projection requires CUDA SM90`. That was checked on 2026-11-05 on B200:
all three jobs hit it, even before this round's changes. So the model-level evidence for the current code can only come
from Hopper.

## Code: one bundle, three branches

`review-followup.bundle` (in this folder) holds three branches that are not pushed yet. It needs three public commits:
`cfec0d98` (`test-h3`), `b1da8b64` (#569 head on the fork) and `d4e6549a` (#575 head on the fork).

```bash
cd <REPO>                                # a clone of RL-Align/RL-Kernel with the fusheng-ji fork as a remote
git fetch upstream test-h3 && git fetch fork feat/h3-ws2-parallel feat/h3-checkpoint-mapper-vllm
git bundle verify <THIS_DIR>/review-followup.bundle
git fetch <THIS_DIR>/review-followup.bundle 'refs/heads/h3/*:refs/heads/h3/*'
git worktree add --detach <SRC>/followup h3/review-followup    # 0264a4f8: test-h3 + the six follow-up fixes
git worktree add --detach <SRC>/pr569    h3/pr569              # 7316a5cb: #569 + its fixes, merged with test-h3
git worktree add --detach <SRC>/pr575    h3/pr575              # 39cb7d3d: #575 + its fixes, merged with test-h3
```

Check the SHAs after checkout: `followup` 0264a4f8, `pr569` 7316a5cb, `pr575` 39cb7d3d.

## Environment (same as the earlier H100 sections)

- torch 2.13.0+cu130, CUDA 13.0, pinned `MiniMax-H3@42ed227` checkpoint (`<CKPT>/transformer`, 14 shards). The extracted
  `<WEIGHTS>` directory (`h3_conditioning.safetensors`, `h3_block0.safetensors`, `hf/transformer` -> the 14 shards) is
  what `RL_KERNEL_H3_WEIGHTS` points at.
- Provider: diffusers at `f53d552036a0d1bd5570782a39cd40cfabf112bc`, clean (`<PROVIDER>`). The runners check both.
- Optional, pinned as before: vLLM-Omni `12e92800`, verl-omni `efb9dc7f`.
- Build the extensions **on the H100** in each worktree with `RL_KERNEL_H3_STRICT=1` (so `_H3_C` / `h3_full_attention` is
  there), and check that `rl_engine._C.__file__` and `rl_engine._H3_C.__file__` point into that worktree.
  `scripts/run_h100.sh` does all of this.

## Items

| # | File | GPUs | What it settles |
|---|---|---|---|
| 1 | `01-tests-and-cp-hierarchical.md` | 4 (one node) | real-weight H3 suite at the three heads; first GPU run of the hierarchical collective with CP > 1 (the OCR fix in #569) |
| 2 | `02-full-transformer-and-dual-trajectory.md` | 1 + 1 | 33B strict vs provider with the fused QKV backward and the bound video projections; RFC H-C2 (rollout vs train) at 0/16 |
| 3 | `03-full-speed.md` | 1 | strict vs provider step time with the setup moved out of the timed region (#575) |

## Rules (unchanged)

- No push, no PR and no comment on GitHub until Wenbo says so. Evidence goes to `pr_assets` only after "push evidence".
- Never change a test bound or a runner to make a result pass. A failure is reported as a failure, with its log.
- Anything that may reach a PR is redacted: no cluster, job, node, partition, user or personal paths (hosts become `node-a`, `node-b`).
- Commits (if any) use `-s` and carry no AI attribution lines.

## What to send back

One folder per item, with:
- `commit.txt` (the tested SHA, clean tree), `environment.txt` (`nvidia-smi`, torch/CUDA/triton, `_C` / `_H3_C` paths,
  provider commit), `pip-freeze.txt`;
- the raw outputs (`*.json`, junit `*.xml`, logs);
- a short `SUMMARY.md` with the numbers asked for in the item file and any deviation from the expectation.
