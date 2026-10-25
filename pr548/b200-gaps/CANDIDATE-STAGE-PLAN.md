# PR #548: plan and cost of the Qwen3-8B candidate stage on B200 (not started)

Status: **not run.** The stage cannot start with what is on the shared filesystem today.

## What `qualify_candidate.sbatch` needs
| Input | Present now? |
|---|---|
| Clean checkout of the PR head | yes (a detached worktree at the head exists) |
| Frozen `acceptance.json` and six bounds files (`pr548-qwen3-fixed-reference/`) | yes, on `pr_assets` |
| Raw reference directory (rollout, targets, eight HF control reports and tensors; about 955 GiB, 12,866 pinned files); `verify_reference` and `compare_fixed` read it | **no** (not found anywhere under the user's directories) |
| Trainer environment: Python 3.11, PyTorch 2.9.1+cu130, transformers 5.13.1 | **no** (the venv named in the earlier notes is gone) |
| Framework overlay: Megatron Core 0.15.3, DeepSpeed 0.19.7, NumPy 1.26.4 (`requirements-frameworks.txt`) | **no** |
| Separate vLLM 0.15.1 environment (`requirements-vllm.txt`) | **no** (the shared env has vLLM 0.30.0 / torch 2.13, a different stack) |
| `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218` (hashes in `qwen3-8b-manifest.json`) | **no** (not in any HF cache found) |

## Cost if the missing inputs are rebuilt
- Compute: 4 x B200, exclusive, one node. The sbatch asks for 512 GB host memory and a 24 h limit. It runs 16 engine/precision/role/TP
  fixed-rollout checks (1-2 GPUs each), two four-GPU vLLM PPO topologies per framework and a tiny-GPT regression, twice (Megatron, DeepSpeed).
- Disk: the candidate writes per-parameter tensors for the same 1,600-tensor inventories as the reference; the reference was about 955 GiB,
  so budget roughly the same again on a shared filesystem that is at 98% use (about 16 TB free).
- The raw reference cannot be recovered from the committed manifest. Re-running `qualify_reference.sbatch` would sample a new vLLM rollout
  (the preempted third pass of the original run already differed from the fourth), so its bounds would differ from the frozen ones. That would
  be a new reference needing its own freeze commit and an amendment to the PR, which is a decision for the PR author, not a gap-filling rerun.

## What was run instead
Only the unit tests of the stack at the head (`run.sh` here), clearly labelled as such. No Qwen3 PPO candidate evidence is claimed.
