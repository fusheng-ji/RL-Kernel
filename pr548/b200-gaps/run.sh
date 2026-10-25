#!/bin/bash
# PR #548 on two B200: the unit tests of the TP PPO stack at the PR head, in the shared torch 2.13 environment
# (no Megatron Core / DeepSpeed overlay, no vLLM sidecar environment). Needs GAPS, WT, OUT.
# The Qwen3-8B fixed-rollout candidate stage (qualify_candidate.sbatch) is NOT run here; see CANDIDATE-STAGE-PLAN.md.
source "$(dirname "$0")/../../b200-gaps-common/common.sh"
env_header
export CUDA_VISIBLE_DEVICES=0,1
run pytest-ppo-unit $PY -m pytest -p no:cacheprovider -q -rs \
  tests/integrations/common/test_ppo.py tests/integrations/common/test_tp_scalar.py \
  tests/integrations/engines/train/test_ppo_adapters.py tests/integrations/engines/train/test_ppo_bf16.py \
  tests/integrations/engines/train/test_qwen3_ppo_checkpoint.py tests/integrations/engines/train/test_qwen3_vllm_install.py
finish
