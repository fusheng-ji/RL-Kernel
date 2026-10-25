#!/bin/bash
# Qwen-Image AdaLN PRs (#537, #538) on two B200: test suite and the fixed-mode acceptance chain at the PR head.
# Needs GAPS, WT, OUT, MODEL. The extension in WT was built with KERNEL_ALIGN_USE_FAST_MATH=0 TORCH_CUDA_ARCH_LIST=10.0.
source "$(dirname "$0")/../../b200-gaps-common/common.sh"
env_header
# Full Qwen-Image checkout (the HF cache snapshot only holds configs); the revision is checked through the tensor hashes in report.json.
MODEL=${MODEL:?set MODEL to a Qwen-Image checkout}
Q=benchmarks/models/qwenimage
mkdir -p "$OUT/fixed"
[ -n "${SKIP_PYTEST:-}" ] || CUDA_VISIBLE_DEVICES=0,1 run pytest $PY -m pytest -p no:cacheprovider tests/models/qwenimage tests/runtime/test_kernel_registry.py \
  tests/contracts/test_tolerance_contract.py tests/build/test_build_platform_collectives.py \
  tests/validation/operators/test_ws1_gtest_gpu.py tests/runtime/test_layout_compatibility.py -q -rs
export CUDA_VISIBLE_DEVICES=0
FP64=; grep -q fp64-check $Q/adaln_evidence.py && FP64=--fp64-check
run evidence $PY $Q/adaln_evidence.py --model "$MODEL" --output "$OUT/report.json" --repeats 10 $FP64
run prior-art $PY $Q/adaln_prior_art.py --model "$MODEL" --output "$OUT/prior-art.json"
for seed in 0 127 2026; do for phase in accuracy accuracy_large; do
  run fixed-$phase-$seed $PY $Q/adaln_reuse_bi_audit.py --mode fixed --phase $phase --seed $seed --model "$MODEL" --output "$OUT/fixed/$phase-$seed.json"
done; done
for phase in bi workload parameters; do
  run fixed-$phase $PY $Q/adaln_reuse_bi_audit.py --mode fixed --phase $phase --model "$MODEL" --output "$OUT/fixed/$phase-0.json"
done
run cancellation-screen $PY $Q/adaln_parameter_screen.py --output "$OUT/fixed/cancellation-screen.json"
run validate $PY $Q/validate_adaln_repair.py "$OUT/fixed"
finish
