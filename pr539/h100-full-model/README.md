# H3 full 50-block model on Hopper (H100, sm90): numerical qualification

One packed FL2VA sample (about 12 tokens per block, 56 heads, head_dim 128) through the real MiniMax-H3 transformer
(50 blocks, pinned checkpoint revision `42ed227ee7df40d41602854ae760620d6eb651fe`, diffusers provider commit
`f53d552036a0d1bd5570782a39cd40cfabf112bc`), forward and full backward, 638 parameter gradients per run.
Candidate source commit: `58e08219a506fb55b7242e52a95b09778b5bb3b7`. NVIDIA H100 80GB, torch 2.13.0+cu130.
All reference and provider numbers below were measured on H100; nothing is carried over from B200.

## Result

| Check | Result |
|---|---|
| Block-0 golden gates (no skips) | 9 passed, 0 skipped (`block0.xml`) |
| FP32 reference, BF16 provider (default, sdpa-flash, sdpa-efficient, cuBLASLt) | complete, H100 bounds frozen (`full-model-thresholds.json`) |
| Ensemble rule calibrated on H100 | k = 4.0 (B200 rule: 2.6); leave-one-out worst ratios 0.99 / 0.99 / 2.77 / 3.94 (`full-model-ensemble-rule.json`) |
| Candidate execution | complete: forward 2.2 s, backward 379 s, peak 67 GB |
| Candidate vs single default-provider bound (1.5x provider error + floor) | **fails**: 347 of the checked tensors exceed it, first drift at `nodes.block.33` (`candidate-compare.json`) |
| Candidate vs ensemble rule (relative L2 <= k x max over providers + floor) | **passes**, 0 failures, worst ratio 0.577 (`candidate-ensemble-check.json`) |
| Output error vs FP32 | video 0.0063, audio 0.0198 (default provider: 0.0064 / 0.0214) |
| Generation preflight (128x128, 1 step, seed 42) | completed, `generation_complete_comparison_pending`; this is a flow check, not the three-seed generation evidence (`preflight/`) |

Acceptance for this stage is the ensemble rule. The single-provider bound is reported for transparency; see below for why it is not the criterion.

## Why the single-provider bound fails

* Outputs and input gradients of the candidate are as good as or better than the default provider; only the parameter gradients of the attention weights in the 50 main blocks are about 1.65x higher (relative L2 about 0.5 vs 0.3). FFN, AdaLN and token-refiner gradients are on par or better.
* The error baseline is set in the backward of the last block. The gradient error w.r.t. the block output is 0.021 for both candidate and provider at block 49 and jumps to 0.39 (provider) / 0.78 (candidate) at block 48; the ratio then stays between 1.6 and 2.0 for every earlier block (`layer_curves.png`).
* Block 49's attention is ill-conditioned: maximum logit about 3189 and mean softmax entropy 0.57, versus 14 to 536 and 1.0 to 1.5 in other blocks. A bf16-sized (2^-9) perturbation of q/k/v/dO changes block 49's dk by 0.16 (median) and up to 0.65 (95th percentile), versus 0.003 to 0.005 elsewhere.
* With identical inputs the candidate attention is more accurate than SDPA: block 49 out/dq/dk/dv error 0.0014/0.0010/0.0016/0.0016 versus flash 0.0015/0.0092/0.0043/0.0018. The candidate kernels are not the source of the spread.
* The other BF16 provider variants spread widely at the same place (block-48 gradient error: cuBLASLt 0.39, efficient 0.45, flash 0.68); the candidate (0.78) sits at the high end of that distribution, inside the ensemble bound.
* Using fp32 q/k only inside block 49 was evaluated and gives no expected benefit (the 0.03 error of q/k is inherited from the earlier blocks; single-step rounding is 0.0017), so no kernel or numerical-contract change was made.

## Files

* `full-model-thresholds.json`, `full-model-ensemble-rule.json`, `reference-manifest-h100.json`: frozen H100 bounds, rule, and FP32 reference inventory.
* `cmp-*.json`: each provider variant vs the FP32 reference (inputs to the ensemble calibration).
* `candidate-compare.json`, `candidate-ensemble-check.json`: candidate evaluation.
* `reports/*.report.json`: run reports; local paths and the scheduler job id are redacted, so their file hashes differ from the hashes referenced inside the comparison JSONs.
* `preflight/`: generation preflight report, trace and three frames.
