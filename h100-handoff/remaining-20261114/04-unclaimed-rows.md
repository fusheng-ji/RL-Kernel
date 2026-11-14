# 04. RFC #420 rows nobody has claimed (no ROCm, no dependency on someone else's row)

These rows have an empty owner column in the RFC #420 table (body last edited 2026-10-30), and no PR on `test-h3` covers
them. Each one needs a claim comment on #420 first. Wenbo posts it, and the comment names only #420 PRs. Out of scope:
every ROCm column, `final_e2e_diffusers_rocm`, `rocm_perf_closeout`.

## Can start now

| Row | Goal (RFC text) | Start from | Done when |
|---|---|---|---|
| `stochastic_transition_contract` (Phase C) | "Declare SDE, noise law and exact transition logprob before implementing RL logp" | `rl_engine/models/minimax_h3/transition.py` (`h3-cps-transition-v1`, #574) already implements verl-omni's CPS step, bitwise equal to `FlowMatchSDEDiscreteScheduler.sample_previous_step`. The noise-law question in #574 is still unanswered | a contract doc (`docs/operators/`): SDE type, noise law (`noise_level`, SDE window, the generator per step vs per sample), log-prob formula and reduction order, what "bitwise" means for log-prob. If the maintainers' answer differs from #574, a code change follows |
| `sglang_runtime_adapter_h3` (integration) | "Bind and read back actual strict H3 operators in pinned SGLang support. Verify upstream support first" | `rl_engine/integrations/engines/rollout/vllm/h3_omni.py` (`install_h3_rollout`, #575) is the vLLM-Omni pattern | step 1 is a written answer: does a pinned SGLang release serve MiniMax-H3, and with which module names? If not, the row is blocked, and the comment says so |
| `ws2_pipeline_parallel` (WS2 follow-up, optional) | "PP2/PP4 layer partition with exact activation handoff" | `rl_engine/validation/models/h3_parallel_model.py` (item 03) chains blocks on one grid | PP2 and PP4 over the 50 blocks, the handed-off activations and gradients bitwise equal to the single-GPU run, on 2 x 4 H100 |
| `qwen3vl_layer50_contract` (Phase B) | tokenizer, special tokens, multimodal processor, exact layer-50 hidden-state extraction | new model family. Check what the repo already has for Qwen3-VL before writing a golden | layer-50 hidden states bitwise repeatable and within the FP64 golden bound, at the pinned revision |
| `visual_vae_contract` (Phase B) | encode/decode, latent normalization, `f16t4d24` layout, tiling | `rl_engine/reference/packing/video_patch_pack_unpack.py` (#550) for the latent layout | encode/decode repeatable, tiled equals untiled where the contract says so |
| `audio_vae_contract` (Phase B) | stereo encode/decode, 40 Hz latent layout | — | same as above for audio |

## Need other rows first

| Row | Waits for |
|---|---|
| `fl2va_pipeline_contract`, `ref2va_pipeline_contract` | `h3_packed_layout_schema` (#666, BPbruce, open) |
| `phase_b_full_chain` | the five Phase B rows above |
| `final_e2e_diffusers_cuda` | the block rows that are still interim: `h3_attention_o_gemm` / `h3_ffn_gate_up_gemm` (#554), `h3_ffn_down_gemm` (#535), `h3_qk_rmsnorm_d128` and `h3_swiglu` (no PR yet), `h3_mm_rope_3axis_partial` (#656 draft). Attention is bound since `4f7c5d11` |
| `cuda_perf_closeout` | RFC §9: after the correctness closeout |

## Rows owned by others that Wenbo can review

- #554 / #535 (zzf2024, changes requested). Their O / down GEMMs reduce K in a single sweep. The row-parallel contract of
  the merged #569 is the 8-unit balanced tree, and "TP = 1 is that definition, not the sweep"
  (`docs/operators/h3-parallel.md`, "The canonical reduction"). RFC §4.8 requires TP / CP / SP to be byte-equal to WS1,
  so one of the two has to change before #554 / #535 can be bound in the block.
- #471 (changes requested), #476 (approved, not merged), #545, #666, and the drafts #656 / #663.

## RFC table (Wenbo decides)

- The nine rows that #569 implements have no owner: `tp_attention_7168`, `tp_ffn_14336`, `cp_full_attention`,
  `cp_metadata_coshard`, `cp_arbitrary_length`, `distributed_packed_io`, `ws2_tp2_tp4_tp8`, `ws2_tp4_cp2`,
  `ws2_tp4_cp2_sp`.
- The status column says `OPEN` for merged rows. Only `video_patch_pack_unpack` says `Finished`.
