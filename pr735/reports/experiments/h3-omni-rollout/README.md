# H3 rollout on the grid (`h3-omni-rollout-v1`): evidence

These files measure branch `feat/h3-omni-rollout-model`, which is stacked on `feat/h3-parallel-lora` `f534f2a5`. Each run's commits are in its `commits.txt`.

The TP4 runs used the vLLM-Omni environment of the final run: torch 2.14.0+cu130, Diffusers 0.43.0, vLLM 0.33.0 and vLLM-Omni `6039984e` (`env.txt`). RL-Kernel builds on torch 2.14 only with the C++20 build fix, so each run's checkout is this branch plus that fix; the fix touches only `build_tools/extensions.py`. Each run had one node with 8 x B200 to itself (`gpus.txt`). vLLM-Omni ran the released checkpoint's FL2VA partition at TP4 on GPUs 0-3, and the canonical replay ran on GPU 4. The tool's defaults for `--tp`, `--max-num-seqs`, `--steps` and the request size are the values these runs used (see "Requests" below).

Host names and local paths are replaced by `<env>` (the torch-2.14 environment), `<repo-env>` (the repo environment), `<head>` / `<base>` (the checkouts), `<MiniMax-H3>` (the checkpoint), `<home>` and `<tmp>`.

| Path | What |
|---|---|
| `tp4-lora-random/` | `tools/validation/models/h3_omni_rollout_report.py engine --checkpoint <MiniMax-H3> --engine-model <FL2VA partition> --tp 4 --max-num-seqs 2 --lora-rank 64` (alpha 128 and seed 7 are the defaults) on GPUs 0-3, then `canonical` on GPU 4 and `actor` without a GPU; B drawn from N(0, 0.02) |
| `tp4-lora-zero/` | the same with `--lora-b-zero` (the recipe's initial B) |
| `tp4-native/` | `native --compare-calls tp4-lora-zero/calls` on GPUs 0-3: vLLM-Omni's own DiT at TP4 (no binding, no LoRA), compared with the binding's recorded calls of `tp4-lora-zero/` |
| `suite/suite-head.txt`, `suite-base.txt` | `pytest -q tests/models/minimax_h3`, run at the same time: head `8ecfed97` on GPUs 0-3 and base `f534f2a5` on GPUs 4-7. Both in the repo environment (torch 2.13, Diffusers 0.40.0.dev0) |
| `suite/failed-*.txt` | the failing test ids of each side |
| `suite/omni-cpu.txt`, `omni-gpu.txt` | `tests/models/minimax_h3/test_h3_omni_model.py` alone, first without a GPU (as in CI), then on one B200 |

Requests: T2VA at 448 x 256, 4 s, 2 steps, `flow_shift` 12 and `audio_flow_shift` 3. A and B are two prompts with seeds 11 and 22. The engine stage runs A alone, B alone, then A and B together. With `max_num_seqs=2`, step mode puts A and B into one batched DiT call.

Reading the reports:
- `engine.json`:
  - `ranks_agree`: every TP rank returned the same output digest for every DiT call (`calls_per_rank`).
  - `batched_calls_with_an_alone_twin`: how many calls of the batched A+B request equal, byte for byte, the call with the same inputs from A or B alone.
  - `install[r]` holds rank r's report: `resident_param_bytes` is the bound model's parameters, `engine_parameter_bytes_released` is vLLM-Omni's own DiT freed before the bound model loads.
  - `lora` covers the sync check. The tool first attaches a decoy LoRA (seed 8). It then sends rank 2 shards with one element changed (`tampered`: block 17, `attn.to_v.A[3, 5]`, changed after the sender took its digests). `tampered_sync_refused` is every rank's refusal; `every_rank_kept_the_decoy` checks that no rank changed. Last, the true sync runs: `every_rank_has_the_senders_digests` covers the `shards_per_rank` shards (50 blocks x 13).
- `canonical.json`: each recorded call (`calls/call<n>.pt`, rank 0's inputs and outputs) replayed on `Grid(1, 1)` on one GPU, with the same LoRA. `video_bitwise` / `audio_bitwise` compare with the rollout's output. `in_rollout_digests` checks that the replay's digest is the one every TP rank reported.
- `actor.json`: for each recorded call, RL-Kernel's converter against verl-omni's `prepare_model_inputs` on the same request; `differ` lists every field that is not bitwise equal.
- `native.json` (reported, not a gate): vLLM-Omni's own DiT at TP4 runs request A (seed 11). TP rank 0 saves each DiT call's inputs and outputs to `native-calls/call<n>.pt`, and each is compared with the binding's call of the same index in `tp4-lora-zero/calls/`. `same_inputs` checks `x`, `audio_x`, `img_position_ids`, `unique_timesteps` and `inverse_indices`; only `call0` has the same inputs, because `call1`'s latent comes from each engine's own step 0. `elements_differing`, `max_abs` and `rel_l2` (the L2 norm of the difference over the L2 norm of the binding's output) are given for video and audio.
- `summary.txt`: each stage's exit code and one line of its report.
