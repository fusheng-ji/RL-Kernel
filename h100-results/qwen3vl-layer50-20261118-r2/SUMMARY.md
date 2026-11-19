# qwen3vl_layer50_contract r2 on H100 (2026-11-19 handoff, head `b4dd2962`)

Result: **as expected.** All six required verdict entries are true, the full model is freed before the 50-layer
stack loads, the peak stays well below the card size, and pytest passes. The new release check did not fire.
Nothing in the handoff branch was changed; `git diff --stat c07aa5dd HEAD -- . ':!h100-handoff'` printed nothing.

| Check | Expected | Observed |
|---|---|---|
| `contract_equals_diffusers_all` | true | true |
| `contract_on_full_equals_diffusers_all` | true | true |
| `hidden_states_50_is_diffusers_all` | true | true |
| `off_by_one_distinguishable_all` | true | true |
| `contract_repeatable_all` | true | true |
| `contract_batch_composition_invariant` | true | true |
| `hf_padded_batch_bitwise_all` (informational) | n/a | false |
| `memory.allocated_after_full_model_free_gib` | below 1.0 | 0.03125 |
| `memory.peak_allocated_full_model_gib` | n/a | 64.08 |
| `nvidia-smi` peak `memory.used` | below 95830 MiB | 66487 MiB |
| pytest | 52 passed, no skips | 52 passed in 156 s |

Other memory fields: `full_model_param_gib` 62.13, `layer50_param_gib` 49.42.

## Notes

- GPU: NVIDIA H100, 95830 MiB; driver, torch 2.13.0+cu130 and transformers 5.17.0 are in `nvidia-smi.csv` and
  `pip-list-torch-transformers.txt`. Diffusers at `f53d552036a0d1bd5570782a39cd40cfabf112bc`, checkpoint
  `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe` (same as run 1).
- The evidence step ran twice with the same command. In the first run my memory sampler watched the wrong card
  (`nvidia-smi -i 0` while the job used the card with index 1), so its `peak-memory.txt` was meaningless. I reran
  with the sampler on the right card; this folder is that second run. The first run's `verdict` and `memory`
  blocks are identical to the second's, and its pytest also gave 52 passed.
- The sampler writes `timestamp, memory.used [MiB], memory.total [MiB]` every 500 ms (`nvidia-smi-memory.csv`);
  the peak is its maximum.
- Host names, paths and the job id are replaced with `{...}` placeholders; hashes are untouched.
