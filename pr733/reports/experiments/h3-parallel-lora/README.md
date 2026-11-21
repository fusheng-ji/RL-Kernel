# H3 LoRA on the TP x CP grid (`h3-parallel-lora-v1`): evidence

This is the measurement of branch `feat/h3-parallel-lora` at `681a3a7a`, stacked on `feat/h3-sharded-model` `8132b974` (`commits.txt`). It ran on one node with 8 x B200 (`gpus.txt`), allocated to the job alone, in the repo environment: torch 2.13.0+cu130 and the H3 provider pin, Diffusers 0.40.0.dev0 (`env.txt`). Host names and local paths are replaced by `node0`, `<env>`, `<head>` and `<base>`.

| Path | What |
|---|---|
| `grids/<run>/report.json`, `run.log` | one run of `tools/validation/models/h3_parallel_full_model.py --checkpoint <MiniMax-H3> --attention h3_full --provider-boundaries --sharded --lora-rank 64 --lora-alpha 128 --lora-seed 7`, with the flags below |
| `suite/suite-head.txt`, `suite-base.txt` | `pytest -q tests/models/minimax_h3`, head on GPUs 0-3 and base (`8132b974`) on GPUs 4-7, at the same time |
| `suite/failed-*.txt` | the failing test ids of each side |
| `accuracy/` | `lora_vs_strict_probe.py`: per target, the canonical (`Grid(1, 1)`) LoRA projection against #574's `strict_lora_linear` and FP64, on the released block-0 weights, one B200; `plot_prc.py` draws `prc-lora-accuracy.png` from `report.json` |
| `prior-art/` | `tp_lora_prior_art_probe.py`: the cross-rank sums of peft's `LoraParallelLinear` / vLLM's LoRA layers (partial products summed by an all-reduce) against TP = 1; a local B200 shared with other jobs (a byte comparison, not a timing) |

| Run | Flags |
|---|---|
| `lora_random_tp2_40` / `lora_zero_tp2_40` | `--tp 2 --size 40` |
| `lora_*_tp4_sp_928` | `--tp 4 --sp --size 928` |
| `lora_*_tp4_cp2_ring_925` | `--tp 4 --cp 2 --cp-mode ring --size 925` |
| `lora_*_tp4_cp2_sp_ring_928` | `--tp 4 --cp 2 --sp --cp-mode ring --size 928` |
| `lora_*_tp4_cp2_sp_ulysses_928` | `--tp 4 --cp 2 --sp --cp-mode ulysses --size 928` |
| `lora_*_tp8_sp_928` | `--tp 8 --sp --size 928` |
| `inject_lora_tp4_sp_928` | `--tp 4 --sp --size 928 --inject 17:700:2` (negative control, B random) |
| `inject_lora_tp4_cp2_ring_925` | `--tp 4 --cp 2 --cp-mode ring --size 925 --inject 17:5:3` (negative control, B random) |

`lora_random_*` draw B from N(0, 0.02); `lora_zero_*` add `--lora-b-zero` (the recipe's initial B).

Reading a `report.json`:
- `all_bitwise` and each rank's `checks` hold the result; `lora_gradients_bitwise` covers every LoRA shard's gradient;
- `lora_shards_compared` is the LoRA gradient leaves of the rank (50 blocks x 13), `lora_shards_mismatched` lists at most the first 10;
- `weight_shards_compared` counts every gradient leaf at this head (350 base shards + 650 LoRA leaves = 1000); `f534f2a5` makes it the base shards only, as without LoRA. `weight_shards_mismatched` lists base shards only, at most the first 10;
- `lora_bytes` is the LoRA tensors the rank holds, for the canonical model and for its grid;
- `first_drift_forward` / `first_drift_backward` give the first differing block, row and owner;
- `sharded.parallel_peak_bytes` is the peak allocation during the grid run.
