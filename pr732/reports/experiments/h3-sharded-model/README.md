# H3 weight-sharded full model: evidence

This is the measurement of branch `feat/h3-sharded-model` at head `8132b974`, on `test-h3` `bb2a3ee1` (`commits.txt`). It ran on one node with 8 x B200 (`gpus.txt`), in the repo environment: torch 2.13.0+cu130 and the H3 provider pin, Diffusers 0.40.0.dev0 (`env.txt`). Host names and local paths are replaced by `node0`, `<env>`, `<head>` and `<base>`.

| Path | What |
|---|---|
| `grids/<run>/report.json`, `run.log` | one run of `tools/validation/models/h3_parallel_full_model.py --checkpoint <MiniMax-H3> --attention h3_full --provider-boundaries`, with the flags below |
| `suite/suite-head.txt`, `suite-base.txt` | `pytest -q tests/models/minimax_h3`, head on GPUs 0,1 and base on GPUs 2,3, at the same time |
| `suite/failed-*.txt` | the failing test ids of each side |
| `diffusers-local-shard/` | `cut_shard` against Diffusers 0.43.0's `_local_shard` under `MiniMaxH3Transformer3DModel._tp_plan`, on the released checkpoint, CPU, torch 2.14.0 |

| Run | Flags |
|---|---|
| `harness_tp4_sp_928` | `--tp 4 --sp --size 928` (#701 harness, full weights, control) |
| `sharded_tp2_40` | `--tp 2 --size 40 --sharded` |
| `sharded_tp4_sp_928` | `--tp 4 --sp --size 928 --sharded` |
| `sharded_tp4_cp2_ring_925` | `--tp 4 --cp 2 --cp-mode ring --size 925 --sharded` |
| `sharded_tp4_cp2_sp_ring_928` | `--tp 4 --cp 2 --sp --cp-mode ring --size 928 --sharded` |
| `sharded_tp4_cp2_sp_ulysses_928` | `--tp 4 --cp 2 --sp --cp-mode ulysses --size 928 --sharded` |
| `sharded_tp8_sp_928` | `--tp 8 --sp --size 928 --sharded` |
| `inject_sharded_tp4_sp_928` | `--tp 4 --sp --size 928 --sharded --inject 17:700:2` (negative control) |
| `inject_sharded_tp4_cp2_ring_925` | `--tp 4 --cp 2 --cp-mode ring --size 925 --sharded --inject 17:5:3` (negative control) |

Reading a `report.json`:
- `all_bitwise` and each rank's `checks` hold the result;
- `first_drift_forward` / `first_drift_backward` give the first differing block, row and owner;
- `sharded.resident_param_bytes` is the weights a rank holds (loaded parameters plus stored shards);
- `sharded.parallel_peak_bytes` is the peak allocation during the grid run;
- `weight_shards_mismatched` lists at most the first 10.
