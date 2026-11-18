# Pipeline contracts, item 1 only: identify both transformer partitions (2026-11-18)

- Code: `handoff/h3-h100-pipeline-contracts-20261118` @ `d83c5d58`, detached worktree, `git status --porcelain --untracked-files=no` empty before and after. No patch. No GPU used (CPU file reads only); the H100 node was used only as a host with fast storage access. Driver and GPU: see `nvidia-smi.txt`.
- Environment: torch 2.13.0+cu130, python 3.12.15 (the `rlk_h3` venv), `PYTHONPATH=<worktree>`.
- Weights: Hub `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`. `transformer/` came from an earlier `hf download` of that revision (14 shards, 62 GiB); `transformer_ref/` was downloaded with `hf download ... --include 'transformer_ref/*' --local-dir` for this run (14 shards, 62 GiB). It was deleted afterwards because the disk quota was full; it has to be fetched again for item 2.

## Results
| Directory | Hub metadata (`item1-hub-metadata.log`) | Full hash, all shards read (`item1-full-hash.log`) | Expected |
|---|---|---|---|
| `transformer` | `FL2VA` | `FL2VA` (64 s) | `FL2VA` |
| `transformer_ref` | `Ref2VA` | `Ref2VA` (66 s) | `Ref2VA` |

No `VariantMismatch`. As expected.

## Deviations from the handoff command (read this)
1. The handoff passes `<ckpt>/transformer` and `<ckpt>/transformer_ref` from one checkpoint root. Here the two directories live in two different `--local-dir` roots, so the command was run with two paths, one per root. The Hub metadata lookup walks the directory parents for `.cache/huggingface/download`, so each path was given with its real root.
2. With real `--local-dir` downloads, `identify_transformer_dir(..., full_hash=True)` still uses the Hub metadata when it exists, so that call alone does not read the shards. To actually hash the shards, the same call was repeated on a directory of symlinks to the two partitions (`full-hash(read all shards)` in the log): the parents of a symlinked directory carry no metadata, and the shard files are not symlinks, so the code falls through to `_file_sha256`. The 64 s and 66 s timings are the real read of about 62 GiB each (page cache was probably warm for `transformer_ref`, which had just been downloaded). No repository file was changed.
3. The first launch with the symlinked layout and no `full_hash` raised `ValueError: no Hub metadata or cache blob name` (the documented behaviour for a directory without metadata). It is not a result.

Items 2 and 3 of the handoff were not run: item 2 waits for the dependent rows, item 3 needs ROCm.
