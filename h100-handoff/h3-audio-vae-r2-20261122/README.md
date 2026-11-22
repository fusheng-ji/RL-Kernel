# H100 handoff: `audio_vae_contract` (#722) at `25d95ccd` (2026-11-22)

**Why.** The H100 evidence in #722's description was measured on an earlier benchmark (`a1177ddd`, results `1264ddc1`). Since then the review rounds changed what the benchmark measures and how the module restores precision state, so that section is marked superseded. This run replaces it.

What changed since the last H100 run:
- **vLLM-Omni encode** runs inside `torch.jit.optimized_execution(False)`, as vLLM-Omni `aae3f386` does. The old H100 number "first encode call differs" for vLLM-Omni came from the benchmark.
- **Cold start** runs one fresh child process per path and direction. Its input now lives in a per-run temporary directory next to `--out`, which is removed afterwards.
- **Invariance and parity** read the final batch slot (`batch8_last`) and compare sha256, i.e. bytes. The FP64 reference stays FP64.
- **The module** disables autocast inside `execution_scope` and restores every backend's own `fp32_precision` after it. It also refuses a decoder whose upsampling is not exact; the released model passes.
- **B200** at `53b944ed` gives the same strict bytes as before; `25d95ccd` changes nothing on the measured paths.

**Code.** Run from a detached, clean checkout of `25d95ccd`, which is #722's head and is on the fork as `feat/h3-audio-vae-contract`. Do not run from this handoff commit: the report records the source commit and must say `25d95ccd`.

## Inputs (as in the last H100 runs)

- `<E1>`: the repo environment of the last rounds (torch 2.13.0+cu130; `results/pip-list-filtered-E1.txt` on `handoff/h100-h3-gemm-rows-results-20261122`). matplotlib is needed for `--plot`; without it the two chart tests skip.
- `<AUDIO_VAE>`: the released `FL2VA/audio_vae/` (Hub `MiniMaxAI/MiniMax-H3` @ `42ed227e`; the code checks `model.safetensors`' sha256). The last round's `<CKPT>/FL2VA/audio_vae` is fine.
- `<PROVIDER>`: Diffusers at `f53d5520`, the same `src/` tree as before.
- No SGLang: leave `--sglang` out. The SGLang rows stay B200-only.

## Run (one H100, about 45 min; the cold start starts 6 child processes per run)

```bash
git fetch https://github.com/fusheng-ji/RL-Kernel.git feat/h3-audio-vae-contract
git worktree add --detach <W> 25d95ccd
cd <W>
<E1>/bin/python setup.py build_ext --inplace          # as in the last rounds; the audio VAE itself uses no _C kernel
git rev-parse --short=8 HEAD                        # 25d95ccd
git status --porcelain --untracked-files=no | wc -l  # 0
export RL_KERNEL_H3_AUDIO_VAE=<AUDIO_VAE>
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W>:<PROVIDER>/src <E1>/bin/python -m pytest -q -rs -p no:cacheprovider \
  tests/models/minimax_h3/test_h3_audio_vae.py --junitxml=<OUT>/pytest-audio-vae-h100.xml | tee <OUT>/pytest-h100.txt
A="--checkpoint $RL_KERNEL_H3_AUDIO_VAE --diffusers-src <PROVIDER>/src"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W> <E1>/bin/python benchmarks/models/benchmark_h3_audio_vae.py $A --repeats 10 --plot \
  --out <OUT>/h3-audio-vae-h100.json | tee <OUT>/run-a.txt
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W> <E1>/bin/python benchmarks/models/benchmark_h3_audio_vae.py $A --repeats 0 \
  --out <OUT>/h3-audio-vae-h100-rerun.json | tee <OUT>/run-b.txt
PYTHONPATH=<W> <E1>/bin/python benchmarks/models/benchmark_h3_audio_vae.py --out <OUT>/h3-audio-vae-h100.json --markdown \
  > <OUT>/tables-h100.md
git status --porcelain --untracked-files=no | wc -l  # still 0
```

The two benchmark runs must not run at the same time on the same GPU. Run them on an idle GPU, because the latency is reported.

## Expected

1. **pytest:** 58 passed. With matplotlib, the 2 chart tests also run (60 passed); without it they skip. Nothing fails. (B200 at `25d95ccd`, without matplotlib: 58 passed, 2 skipped.)
2. **`h3-audio-vae-h100.json`:**
   - `schema` is `rlkernel.h3.audio_vae_bench.v2`;
   - `provenance.source` is `25d95ccd…` with `clean: true`;
   - `checkpoint.remote_model_safetensors_sha256` equals `model_safetensors_sha256`.
3. **`ours`, every case:** `encode` / `decode` `batch_invariant: true`, every `bitwise_equal: true`, every `max_abs` 0, and `cold_start.ours` both `*_bitwise: true`.
4. **`cold_start["vllm-omni"].encode_first_vs_third_bitwise`:** B200 gives true. This value replaces the old H100 vLLM-Omni encode cold-start number (5.8e-06).
5. **Across the two processes:** `cases[*].rows_sha256` and `paths.*.{encode,decode}.sha256` are equal between the two JSONs.
6. **No leftovers:** no `h3-audio-vae-cold-start-*` directory is left in `<OUT>` after either run.
7. **Not expected:** H100 hashes equal to the B200 hashes. The contract claims reproducibility per platform.

Report every deviation as it is, with the log. Do not edit code, tests or thresholds.

## Send back

Push one branch, `handoff/h100-h3-audio-vae-r2-results-20261122`, based on this handoff commit, holding `results/`:
- `h3-audio-vae-h100.json`, `h3-audio-vae-h100-rerun.json`;
- the four charts `h3-audio-vae-h100-{bi,accuracy,coldstart,latency}.png`;
- `tables-h100.md`, `pytest-h100.txt`, `pytest-audio-vae-h100.xml`, `run-a.txt`, `run-b.txt`;
- `commit.txt` (HEAD and tree), `env.txt`, `nvidia-smi*.txt`, the tree status before and after, `c-file.txt`, `pip-list-filtered-E1.txt`;
- `SUMMARY.md`: one verdict per item of "Expected", and every deviation.

Redact before committing: cluster, job, node, partition, user name, conda environment name, `/home/…` and `/data/…` paths, e-mail addresses; write host names as `node-a`. Add `*.log` with `git add -f`. Commit with `git commit -s` and no AI attribution line. No PR, no comment.
