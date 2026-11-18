# Update 2026-11-18 to the Flink-round H100 handoff

This adds to `../README.md`; it does not replace it. Rules 1-7 of the 2026-11-14 handoff still apply: clean detached tree, `rl_engine._C.__file__` checked before every GPU run, idle node, everything redacted before it goes back.

## A. Heads

`../README.md` ("State of the code", updated 2026-11-18) has the pushed heads and steps 1-4. This file adds only item B.

A second push (the local fixes in section C) may land before the run. If it does, use the new heads. Section C changes only the chain-gate script, the CI workflow, a test file name and the benchmark runners, not model code.

## B. New item: #507 attention prior art on H100, with the timing fix

**Why.** OpenCodeReview found two problems in #551's attention runner, `benchmarks/models/qwen3_next/attention_candidates.py`. The per-batch setup ran inside the timed region:
- FlashInfer's `plan()` ran on every call.
- The Triton 2D path built its paged KV cache, block table and `cu_seqlens` on every call.

The fix is local commit `73e5b5fb`: a `prepare(seqs) -> launch` split, so only the kernel launch is timed. The #507 B200 table is being re-measured with it. The H100 table in #507's description (2026-11-14, runner `58aa50c7`) has the same problem, so it needs the same rerun. Batch-invariance verdicts should not change. The FlashInfer and Triton 2D latencies should drop. The other rows should move by noise only.

**Code.**
- The bundle in this directory, `qwen3-next-ocr-20261118.bundle`, holds `fix/551-flink-r1` @ `2c0cd1e1`. That is #551's head `ea999fd4` plus the merged OCR fixes of the whole stack.
- Prerequisites: `ea999fd4` and `62487710`, both on the fork.
- The commits are local, not pushed. When they are pushed as a fast-forward, the SHA stays `2c0cd1e1`. Until then the result is held, not published.

```bash
cd <REPO>                                   # clone of the fork (fusheng-ji/RL-Kernel)
git fetch origin feat/qwen3-next-logprob-benchmarks feat/ws1-qwen3-next-distributed-logprob-ep
git bundle verify <THIS_DIR>/qwen3-next-ocr-20261118.bundle
git fetch <THIS_DIR>/qwen3-next-ocr-20261118.bundle 'refs/heads/fix/551-flink-r1:refs/heads/fix/551-flink-r1'
git worktree add --detach <SRC> fix/551-flink-r1          # 2c0cd1e1
cd <SRC> && git status --porcelain --untracked-files=no   # must be empty
```

**Environment.** Match the B200 rerun, so the two tables use the same libraries. The report records every version; send it back as is.
- torch 2.13.0+cu130, vLLM 0.30.0, FlashInfer 0.6.18.post1, FlashAttention-4 (CuTe) as installed with vLLM.
- Transformer Engine 2.20.2.
- Megatron-LM at `52fbcbc18` (`megatron.core` 0.20.0), on `PYTHONPATH`.

  The table in #507's description was measured with TE 2.16.1 and Megatron-core 0.16. If the H100 environment still has those versions, run with them and say so: the description will then state the versions per table.
- `ninja` on `PATH`, because FlashInfer JIT-compiles on first use. `TRITON_F32_DEFAULT=ieee`.

**Run** (one H100, idle; about 20 min plus the extension build):

```bash
cd <SRC>
export TRITON_F32_DEFAULT=ieee PYTHONPATH=<SRC>:<MEGATRON_LM>
python setup.py build_ext --inplace > <OUT>/build.log 2>&1
python -c "import rl_engine._C as c; print(c.__file__)"        # must be under <SRC>
git status --porcelain --untracked-files=no                      # still empty
A=benchmarks/models/qwen3_next/qwen3_next_attention_prior_art.py
python $A --only rl_kernel_cuda,torch_sdpa,vllm_fa2_auto,vllm_fa2_split1,vllm_triton_2d,flashinfer,fa4_cute,te_training,megatron_local \
  --out <OUT>/report-main.json > <OUT>/attn-main.log 2>&1
python $A --only te_fused --merge <OUT>/report-main.json --out <OUT>/report.json > <OUT>/attn-te.log 2>&1
python benchmarks/models/qwen3_next/plot_qwen3_next_attention_prior_art.py <OUT>/report.json > <OUT>/plot.log 2>&1
```

`te_fused` runs in its own process because cuDNN-frontend refuses to start when two CUDA runtimes can be loaded (a CUDA 12 `libcudart` next to CUDA 13's). If the first command can also run `te_fused` in the H100 environment, adding it to the `--only` list is fine; then skip the `--merge` step and name the output `report.json`.

**Expected.**
- `report.json` has `rl_kernel_commit` = `2c0cd1e1…` and `tracked_tree_dirty: false`.
- Batch-invariance verdicts equal the current H100 table in #507:
  - RL-Kernel and Megatron local: yes, forward and backward.
  - SDPA, FA-2 auto, FA-4 and TE training: no.
  - FA-2 split=1, Triton 2D, FlashInfer and TE fused: yes.
- Max abs error vs FP64 about 6.7e-3 (Megatron local 1.3e-2).
- Lower prefill latencies for FlashInfer and Triton 2D than 364 / 420 / 793 µs and 412 / 527 / 1949 µs.
- Any candidate reported as unavailable or failed is a finding: send its log.

**Send back:**
- `report.json`, `report-main.json` and `figure.png`;
- the four logs;
- `nvidia-smi --query-gpu=name,driver_version --format=csv`.

Strip host names and paths first. The JSON holds no paths, but the logs may.

## C. What the local fixes in the bundle change (for reading results)

Model code is unchanged. The fixes cover:
- **#543:** the chain gate rejects a non-boolean `passed` (exit 2); `test_qwen3_next_vllm_attention_metadata.py` renamed to `check_…`; the provider GPU job runs every `check_` module. #503's own fix `8e33c453` (empty-input bias-grad gate, already pushed on #503) is merged into #543 and up the stack.
- **#551:**
  - the attention timing fix (above);
  - `--merge` refuses two reports that both measured a candidate;
  - the prior-art CLIs reject a missing `--checkpoint` / `--output` / `--out`.

Steps 1-3 of `../README.md` are unaffected unless they are run at these commits.

## Return path

Push results to a new fork branch `handoff/h100-qwen3-next-results-20261118`, under `results/<step or item>/`. Use `git add -f` for `*.log`, because `.gitignore` drops them.
