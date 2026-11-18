# H100 handoff: `visual_vae_contract` (RFC #420 Phase B, MiniMax-H3), 2026-11-18

Hardware: one H100 (sm90) with at least 40 GB free. ROCm is out of scope. Paths are
placeholders (`<REPO>`, `<CKPT>`, `<PROVIDER>`, `<OUT>`).

## Why this is needed

The contract was validated on B200 (sm100) only. Nothing in it is Hopper-gated: no
`cc == (9, 0)` check, and no strict H3 kernel. The RFC asks for CUDA evidence, though, and
the kernels behind the reused Diffusers network differ by architecture:

- On B200, the decoder attention ran `cudnn ... sdpa_sm100_flash_fprop_f16`.
- On B200, the encoder convolutions ran `cutlass3x_sm100 ... tf32 implicit_gemm`.

Hopper picks other kernels. So the H100 run has to show three things:

1. Run-to-run determinism.
2. Batch-row independence.
3. Bitwise parity with the Diffusers recipe.

It does not have to show that H100 bits equal B200 bits; the contract does not claim that.

## Code

This branch, `handoff/h3-h100-visual-vae-20261118`, is `feat/h3-visual-vae-contract` @ `1e1ff922`
(base `10552d82`, `test-h3`) plus one commit that adds this folder. Run from a checkout of it:

```bash
git fetch https://github.com/fusheng-ji/RL-Kernel handoff/h3-h100-visual-vae-20261118
git worktree add --detach <SRC> FETCH_HEAD
```

No extension build is needed: the contract is pure PyTorch. The `_C` state does not
matter for these tests.

## Environment

- torch 2.13.0+cu130, CUDA 13.0, the same versions as the B200 run.
- Diffusers source at `f53d552036a0d1bd5570782a39cd40cfabf112bc` (`<PROVIDER>`). The tests and
  the evidence tool only need `<PROVIDER>/src` on `PYTHONPATH`; `accelerate` is not needed.
- `safetensors` and `matplotlib` (the latter for the charts only).
- The released video VAE from `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`:
  only `vae/*` (10.4 GB). For example:

```bash
python -c "from huggingface_hub import snapshot_download as s; s('MiniMaxAI/MiniMax-H3', \
  revision='42ed227ee7df40d41602854ae760620d6eb651fe', local_dir='<CKPT>', allow_patterns=['vae/*'])"
```

## Run

```bash
cd <SRC>
export RL_KERNEL_H3_VAE_DIR=<CKPT>/vae PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 1. tests (CPU + CUDA + released weights)
PYTHONPATH=.:<PROVIDER>/src python -m pytest -q -rs tests/models/minimax_h3/test_h3_visual_vae.py \
    > <OUT>/pytest-visual-vae-h100.txt 2>&1
# 2. evidence at 480x832 (verifies shard sha256 first), then charts
PYTHONPATH=.:<PROVIDER>/src python tools/validation/models/h3_visual_vae_evidence.py \
    --vae-dir <CKPT>/vae --out <OUT>/evidence-h100.json --verify-sha256 --timing-repeats 7
python tools/validation/models/plot_h3_visual_vae.py <OUT>/evidence-h100.json --out-dir <OUT>
```

## Expected

- **pytest.** On B200 the result was 208 passed, 0 failed, 0 skipped (`1e1ff922`, released weights; the B200
  evidence is in `b200-reference/`). H100 should give the same count.
- **Evidence exit code.** It is 0 when every required verdict holds: `run_to_run`, `layout`,
  `diffusers_parity_pinned`, `contract_batch_invariant`, `tiling_deterministic`,
  `lut_equals_cuda_release`, `normalization_cpu_cuda_bitwise`.
- **Informational verdicts.** These are recorded but not required:
  - `diffusers_batched_batch_invariant_default_flags` was `false` on B200. Record the H100
    value, whichever it is.
  - `diffusers_parity_default_flags` was `true` on B200.
- **Kernel names.** `kernels` in the JSON lists the H100 conv and attention kernels; they should
  differ from B200.

## Send back

- `<OUT>/pytest-visual-vae-h100.txt`
- `<OUT>/evidence-h100.json`
- the two PNGs
- `nvidia-smi --query-gpu=name,driver_version --format=csv`

Before sending, strip host names and paths. The JSON already contains no paths.

Push them as fork branch `handoff/h3-h100-visual-vae-results-20261118` (folder
`h100-results/visual-vae-20261118/`). No PR and no comment.
