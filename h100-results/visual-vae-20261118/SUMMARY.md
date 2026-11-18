# visual_vae_contract on H100 (2026-11-18)

- Code: `handoff/h3-h100-visual-vae-20261118` @ `f299a32b` (detached worktree, `git status --porcelain --untracked-files=no` empty before and after, see `header.txt`, `status-after.txt`). No patch applied. No `_C` build (pure PyTorch).
- Hardware: 1 x NVIDIA H100 (94 GB class, 95830 MiB), driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1, python 3.12.15, Diffusers source `f53d552036a0d1bd5570782a39cd40cfabf112bc` on `PYTHONPATH`. Released VAE weights at `MiniMaxAI/MiniMax-H3@42ed227` (`vae/`), `--verify-sha256` passed. Node had no other job of ours; another job of this allocation (audio VAE, task C) ran concurrently in a separate one-GPU Slurm step on the same node (the physical device of each step was not recorded).
- Commands (paths replaced by placeholders):
  - `RL_KERNEL_H3_VAE_DIR=<CKPT>/vae PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.:<PROVIDER>/src python -m pytest -q -rs tests/models/minimax_h3/test_h3_visual_vae.py`
  - `PYTHONPATH=.:<PROVIDER>/src python tools/validation/models/h3_visual_vae_evidence.py --vae-dir <CKPT>/vae --out evidence-h100.json --verify-sha256 --timing-repeats 7`
  - `python tools/validation/models/plot_h3_visual_vae.py evidence-h100.json --out-dir .`

## Results
| Item | Expected | Observed |
|---|---|---|
| pytest | 208 passed | **208 passed, 0 failed, 0 skipped** (70.6 s) |
| evidence tool exit code | 0 | **0** |
| required verdicts (`run_to_run`, `layout`, `diffusers_parity_pinned`, `contract_batch_invariant`, `tiling_deterministic`, `lut_equals_cuda_release`, `normalization_cpu_cuda_bitwise`) | all true | **all true** |
| `diffusers_batched_batch_invariant_default_flags` (informational) | false on B200 | **false** on H100 |
| `diffusers_parity_default_flags` (informational) | true on B200 | **true** on H100 |
| kernels | differ from B200 | H100: decoder attention `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16...` (432 calls), encoder convs `sm90_xmma_fprop_implicit_gemm_..._tf32f32` (B200 had `sdpa_sm100...` and `cutlass3x_sm100...`) |

Peak memory 18.1 GiB. Median latency (ms, H100): contract encode_keyframe 97.4, encode_video 1849, decode_video 1126 (tiled).

No deviation from the expected values. Files: `pytest-visual-vae-h100.txt`, `evidence-h100.json`, `h3-visual-vae-accuracy.png`, `h3-visual-vae-latency.png`, logs, `nvidia-smi.txt` (query + topology).

## Draft "H100" paragraph
On one H100 (driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, Diffusers `f53d552`, released weights `42ed227`), at `1e1ff922` (handoff `f299a32b`): `test_h3_visual_vae.py` 208 passed / 0 failed / 0 skipped; the evidence tool exits 0 with all required verdicts true; the Diffusers batched default-flags batch-invariance verdict is false, as on B200. The H100 conv and attention kernels are SM90 ones (cuDNN `sdpa_sm90` wgmma, `sm90_xmma` tf32 implicit GEMM). Bytes are not claimed equal to B200. No patch applied.
