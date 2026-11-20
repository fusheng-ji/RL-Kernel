# VisualVAE follow-up to RL-Align/RL-Kernel#721: test logs (2026-11-20)

One job on one B200 with no other job on the GPU ran both trees in turn: `head` = `fix/h3-visual-vae-followup-20261120` at `34e3da81`, `base` = `test-h3` at `12e81c36` (`commits.txt`, `gpu.txt`).

| File | What |
|---|---|
| `<side>-priorart-b200.txt` | `pytest tests/models/minimax_h3/test_h3_visual_vae.py tests/models/minimax_h3/test_h3_visual_vae_tools.py`, torch 2.13.0+cu130, Diffusers `f53d552` source on `PYTHONPATH`, released VAE |
| `<side>-rlkernel-cpu.txt` | the same files in the RL-Kernel environment (no Diffusers), CUDA hidden |
