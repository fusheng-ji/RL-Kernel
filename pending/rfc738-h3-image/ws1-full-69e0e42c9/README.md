# RFC #738 `ws1_full_h3_image_t2i` + `ws1_full_h3_image_edit` on B200

Code: `feat/h3-image-ws1-full` at `69e0e42c9`, clean tree, strict build (`RL_KERNEL_H3_STRICT=1`), one NVIDIA B200 per group (CUDA 13.0, torch 2.13.0, Triton 3.7.1, Diffusers 875373ac). Released MiniMax-H3-Image weights.

Command, per group (`t2i`: `t2i_upscaled,t2i_example`, controls on `t2i_upscaled`, training at steps 0-5; `edit`: `edit_two_refs,edit_example`, controls on `edit_two_refs`, training at step 5):

```bash
python tools/validation/models/h3_image_trajectory.py --fixtures <fixtures> --controls <fixture> --train-steps <steps> --out run-a
python tools/validation/models/h3_image_trajectory.py --fixtures <fixtures> --strict-only --against run-a --out run-b
```

| Path | What |
|---|---|
| `<group>/run-a/summary.json` | bindings, per fixture: strict chain, Diffusers arm and kernel variants, per-step distance, PSNR, controls, training |
| `<group>/run-a/<fixture>.records.json` | inputs digest, six step records, final latent and decoded image digests |
| `<group>/run-a/<fixture>.{strict,diffusers}.json` | the sealed `h3_image_evidence_schema` artifact of each arm |
| `<group>/run-b/` | the strict arm again in a second process; `equal_to_other_process` compares the records with run-a |
| `t2i/pytest-new.txt`, `t2i/pytest-suite.txt` | `test_h3_image_ws1_full.py` (17 passed), `tests/models/minimax_h3_image` (531 passed) |
| `t2i/run-a/t2i_upscaled.{strict,diffusers}.png` | decoded 1024x768 outputs |
| `charts/ws1-full-latent-distance.png`, `charts/table.md` | per-step distance to the Diffusers default: strict vs Diffusers flash / memory-efficient SDPA |
| `charts/ws1-full-outputs.jpg` | strict and Diffusers outputs, all four fixtures, downscaled |

Both groups report `ok: true` in both processes.
