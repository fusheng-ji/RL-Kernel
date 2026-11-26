# RFC #738 `h3_image_pdd_head_contract` + `ws1_one_h3_image_step` on B200

Code: `feat/h3-image-one-step` at `b685c4991` (on `feat/h3-image-pdd-head` `984ee3d2e`), clean tree, strict build (`RL_KERNEL_H3_STRICT=1`), one NVIDIA B200 (CUDA 13.0, torch 2.13.0, Triton 3.7.1, Diffusers 875373ac). Released MiniMax-H3-Image weights. Bindings: 22 row, 14 interim, 3 reference.

```bash
pytest tests/models/minimax_h3_image/test_h3_image_pdd_head.py tests/models/minimax_h3_image/test_h3_image_step.py   # pytest-new.txt
python tools/validation/models/h3_image_step.py capture --out inputs
python tools/validation/models/h3_image_step.py replay --inputs inputs --out run-a --fixtures t2i_upscaled --plain-autograd
python tools/validation/models/h3_image_step.py replay --inputs inputs --out run-a-edit --fixtures edit_two_refs
python tools/validation/models/h3_image_step.py replay --inputs inputs --out run-b --fixtures t2i_upscaled --steps 0,5 --against run-a
pytest tests/models/minimax_h3_image   # pytest-suite.txt
```

| Path | What |
|---|---|
| `pytest-new.txt` | the PDD head and one-step tests: 83 passed |
| `pytest-suite.txt` | `tests/models/minimax_h3_image`: 514 passed |
| `inputs/*.inputs.json` | digests of the captured inputs of the six released calls (the tensors are not uploaded) |
| `run-a/`, `run-a-edit/` | per fixture: bindings, 61-node digests per step, controls (repeat, batch, permutation, next head), training step, distance to Diffusers and its kernel variants |
| `run-b/` | steps 0 and 5 in a second process, compared with `run-a` |
| `charts/h3-image-one-step-per-node.png` | step 0, distance to Diffusers per node next to Diffusers' own kernel spread |

The two fixes found on H100 later (`d3bc9fe41`, `f4171b776`) do not change these B200 bytes: see `../ws1-full-69e0e42c9/after-h100-fixes-4cfa3cbfc/`.
