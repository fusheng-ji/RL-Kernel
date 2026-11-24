# RFC #738 MiniMax-H3-Image PR evidence

These are the evidence files for the H3-Image row PRs that target `test-h3-image`. The row branches sit on the `test-h3` sync commit `bf297aaa3`.

| Folder | What | Where it ran |
|---|---|---|
| `charts/` | every PR chart (PNG), the table behind each one (`tables.md`), the data (`data-b200-104eb599f.json`) and the two scripts that made them | B200, data computed at the stack head `104eb599f` |
| `stack-head-104eb599f/` | full image suite (`pytest-image.txt`, 460 passed), Phase B chain on the released weights (`chain/`, 4 fixtures × 2 arms + `summary.json`; two fixtures are the release's example requests, the other two are built from its example inputs), reference-conditioning, VAE and text evidence | B200, stack head `104eb599f` (contains every row) |
| `real-vae-check-b200.json` | released VAE + reference conditioning vs the full Diffusers edit blocks | B200, earlier integrated head with identical row files |
| `vae/` | the `image_vae_contract` charts and their tables | B200 and H100, identical VAE row files |
| `h100/` | H100 (Hopper, 94 GB, sm_90) results of the H3-Image handoff: cross-environment probe (T2), VAE (T3), text (T4), reference conditioning (T5) and the Phase B chain (T6) | H100. T2-T5 ran source files identical to these PRs. T6 ran the Phase B chain one commit before its last one (public twin `7c32d915d`), so `full_chain.py` and its runner changed afterwards. Not run on H100: `image_sigma_pdd_schedule` (#777), `image_velocity_step` (#778), and the final Phase B commit that uses them. The Diffusers gates in the test files of #770 and #772, and the sealed H100 text digests of #776, were added after these runs |

Environment: torch 2.13.0+cu130, Diffusers 875373ac (huggingface/diffusers#15230), MiniMax-H3-Image `8ced6006`. The `stack-head` VAE and text evidence ran on a B200 that also held another process (19 GB resident, 0% utilization when sampled before the run), so treat those timings as indicative. The B200 run behind `vae/` started on a card showing 0 MiB used and 0% utilization. Paths are redacted.
