# PR E at `b1d3e1b2` on B200 (2026-11-23)

The grid training-step branch after merging the bot-round heads of #735 (`0271ee92`), #741 (`0f2de999`), #742 (`8eb80c6a`), #743 (`bcba86ca`) and `test-h3` `fc4989f7`, with the LoRA scale in the actor's export (`f3da4b98`, `2ee1ef44`). Paths, nodes and job ids are redacted.

- `g5/`: G5, TP4 x CP2 with SP on 8 x B200, 2 optimizer steps of 3 passes. One folder per case with `report.json` and `run.log`. Every non-inject case is bitwise on all 8 ranks and on repeat. The injected error fails rank 2 only.
- `g6-rr/`: G6 R/R, 8 RL steps of the verl-omni H3 recipe (T2VA, 121 frames, 10 inference steps, 16 samples) on the fork `rl-kernel/h3` `e62c4bd`. On all 8 steps the rollout digests match (0 of 192 differ), the training replay digests match (0 of 96 differ) and `logprob_abs_diff_max` is 0. `train.log` holds the full log; `gate-lines.txt` holds the per-step metric lines.
- `suite/`: `pytest tests/models/minimax_h3` with 4 GPUs per side, head against `test-h3` `fc4989f7`. `failed-*.txt` lists the failures on each side; the one failure is on both.
- `h3-grid-g6-pp-vs-rr-b200.png` / `.json`: step time per phase, mean of steps 2-8. P/P is the earlier run in `../g6-pp/`; R/R is `g6-rr/`. Made by `../../plot_g6.py`.
