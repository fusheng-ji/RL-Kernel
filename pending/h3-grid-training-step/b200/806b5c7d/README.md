# PR E (#760) at `806b5c7d` on B200 (2026-11-23)

#760 after its second bot round on 2026-11-23. This head:
- merges #741 `6e482907`, #742 `4bbb7fc0` and #743 `d11a0baf`;
- puts the refiner's gate/up LoRA base on the gate/up row's fixed kernel (`7f0e996a`);
- gathers export write failures (`3c91fc23`) and tightens the G5 verdict (`fb05788b`).

The refiner LoRA's bytes changed, so G5 and G6 were rerun. Paths, nodes and job ids are redacted.

- `g5/`: G5, TP4 x CP2 with SP on 8 x B200, 2 optimizer steps of 3 passes. Every non-inject case is bitwise on all 8 ranks and on repeat, with finite norms. The injected error fails rank 2 only.
- `g6-rr/`: G6 R/R, 8 RL steps of the verl-omni H3 recipe on the fork `rl-kernel/h3` `e62c4bd`. On every step: rollout digests 0 of 192 differ, training replay 0 of 96 differ, `logprob_abs_diff_max` 0.
- `suite/`: `pytest tests/models/minimax_h3`, head against `test-h3` `3d137858`. The one failure (the refiner's SwiGLU tolerance case) is on both sides.
- `h3-grid-g6-pp-vs-rr-b200.png` / `.json`: step time per phase, mean of steps 2-8. P/P is `../g6-pp/`; R/R is `g6-rr/`.
