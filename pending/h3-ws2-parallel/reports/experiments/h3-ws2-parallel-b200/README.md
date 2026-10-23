# B200 results for #569 (WS2 parallel layouts), one 8 x B200 node

Measured on NVIDIA B200 (sm100), driver 580.126, CUDA 13.0, torch 2.13.0+cu130, pinned `MiniMax-H3@42ed227` weights, clean tree at `5ae9f5eb`
(the #569 head), single node with 8 GPUs, one process per GPU. Commands as in the H100 evidence (`tools/validation/models/h3_parallel_run.py`),
host names replaced by `node-a`.

- `tests.json`: `test_h3_parallel.py` and `test_h3_projection_rows.py`, 128 passed, 0 failed, 0 skipped; the 8-GPU NCCL cases ran on this node.
- One report per layout: `tp2`, `tp4`, `tp4_cp2`, `tp4_cp2_sp`, `tp8_ipc`, `tp8_hierarchical`. All six are bitwise equal to the single-GPU canonical
  block (forward and input gradient; the weight-gradient shards as well for the layouts without CP).
- `tp8_ipc` runs the TP8 group with the node-local CUDA-IPC fixed tree (`--local-size 8`). `tp8_hierarchical` runs the same TP8 group with
  `--local-size 4`, which selects the hierarchical IPC + NCCL tree and treats the 8 GPUs as 2 groups of 4 **inside one physical node**. It exercises
  the hierarchical code path, not a network between nodes: the cross-node collective was not measured on B200.
- The job was preempted once (the first attempt stopped inside the third layout and was requeued); the reports here are all from the second,
  complete attempt, which also ran the tests again.
- Not run here: the two inherited tests that the PR text lists as skipped for lack of eight GPUs (`test_h3_tp_adaln.py`, `test_h3_sp_norm_adaln.py`),
  and any Hopper 8-GPU single-node run.
