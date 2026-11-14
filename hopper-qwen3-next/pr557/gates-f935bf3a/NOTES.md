#557 head `f935bf3ae70eb30580e38d83207b20fd7d407faa`, 4 x H100, no patch, detached worktree, `git status --short` empty before and after.
`tests/models/qwen3_next/test_*.py` + `check_*.py`: 871 passed, 0 skipped, 0 failed (the head has more tests than the 814 of `a705d287`).
The six gates (adapter, ep_moe, tp_attention, tp_gdn, tp_head, tp_moe), each as a 4-rank torchrun on the official checkpoint: all rc=0 (`rc.txt`), rank files under `gates/`.
`PYTHONPATH` was set to the worktree (the venv's editable install points at another worktree). EP matrix at the stack top: see `../../pr551/ep_matrix-2c699e3f`.
