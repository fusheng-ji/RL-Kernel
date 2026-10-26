# B200 evidence: PR #469 at its head `e8daba53`, idle GPU

PR #469, fork branch `feat/ws1-c6-gdn-recurrent-golden`, head `e8daba53` (full `e8daba53591bdf0afb5ebd084a1e9c1f2eaacb85`), sm_100, labelled B200 only. One NVIDIA B200 with no other process on it, torch 2.13.0+cu130, clean tree (`git status` empty before and after, `environment.txt`), extension built in place and checked to load from the checkout. Runner: `pr469/b200-idle-rerun/run.sh` on this branch; commands and exit codes in `commands.tsv` / `exitcodes.tsv` (3 of 3 exit 0).

Why: the earlier B200 set at this head (`b200-pr469-head-e8daba53-b200`) was run locally while the GPU was shared.

| Run | Result |
|---|---|
| unit suite (five paths in the PR's Tests) | 312 passed, 0 failed, 73 skipped |
| provider checks (`check_qwen3_next_norm_providers.py`, `check_gdn_recurrent_golden.py`) | 48 passed |
| `ws1_gdn_provider_agreement.py` | rc 0; the JSON is identical to the shared-GPU run except the recorded checkout path |

So the shared GPU did not change any recorded value; this set replaces it as the idle-GPU record.
