# Qwen3-Next follow-up (`feat/qwen3-next-followup-20261119`): B200 and H100 evidence

Every run here is at `8284ce2b`, the follow-up's last commit before it was merged with `test-qwennext` `12d50412`. That merge adds only #622's strict-provider files, which these runs do not use. Paths, user, node, group, job and environment names are redacted.

## `b200/`: 4 × B200

- `tp4-chain/`:
  - `A/`: the user-site guard, with a planted user-site `apache-tvm-ffi` and then clean.
  - `B/`: the TP4 chain (`preflight`, `layers`, `l2`, `bi` full); each run's `run.json` and `result.json`, `bi-matrix.json`, chain report and gate logs. The gate inside the job could not run, because the node's `python3` had no pytest. `gate-rerun-local.log` is the same gate run with the environment's python: rc 0.
  - `D/`: the `_C` build in the CI checkout.
- `ep4-chain/`: the EP4 chain (`C/`: `preflight`, `layers`, `l2`).
- `ci-steps-and-cpu-suite/`:
  - `D/`: the provider-GPU workflow's two `run:` blocks, extracted as `provider.sh` and `four-gpu.sh`, and their logs.
  - `E/`: the CPU suite.

An earlier B200 round is not included. It was void because other jobs on the shared node cleared `/tmp` during the build.

## `h100/`: 4 × H100

The tester's round, unchanged. `SUMMARY.md` has one row per step, A to E. Two first attempts that met another job on the same node are kept in `not-green-attempts/`; their reruns on an idle node passed.
