# H100 handoff: `final_e2e_diffusers_cuda`, the recipe's T2VA workload (2026-11-23)

## Why

RFC #420's `final_e2e_diffusers_cuda` compares pinned Diffusers with RL-Kernel over the full sampling trajectory, and reports the time split. The strict path needs SM90:
- the #535 / #554 GEMM rows;
- the strict boundary rows.

So the evidence comes from one H100.

The tool is `tools/validation/models/h3_e2e_diffusers.py` (local branch, in the bundle). It runs the pinned T2VA pipeline as two arms over the 8 cases of `examples/models/minimax_h3/e2e-t2va-contract.json`:
- `provider`: the pipeline as released;
- `candidate`: the transformer on the RL-Kernel strict graph, with strict boundaries, and both schedulers on the `h3_sigma_schedule` / `h3_ode_step` rows where those are bitwise.

The workload is the verl-omni H3 T2VA recipe's rollout settings:
- 256x384;
- 121 frames requested (the pipeline renders 124);
- 10 inference steps;
- no CFG;
- 4 prompts x 2 seeds.

`compare` then:
- **gates** every candidate repeat bitwise against the first run, at every step;
- **reports** provider against candidate.

A B200 smoke of the same code passed. It ran with provider boundaries and the GEMM rows interim, so it is plumbing only:
- candidate repeat bitwise, 36 of 36 tensors;
- provider repeat bitwise;
- P/R drift from step 0's video velocity;
- steady transformer call 6.83 s (candidate) against 0.22 s (provider).

| Task | Code | Hardware | Time (est.) |
|---|---|---|---|
| **E1** preflight: tests, case 0 both arms, compare | `refs/h100/h3-e2e-integrated` | 1 H100 | about 45 min (two checkpoint hashes included) |
| **E2** all 8 cases, both arms, compare | the same checkout | 1 H100 | about 1.5 to 2 h |

Paths are placeholders, as in the earlier rounds:
- `<REPO>`: a clone of the fork;
- `<THIS_DIR>`: this folder;
- `<OUT>`: where results are written;
- `<RUNS>`: where run outputs go (large: tensors and media, about 0.5 GB per run; keep them, but do not send them);
- `<CKPT>`: the **Diffusers-format** H3 checkpoint, revision `42ed227ee7df40d41602854ae760620d6eb651fe`, with `scheduler/`, `audio_scheduler/`, `vae/`, `audio_vae/`, `text_encoder/`, `tokenizer/`, `processor/` and `transformer/` (the one `h3_full_generation.py` ran on);
- `<WEIGHTS>`: the directory that holds `hf_full/`, as in the earlier rounds (the GPU tests read it);
- `<PROVIDER>`: the Diffusers checkout at `f53d552036a0d1bd5570782a39cd40cfabf112bc`, clean;
- `<E>`: the environment that ran `h3_full_generation.py`.

## Rules (unchanged from the earlier H3 rounds)

- **Clean trees.** Use detached worktrees. `git status --porcelain --untracked-files=no` must be empty before and after every step. Change no file.
- **Check `_C`.** Before every GPU step, check that `rl_engine._C.__file__` is inside the checkout.
- **Idle hardware.** Use an idle H100 with nothing else on it, and record `nvidia-smi` at the start of each step.
- **Report failures as they are.** A failure is reported as a failure, with its log. Never change a script, a test or a bound to make a result pass. Failed attempts go, as they are, into `not-green-attempts/`.
- **Push only the results branch.** No PR, no comment.
- **Redact everything sent back.**
  - Remove: cluster, job, node, partition, user, conda environment name, `/home/…` or `/data/…` path, email. Hosts become `node-a`.
  - `git add -f` the `*.log` files.
  - Commit with `git commit -s`, with no AI attribution lines.

## Code

```bash
cd <REPO>
git fetch https://github.com/fusheng-ji/RL-Kernel.git handoff/h100-h3-e2e-t2va-20261123 \
  feat/h3-strict-gemm-rows feat/h3-pack-noise-rows
git fetch https://github.com/RL-Align/RL-Kernel.git test-h3
git bundle verify <THIS_DIR>/h3-e2e-t2va-20261123.bundle
git fetch <THIS_DIR>/h3-e2e-t2va-20261123.bundle 'refs/h100/*:refs/h100/*'
git worktree add --detach <W> refs/h100/h3-e2e-integrated
```

`refs.txt` lists both refs with their commits. The bundle's prerequisites are `test-h3` `3d137858` and the fork heads of #741 (`0f2de999`) and #743 (`bcba86ca`).

| Ref | Commit | What |
|---|---|---|
| `refs/h100/h3-e2e-t2va` | `b4ba5cee` | the PR branch: the scheduler binding (`h3_strict_scheduler.py`), the e2e tool, its contract and tests, on `test-h3` `3d137858` |
| `refs/h100/h3-e2e-integrated` | `65ba0387` | the same, with #741 (GEMM rows in the strict block) and #743 (pack row) merged: **run this one** |

Build and check, in `<W>`:

```bash
RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 <E>/bin/python setup.py build_ext --inplace
export PYTHONPATH=<W>:<PROVIDER>/src RL_KERNEL_H3_WEIGHTS=<WEIGHTS>
<E>/bin/python -c "import accelerate, diffusers, torch; from transformers import Qwen3VLProcessor; \
from rl_engine.backends.extension import _C; print(torch.__version__, diffusers.__file__, _C.__file__, \
torch.cuda.get_device_name(0))" > <OUT>/env.txt 2>&1
```

`diffusers` must come from `<PROVIDER>/src`. `_C` must be inside `<W>`. If `accelerate` or `Qwen3VLProcessor` is missing from `<E>`, stop and report it; do not install into a shared environment.

---

## E1. Preflight (1 H100)

```bash
cd <W>; T=tools/validation/models/h3_e2e_diffusers.py
CUDA_VISIBLE_DEVICES=0 <E>/bin/python -m pytest -q -p no:cacheprovider -rfs \
  tests/models/minimax_h3/test_h3_e2e_diffusers.py tests/models/minimax_h3/test_h3_strict_scheduler.py \
  tests/models/minimax_h3/test_h3_full_model.py tests/models/minimax_h3/test_h3_packed_forward.py \
  tests/models/minimax_h3/test_h3_one_block.py > <OUT>/E1/tests.txt 2>&1
CUDA_VISIBLE_DEVICES=0 <E>/bin/python $T run --checkpoint <CKPT> --output <RUNS>/provider \
  --arm provider --cases 0 --repeats 2 > <OUT>/E1/provider.log 2>&1
CUDA_VISIBLE_DEVICES=0 <E>/bin/python $T run --checkpoint <CKPT> --output <RUNS>/candidate \
  --arm candidate --cases 0 --repeats 2 > <OUT>/E1/candidate.log 2>&1
<E>/bin/python $T compare --provider <RUNS>/provider --candidate <RUNS>/candidate \
  --output <OUT>/E1/compare --cases 0 > <OUT>/E1/compare.log 2>&1
```

Use no `--provider-boundaries` and no `--skip-checkpoint-hash`: both would make the result plumbing only. Each `run` hashes every checkpoint file first, which takes a while.

**Expected:**
- **`tests.txt`.** Everything passes. `test_gemm_rows_run_their_operators_on_sm90` runs (it is not skipped).
- **`compare.log`'s last line.** `gate_candidate_repeats_bitwise`, `gate_every_case_has_a_repeat`, `all_finite` and `final_evidence` are all `true`.
- **Each candidate `manifest.json`:**
  - `strict_boundaries: true`;
  - `trace.bindings` has `o_proj`, `ffn_gate_up`, `ffn_down` and `attention` as `row`;
  - `trace.boundary_binding` is `strict rows + provider`;
  - `scheduler_rows`: `scheduler` 9 `row` / 0 `provider`; `audio_scheduler` 8 `row` / 1 `provider` (step 8, where the provider's timestep sigma is not the grid sigma bitwise).
- **Known and expected:**
  - **Still `interim` or `reference`:** QK RMSNorm, MM-RoPE, SwiGLU and `packed_index_select`; their rows are not implemented yet.
  - **`packed_index_copy` reports `provider`:** model offload wraps the transformer's `forward`, so the pack row does not bind (the bytes are the same either way).
  - **Provider against candidate is not bitwise.** That is the reported comparison, not a failure.

**If the candidate runs out of memory**, or its steady transformer call (`timing.transformer_steady.mean_seconds`) is far above the first call's share (weights being moved every call), rerun E1's candidate with `--offload-margin 8GB` into a new `<RUNS>` folder. Report both attempts. A failing attempt goes into `not-green-attempts/`.

## E2. All 8 cases (1 H100)

Same commands with `--cases` left out and the same `<RUNS>` folders. Case 0's finished runs are skipped. Then compare all cases:

```bash
CUDA_VISIBLE_DEVICES=0 <E>/bin/python $T run --checkpoint <CKPT> --output <RUNS>/provider --arm provider --repeats 2 > <OUT>/E2/provider.log 2>&1
CUDA_VISIBLE_DEVICES=0 <E>/bin/python $T run --checkpoint <CKPT> --output <RUNS>/candidate --arm candidate --repeats 2 [--offload-margin as in E1] > <OUT>/E2/candidate.log 2>&1
<E>/bin/python $T compare --provider <RUNS>/provider --candidate <RUNS>/candidate --output <OUT>/E2/compare > <OUT>/E2/compare.log 2>&1
```

**Expected:** the same four `true` flags, over 8 cases. For each case, `report.json` holds the first drift, the per-step differences, the media PSNR / relative L2 and the time split.

## What to send back

Push the branch `handoff/h100-h3-e2e-t2va-results-20261123`, with one folder per task (`E1/`, `E2/`), containing:
- `env.txt`, `nvidia-smi` at the start, and `pip list` of `<E>` filtered to torch, triton, diffusers, transformers, accelerate;
- `tests.txt` and every `*.log`;
- `compare/report.json`;
- for every run, a copy of `manifest.json` and `trajectory.json`, as `runs/<arm>/case<N>/r<k>/`.

Do **not** send `tensors.safetensors` or `*.npy`: they are large, and `report.json` already holds the drift numbers computed from them. Keep `<RUNS>` until the results are verified.
