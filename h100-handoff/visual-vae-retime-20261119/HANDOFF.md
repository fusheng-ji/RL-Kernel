# H100 handoff: re-time the `visual_vae_contract` decode at `0028ad7d` (RFC #420 Phase B), 2026-11-19

Hardware: one **idle** H100 (sm90) with at least 40 GB free, the same kind of card as the
first H100 run. ROCm is out of scope. Paths are placeholders (`<SRC>`, `<CKPT>`, `<PROVIDER>`,
`<OUT>`). About 15 minutes, most of it the evidence run.

## Why

The first H100 run (handoff `f299a32b`, results `0eadf0fd`) measured the code at `1e1ff922`.
There the batch-1 decode took **1126 ms** against **491 ms** for the Diffusers recipe. Most of
the gap was host-side hashing of the output: it was hashed twice, through copies.

`5116eca6` hashes each output once, in place. On an idle B200 it brought the decode to
402.7 ms against 339.1 ms for Diffusers, with all 30 sha256 fields of the evidence unchanged.
The H100 decode was never re-timed, so the PR still quotes the old H100 number. This run
replaces it.

Nothing else is being tested. The outputs must not change.

## Code

This branch, `handoff/h3-h100-visual-vae-retime-20261119`, is `feat/h3-visual-vae-contract` @
`0028ad7d` (the PR head) plus one commit that adds this folder. Changes since the first H100 run:

- `5116eca6`: the hashing change in `rl_engine/models/minimax_h3/visual_vae.py`, plus tests.
- `0028ad7d`: chart tool only.

```bash
git fetch https://github.com/fusheng-ji/RL-Kernel handoff/h3-h100-visual-vae-retime-20261119
git worktree add --detach <SRC> FETCH_HEAD
```

No extension build is needed: the contract is pure PyTorch.

## Environment

The same as the first H100 run:

- torch 2.13.0+cu130 and CUDA 13.0;
- Diffusers source at `f53d552036a0d1bd5570782a39cd40cfabf112bc` (`<PROVIDER>/src` on
  `PYTHONPATH`);
- `safetensors` and `matplotlib`;
- the released `vae/` from `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`.

Reuse the first run's `<CKPT>` if it is still there.

## Run

Make sure no other process is using the card before starting; the timings are the point of
this run.

```bash
cd <SRC>
export RL_KERNEL_H3_VAE_DIR=<CKPT>/vae PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu --format=csv > <OUT>/gpu-before.txt
# 1. tests at the PR head (CPU + CUDA + released weights)
PYTHONPATH=.:<PROVIDER>/src python -m pytest -q -p no:cacheprovider -rs \
    tests/models/minimax_h3/test_h3_visual_vae.py > <OUT>/pytest-visual-vae-h100.txt 2>&1
# 2. evidence at 480x832, same arguments as the first run
PYTHONPATH=.:<PROVIDER>/src python tools/validation/models/h3_visual_vae_evidence.py \
    --vae-dir <CKPT>/vae --out <OUT>/evidence-h100.json --verify-sha256 --timing-repeats 7 \
    > <OUT>/evidence.stdout 2>&1; echo "evidence exit=$?"
# 3. charts and the comparison with the first H100 run
python tools/validation/models/plot_h3_visual_vae.py <OUT>/evidence-h100.json --out-dir <OUT>
H=h100-handoff/visual-vae-retime-20261119
python $H/compare_h100.py $H/reference/evidence-h100-1e1ff922.json <OUT>/evidence-h100.json \
    > <OUT>/compare.txt; echo "compare exit=$?"
```

## Expected

- **pytest:** 221 passed, the same as B200 at `5116eca6`. The first H100 run had 208 passed; the
  13 extra tests come from `5116eca6`.
- **Evidence exit code 0:** every required verdict is true.
- **`compare.txt` ends in `RESULT: PASS`:**
  - all 7 required verdicts are true;
  - the two informational verdicts are the same as before
    (`diffusers_batched_batch_invariant_default_flags` false, `diffusers_parity_default_flags`
    true);
  - **0 of the 30 sha256 fields differ** from the first H100 run.

  A differing digest is a finding: stop and send everything back.
- **Timing:**
  - `contract / decode_video` is well below 1126 ms;
  - `diffusers_default_flags / decode_video` stays near 491 ms. If it doesn't, the card was
    not in the same state as the first run; say so.

  On B200, the remaining gap (402.7 vs 339.1 ms) is the one in-place hash and the device-to-host
  copy that Diffusers does not do.
- **`rl_kernel_dirty` should be `false`.** If it is `true`, run `git -C <SRC> diff --summary`.
  - Mode-only changes (`100755 => 100644`, as seen on other H100 checkouts) are harmless; note
    them.
  - Any content change makes the run invalid.

## Send back

- `<OUT>/pytest-visual-vae-h100.txt`;
- `<OUT>/evidence-h100.json` and `<OUT>/evidence.stdout`;
- `<OUT>/compare.txt` and `<OUT>/gpu-before.txt`;
- the PNGs and `.md` tables from step 3.

Strip host names, user names, job ids and paths before sending. The JSON contains no paths, and
the PNGs are written without metadata.

Push them as fork branch `handoff/h3-h100-visual-vae-retime-results-20261119`, in folder
`h100-results/visual-vae-retime-20261119/`, as one signed-off commit on top of this branch.
No PR and no comment.

## Reference files (`reference/`)

Both files were already published on `pr_assets` `1819e668`.

- `evidence-h100-1e1ff922.json`: the first H100 run, from results `0eadf0fd`.
- `evidence-b200-5116eca6.json`: idle-B200 run at `5116eca6`.
