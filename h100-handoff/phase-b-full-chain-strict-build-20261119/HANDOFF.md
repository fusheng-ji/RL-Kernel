# Optional H100 follow-up for `phase_b_full_chain`: the strict arm with `h3_full_attention` (2026-11-19)

Nothing in the pushed handoff (`7e22c736`) or its results
(`c86c37b7`) changes. This follow-up is optional: the PR already reports the strict arm as run.

## What it would add

The first H100 round ran the strict graph (`R/R`, bound, no fallback) with every RL-Kernel
candidate op except attention: `_C` was built without `RL_KERNEL_H3_STRICT=1`, so
`h3_full_attention` was unavailable and attention ran as `QueryChunkAttention`
(`--attention-query-chunk 4096`), the interim attention of `h3_full_model.py`. The handoff did
not ask for the strict build.

A rerun of items 1 and 2 with the strict build would add:

1. the strict arm's repeat equality (A vs B, fresh processes) and injection localization
   (A vs C) with **all seven ops** from RL-Kernel, including the H3 attention kernel
   (`H3FullAttentionOp`, profile `h3-cuda-strict-v1`);
2. peak memory and time of the strict arm with that kernel, against the query-chunk round
   (Ref2VA filled the card there: 95,317 of 95,830 MiB);
3. the query-chunk round against the strict-build round, one card, as a localization of the
   attention change (informational; the first drift is expected in the first block at step 0).

It also collects two small facts for the PR: the H100 host's Pillow (the toy fixtures'
`media_preprocess` drift) and the chain tests at the new head, where that drift no longer fails.

## Code

Fork branch `handoff/h3-h100-phase-b-full-chain-strict-build-20261119`:
`feat/h3-phase-b-full-chain` @ `f26fa64e` plus one commit that adds this folder (this file, the
unchanged runner `run_h100_full_chain.sh` from `7e22c736`, and `text-sha256/`). The tree outside
`h100-handoff/` is exactly `f26fa64e`.

```bash
cd <REPO>                       # clone of RL-Align/RL-Kernel, remote "upstream"
git fetch upstream test-h3      # must reach 10552d82
git fetch https://github.com/fusheng-ji/RL-Kernel.git \
    '+refs/heads/handoff/h3-h100-phase-b-full-chain-strict-build-20261119:refs/heads/handoff/h3-h100-phase-b-full-chain-strict-build-20261119'
git worktree add --detach <SRC> handoff/h3-h100-phase-b-full-chain-strict-build-20261119
git -C <SRC> log --oneline -2   # the handoff commit on top of f26fa64e; stop if not
git -C <SRC> diff --stat f26fa64e HEAD -- . ':!h100-handoff'   # must print nothing
G=<SRC>/h100-handoff/phase-b-full-chain-strict-build-20261119/run_h100_full_chain.sh
```

- Set `SRC`, `PY`, `CKPT`, `AUDIO_VAE`, `PROVIDER` and `TEXT` as in round 1. `TEXT` is round 1's
  `text/` (the pinned files from `7e22c736`). Check it (each file in `text-sha256/` holds the bare
  hash):

  ```bash
  for w in fl2va ref2va; do
    [[ $(sha256sum $TEXT/$w-text.safetensors | cut -d' ' -f1) == \
       $(cat <SRC>/h100-handoff/phase-b-full-chain-strict-build-20261119/text-sha256/$w-text.sha256) ]] \
      && echo "$w ok" || echo "$w MISMATCH: stop"
  done
  ```
- `<ROUND1>` is round 1's output folder (`01/`, `02/` hold the query-chunk A runs).
- **Since `a60a1a9c`, `run` requires `--text-source`.** The runner passes extra arguments
  through, so every `run` below adds `--text-source pinned`. Without it the run stops at
  argument parsing.

## Items (one H100; Ref2VA B and C may run on two cards at once, as before)

### F0. Strict build and environment

In `<SRC>`, an isolated worktree (`_C` of a shared env resolves to the last editable
install; check `_C` in `environment.txt`):

```bash
cd <SRC>
RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=2 \
  $PY setup.py build_ext --inplace
$PY -m pytest -q tests/backends/cuda/test_h3_full_attention.py > <OUT>/F0/h3-attention-tests.txt 2>&1
$G env <OUT>/F0
```

Expected: `environment.txt` shows capability `(9, 0)`, `_C` inside `<SRC>` and
**`h3_full_attention True`**; the attention tests pass. If `h3_full_attention` is still
`False`, stop and send `F0/`: the rest is the first round again.

### F1. Chain tests and Pillow at `f26fa64e`

```bash
$G tests <OUT>/F1
$PY -c "import PIL, sys; print(PIL.__version__, PIL.__file__); from PIL import features; features.pilinfo(supported_formats=False)" > <OUT>/F1/pillow.txt 2>&1
$PY - > <OUT>/F1/pillow-probe.txt <<'PY'
import hashlib, numpy as np
from PIL import Image
im = Image.fromarray(np.random.default_rng(1).integers(0, 256, (80, 48, 3), dtype=np.uint8))
a = np.ascontiguousarray(np.array(im.resize((64, 64), Image.Resampling.LANCZOS)).transpose(2, 0, 1)[None, :, None])
h = hashlib.sha256(f"torch.uint8|{tuple(a.shape)}|".encode()); h.update(a.tobytes()); print(h.hexdigest())
PY
```

Expected: `tests.txt` **43 passed, 5 skipped** (3 need `RL_KERNEL_H3_QWEN3VL_DIR`, 2 are the
opt-in bitwise replays of the committed fixtures). On the sealing host the probe prints
`47b12eaffdd9d3837c8f437521efe6b46d3457fb2ee314cff02326b24914422f`; any other value on H100
confirms the first round's `media_preprocess` drift is Pillow on that host, and `pillow.txt`
says which Pillow.

### F2. Strict FL2VA with the strict build

As item 1 of the pushed handoff, without `--attention-query-chunk`:

```bash
O=<OUT>/F2
$G run $O fl2va A --text-source pinned --mode strict --offload --keep-tensors \
    --capture 1:transformer_blocks.24,1:transformer_blocks.25,1:transformer_blocks.26 --tensors $O/fl2va-A.safetensors
$G run $O fl2va B --text-source pinned --mode strict --offload
$G run $O fl2va C --text-source pinned --mode strict --offload --keep-tensors \
    --capture 1:transformer_blocks.24,1:transformer_blocks.25,1:transformer_blocks.26 --tensors $O/fl2va-C.safetensors \
    --inject '{"stage": "denoise", "step": 1, "module": "transformer_blocks.25", "row": 3000}'
$G compare $O fl2va-A-B $O/fl2va-A.json $O/fl2va-B.json
$G compare $O fl2va-A-C $O/fl2va-A.json $O/fl2va-C.json \
    --tensors-lhs $O/fl2va-A.safetensors --tensors-rhs $O/fl2va-C.safetensors
$G compare $O fl2va-querychunk-vs-strictbuild <ROUND1>/01/fl2va-A.json $O/fl2va-A.json
```

Expected:

- `fl2va-A-report.json`: `execution.case_id = "R/R"`, `bound = true`, `fell_back = null`,
  8 `row` entries in `boundary_bindings`, and
  **`actual_backends.attention` ends in `H3FullAttentionOp`**.
- `cmp-fl2va-A-B.json`: `first_drift = null`, hard gate `pass`, intermediates and decoded
  media equal.
- `cmp-fl2va-A-C.json`: `denoise`, step 1, `transformer_blocks.25`, row 3000 (target video,
  latent frame 18, patch (6, 3)), as in round 1.
- `cmp-fl2va-querychunk-vs-strictbuild.json` (informational): every stage before `denoise`
  equal; the first drift in `denoise`, step 0. Record the module.

### F3. Strict Ref2VA with the strict build

Same as F2 with `ref2va`, row 20000 (condition row 11683), `<ROUND1>/02/ref2va-A.json` for the
last compare. Use `--offload`; if it runs out of memory, use `--stream-blocks` instead of
`--offload` for all three runs and say so.

### F4. Memory

Sample `nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -i <card> -l 2`
into `<OUT>/mem/<item>.csv` for each card used, as in round 1 (`unix_time,MiB` lines are
also fine).

## Send back

`<OUT>/F0` … `F3`, `<OUT>/mem/`, logs and a short SUMMARY like round 1's, as results branch `handoff/h3-h100-phase-b-full-chain-strict-build-results-20261119`, folder
`h100-results/phase-b-full-chain-strict-build-20261119/` (redacted: no host names, paths, user or job ids). Do not send the `*.safetensors`.
