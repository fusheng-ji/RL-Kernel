# H100 handoff: `phase_b_full_chain` strict end-to-end runs (RFC #420 Phase B, MiniMax-H3), 2026-11-18

Hardware: one H100 (sm90, 80 GB) for everything; a second card on the same node lets item 3
run beside item 2. ROCm is out of scope. Paths are placeholders (`<REPO>`, `<SRC>`, `<CKPT>`,
`<AUDIO_VAE>`, `<PROVIDER>`, `<OUT>`, `<TEXT>`, `<THIS_DIR>`).

## Why

On B200 the chain ran end to end with provider arithmetic for the transformer (`P/P`),
4 released blocks and all 50, FL2VA and Ref2VA (see `../evidence/`). The strict RL-Kernel graph
cannot run there: the strict video rows need cc 9.0 (#646), so a strict run on B200 fails at
`proj_in`. The Phase B exit criteria need the strict arm:

- one sealed end-to-end fixture each for FL2VA and Ref2VA;
- latent trajectory equality as the hard gate.

So this handoff produces the strict (`R/R`) fixtures on Hopper and checks four things:

1. Two strict runs of the same request in separate processes are byte-equal at every stage
   (H-C0 / H-C2 at chain level).
2. An injected one-ULP drift inside block 25 is localized to its step, block and packed row
   on the strict graph.
3. Strict versus provider on the same H100: where the two arms first diverge (informational).
4. The test file passes on Hopper.

## Code

`<THIS_DIR>` is `h100-handoff/phase-b-full-chain-20261118/` in the handoff branch
`handoff/h3-h100-phase-b-full-chain-20261118`. That branch is `feat/h3-phase-b-full-chain` @
`c9666ada` plus one commit that adds this folder (the doc, the runner script, the B200 reference
artifacts), plus one commit that adds the pinned text features under `text/`. The handoff
branch is on the fork; `feat/h3-phase-b-full-chain` itself is not pushed. The branch merges the four sibling
Phase B branches (visual VAE, audio VAE, FL2VA and Ref2VA pipeline contracts) on top of
`test-h3` @ `10552d82`.

```bash
cd <REPO>                                   # clone of RL-Align/RL-Kernel, remote "upstream"
git fetch upstream test-h3                  # must reach 10552d82
# from the fork:
git fetch https://github.com/fusheng-ji/RL-Kernel.git \
    handoff/h3-h100-phase-b-full-chain-20261118:refs/heads/handoff/h3-h100-phase-b-full-chain-20261118
git worktree add --detach <SRC> handoff/h3-h100-phase-b-full-chain-20261118
git -C <SRC> log --oneline -3               # two handoff commits on top of c9666ada; stop if not
```

The runs only read `rl_engine/` and `tools/`, which are identical to `c9666ada`.

Build the extension in place on the H100, as for the earlier H3 handoffs (`ninja` on `PATH`):
`cd <SRC> && $PY setup.py build_ext --inplace`. The strict graph needs `_C` with the H3 rows.
The editable `_C` of a shared env resolves to the last `pip install -e`, so check `_C` in
`environment.txt` (item 0). It must point into `<SRC>`.

## Environment

- torch 2.13.0+cu130, CUDA 13.0: the same stack as the 2026-11-14 H100 runs.
- Provider: a clean diffusers checkout at `f53d552036a0d1bd5570782a39cd40cfabf112bc`
  (`<PROVIDER>`); the runner puts `<PROVIDER>/src` on `PYTHONPATH`. `accelerate` is not needed.
- `safetensors`.
- Checkpoint `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`. Only these parts
  are read:
  - `transformer/` and `transformer_ref/` (14 shards each, about 62 GB each). Download them with
    `hf download --local-dir`, so the Hub metadata is kept: the runner identifies each partition
    from it and refuses the wrong one.
  - `vae/` (10.4 GB).
  - `FL2VA/audio_vae/` in the Hub layout (`config.json`, `metadata.json`, `model.safetensors`).
    FL2VA and Ref2VA ship the same audio VAE bytes.

  ```bash
  hf download MiniMaxAI/MiniMax-H3 --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
      --local-dir <CKPT> --include 'transformer/*' 'transformer_ref/*' 'vae/*' 'FL2VA/audio_vae/*'
  ```
- Text features: `fl2va-text.safetensors` (2.7 MB) and `ref2va-text.safetensors` (84 MB). They
  are in `<THIS_DIR>/text/`, together with their `*.sha256`, so `TEXT=<THIS_DIR>/text`.
  `<THIS_DIR>/text-sha256/` holds the same sha256 for
  checking.
  - They were pinned on the B200 host with Diffusers' own `MiniMaxH3FL2VATextEncoderStep` /
    `MiniMaxH3Ref2VATextEncoderStep` and the released Qwen3-VL (BF16, layer 50).
  - The runner checks each file's sha256 (`<TEXT>/*.sha256`, copies of `<THIS_DIR>/text-sha256/`).
  - It also checks that each file belongs to the fixture request, and refuses any other.
  - Do not re-encode the text: both arms must read the same bytes.

```bash
export SRC=<SRC> PY=<python> CKPT=<CKPT> AUDIO_VAE=<CKPT>/FL2VA/audio_vae PROVIDER=<PROVIDER> TEXT=<TEXT>
G=<THIS_DIR>/run_h100_full_chain.sh
```

## Items

### 0. Environment and tests

```bash
$G env   <OUT>/00
$G tests <OUT>/00
```

Expected:

- `environment.txt` shows capability `(9, 0)`, `_C` inside `<SRC>`, and
  `h3_full_attention True`. If the last is `False`, use the query-chunk fallback in item 1.
- `tests.txt` shows 40 passed and 0 skipped. On B200 it was 40 passed with Diffusers
  importable, and the Euler step is bitwise equal to `MiniMaxH3Scheduler` on CPU and CUDA.

### 1. Strict FL2VA: repeat and injected drift

`--offload` keeps the VAE and the transformer on the CPU while the other one runs. That is
peak 66 GB plus activations; FL2VA is 5,041 packed rows.

```bash
O=<OUT>/01
$G run $O fl2va A --mode strict --offload --keep-tensors \
    --capture 1:transformer_blocks.24,1:transformer_blocks.25,1:transformer_blocks.26 --tensors $O/fl2va-A.safetensors
$G run $O fl2va B --mode strict --offload
$G run $O fl2va C --mode strict --offload --keep-tensors \
    --capture 1:transformer_blocks.24,1:transformer_blocks.25,1:transformer_blocks.26 --tensors $O/fl2va-C.safetensors \
    --inject '{"stage": "denoise", "step": 1, "module": "transformer_blocks.25", "row": 3000}'
$G compare $O fl2va-A-B $O/fl2va-A.json $O/fl2va-B.json
$G compare $O fl2va-A-C $O/fl2va-A.json $O/fl2va-C.json \
    --tensors-lhs $O/fl2va-A.safetensors --tensors-rhs $O/fl2va-C.safetensors
```

Expected:

- **Run A report.** `execution.case_id = "R/R"`, `bound = true`, `fell_back = null`.
  `boundary_bindings` has eight `row` entries (test-h3 `10552d82` includes #700).
  `actual_backends` lists the CUDA ops.
- **`cmp-fl2va-A-B.json`.** `first_drift = null`; every stage has `equal = true`;
  `gate.hard_gate = "pass"`; `transformer_intermediates_equal = true`; decoded media equal.
- **`cmp-fl2va-A-C.json`.**
  - `first_drift.stage = "denoise"`, `step = 1`.
  - `transformer.module = "transformer_blocks.25"`, `transformer.first_row = 3000`,
    `differing_rows = 1`.
  - `transformer.row` is target video, latent frame 18, patch (6, 3). The same row on B200 had
    `position_thw` `[359.0, 22.047…, 3.905…]`.
  - `gate.hard_gate = "fail"`.
  - Velocities and latents may stay equal if the one-ULP change is rounded away before the
    output heads. The report then has `trajectory = null` and says so. That is a valid outcome:
    record which one happened.

### 2. Strict Ref2VA: repeat and injected drift

Ref2VA is 26,139 packed rows: an image reference at its 2048 px short edge, a 22-frame
768x1344 video with a soundtrack, an audio reference, and the 256x448 target. Use `--offload`
first. If the strict run runs out of memory, retry the same three runs with `--stream-blocks`
instead of `--offload`.

- `--stream-blocks` keeps the blocks in pinned host memory and moves one block at a time to the
  GPU: same weight bytes, same kernels.
- It is checked on B200 for provider runs only; for strict it is new. If it fails, report the
  error, then try `--offload --attention-query-chunk 4096`.

```bash
O=<OUT>/02
$G run $O ref2va A --mode strict --offload --keep-tensors \
    --capture 1:transformer_blocks.24,1:transformer_blocks.25,1:transformer_blocks.26 --tensors $O/ref2va-A.safetensors
$G run $O ref2va B --mode strict --offload
$G run $O ref2va C --mode strict --offload --keep-tensors \
    --capture 1:transformer_blocks.24,1:transformer_blocks.25,1:transformer_blocks.26 --tensors $O/ref2va-C.safetensors \
    --inject '{"stage": "denoise", "step": 1, "module": "transformer_blocks.25", "row": 20000}'
$G compare $O ref2va-A-B $O/ref2va-A.json $O/ref2va-B.json
$G compare $O ref2va-A-C $O/ref2va-A.json $O/ref2va-C.json \
    --tensors-lhs $O/ref2va-A.safetensors --tensors-rhs $O/ref2va-C.safetensors
```

Expected: as in item 1, except for the row. Packed row 20000 is a *reference video condition*
row (`role = "condition"`, `condition_row = 11683` on B200), not a target row. On B200 its
one-ULP change at block 1 was rounded away before the output heads. The probe localized it:
step 1, the injected block, row 20000. `trajectory` was `null`, the hard gate passed and
`transformer_intermediates_equal` was `false`. Either outcome is valid; record which one
happened.

- If a stage before `denoise` differs between A and B, that is a finding. For example,
  `visual_vae_encode` for the 2048 px image, which encodes in tiles; report it with its
  `first_index`.
- The run refuses `transformer/` for Ref2VA and `transformer_ref/` for FL2VA (H16). Do not
  work around this.

### 3. Strict versus provider on the same card (informational)

```bash
O=<OUT>/03
$G run $O fl2va P --offload            # provider arithmetic, same request
$G run $O ref2va P --offload
$G compare $O fl2va-strict-vs-provider <OUT>/01/fl2va-A.json $O/fl2va-P.json
$G compare $O ref2va-strict-vs-provider <OUT>/02/ref2va-A.json $O/ref2va-P.json
```

Expected:

- Every stage before `denoise` is equal, because both arms use the same contracts on the
  same card.
- The first drift is in `denoise` at step 0, at the first module the strict graph computes
  differently from the provider; it is likely `proj_in` or `context_embedder`.
- Record the module, `max_abs_diff`, and whether the decoded media still differ.
- No verdict is attached. This is the R/R versus P/P localization the RFC asks reports to
  give.

### 4. H100 against the B200 artifacts (informational)

`<THIS_DIR>/b200-reference/{fl2va,ref2va}-A.json` are the B200 provider runs (all 50 blocks).

```bash
O=<OUT>/04
$G compare $O fl2va-h100-vs-b200 <OUT>/03/fl2va-P.json <THIS_DIR>/b200-reference/fl2va-A.json
$G compare $O ref2va-h100-vs-b200 <OUT>/03/ref2va-P.json <THIS_DIR>/b200-reference/ref2va-A.json
```

Expected:

- `request` through `text_conditions` are equal: CPU preprocessing and the pinned text bytes.
- The first drift is most likely `visual_vae_encode`. The VAE's cuDNN and SDPA kernels are
  architecture-specific, and the visual VAE row does not claim bits across GPU generations.
- Record what it is. It is not a failure.

### Drift maps (optional)

For any compare above, `tools/validation/models/h3_full_chain.py plot LHS.json RHS.json --png
X.png --report X.json` draws the per-step digest equality (one cell per module output and per
trajectory tensor). Send the PNGs of items 1–3.

## Rules

- Never change a bound, a runner or the code to make a result pass. A failure is reported
  with its log.
- No PR, no PR edit, no comment. The only push is the results branch below.
- Anything that may leave the machine is redacted. That means no cluster, job, node,
  partition, user or personal paths. The artifacts record versions, device names and
  compute capability only. Check the logs before sending them.
- The `*.safetensors` tensor files (1–2 GB each) stay on the machine. Send them only if a
  compare asks for a second look.

## Send back

`<OUT>/00` … `<OUT>/04`, without `*.safetensors` and `tmp/`, plus a `SUMMARY.md` with one row
per item:

- the `first_drift` (stage, step, module, row);
- `gate.hard_gate`;
- `transformer_intermediates_equal`;
- decoded-media equality;
- `execution.case_id`, `bound` and `fell_back`;
- the peak memory, if `nvidia-smi` showed it.

Push them as fork branch `handoff/h3-h100-phase-b-full-chain-results-20261118` (folder
`h100-results/phase-b-full-chain-20261118/`), redacted as above.
