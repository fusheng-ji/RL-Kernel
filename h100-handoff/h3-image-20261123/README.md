# H100 handoff: MiniMax-H3-Image rows of RFC #738 on Hopper (2026-11-23)

## Why

Nine RFC #738 rows are implemented and checked on B200 against the pinned Diffusers `875373ac` (huggingface/diffusers#15230). Each PR also needs a Hopper column. This round supplies it:
- the same tests and parity checks on H100;
- a cross-hardware comparison of every digest that should not depend on the GPU;
- VAE latency and memory on H100.

None of the rows is a PR yet, so the code comes in the bundle. All tasks are **single-GPU**.

**Added later the same day: T5** for a tenth row, `image_reference_conditioning`. It ships in a second, small bundle, `h3-image-t5-20261123.bundle`. T1–T4, the first bundle and their refs are unchanged; if they are already running or done, only add T5.

**Added after that: T6**, for `phase_b_image_full_chain`: the whole T2I and edit chain on the released weights. It ships in a third bundle, `h3-image-t6-20261123.bundle`. It needs about 66 GB more weights (`transformer_diffusers/`) and `accelerate` in `<V>`. T1–T5 are unchanged.

| Task | Code | Hardware | Environment | Time (est.) |
|---|---|---|---|---|
| **T1** suite at the integrated head, plus no new failures in `tests/models` | `refs/h100/h3-image-integrated` and `test-h3` `3f93cae8` | 1 H100 | `<E1>` and `<V>` | about 40 min |
| **T2** cross-environment digest probe against the B200 reference | `refs/h100/h3-image-integrated` | 1 H100 | `<V>` | about 10 min |
| **T3** `image_vae_contract` evidence (parity, tiling, determinism, latency) | `refs/h100/h3-image-vae-contract` | 1 H100 | `<V>` | about 30 min |
| **T4** `image_text_presentation_contract` evidence (sealed layer-50 fixtures) | `refs/h100/h3-image-text-presentation` | 1 H100 80 GB | `<V>` | about 30 min |
| **T5** `image_reference_conditioning` tests, evidence and the real-VAE check | `refs/h100/h3-image-integrated-refcond` | 1 H100 | `<E1>` (build) and `<V>` | about 20 min |
| **T6** `phase_b_image_full_chain` on the released weights, 4 fixtures, plus its tests | `refs/h100/h3-image-phase-b-full-chain` | 1 idle H100 80 GB | `<V>` + `accelerate` | about 30 min, plus the download |

Paths are placeholders, as in the earlier rounds:
- `<REPO>`: a clone of the fork;
- `<THIS_DIR>`: this folder;
- `<OUT>`: where results are written;
- `<E1>`: the environment of the last round (torch 2.13.0+cu130);
- `<V>`: the new venv built in "Environment" below;
- `<IMG>`: the MiniMax-H3-Image download (see "Weights");
- `<CKPT>`: the MiniMax-H3 checkpoint of the earlier rounds, revision `42ed227ee7df40d41602854ae760620d6eb651fe`. Its `text_encoder/` is byte-identical to MiniMax-H3-Image's, and T4 checks this with `--verify-weights`.

## Rules (unchanged from the earlier H3 rounds)

- **Clean trees.** Use detached worktrees. `git status --porcelain --untracked-files=no` must be empty before and after every step. Change no file.
- **Check `_C`.** Before every GPU step, check that `rl_engine._C.__file__` is inside the checkout. (The image rows do not use `_C`, but `tests/models` does.)
- **Idle hardware.** Use an idle GPU with nothing else on it, and record `nvidia-smi` at the start of each task.
- **Report failures as they are.** A failure is reported as a failure, with its log. Never change a script, a test or a bound to make a result pass. Failed attempts go, as they are, into `not-green-attempts/`.
- **Push only the results branch.** Use `handoff/h100-h3-image-results-20261123`. No PR, no comment.
- **Redact everything sent back.**
  - Remove: cluster, job, node, partition, user, conda environment name, `/home/…` or `/data/…` path, email. Hosts become `node-a`.
  - `git add -f` the `*.log` files.
  - Commit with `git commit -s`, with no AI attribution lines.

## Code

```bash
cd <REPO>
git fetch https://github.com/fusheng-ji/RL-Kernel.git handoff/h100-h3-image-20261123
git fetch https://github.com/RL-Align/RL-Kernel.git test-h3      # needs 3f93cae8b
git checkout FETCH_HEAD -- h100-handoff/h3-image-20261123        # or read it from the branch
git bundle verify <THIS_DIR>/h3-image-20261123.bundle
git fetch <THIS_DIR>/h3-image-20261123.bundle 'refs/h100/*:refs/h100/*'
# T5 only, after the first bundle:
git bundle verify <THIS_DIR>/h3-image-t5-20261123.bundle
git fetch <THIS_DIR>/h3-image-t5-20261123.bundle 'refs/h100/*:refs/h100/*'
# T6 only, after the T5 bundle:
git bundle verify <THIS_DIR>/h3-image-t6-20261123.bundle
git fetch <THIS_DIR>/h3-image-t6-20261123.bundle 'refs/h100/*:refs/h100/*'
```

`refs.txt` lists every ref with its commit. The first bundle's only prerequisite is `test-h3` `3f93cae8b`. The T5 bundle needs `e22c9d26` and `89061319`, both from the first bundle. The T6 bundle needs `be983dcc` from the T5 bundle.

| Ref | Commit | Row |
|---|---|---|
| `refs/h100/h3-image-arch-fingerprint` | `1ed43e3f` | `h3_image_arch_fingerprint` (base of every other ref) |
| `refs/h100/h3-image-position-grid` | `72548238` | `image_position_grid` |
| `refs/h100/h3-image-evidence-schema` | `6d8e5c3b` | `h3_image_evidence_schema` (on the position grid) |
| `refs/h100/h3-image-canvas-contract` | `a030ff8b` | `image_canvas_contract` |
| `refs/h100/h3-image-packed-layout` | `fb79ec91` | `image_packed_layout_schema` (on the position grid) |
| `refs/h100/h3-image-patch-pack` | `e22c9d26` | `image_patch_pack_unpack` (on the packed layout) |
| `refs/h100/h3-image-vae-contract` | `a8f05c22` | `image_vae_contract` + `image_vae_tiling_contract` |
| `refs/h100/h3-image-text-presentation` | `b1617e59` | `image_text_presentation_contract` |
| `refs/h100/h3-image-integrated` | `89061319` | all of the above merged (merge commits only, no new code) |
| `refs/h100/h3-image-reference-conditioning` | `107726a7` | `image_reference_conditioning` (on the patch pack; T5 bundle) |
| `refs/h100/h3-image-integrated-refcond` | `be983dcc` | `h3-image-integrated` plus the merge of the row above (T5 bundle) |
| `refs/h100/h3-image-phase-b-full-chain` | `b4ef2ddc` | `phase_b_image_full_chain` on `h3-image-integrated-refcond` (T6 bundle) |

## Environment

The image rows need Diffusers at the commit that added MiniMax-H3-Image. Install it into a **new** venv on top of `<E1>` and leave `<E1>` untouched:

```bash
<E1>/bin/python -m venv --system-site-packages <V>
<V>/bin/pip install --no-deps "git+https://github.com/huggingface/diffusers@875373ac7f5856071e345a3e394c74efccc6b9e6"
<V>/bin/python -c "import diffusers; from diffusers import MiniMaxH3ImageTransformer2DModel, AutoencoderKLMiniMaxH3Image; print(diffusers.__version__)"   # 0.42.0.dev0
<V>/bin/python -m pip list > <OUT>/pip-list.txt
```

`reference/pip-list-b200-filtered.txt` holds the B200 versions (torch 2.13.0, numpy 2.3.5, Pillow 12.3.0, transformers 5.17.0). Importing Diffusers can take a few minutes on a network filesystem; that is expected.

## Weights

`<IMG>` is the part of `MiniMaxAI/MiniMax-H3-Image` at revision `8ced60068623def5b74b4d10558117cb61d6482d` that the rows read, about 2.5 GB. The transformer and text-encoder weights are not needed: T4 reads the text encoder from `<CKPT>`.

```bash
<V>/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    "MiniMaxAI/MiniMax-H3-Image",
    revision="8ced60068623def5b74b4d10558117cb61d6482d",
    local_dir="<IMG>",
    allow_patterns=[
        "*.json", "processor/*", "tokenizer/*", "scheduler/*", "vae_diffusers/*",
        "pdd_fused_heads.safetensors",
    ],
)
EOF
```

The SHA-256 of `vae_diffusers/diffusion_pytorch_model.safetensors` must be `5bb2809698c3989033f719ece2243f277fefedef1dd9a1d05f3d4136b2f93b35`. T3's `--verify-sha256` checks it.

---

## T1. Suite at the integrated head (`<E1>`, `<V>`)

```bash
git worktree add --detach <W> refs/h100/h3-image-integrated
git worktree add --detach <B> 3f93cae8b
for R in <W> <B>; do (cd $R && RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 <E1>/bin/python setup.py build_ext --inplace); done
cd <W>
export RL_KERNEL_H3_IMAGE_DIR=<IMG> RL_KERNEL_H3_IMAGE_VAE_DIR=<IMG>/vae_diffusers RL_KERNEL_H3_IMAGE_TEXT_ENCODER_ROOT=<CKPT>
# a. image rows, with the pinned Diffusers and the released weights
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W> <V>/bin/python -m pytest -q -p no:cacheprovider -rs \
  tests/models/minimax_h3_image > <OUT>/T1/image-diffusers.txt 2>&1
# b. image rows without Diffusers (the CI view)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W> <E1>/bin/python -m pytest -q -p no:cacheprovider -rs \
  tests/models/minimax_h3_image > <OUT>/T1/image-nodiffusers.txt 2>&1
# c. fingerprint golden
PYTHONPATH=<W> <E1>/bin/python tools/validation/models/h3_image_fingerprint.py check > <OUT>/T1/fingerprint.txt 2>&1
# d. tests/models at head and at base, with RL_KERNEL_H3_WEIGHTS set as in the earlier rounds
for R in <W> <B>; do (cd $R && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R <E1>/bin/python -m pytest -q -p no:cacheprovider --tb=no -rf \
  tests/models > <OUT>/T1/models-$(basename $R).txt 2>&1); done
```

**Expected:**
- **a.** Nothing fails. On B200 this was `320 passed`, with 0 skipped. That includes the 3 CUDA tests of `test_h3_image_patch_pack.py` and the released-weight VAE and text tests (the text test needs about 70 GB of free GPU memory). Any skip on H100 is listed with its reason; send it as it is.
- **b.** Nothing fails. The Diffusers-gated tests skip. On B200 this was `264 passed, 56 skipped`.
- **c.** `ok: 0eefe73a8ffa69ac3a4cf8bebcf3b770d3bbf48c21320df1fe72f2ec5ad02579`.
- **d.** No test fails only at head. Send both files; the comparison is made here.

## T2. Cross-environment digest probe (`<V>`)

```bash
cd <W>
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W> <V>/bin/python <THIS_DIR>/h3_image_cross_env_probe.py \
  --out <OUT>/T2/probe-h100.json --compare <THIS_DIR>/reference/probe-b200-89061319-v2.json \
  > <OUT>/T2/probe.log 2>&1; echo "exit=$?" >> <OUT>/T2/probe.log
```

The probe hashes the exact bytes produced by:
- the fingerprint;
- 1,425 position grids (float64 and their float32 cast);
- 46,117 canvas plans and 4 LANCZOS resizes;
- 3 packed layouts and one permutation of each;
- patch-pack rows of 5 shapes x 3 dtypes on CPU and on CUDA;
- the sealed test-loop trajectories.

It also measures the permutation drift of a tiny Diffusers transformer on CUDA in float64 and float32.

**Expected:**
- `hardware-independent digests differing: 0`, exit 0.
- Every `cuda_equals_cpu` and `cuda_round_trip` is `True`.
- With Pillow 12.3.0 the resize digests also match; with another Pillow version a resize difference is expected and only reported.
- `permutation_drift` is reported, not compared: `repeat_bitwise_equal` true, and `max_abs` small (B200 values are in the reference JSON).

A hardware-independent digest that differs is the main finding of T2. Send it as is.

*Round 1 (H100, Pillow 13.0.0) found `canvas.plans` differing. That digest hashes the plan artifacts, and a plan that resizes records the Pillow version, so it is Pillow-dependent. Recomputing it here with the version string set to 13.0.0 reproduces the H100 value exactly. The probe now lists `canvas.plans` as Pillow-dependent and adds `canvas.plans_geometry` (the same artifacts without the `pillow` field), which must match everywhere. The reference is `probe-b200-89061319-v2.json`, from the same commit; the old reference is kept.*

## T3. `image_vae_contract` evidence (`<V>`)

```bash
git worktree add --detach <W3> refs/h100/h3-image-vae-contract
cd <W3>
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W3> RL_KERNEL_H3_IMAGE_VAE_DIR=<IMG>/vae_diffusers <V>/bin/python -m pytest -q -p no:cacheprovider -rs \
  tests/models/minimax_h3_image > <OUT>/T3/pytest-image-vae.txt 2>&1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W3> <V>/bin/python tools/validation/models/h3_image_vae_evidence.py \
  --vae-dir <IMG>/vae_diffusers --out <OUT>/T3/evidence-h100-a8f05c22.json --verify-sha256 --timing-repeats 7 \
  > <OUT>/T3/evidence.log 2>&1
PYTHONPATH=<W3> <V>/bin/python tools/validation/models/h3_image_vae_evidence.py --check-manifest >> <OUT>/T3/evidence.log 2>&1
```

No charts are needed from the H100 side; they are made here from the JSON.

**Expected** (B200 values are in `reference/vae-evidence-b200-a8f05c22f.json`):
- **Verdicts.** Every `verdicts` entry is true except `diffusers_batched_batch_invariant_default_flags`. That one is false on B200 too: the Diffusers batched encode is not batch-invariant, and the contract's per-row path is.
- **Bitwise parity on H100.** `diffusers_parity_pinned`, `diffusers_parity_default_flags`, `run_to_run`, `contract_batch_invariant`, `tiling` and `normalization_cpu_cuda_bitwise` hold.
- **Latency and memory** are recorded at the 2048² canvas. The B200 medians are: tiled decode 1486 ms, untiled 860 ms; reference encode 264 ms tiled, 103 ms untiled. B200 peak memory is 9.4 GiB.

Whether the H100 output digests equal the B200 ones is a finding either way; the comparison is made here. Send the JSON and the log.

## T4. `image_text_presentation_contract` evidence (`<V>`)

```bash
git worktree add --detach <W4> refs/h100/h3-image-text-presentation
cd <W4>
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W4> <V>/bin/python tools/validation/models/h3_image_text_presentation.py evidence \
  --root <IMG> --text-encoder-root <CKPT> --verify-weights --out <OUT>/T4/evidence-h100-b1617e591.json \
  > <OUT>/T4/evidence.log 2>&1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W4> RL_KERNEL_H3_IMAGE_DIR=<IMG> RL_KERNEL_H3_IMAGE_TEXT_ENCODER_ROOT=<CKPT> \
  <V>/bin/python -m pytest -q -p no:cacheprovider -rs tests/models/minimax_h3_image > <OUT>/T4/pytest-h3-image.txt 2>&1
```

**Expected** (B200 values are in `reference/text-evidence-b200-b1617e591.json`):
- **Equality with Diffusers.** `all_equal_diffusers` is true: on H100, every case (`t2i`, `t2i_truncated`, `edit`, `edit_swapped`) equals the pinned Diffusers step bitwise, for both the full 64-layer model and the 50-layer stack.
- **Sealed fixtures.** `all_cpu_digests_match_sealed` is true.
- **Weights.** `--verify-weights` confirms that `<CKPT>/text_encoder` has the released hashes.
- **Memory.** The B200 peak for the 50-layer stack was 50.6 GiB.
- **Pytest.** `test_release_fixtures_bitwise_against_diffusers` runs and passes. It needs about 70 GB of free GPU memory and skips with that reason otherwise, so use an idle 80 GB GPU. H100 feature digests are not sealed yet, so on H100 the test checks equality with Diffusers but has no sealed digest to compare against. Everything else passes.

## T5. `image_reference_conditioning` (`<E1>` build, `<V>`)

The row noises the encoded edit references to `t = 0.999` through the registered #420 `conditioning_noise_mix` operator and records every draw from the request generator. T5 needs `_C` built in its own checkout: the CUDA kernel arm of the tests uses it.

```bash
git worktree add --detach <W5> refs/h100/h3-image-integrated-refcond
cd <W5>
RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 <E1>/bin/python setup.py build_ext --inplace
PYTHONPATH=<W5> <E1>/bin/python -c "import rl_engine._C as c; print(c.__file__, hasattr(c, 'conditioning_noise_mix_forward'))" > <OUT>/T5/ext.txt
# a. tests, with the pinned Diffusers
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W5> <V>/bin/python -m pytest -q -p no:cacheprovider -rs \
  tests/models/minimax_h3_image/test_h3_image_reference_conditioning.py > <OUT>/T5/pytest-refcond.txt 2>&1
# b. fixture digests per execution arm, and the probe arms
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W5> <V>/bin/python tools/validation/models/h3_image_reference_conditioning.py \
  --device cuda --out <OUT>/T5/refcond-evidence-h100-107726a79.json > <OUT>/T5/refcond-evidence.log 2>&1; echo "exit=$?" >> <OUT>/T5/refcond-evidence.log
# c. released VAE + this row against the full pinned Diffusers edit blocks
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W5> <V>/bin/python <THIS_DIR>/h3_image_refcond_real_vae_check.py \
  --vae-dir <IMG>/vae_diffusers --out <OUT>/T5/refcond-real-vae-h100-be983dcca.json > <OUT>/T5/refcond-real-vae.log 2>&1; echo "exit=$?" >> <OUT>/T5/refcond-real-vae.log
```

**Expected** (B200 values are in `reference/refcond-evidence-b200-107726a79.json` and `reference/refcond-real-vae-b200-be983dcca.json`):
- **ext.txt.** The path is inside `<W5>`, and the second field is `True`.
- **a.** `46 passed`, 0 skipped. That includes the 7 CUDA tests, 2 of which are the CUDA-kernel arm. A skip names its reason; send it as it is.
- **b.** `exit=0` and `arms not equal to the CPU reference or failing the fixed-row loop: none`.
  - **Rows from a CPU generator** should not depend on the GPU: arms `cpu`, `cuda-cpu_generator` and `cuda_kernel-cpu_generator`. Their `rows_sha256` should equal the B200 values for all four fixtures.
  - **The `probes` block** is computed on the CPU and should equal the B200 block exactly.
  - **Rows from a CUDA generator** (`*-cuda_generator`) may differ from B200, because the CUDA generator's output can depend on the device. Those are reported, not compared. Within the H100 file, `cuda-cuda_generator` and `cuda_kernel-cuda_generator` should give the same digest.
- **c.** `exit=0` and `all_green True`. In all 4 arms, the reference rows, target rows and final generator state equal Diffusers bitwise, and the rows stay fixed over the six steps. The four arms are default and pinned flags, each with a CPU and a CUDA generator. `rows_sha256` there depends on the GPU's VAE convolutions and is only reported.

## T6. `phase_b_image_full_chain` on the released weights (`<V>`)

The chain runs one T2I or edit request on the RL-Kernel contracts, from the prompt and reference images to the output bytes, and seals one artifact per stage. The pinned Diffusers modular pipeline runs block by block on the same loaded weights, and the two artifacts are compared stage by stage. The runner works in two phases:
1. It loads the 64-layer Qwen3-VL once and computes the text for both arms, then frees it.
2. It loads the transformer and the VAE.

The four fixtures are the release's own examples:
- `t2i_example`: `examples/request.t2i.json`, 4736x3552;
- `t2i_upscaled`: the same prompt at 1024x768, so the canvas is upscaled and resized back;
- `edit_example`: the `examples/edit.py` defaults: `reference1.png`, 4096x4096;
- `edit_two_refs`: `reference1.png` plus `assets/showcase/photo.jpg`, 2048x2048.

Extra setup, once. `from_pretrained` keeps the checkpoint's FP32 modules only when `accelerate` is installed:

```bash
<V>/bin/pip install --no-deps accelerate==1.16.0
<V>/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('MiniMaxAI/MiniMax-H3-Image', revision='8ced60068623def5b74b4d10558117cb61d6482d', local_dir='<IMG>', allow_patterns=['transformer_diffusers/*', 'examples/*', 'assets/showcase/photo.jpg'])"
```

`transformer_diffusers/` is about 66 GB.

```bash
git worktree add --detach <W6> refs/h100/h3-image-phase-b-full-chain
cd <W6>
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W6> <V>/bin/python tools/validation/models/h3_image_full_chain.py \
  --root <IMG> --text-encoder-root <CKPT> --out <OUT>/T6/run --repeat > <OUT>/T6/evidence.log 2>&1; echo "exit=$?" >> <OUT>/T6/evidence.log
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W6> <V>/bin/python -m pytest -q -p no:cacheprovider -rs \
  tests/models/minimax_h3_image/test_h3_image_full_chain.py > <OUT>/T6/pytest-full-chain.txt 2>&1
```

**Expected** (B200 values are in `reference/phase-b-summary-b200-b4ef2ddc4.json`):
- **evidence.log.** `exit=0` and `all_gates_pass True`.
- **Per fixture.** `gate` and `secondary_equal` are true and `first_drift` is null: RL-Kernel equals Diffusers at every stage on the same GPU, output bytes included. `run_to_run_equal` is true.
- **Memory.** Peak memory on B200 was 64.7–67.2 GiB per fixture, so use an idle 80 GB card. Out of memory is a finding; send it as it is.
- **Time.** On B200 the whole job took 8 minutes. Per fixture, the RL-Kernel chain took 3–15 s and Diffusers took 4–19 s.
- **Digests.** H100 `final_latent_sha256` and `output_sha256` are expected to differ from B200, because BF16 matmuls are hardware-specific. The comparison is made here.
- **Pytest.** `26 passed`.

Send back `summary.json`, the eight `*.json` artifacts and both logs. Do **not** commit the `*.png` files: they are about 110 MB, and their bytes are hashed in the artifacts.

## Sending results back

```
results/
  SUMMARY.md       one line per task: command, exit code, headline numbers
  env.txt          torch / CUDA / driver / GPU name; nvidia-smi at the start of each task
  pip-list.txt     <V>, redacted
  T1/ T2/ T3/ T4/ T5/ T6/  every file named above (T6: no PNGs)
  not-green-attempts/
```

Push to `handoff/h100-h3-image-results-20261123` on the fork.
