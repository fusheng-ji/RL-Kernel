# H100 handoff r2: `qwen3vl_layer50_contract` (RFC #420 Phase B, MiniMax-H3), 2026-11-19

Rerun of the 2026-11-18 handoff. Run 1 (results branch
`handoff/h3-h100-qwen3vl-layer50-results-20261118` @ `e0c1cfd4`) passed pytest (50 passed) but
the evidence script ran out of GPU memory at stage 7: the full 64-layer model was still
referenced when the 50-layer contract model loaded.

Hardware: one H100 with 95 GB (95830 MiB, the card run 1 used). ROCm is out of scope. Paths are
placeholders (`<REPO>`, `<CKPT>`, `<PROVIDER>`, `<OUT>`).

## What changed since run 1

**Only the memory release in the evidence script.** Head `c07aa5dd`, one commit on top of
`97ba3b82`:

- `tools/validation/models/h3_qwen3vl_evidence.py`: stages 3-6 (the full 64-layer model) move
  into a function that returns plain JSON data only. A new `release_after` drops the model and
  fails with a clear `RuntimeError` if the model is still referenced, or if more than
  `--max-retained-gib` (default 1.0) stays allocated, before the 50-layer stack loads. The
  evidence JSON's `memory` block gains `allocated_after_full_model_free_gib` and
  `peak_allocated_full_model_gib`. No measurement, case or verdict changed.
- `tests/models/minimax_h3/test_h3_qwen3vl_layer50.py`: two CPU tests for that release. The
  test count is now 52 (was 50).
- `rl_engine/` is unchanged, so the contract itself is the code run 1 tested.

Expected peak: one model at a time, about 62 GiB of weights plus activations, so the run fits a
95 GB card.

## Code

Fork branch `handoff/h3-h100-qwen3vl-layer50-r2-20261119`: `feat/h3-qwen3vl-layer50-contract`
@ `c07aa5dd` plus one commit that adds this folder. The tree outside `h100-handoff/` is exactly
`c07aa5dd`.

```bash
cd <REPO>                       # clone of RL-Align/RL-Kernel, remote "upstream"
git fetch upstream test-h3      # must reach 10552d82
git fetch https://github.com/fusheng-ji/RL-Kernel.git \
    '+refs/heads/handoff/h3-h100-qwen3vl-layer50-r2-20261119:refs/heads/handoff/h3-h100-qwen3vl-layer50-r2-20261119'
git worktree add --detach <SRC> handoff/h3-h100-qwen3vl-layer50-r2-20261119
git -C <SRC> log --oneline -2   # the handoff commit on top of c07aa5dd; stop if not
git -C <SRC> diff --stat c07aa5dd HEAD -- . ':!h100-handoff'   # must print nothing
```

No extension build is needed, and the `_C` state does not matter.

## Environment

The same as run 1: torch 2.13.0+cu130, transformers 5.17.0, Diffusers source at
`f53d552036a0d1bd5570782a39cd40cfabf112bc` (`<PROVIDER>`), numpy, Pillow, matplotlib, and the
`MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe` download from run 1 (`<CKPT>`,
`text_encoder/`, `tokenizer/`, `processor/`).

## Run

The same commands as run 1, plus a memory sampler around the evidence step.

```bash
cd <SRC>
export RL_KERNEL_H3_QWEN3VL_DIR=<CKPT> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 0. record GPU memory every 500 ms while the evidence runs
nvidia-smi -i 0 --query-gpu=timestamp,memory.used,memory.total --format=csv -lms 500 \
    > <OUT>/nvidia-smi-memory.csv &
SAMPLER=$!
# 1. evidence: the full 64-layer model, freed (and checked freed) before the 50-layer model loads
PYTHONPATH=.:<PROVIDER>/src python tools/validation/models/h3_qwen3vl_evidence.py \
    --root <CKPT> --out <OUT>/evidence-h100.json 2>&1 | tee <OUT>/evidence.log
kill $SAMPLER
# peak memory.used over the run, in MiB
awk -F', ' 'NR>1 {sub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print "peak_mib", m}' \
    <OUT>/nvidia-smi-memory.csv | tee <OUT>/peak-memory.txt
python - <<'PY' | tee -a <OUT>/peak-memory.txt
import json; print(json.dumps(json.load(open("<OUT>/evidence-h100.json"))["memory"], indent=1))
PY
python tools/validation/models/plot_h3_qwen3vl.py <OUT>/evidence-h100.json --out-dir <OUT> --label H100
# 2. tests (CPU tiny model + released files + released weights)
PYTHONPATH=.:<PROVIDER>/src python -m pytest -q -rs tests/models/minimax_h3/test_h3_qwen3vl_layer50.py \
    2>&1 | tee <OUT>/pytest-qwen3vl.txt
```

If step 1 fails with `full model still referenced` or `GiB still allocated`, that is the new
check firing: send back the log as is, do not retry with other flags.

## Expected

- `evidence.log` prints a `verdict` block. These entries must be `true`:
  - `contract_equals_diffusers_all`;
  - `contract_on_full_equals_diffusers_all`;
  - `hidden_states_50_is_diffusers_all`;
  - `off_by_one_distinguishable_all`;
  - `contract_repeatable_all`;
  - `contract_batch_composition_invariant`.
- `hf_padded_batch_bitwise_all` is informational (prior art).
- `memory.allocated_after_full_model_free_gib` is below 1.0, and the `nvidia-smi` peak stays
  under the 95 GB card size (about 62 GiB of weights plus activations is expected).
- pytest: 52 passed, no skips.

## Send back

Into `h100-results/qwen3vl-layer50-20261118-r2/` on the same results branch
(`handoff/h3-h100-qwen3vl-layer50-results-20261118`):

- `<OUT>/`: `evidence-h100.json`, `evidence.log`, the four `h3-qwen3vl-*-h100.png` charts, `pytest-qwen3vl.txt`,
  `nvidia-smi-memory.csv`, `peak-memory.txt`;
- `nvidia-smi` and `pip list` (torch, transformers).

Strip host names, paths and job ids before uploading anything.
