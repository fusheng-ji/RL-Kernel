# H100 handoff: `qwen3vl_layer50_contract` (RFC #420 Phase B, MiniMax-H3), 2026-11-18

Hardware: one H100 80 GB (sm90) with about 72 GB free. ROCm is out of scope. Paths are
placeholders (`<REPO>`, `<CKPT>`, `<PROVIDER>`, `<OUT>`).

## Why this is needed

The contract was validated on B200 (sm100) only. Nothing in it is Hopper-gated: no
`cc == (9, 0)` check and no strict H3 kernel. The contract is the unmodified HF Qwen3-VL
forward, stopped after decoder layer index 49. The RFC asks for CUDA evidence, and the
cuBLAS GEMMs and the SDPA backend behind that forward differ by architecture. The H100 run
has to show:

1. The contract feature is bitwise equal to Diffusers `get_qwen3vl_prompt_embeds`
   (`hidden_states[50]`) on H100.
2. Run-to-run determinism.
3. Batch-composition independence of `encode_many`.

It does not need H100 bits to equal B200 bits; the contract does not claim that.

## Code

This branch, `handoff/h3-h100-qwen3vl-layer50-20261118`, is `feat/h3-qwen3vl-layer50-contract`
@ `be62fe82` (base `10552d82`, `test-h3`) plus one commit that adds this folder. Run from a
checkout of it:

```bash
git fetch https://github.com/fusheng-ji/RL-Kernel handoff/h3-h100-qwen3vl-layer50-20261118
git worktree add --detach <SRC> FETCH_HEAD
```

The contract is pure PyTorch plus transformers. No extension build is needed, and the `_C`
state does not matter.

## Environment

- torch 2.13.0+cu130, CUDA 13.0, transformers 5.17.0 (the B200 versions). transformers must
  ship `Qwen3VLForConditionalGeneration`. `accelerate` is not needed.
- A Diffusers source checkout at `f53d552036a0d1bd5570782a39cd40cfabf112bc`
  (`<PROVIDER>`): put `<PROVIDER>/src` on `PYTHONPATH`.
- `numpy`, `Pillow`, `matplotlib` (the last one for the charts only).
- The released text encoder, tokenizer and processor from
  `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`, about 67 GB:

```bash
python -c "from huggingface_hub import snapshot_download as s; s('MiniMaxAI/MiniMax-H3', \
  revision='42ed227ee7df40d41602854ae760620d6eb651fe', local_dir='<CKPT>', \
  allow_patterns=['text_encoder/*', 'tokenizer/*', 'processor/*'])"
```

## Run

```bash
cd <SRC>
export RL_KERNEL_H3_QWEN3VL_DIR=<CKPT> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 1. evidence: the full 64-layer model (~66 GB), freed before the 50-layer contract model loads
PYTHONPATH=.:<PROVIDER>/src python tools/validation/models/h3_qwen3vl_evidence.py \
    --root <CKPT> --out <OUT>/evidence-h100.json 2>&1 | tee <OUT>/evidence.log
python tools/validation/models/plot_h3_qwen3vl.py <OUT>/evidence-h100.json --out-dir <OUT> --label H100
# 2. tests (CPU tiny model + released files + released weights)
PYTHONPATH=.:<PROVIDER>/src python -m pytest -q -rs tests/models/minimax_h3/test_h3_qwen3vl_layer50.py \
    2>&1 | tee <OUT>/pytest-qwen3vl.txt
```

If 80 GB is not enough for the full model, run the evidence with `--repeats 2`. The
reference phase keeps only the 66 GB of weights plus small activations.

## Expected

- `evidence.log` prints a `verdict` block. These entries must be `true`:
  - `contract_equals_diffusers_all`;
  - `contract_on_full_equals_diffusers_all`;
  - `hidden_states_50_is_diffusers_all`;
  - `off_by_one_distinguishable_all`;
  - `contract_repeatable_all`;
  - `contract_batch_composition_invariant`.
- `hf_padded_batch_bitwise_all` is informational (prior art). The B200 evidence job has
  not run yet, so there is no B200 reference to compare with.
- pytest: everything passes, with no skips other than the "needs CUDA" ones (there are none
  on a GPU host).

## Send back

`<OUT>/` (evidence JSON, log, both charts, pytest log), plus `nvidia-smi` and `pip list`
(torch, transformers). Strip host names and paths, then push it as fork branch
`handoff/h3-h100-qwen3vl-layer50-results-20261118` (folder `h100-results/qwen3vl-layer50-20261118/`).
No PR and no comment.
