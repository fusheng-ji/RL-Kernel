# qwen3vl_layer50_contract on H100 (2026-11-18 handoff)

Result: **NOT as expected.** `h3_qwen3vl_evidence.py` ran out of GPU memory in stage 7, so there is no
`evidence-h100.json`, no `verdict` block and no charts. pytest passed. Nothing in the handoff branch was changed.

| Step | Result |
|---|---|
| `h3_qwen3vl_evidence.py` (default `--repeats 5`) | CUDA OOM at `load_release_text_encoder(...)` (stage 7, the 50-layer contract model). Log: `attempt1-repeats5/evidence.log` |
| `h3_qwen3vl_evidence.py --repeats 2` (HANDOFF fallback) | Same OOM at the same place. Log: `evidence.log` |
| `plot_h3_qwen3vl.py --label H100` | Not possible, no evidence JSON (`FileNotFoundError`). Logs: `plot.log`, `attempt1-repeats5/plot.log` |
| pytest `test_h3_qwen3vl_layer50.py` | 50 passed, 0 failed, 0 skipped (`pytest-qwen3vl.txt`; first run also 50 passed) |
| 6 verdict entries | Not produced (the script died before writing them) |

## Observed

- GPU: NVIDIA H100, 95830 MiB each (see `nvidia-smi.csv`). The OOM message says 92.24 GiB were allocated by
  PyTorch at the failing allocation. Stages 1-6 (the full 64-layer model, about 66 GB) finished; the failure is when
  the 50-layer contract model is loaded afterwards.
- So the full model is still resident when stage 7 loads. The script does `del full, hs` before stage 7, but in the
  stage 2 loop `fn = lambda ids=ids, vision=vision, full=full: ...` binds `full` as a default argument, and `fn` is
  still alive after the loop. This is a reading of the code, not a verified fix. I did not modify or patch anything.
- `--repeats 2` does not help, because the memory is the retained model, not the repeats.
- On a card with 120 GB+ free this would probably pass; on 80 or 94 GB it cannot.

## Environment

- torch 2.13.0+cu130, transformers 5.17.0 (the versions named in HANDOFF); see `pip-list-torch-transformers.txt`.
- Diffusers source checkout at `f53d552036a0d1bd5570782a39cd40cfabf112bc`. Released checkpoint
  `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe` (`text_encoder/`, `tokenizer/`, `processor/`,
  downloaded earlier with Hub metadata, revision confirmed from the metadata).
- Host names, paths and the job id are replaced with `{...}` placeholders.
