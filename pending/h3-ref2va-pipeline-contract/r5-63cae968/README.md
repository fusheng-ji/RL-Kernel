# ref2va pipeline contract, re-measured at `63cae968` (RL-Align/RL-Kernel#725, 2026-11-20)

The report was re-measured against the pinned Diffusers source, `huggingface/diffusers@f53d552` (0.40.0.dev0). The earlier report in the parent folder imported the installed Diffusers 0.43.0. `environment.provider_source_sha256` records the sha256 of the imported Diffusers files the comparison uses.

| File | What |
|---|---|
| `ref2va-report.json` | `tools/validation/models/h3_pipeline_contract_report.py --row ref2va --charts ...` at `63cae968`, clean tree, on one 8 x B200 node allocated to this job alone (CPU preprocessing). PYTHONPATH holds the checkout and the `f53d552` `src/`. |
| `*.png` | the charts that run wrote from the JSON, one per Results table |
| `tests/pytest-priorart-pinned.txt` | `pytest test_h3_packed_layout.py test_h3_ref2va_pipeline.py test_h3_pipeline_contract_report.py` with the `f53d552` source, SGLang 0.5.21 and accelerate, offline (the tiny-pipeline tests skip: the local cache has no processor) |
| `tests/pytest-priorart-pinned-tiny-hub.txt` | the same files, `-k tiny`, with Hub access |
| `tests/pytest-priorart-pinned-tinyfile-hub.txt` | `pytest test_h3_tiny_pipeline.py`, same environment |
| `tests/pytest-rlkernel.txt` | the first three files in the RL-Kernel environment (no Diffusers or SGLang) |

The tests ran on a shared B200 host, not on the timing node.
