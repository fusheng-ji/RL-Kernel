# H100 results: `audio_vae_contract` (#722) at `25d95ccd`, handoff `cb058e1c` (2026-11-22)

**Verdict: every item of "Expected" holds.**
- **pytest:** 60 passed. matplotlib is installed, so the two chart tests ran.
- **`ours`:** batch invariant and bitwise across all four clip lengths, and its cold start is bitwise.
- **vLLM-Omni encode cold start:** now bitwise on H100 (`encode_first_vs_third_bitwise: true`, max abs 0.0). This replaces the old H100 number, 5.8e-06.
- **The two processes:** they agree byte for byte.
- **Leftovers:** no cold-start directory is left.
- **B200:** none of the H100 hashes equals the B200 hashes, which the contract allows.

## Setup

- **Source:**
  - a detached, clean worktree at `25d95ccd` (`feat/h3-audio-vae-contract`), not the handoff commit;
  - `commit.txt` holds HEAD `25d95ccd84ee4490aab788c205c1d8f67432a769` and tree `dafaa56b8088a274fd233bafe1d54e912cc2f174`;
  - both JSONs record `provenance.source` = `25d95ccd…` with `clean: true`.
- **Hardware:** one H100 (GPU 0 of `node-a`). The node's other three GPUs were idle, and only this run's processes were on the node's GPUs (`nvidia-smi-before-*.txt`, `other-processes-*.txt`).
  - Every step ran alone, one after another: pytest, `run-a`, `run-b`, then the Markdown export. The two benchmark runs did not overlap (`steps.txt`).
- **`<E1>`:** torch 2.13.0+cu130, CUDA 13.0, cuDNN 9.20.0 (`env.txt`; `pip-list-filtered-E1.txt`, which now also lists matplotlib 3.11.2).
- **`<AUDIO_VAE>`:** the last round's `<CKPT>/FL2VA/audio_vae`, the full released directory. Its `model.safetensors` sha256 is `37dddc2f…`, equal to the remote one in both JSONs.
- **`<PROVIDER>`:** Diffusers `f53d5520`, clean `src/`.
- **No SGLang.**
- **`_C`:** built in the worktree; `rl_engine._C.__file__` was inside it after the build and before every step (`c-file*.txt`).
- **Tree status:** `git status --porcelain --untracked-files=no` was empty before and after the build and every step (`status-*.txt`, `status-final-count.txt` = 0).

## Expected, item by item

| # | Expected | Result | Verdict |
|---|---|---|---|
| 1 | pytest: 58 passed, or 60 with matplotlib; nothing fails | `60 passed, 4 warnings` (`pytest-h100.txt`, `pytest-audio-vae-h100.xml`); no skips | holds |
| 2 | `schema` v2; `provenance.source` `25d95ccd…`, `clean: true`; remote checkpoint sha256 = local | both JSONs: `rlkernel.h3.audio_vae_bench.v2`, `25d95ccd84ee…`, `clean: true`; `37dddc2f…` = `37dddc2f…` | holds |
| 3 | `ours`, every case: `encode` / `decode` `batch_invariant: true`, every `bitwise_equal: true`, every `max_abs` 0; `cold_start.ours` both `*_bitwise: true` | 4 cases (22, 121, 241, 481 video frames; 0.9 / 5 / 10 / 20 s): no exception in either JSON; `cold_start.ours` encode and decode true | holds |
| 4 | `cold_start["vllm-omni"].encode_first_vs_third_bitwise` (B200: true) | `true`, max abs 0.0, in both runs; decode also true | holds; replaces H100's old 5.8e-06 |
| 5 | `cases[*].rows_sha256` and `paths.*.{encode,decode}.sha256` equal between the two JSONs | 28 hashes compared (4 x rows, 4 x 3 paths x 2 directions): all equal | holds |
| 6 | no `h3-audio-vae-cold-start-*` directory left in `<OUT>` | none after pytest, `run-a`, `run-b` and the export (`leftovers-after-*.txt`) | holds |
| 7 | not expected: H100 hashes equal to B200's | against #722's B200 JSON (`53b944ed`, linked from the PR on the fork's `pr_assets` commit `44af60a9`): 0 of 28 hashes equal; the checkpoint sha256 is equal | holds: per-platform bytes, as the contract claims |

`check.txt` (`scripts/check_avae.py.txt`) has the item-by-item output.

## Other numbers (H100, from `tables-h100.md`)

- **Batch invariance.**
  - Diffusers is not batch invariant at any length.
  - vLLM-Omni's encode is invariant at 0.9 s only: max abs 1.1e-06 to 4.1e-06 at the longer clips. Its decode is invariant at no length.
  - `ours` is `= 0` everywhere.
- **Parity with `ours`.**
  - The vLLM-Omni encode is bitwise equal to `ours` in 3 of 4 cases; its decode in 0 of 4.
  - Diffusers is bitwise equal in 0 of 4 cases for both directions.
- **Cold start.** First call against third call is `= 0` for all three paths, both directions.
- **Latency** (median ms, `--repeats 10`):

  | Path | 0.9 s | 20 s |
  |---|---|---|
  | `ours` encode | 63.8 | 143.0 |
  | `ours` decode | 84.5 | 248.4 |
  | Diffusers encode | 7.1 | 68.2 |
  | Diffusers decode | 45.5 | 182.0 |
  | vLLM-Omni encode | 57.9 | 146.6 |
  | vLLM-Omni decode | 48.3 | 180.4 |

## Deviations

1. **Build flags.** `setup.py build_ext --inplace` ran with `RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1`, as in the last H3 rounds. The README's line has no environment variables; the audio VAE uses no `_C` kernel. The build log is `build.log`.
2. **Run time.** The whole run (pytest, both benchmark runs and the export) took about 6 minutes, not the README's 45.
3. **The B200 reference for item 7.** It is not in the handoff or the source tree. It was read from the JSON that #722's description links (`pending/h3-audio-vae-contract/r2-53b944ed/h3-audio-vae-b200.json` at `44af60a9`) and is not part of `results/`.
4. **Extra records:** `rc.txt`, `steps.txt`, `leftovers-after-*.txt`, `check.txt` and `scripts/`.
5. **Redaction.** It changed no JSON, table or chart: they are byte-identical to the run's output.
