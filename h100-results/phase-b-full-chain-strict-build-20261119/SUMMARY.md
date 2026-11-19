# phase_b_full_chain strict build on H100 (2026-11-19 follow-up, head `d31727d1`)

H100 (cc 9.0), 95830 MiB per card; torch 2.13.0+cu130, transformers 5.17.0 (`pip-list-torch-transformers.txt`,
`nvidia-smi.csv`). `_C` was built in an isolated worktree with
`RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=2`. Nothing in the branch was changed.
All strict runs use `--mode strict --offload --text-source pinned`, with no `--attention-query-chunk`.
The pinned text files matched their sha256 (`F0/text-sha-check.txt`).

| Item | Result | first_drift | `gate.hard_gate` | `transformer_intermediates_equal` | Decoded media equal (audio/video) | `case_id` / `bound` / `fell_back` | Peak GPU memory |
|---|---|---|---|---|---|---|---|
| F0 | `h3_full_attention True`, cc (9, 0), `_C` inside the worktree; attention tests 36 passed | n/a | n/a | n/a | n/a | n/a | n/a |
| F1 | chain tests **43 passed, 5 skipped** (as expected) | n/a | n/a | n/a | n/a | n/a | n/a |
| F2 FL2VA A vs B | as expected | null | pass | true | true / true | R/R, true, null | 66,495 MiB |
| F2 FL2VA A vs C (inject block 25, row 3000) | localized | denoise, step 1, `transformer_blocks.25`, row 3000 | fail | false | false / false | R/R, true, null | 66,495 MiB |
| F2 FL2VA query-chunk vs strict build | informational | denoise, step 0, `token_refiner` | fail | false | false / false | n/a | n/a |
| F3 Ref2VA A vs B | as expected | null | pass | true | true / true | R/R, true, null | A 90,707 MiB; B and C 76,747 MiB each |
| F3 Ref2VA A vs C (inject block 25, row 20000) | localized | denoise, step 1, `transformer_blocks.25`, row 20000 | fail | false | false / false | R/R, true, null | same as above |
| F3 Ref2VA query-chunk vs strict build | informational | denoise, step 0, `token_refiner` | fail | false | false / false | n/a | n/a |

## Details

- **Attention backend.** In every `*-report.json` the key `actual_backends.attention_h3` is
  `rl_engine.backends.cuda.attention.h3_full_attention.H3FullAttentionOp`. The key `actual_backends.attention`
  still reads `DeterministicAttentionOp`: it is the interim op that `CandidateOps` always creates (the handoff
  expected `attention` to end in `H3FullAttentionOp`; the H3 op is under `attention_h3`, and the code in
  `h3_block._cand_attention` uses it first when present). There is no query-chunk wrapper in these runs.
- **Boundary bindings:** 8 `row` entries in each A, B and C report.
- **F2 A vs C.** `differing_rows = 1`, target video, latent frame 18, patch (6, 3),
  `position_thw = [359.0, 22.047..., 3.905...]`. As in round 1, the one-ULP change reaches the velocities
  (8 of 12 trajectory entries differ), so the hard gate fails.
- **F3 A vs C.** `differing_rows = 1`, role `condition`, `condition_row = 11683`. Again 8 of 12 trajectory entries
  differ, as in round 1.
- **Query-chunk against strict build (informational).** The first drift is at denoise step 0 in `token_refiner`
  for both workflows, with all 12 trajectory entries differing for Ref2VA. The handoff expected the first block;
  the first differing module reported is `token_refiner`, before the transformer blocks.
- **Memory and time.** FL2VA: 66.5 GB peak, `chain_s` 184 s (A), against 325.7 s in the query-chunk round.
  Ref2VA: 90.7 GB peak for A (which keeps and captures tensors), against 95.3 GB (card full) in the query-chunk round;
  `chain_s` 1214 s (A), against 2527 s. B and C of Ref2VA ran at the same time on two cards, so their times
  (1087 s and 1052 s) are not comparable. FL2VA ran alone on one card while Ref2VA A ran on another node.
  `--offload` was enough; `--stream-blocks` was not needed.
- **Pillow (F1).** Pillow 13.0.0 (`F1/pillow.txt`). The probe prints
  `417bac1c42903f0f9b840f6e288b32eaf21f6aed7b9d260456ea00780d42a3e0`, not the sealing host's
  `47b12eaffdd9d3837c8f437521efe6b46d3457fb2ee314cff02326b24914422f`. Run 1's `media_preprocess` test
  failures are gone at this head (43 passed).
- **Samplers (F4).** `mem/*.csv` holds `unix_time,MiB` lines every 2 s for the card each item used
  (`F2-fl2va` for the FL2VA runs; `F3-ref2va-A`, `-B`, `-C` for the Ref2VA runs). The Ref2VA B and C peaks are equal.
- **Process notes.** A first launch of the Ref2VA driver hung on its own memory samplers before starting any run
  and was stopped and restarted; no result comes from it. The Ref2VA compares were run by hand after the driver's
  final `wait` blocked on the sampler loops; the commands are the same as the handoff's.
- Host names, paths and the job id are replaced with `{...}` placeholders; hashes are untouched. No `*.safetensors`.
