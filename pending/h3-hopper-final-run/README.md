# H3 final-run PRs on Hopper (H100), 2026-11-22

H100 results for #730, #731, #732, #733 and #735 (RFC #420 `final_distributed_h3_run`). Two rounds, each run from a handoff branch with clean, unedited trees (`commit.txt` and tree status in every step). All files are redacted: home, data and environment paths, user, host, cluster and job identifiers are replaced by placeholders; the binary call captures of S2 are left out (their digests are in `engine.json`).

| Path | Code | What |
|---|---|---|
| `round1/strict/S1/` | #732 `6c7e424c`, #733 `4e67d5d7` | the full model with strict boundaries (SM90 video projections) on 2, 4 and 2 x 4 H100: 19 grid runs against `Grid(1, 1)`, 4 injected-error controls, the suites |
| `round1/strict/S2/` | #735 `4a4fe0c0` + #734 `dcc8535a` | vLLM-Omni TP4 rollout with strict boundaries on 4 H100: `engine.json`, `canonical.json`, `actor.json` (B random and B = 0), the omni tests |
| `round1/strict/SUMMARY.md` | | the H100 side's summary of S1 / S2 with every deviation |
| `round1/pra/` | #731 `b446400f` on #730 `02b123e0` | A: timing, norm partitions; A2: #730's row-mean bytes against the B200 reference; B: verl-omni comparisons; C: TP2 / TP4 / TP8 / TP4 x CP2 norm; D: suites; `SUMMARY.md` |
| `round2/H4-731-735/` | #731 `f67d96d1`, #735 `f3dbae37` (+ #734) | the PRs' tests at their round-2 heads (verl-omni compare, log-prob batch, transition, omni CPU / GPU) |
| `round2/H5-733/` | #733 `0b5b87c3` | the two ring-attention suite tests that round 1 ran without weights |
