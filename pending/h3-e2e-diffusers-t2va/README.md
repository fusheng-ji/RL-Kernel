# RFC #420 `final_e2e_diffusers_cuda`: the recipe's T2VA workload on 1 x H100 (2026-11-23)

Code: `refs/h100/h3-e2e-integrated` `65ba0387`. That is the PR branch `feat/h3-e2e-diffusers-t2va` (`b4ba5cee`, on `test-h3` `3d137858`) with #741 (`0f2de999`) and #743 (`bcba86ca`) merged. Pinned Diffusers `f53d552`, model revision `42ed227`. Hosts, paths and environment names are redacted.

- `h100/E1/`: preflight, case 0 for both arms with 2 repeats each, the GPU tests, and the compare.
- `h100/E2/`: all 8 cases (4 prompts x seeds 42 / 43), both arms, 2 repeats each. `compare/report.json` is the full report. `runs/<arm>/case<N>/r<k>/` holds each run's `manifest.json` (provenance, operator trace, scheduler-row counts, time split, peak memory) and its sealed `trajectory.json`.
- `h100/SUMMARY.md`: the H100 operator's summary.
- `h3-e2e-t2va-drift-h100.png`: max |R - P| of each latent after each scheduler step, median and range over the 8 cases.
- `h3-e2e-t2va-time-h100.png`: the time per phase.
- `plot_e2e.py`: makes both charts from `h100/E2/compare/report.json`.
