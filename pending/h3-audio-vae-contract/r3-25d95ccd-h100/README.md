# `audio_vae_contract` (#722) on H100 at `25d95ccd` (2026-11-22)

The review round's H100 rerun, which replaces the earlier H100 evidence (`a1177ddd`). It ran from the handoff `handoff/h100-h3-audio-vae-r2-20261122` (`cb058e1c`) in a clean detached checkout of `25d95ccd`, on one H100 (cc 9.0), torch 2.13.0+cu130. Redacted: host names are `node-a`.

| File | What |
|---|---|
| `h3-audio-vae-h100.json`, `h3-audio-vae-h100-rerun.json` | the benchmark (`--repeats 10 --plot`) and its rerun in a second process (`--repeats 0`); 28 hashes equal between them |
| `h3-audio-vae-h100-{bi,accuracy,coldstart,latency}.png`, `tables-h100.md` | the charts and their tables |
| `pytest-h100.txt`, `pytest-audio-vae-h100.xml` | `test_h3_audio_vae.py`: 60 passed |
| `SUMMARY.md`, `check.txt` | the H100 side's verdict for each expected item, and its checks |
| `commit.txt`, `env.txt`, `pip-list-filtered-E1.txt`, `run-a.txt`, `run-b.txt` | provenance and the run logs |
