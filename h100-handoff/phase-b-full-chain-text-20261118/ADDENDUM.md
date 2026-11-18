# Optional addendum to the `phase_b_full_chain` H100 handoff: the qwen3vl text source (2026-11-18)

**Nothing in the pushed handoff (`handoff/h3-h100-phase-b-full-chain-20261118` @ `7e22c736`)
changes.** It runs its own code with the pinned text features, and
the strict items stay as written. This addendum needs the newer branch head
`feat/h3-phase-b-full-chain` @ `c0e79f5d`, which merges `qwen3vl_layer50_contract` and adds the
`text-check` command. This branch, `handoff/h3-h100-phase-b-full-chain-text-20261118`, is that
commit plus one commit that adds this folder:

```bash
git fetch https://github.com/fusheng-ji/RL-Kernel handoff/h3-h100-phase-b-full-chain-text-20261118
git worktree add --detach <SRC> FETCH_HEAD
```

Use a worktree of its own, not the strict handoff's worktree. It is worth running only if the H100 side already has the Qwen3-VL
`text_encoder/` (about 66 GB; 50 of 64 layers are loaded) — e.g. from the qwen3vl handoff.

## Why

On B200, `qwen3vl_layer50_contract` run on CPU reproduces the pinned text features bit for bit;
run on CUDA it gives different (repeatable) bytes, mostly in the vision tokens (relative L2 0.038
FL2VA, 0.46 Ref2VA). Two open facts:

1. Does the contract on the H100 node's **CPU** also reproduce the pinned bytes (CPU portability
   of the pinned fixtures)?
2. What does the contract on **H100 CUDA** give, relative to the pinned bytes and to B200 CUDA?

## Run (one H100, about 60 GB free for the CUDA part)

```bash
export SRC=<worktree of this branch> PY=<python> PROVIDER=<diffusers f53d552>
export PYTHONPATH="$SRC:$PROVIDER/src" HF_HUB_OFFLINE=1
cd $SRC
for dev in cpu cuda; do for wf in fl2va ref2va; do
  $PY -W ignore tools/validation/models/h3_full_chain.py text-check --workflow $wf \
    --text-root <dir with text_encoder/ tokenizer/ processor/> --pinned <TEXT>/$wf-text.safetensors \
    --device $dev --report <OUT>/05/$wf-$dev.json
done; done
```

`<TEXT>` is the `h100-handoff/phase-b-full-chain-20261118/text/` folder of the strict handoff branch
(`handoff/h3-h100-phase-b-full-chain-20261118`): the same pinned files as the strict items (sha256 `f4e69532…` FL2VA,
`c32af751…` Ref2VA). The pinned files are only read.

## Expected / record

- `same_conditioner_input = true` and `tags_equal = true` in all four reports (otherwise the
  presentation differs: a finding).
- CPU: `bitwise_equal` — on B200's host it was `true` for both. A `false` here means CPU BF16
  results differ between the two hosts' CPUs; record `max_abs_diff`, `relative_l2`,
  `differing_text_tokens` / `differing_vision_tokens`.
- CUDA: expected `false`. Record the same fields. For comparison, B200 CUDA gave
  FL2VA relative L2 0.0380 / max |Δ| 832, Ref2VA 0.4615 / 16063, first differing token 0.

## Send back

`<OUT>/05/*.json`, as folder `05/` of the full-chain results branch
(`handoff/h3-h100-phase-b-full-chain-results-20261118`, `h100-results/phase-b-full-chain-20261118/05/`),
with host names and paths stripped. No PR and no comment.
