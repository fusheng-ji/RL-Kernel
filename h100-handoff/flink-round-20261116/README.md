# H100 handoff: Flink's 2026-11-15 asks on the Qwen3-Next stack (2026-11-16)

Rules 1-7 of the 2026-11-14 handoff (fork branch `handoff/h100-evidence-gaps-20261114`, `h100-handoff/evidence-gaps-20261114/README.md`) ("Rules for every run") apply unchanged. Plan and the asks: `plan-2026-11-15.md` (this directory).

Files for step 0 (no repository code, no checkpoint): this directory (`probe.py`, `run_probe.sh`, `b200-reference/`).

## State of the code
**Update 2026-11-18:** all five heads are pushed (2026-11-17, fast-forwards) and are the PR heads; CI is green on each. Steps 1-3 run at these SHAs; re-check each with `gh pr view N --json headRefOid` before starting and test only the SHA the PR shows. The PR descriptions already say "H100 at this head: not run yet" and link the previous failed `preflight`; the results of this handoff replace that line.

| PR | PR head (pushed) | What changed since the last H100 run |
|---|---|---|
| #543 | `85675a74` | Hopper preflight from #557 (`a705d287`, here `af186dd2`, so no `2a8f18e4` patch any more), #503's fixed-order `combine_routes` backward (`e514d42f`), GPU-guard CPU test |
| #544 | `c49419ae` | merge of #543 |
| #549 | `e8195782` | merge of #544, on `b15de6e7` |
| #557 | `62487710` | merge of #549; EP4 gate differentiates through the router and `combine_routes` |
| #551 | `ea999fd4` | merge of #557 |

The base of the stack, #503, is at `8e33c453` (see step 4). #507 (`1fef29b0`) is unchanged.

The no-DDP replay actor (`hopper_recompute_no_ddp.patch`) is not ported (open question 1); until it is, a full-model `l2` on 80 GB needs it applied as a named, declared step.

## Step 0: why does `rollout-smoke` abort on H100? (about 30 min, can run now)
At #543 / #544 / #551 the chain `preflight` failed in `rollout-smoke` (pr_assets `7d4bd38e`, `hopper-qwen3-next/pr551/chain-2c699e3f`): no NVSwitch multicast, so the FlashInfer all-reduce workspace falls back from `mnnvl` to `trtllm`, then every TP rank aborts with `tvm::ffi::Error: TypeAttr __ffi_repr__ is already registered for type index 132`.

**On B200 this does not reproduce** (2026-11-16, same packages: vllm 0.30.0, flashinfer-python 0.6.18.post1, apache-tvm-ffi 0.1.11, torch 2.13.0+cu130). `probe.py` starts a TP4 engine exactly as `qwen3_next_rollout_smoke.py --non-strict-smoke` does, with a small model and dummy weights, a cold FlashInfer JIT cache, and generates 8 tokens. All six cases passed, including the H100 path forced with `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm` (with and without the fusion, with FlashInfer and Triton GDN prefill, with and without GDN layers): `b200-reference/summary.txt`.

Two things the B200 run showed:
- `fuse_allreduce_rms` off does **not** avoid the trtllm workspace: vLLM's standalone FlashInfer all-reduce creates the same workspace. Only `VLLM_ALLREDUCE_USE_FLASHINFER=0` keeps FlashInfer's all-reduce out entirely.
- B200 picked the FLASHINFER attention backend; H100 picked FLASH_ATTN (FA3). So the abort may come from a module only SM90 loads.

Run on 4 x H100, idle node:
```bash
run_probe.sh <rlkernel python> <Qwen3.5-0.8B dir> <Qwen2-1.5B-Instruct dir> <out dir>
```
Only `config.json` and the tokenizer of each model are needed (`load_format="dummy"`). The six cases:

| Case | Model | Setting | Tells us |
|---|---|---|---|
| `q35-auto` | Qwen3.5-0.8B (GDN) | auto backend (falls back to trtllm), fusion on | does a small GDN model reproduce it |
| `q35-auto-nofuse` | Qwen3.5-0.8B | fusion off | fusion alone is not the cause (expected: same as `q35-auto`) |
| `q35-no-fi-ar` | Qwen3.5-0.8B | `VLLM_ALLREDUCE_USE_FLASHINFER=0`, fusion off | does removing FlashInfer all-reduce avoid it |
| `q35-gdntriton` | Qwen3.5-0.8B | `gdn_prefill_backend=triton` | is the FlashInfer GDN prefill JIT module involved |
| `q2-auto` | Qwen2-1.5B (no GDN) | auto | does it need GDN at all |
| `q2-no-fi-ar` | Qwen2-1.5B | `VLLM_ALLREDUCE_USE_FLASHINFER=0` | control |

Then, only if `q35-no-fi-ar` passes and the 80B checkpoint is on the node, the real smoke without FlashInfer all-reduce (report only, not a chain result):
```bash
VLLM_ALLREDUCE_USE_FLASHINFER=0 python tools/validation/models/qwen3_next/qwen3_next_rollout_smoke.py \
  --model <ckpt> --output <out>/rollout-smoke-no-fi-ar.json --tensor-parallel-size 4 --non-strict-smoke
```
at the #551 head `2c699e3f` (the one that failed), detached and clean.

Hand back `header.txt` (it lists every `libtvm_ffi` copy found and whether a default FlashInfer cache exists), `summary.txt`, `steps.txt`, the per-case JSON, and the log of every failing case (redacted). The fix (a code change that turns FlashInfer all-reduce off where there is no multicast, or an environment change) is decided here afterwards (open question 3).

## Steps 1-4: at the pushed heads
Blocked on step 0 for everything that starts vLLM. The EP4 MoE gate in step 3 does not start vLLM and can run before step 0 is solved. With the same commands as the 2026-11-14 handoff, Priority 2 B:
1. #543: chain `preflight`, `layers`, `l2`; the chain report and gate exit code.
2. #544: `l2`, then `bi --bi-suite full` (97 cells).
3. #557: the EP4 gate (4 GPUs, layer-0 weights are enough); the EP4 chain `preflight`, `layers` (with `moe_ep`), `l2` (16 cases); optional `bi` full at EP4.
   - B200 reference for the EP4 gate at local `62487710` (2026-11-16): passed, 10 cases, bitwise on all four ranks, including the new `ep4-vs-ep1-router-combine-backward-{8,64,256,1024}`. This is a pre-push check, not PR evidence.
4. #503 at `8e33c453` (optional, 4 GPUs, no vLLM, no full checkpoint): the same 9 files as `hopper-qwen3-next/pr503/b200-8e33c453/` on `pr_assets` (`tests/models/qwen3_next/test_*.py` + `check_*.py`; command in that directory's `steps.txt`), with the Qwen3-Next SM90 build flags. The H100 runs of #503 so far are at `d5253a5c` and at the candidate tree `ce3f499d`, which is the tree of `e514d42f`; `8e33c453` only gates the empty-input bias gradient of `_SharedLinear.backward` and fixes a docstring. B200 at `8e33c453`: 323 passed, 0 failed, 0 skipped.

If time is short: step 0, then the step 3 EP4 gate, then steps 1, 2, the rest of 3, and 4.

The remaining items of the 2026-11-14 handoff that are outside this stack (#507, #495, #491, #536, #538, #547) are listed with their current heads in `h100-handoff/evidence-gaps-20261114/STATUS-2026-11-18.md` on fork branch `handoff/h100-evidence-gaps-20261114`.

Archive under `pr_assets` `hopper-qwen3-next/pr<N>/<head8>/`; draft one "H100" paragraph per PR. H200: no hardware (open question 2).
